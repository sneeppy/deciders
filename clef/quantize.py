"""Собирает 4-битный Clef с поворотом Адамара.

Качает Cloudflare/clef в кэш Hugging Face и пишет упакованные веса в QUANT_DIR.
Сборка идёт на CPU: процесс прячет CUDA, видеокарту не занимает.
Пиковое потребление памяти — один тензор, на этом сервере с 128 ГБ это помещается.

Запуск:

    docker compose run --rm clef quantize
"""

import gc
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from .hadamard import block_hadamard, block_size, group_size, sign_vector

SOURCE = os.environ.get("CLEF_SOURCE", "Cloudflare/clef")
DEST = Path(os.environ.get("QUANT_DIR", "/models/clef"))
LOOKUP_WEIGHTS = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
}


class ShardReader:
    def __init__(self, root):
        self.root = root
        self.path = None
        self.handle = None

    def get(self, shard, name):
        path = self.root / shard
        if self.path != path:
            if self.handle is not None:
                self.handle.close()
            self.handle = safe_open(path, framework="pt", device="cpu")
            self.path = path
        return self.handle.get_tensor(name)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.path = None


def copy_sidecars(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.name.startswith("model-") and item.name.endswith(".safetensors"):
            continue
        target = dest / item.name
        if target.is_file() and target.stat().st_size == item.stat().st_size:
            continue
        shutil.copy2(item, target, follow_symlinks=True)


def load_manifest(path):
    if not path.is_file():
        return {"complete": False, "source": SOURCE, "tensors": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def quantize_int8(weight):
    rows, hidden = weight.shape
    group = group_size(hidden)
    if group is None:
        return None
    codes = torch.empty(rows, hidden, dtype=torch.int8)
    scales = torch.empty(rows, hidden // group, dtype=torch.float16)
    for start in range(0, rows, 2048):
        chunk = weight[start : start + 2048].float().reshape(-1, hidden // group, group)
        scale = chunk.abs().amax(dim=-1).clamp(min=1e-8) / 127
        quantized = torch.round(chunk / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        codes[start : start + 2048] = quantized.reshape(-1, hidden)
        scales[start : start + 2048] = scale.to(torch.float16)
    return {"codes": codes, "scales": scales}, group


def quantize_hadamard(name, weight):
    out_features, width = weight.shape
    block = block_size(width)
    group = group_size(width)
    if block is None or group is None or width % 2 != 0:
        return None
    signs = sign_vector(name, width).to(torch.float32)
    packed = torch.empty(out_features, width // 2, dtype=torch.uint8)
    scales = torch.empty(out_features, width // group, dtype=torch.float16)
    for start in range(0, out_features, 2048):
        rotated = block_hadamard(weight[start : start + 2048].float() * signs, block)
        grouped = rotated.reshape(-1, width // group, group)
        scale = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 7
        codes = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7)
        nibbles = (codes + 7).to(torch.uint8).reshape(-1, width)
        packed[start : start + rotated.shape[0]] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        scales[start : start + rotated.shape[0]] = scale.to(torch.float16)
        del rotated, grouped, codes, nibbles
    return {"packed": packed, "scales": scales}, block, group


def tensor_file(dest, name):
    return dest / "tensors" / f"{name}.safetensors"


def main():
    print(f"скачиваю {SOURCE}", flush=True)
    snapshot = Path(snapshot_download(SOURCE))
    copy_sidecars(snapshot, DEST)
    (DEST / "tensors").mkdir(parents=True, exist_ok=True)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    manifest_path = DEST / "manifest.json"
    manifest = load_manifest(manifest_path)
    reader = ShardReader(snapshot)
    done = 0
    try:
        names = [name for name in weight_map if not name.endswith(".bias")]
        names += [name for name in weight_map if name.endswith(".bias")]
        for number, name in enumerate(names, start=1):
            entry = manifest["tensors"].get(name)
            if entry and (DEST / entry["file"]).is_file():
                done += 1
                continue
            if name.endswith(".bias"):
                weight_name = name[: -len("bias")] + "weight"
                weight_entry = manifest["tensors"].get(weight_name)
                if weight_entry and weight_entry["kind"] == "hadamard_int4":
                    manifest["tensors"][name] = {"kind": "with_weight", "file": weight_entry["file"]}
                    write_manifest(manifest_path, manifest)
                    done += 1
                    continue
            tensor = reader.get(weight_map[name], name)
            path = tensor_file(DEST, name)
            bundled_bias = None
            if name in LOOKUP_WEIGHTS:
                quantized = quantize_int8(tensor)
                if quantized is None:
                    payload = {"tensor": tensor.contiguous()}
                    kind = "bf16"
                    extra = {}
                else:
                    payload, group = quantized
                    kind = "int8_rows"
                    extra = {"group": group, "shape": list(tensor.shape)}
            elif name.endswith("pos_embed.weight") or tensor.ndim != 2:
                payload = {"tensor": tensor.contiguous()}
                kind = "bf16"
                extra = {}
            else:
                quantized = quantize_hadamard(name, tensor)
                if quantized is None:
                    payload = {"tensor": tensor.contiguous()}
                    kind = "bf16"
                    extra = {}
                else:
                    payload, block, group = quantized
                    kind = "hadamard_int4"
                    extra = {"block": block, "group": group, "shape": list(tensor.shape)}
                    bias_name = name[: -len("weight")] + "bias"
                    if bias_name in weight_map:
                        bundled_bias = bias_name
                        payload["bias"] = reader.get(weight_map[bias_name], bias_name).contiguous()
            save_file(payload, path)
            relative = path.relative_to(DEST).as_posix()
            manifest["tensors"][name] = {"kind": kind, "file": relative, **extra}
            if bundled_bias is not None:
                manifest["tensors"][bundled_bias] = {"kind": "with_weight", "file": relative}
            manifest["complete"] = False
            write_manifest(manifest_path, manifest)
            done += 1
            print(f"{number}/{len(names)} {kind} {tuple(tensor.shape)} {name}", flush=True)
            del tensor, payload
            gc.collect()
    finally:
        reader.close()
    missing = [name for name in weight_map if name not in manifest["tensors"]]
    if missing:
        raise SystemExit(f"не записаны тензоры: {missing[:5]}")
    manifest["complete"] = True
    manifest["source"] = SOURCE
    write_manifest(manifest_path, manifest)
    print(f"готово: {done} тензоров в {DEST}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
