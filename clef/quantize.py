"""Собирает 4-битный Clef с поворотом Адамара.

Качает Cloudflare/clef в кэш Hugging Face и пишет упакованные веса в QUANT_DIR.
Поворот и квантование считаются на GPU, на диск уходит уже упакованный тензор.
Знаки строятся на CPU: тот же вектор заново собирается при загрузке модели.

Запуск:

    docker compose run --rm clef quantize
"""

import gc
import json
import os
import shutil
import sys
from pathlib import Path

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
    def __init__(self, root, device):
        self.root = root
        self.device = device
        self.path = None
        self.handle = None

    def get(self, shard, name):
        path = self.root / shard
        if self.path != path:
            if self.handle is not None:
                self.handle.close()
            self.handle = safe_open(path, framework="pt", device=self.device)
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
        return {"complete": False, "source": SOURCE, "device": "cuda", "tensors": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def to_cpu(tensor):
    return tensor.detach().to(device="cpu").contiguous()


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def quantize_int8(weight):
    rows, hidden = weight.shape
    group = group_size(hidden)
    if group is None:
        return None
    grouped = weight.to(dtype=torch.float32).reshape(rows, hidden // group, group)
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 127
    codes = torch.round(grouped / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return {
        "codes": to_cpu(codes.reshape(rows, hidden)),
        "scales": to_cpu(scale.to(dtype=torch.float16)),
    }, group


def quantize_hadamard(name, weight):
    out_features, width = weight.shape
    block = block_size(width)
    group = group_size(width)
    if block is None or group is None or width % 2 != 0:
        return None
    signs = sign_vector(name, width).to(device=weight.device, dtype=torch.float32)
    rotated = block_hadamard(weight.to(dtype=torch.float32) * signs, block)
    grouped = rotated.reshape(out_features, width // group, group)
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 7
    codes = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7)
    nibbles = (codes + 7).to(dtype=torch.uint8).reshape(out_features, width)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return {"packed": to_cpu(packed), "scales": to_cpu(scale.to(dtype=torch.float16))}, block, group


def tensor_file(dest, name):
    return dest / "tensors" / f"{name}.safetensors"


def main():
    print(f"скачиваю {SOURCE}", flush=True)
    snapshot = Path(snapshot_download(SOURCE))
    copy_sidecars(snapshot, DEST)
    (DEST / "tensors").mkdir(parents=True, exist_ok=True)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA не видна. Квантование считается на видеокарте.")
    manifest_path = DEST / "manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest["tensors"] and not manifest.get("complete") and manifest.get("device") != "cuda":
        raise SystemExit("квантование уже начато на CPU. Очистите каталог весов и запустите quantize заново.")
    if not (manifest.get("complete") and manifest["tensors"] and manifest.get("device") != "cuda"):
        manifest["device"] = "cuda"
    reader = ShardReader(snapshot, "cuda")
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
            with torch.inference_mode():
                if name in LOOKUP_WEIGHTS:
                    quantized = quantize_int8(tensor)
                    if quantized is None:
                        payload = {"tensor": to_cpu(tensor)}
                        kind = "bf16"
                        extra = {}
                    else:
                        payload, group = quantized
                        kind = "int8_rows"
                        extra = {"group": group, "shape": list(tensor.shape)}
                elif name.endswith("pos_embed.weight") or tensor.ndim != 2:
                    payload = {"tensor": to_cpu(tensor)}
                    kind = "bf16"
                    extra = {}
                else:
                    quantized = quantize_hadamard(name, tensor)
                    if quantized is None:
                        payload = {"tensor": to_cpu(tensor)}
                        kind = "bf16"
                        extra = {}
                    else:
                        payload, block, group = quantized
                        kind = "hadamard_int4"
                        extra = {"block": block, "group": group, "shape": list(tensor.shape)}
                        bias_name = name[: -len("weight")] + "bias"
                        if bias_name in weight_map:
                            bundled_bias = bias_name
                            payload["bias"] = to_cpu(reader.get(weight_map[bias_name], bias_name))
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
