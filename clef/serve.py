"""System One для квантованного Clef.

Грузит manifest из QUANT_DIR, матрицы слоёв читает как int4 после поворота Адамара.
Голова решений остаётся BF16. lm_head и входные эмбеддинги — int8 по строкам:
голова индексирует их как таблицу токенов, а не как одну большую матрицу.
"""

import base64
import io
import json
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "0")

import torch

from .hadamard import sign_vector
from .layers import HadamardInt4, Int8Rows

QUANT_DIR = Path(os.environ.get("QUANT_DIR", "/models/clef"))
MAX_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "65536"))


def require_port():
    raw = os.environ.get("PORT", "").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise SystemExit("PORT должен быть целым числом от 1 до 65535 и задаётся в .env")
    return int(raw)


def api_key():
    key = os.environ.get("CLEF_API_KEY", "").strip()
    if not key:
        raise SystemExit("CLEF_API_KEY пустой. Задайте ключ в .env")
    return key


def confidence(probabilities):
    count = len(probabilities)
    if count < 2:
        return 1.0
    top = max(probabilities.values())
    uniform = 1.0 / count
    return (top - uniform) / (1.0 - uniform)


def systemone_answer(question, probabilities):
    rounded = {key: round(value, 6) for key, value in probabilities.items()}
    if question["type"] == "noul":
        return {"type": "noul", "noul": rounded["true"]}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=rounded.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(confidence(rounded), 6),
            "probabilities": {option: rounded[option] for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * rounded[level] for index, level in enumerate(levels)), 6),
        "confidence": round(confidence(rounded), 6),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: rounded[level] for level in levels},
    }


def load_image(value):
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError("картинка должна быть data URL или http(s)-ссылкой")
    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        raw = base64.b64decode(payload)
    elif value.startswith("http://") or value.startswith("https://"):
        with urllib.request.urlopen(value, timeout=20) as response:
            raw = response.read(20 * 1024 * 1024 + 1)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError("картинка больше 20 МБ")
    else:
        raise ValueError("картинка должна быть data URL или http(s)-ссылкой")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def materialize_rope(module, device):
    for child in module.modules():
        inv_freq = getattr(child, "inv_freq", None)
        if inv_freq is None or not getattr(inv_freq, "is_meta", False):
            continue
        if getattr(child, "rope_type", "") == "axial" and hasattr(child, "compute_axial_rope_parameters"):
            compute = child.compute_axial_rope_parameters
        elif hasattr(child, "compute_default_rope_parameters"):
            compute = child.compute_default_rope_parameters
        else:
            continue
        rope_type = getattr(child, "rope_type", "default")
        if rope_type not in ("default", "axial"):
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            frequencies, scale = ROPE_INIT_FUNCTIONS[rope_type](child.config, device=device)
        else:
            frequencies, scale = compute(child.config, device=device)
        child.register_buffer("inv_freq", frequencies, persistent=False)
        child.register_buffer("original_inv_freq", frequencies.clone(), persistent=False)
        child.attention_scaling = scale


def assign_parameter(module, name, tensor):
    parent_name, _, attr = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    parameter = torch.nn.Parameter(tensor, requires_grad=False)
    setattr(parent, attr, parameter)


def load_model(schema):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    manifest = json.loads((QUANT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise SystemExit(f"квантование не закончено: {QUANT_DIR / 'manifest.json'}")
    config = AutoConfig.from_pretrained(QUANT_DIR)
    for part in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if part is not None:
            part._attn_implementation = "sdpa"
    with init_empty_weights():
        backbone = Qwen3_5ForConditionalGeneration(config)
    backbone.config.use_cache = False
    backbone.eval()
    device = torch.device("cuda")
    total = len(manifest["tensors"])
    for number, (name, entry) in enumerate(manifest["tensors"].items(), start=1):
        if entry["kind"] == "with_weight":
            continue
        if number == 1 or number % 50 == 0:
            print(f"читаю {number}/{total} {name}", flush=True)
        payload = load_file(QUANT_DIR / entry["file"], device="cpu")
        module_name = name.removesuffix(".weight")
        if entry["kind"] == "hadamard_int4":
            bias = payload.get("bias")
            layer = HadamardInt4(
                payload["packed"],
                payload["scales"],
                sign_vector(name, entry["shape"][1]),
                None if bias is None else bias.to(device=device, dtype=torch.bfloat16),
                entry["block"],
                entry["group"],
            )
            parent_name, _, attr = module_name.rpartition(".")
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            setattr(parent, attr, layer.to(device))
        elif entry["kind"] == "int8_rows":
            layer = Int8Rows(payload["codes"], payload["scales"], entry["group"]).to(device)
            parent_name, _, attr = module_name.rpartition(".")
            parent = backbone.get_submodule(parent_name) if parent_name else backbone
            setattr(parent, attr, layer)
        elif entry["kind"] == "bf16":
            assign_parameter(backbone, name, payload["tensor"].to(device))
        else:
            raise SystemExit(f"неизвестный вид тензора {entry['kind']} у {name}")
        del payload
        if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
            print(f"загружен {name}", flush=True)
    materialize_rope(backbone, device)
    missing = [
        name
        for name, tensor in list(backbone.named_parameters()) + list(backbone.named_buffers())
        if tensor.is_meta
    ]
    if missing:
        raise SystemExit("на meta-устройстве остались: " + ", ".join(missing[:12]))
    head_config = json.loads((QUANT_DIR / "joint_head_config.json").read_text(encoding="utf-8"))
    head = schema.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(QUANT_DIR / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(QUANT_DIR)
    model = schema.ClefModel(backbone, head).eval()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    print(f"модель на GPU, занято {allocated:.1f} ГБ", flush=True)
    return model, processor


def build_handler(model, processor, schema, key):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

        def _send(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {key}"

        def do_GET(self):
            if self.path.rstrip("/") != "/health":
                self._send(404, {"error": "нет такого пути"})
                return
            if not self._authorized() and self.headers.get("Authorization"):
                self._send(401, {"error": "неверный ключ"})
                return
            self._send(200, {"status": "ok"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "нет такого пути"})
                return
            if not self._authorized():
                self._send(401, {"error": "нужен заголовок Authorization: Bearer"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64 * 1024 * 1024:
                self._send(413, {"error": "тело запроса больше 64 МБ"})
                return
            try:
                request = json.loads(self.rfile.read(length))
                response = answer(model, processor, schema, request)
            except ValueError as error:
                self._send(400, {"error": str(error)})
                return
            except Exception as error:
                self._send(500, {"error": str(error)})
                return
            self._send(200, response)

    return Handler


def answer(model, processor, schema, request):
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("нужны поля model и state")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("нужен хотя бы один вопрос")
    for question_id, question in questions.items():
        if question.get("type") not in schema.QUESTION_TYPES:
            raise ValueError(f"{question_id}: type должен быть noul, choice или score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria не должен быть пустым")
    if request.get("videos"):
        raise ValueError("этот сервер принимает images, но не videos")
    images = request.get("images")
    if images:
        request = dict(request)
        request["images"] = [load_image(image) for image in images]
    encoded = schema.encode_record(
        processor.tokenizer,
        request,
        max_length=MAX_LENGTH,
        processor=processor,
    )
    device = next(model.head.parameters()).device
    with torch.inference_mode():
        logits = model(schema.collate_records([encoded], processor.tokenizer.pad_token_id, device))[0]
    answers = {}
    for question, question_logits in zip(encoded.questions, logits):
        probabilities = dict(
            zip(question.option_ids, question_logits.float().softmax(-1).tolist())
        )
        answers[question.question_id] = systemone_answer(questions[question.question_id], probabilities)
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }


def main():
    port = require_port()
    key = api_key()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA не видна. Полный Clef даже в int4 считается на видеокарте.")
    sys.path.insert(0, str(QUANT_DIR))
    import joint_schema_model

    print(f"гружу Clef из {QUANT_DIR}", flush=True)
    model, processor = load_model(joint_schema_model)
    handler = build_handler(model, processor, joint_schema_model, key)
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"готов, слушаю 0.0.0.0:{port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
