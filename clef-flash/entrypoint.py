import os
import sys
from pathlib import Path

CONFIG = Path(os.environ.get("CLEF_CONFIG", "/opt/jevs/clef-flash/config.yaml"))
DOWNLOAD_DIR = os.environ.get("HF_HUB_CACHE", "/cache/huggingface/hub")


def load_config(path):
    cfg = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            sys.exit(f"не понял строку в {path}: {raw}")
        key, value = line.split(":", 1)
        cfg[key.strip()] = parse_scalar(value.strip())
    return cfg


def parse_scalar(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value == "" or value.lower() in {"null", "~"}:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def require_int(cfg, key):
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        sys.exit(f"{key} должен быть целым числом больше нуля")
    return value


def require_port():
    raw = os.environ.get("PORT", "").strip()
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        sys.exit("PORT должен быть целым числом от 1 до 65535 и задаётся в .env")
    return raw


def main():
    port = require_port()
    api_key = os.environ.get("CLEF_API_KEY", "").strip()
    if not api_key:
        sys.exit("CLEF_API_KEY пустой. Задайте ключ в .env")
    if not CONFIG.is_file():
        sys.exit(f"нет файла конфигурации: {CONFIG}")

    cfg = load_config(CONFIG)
    model = str(cfg.get("model") or "").strip()
    if not model:
        sys.exit("в config.yaml пустое поле model")

    args = [
        "sglang", "serve",
        "--model-path", model,
        "--host", "0.0.0.0",
        "--port", port,
        "--download-dir", DOWNLOAD_DIR,
        "--context-length", str(require_int(cfg, "context_length")),
        "--max-running-requests", str(require_int(cfg, "max_running_requests")),
        "--api-key", api_key,
    ]

    fraction = cfg.get("mem_fraction_static")
    if fraction is not None:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            sys.exit("mem_fraction_static должен быть числом от 0 до 1")
        if not 0 < float(fraction) < 1:
            sys.exit("mem_fraction_static должен быть между 0 и 1")
        args += ["--mem-fraction-static", str(fraction)]

    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
