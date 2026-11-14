# База — официальный образ SGLang для Clef. В нём уже есть torch с CUDA,
# transformers, huggingface_hub, safetensors, pillow и остальной стек.
# Не переустанавливаем его: pip иначе тянет чужой torch с PyPI.
# Карточка Cloudflare/clef проверяла transformers 5.10.2; образ SGLang
# отдаёт Clef на своём (более новом) transformers — так и задумано в cookbook.
FROM lmsysorg/sglang:dev-clef

# accelerate — init_empty_weights при загрузке квантованных весов.
# torchvision — AutoProcessor поднимает видеокласс даже без videos в запросе.
# Ставим только если в образе нет, и только --no-deps под уже стоящий torch.
RUN set -eux; \
    python3 -c "import accelerate" \
      || python3 -m pip install --no-deps "accelerate>=1.1.0"; \
    if ! python3 -c "import torchvision"; then \
      TV="$(python3 -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"0.{int(v[1])+15}.{v[2]}")')"; \
      CU="$(python3 -c 'import torch; c=torch.version.cuda or ""; print(("cu"+c.replace(".","")) if c else "")')"; \
      echo "install torchvision==${TV} index=${CU}"; \
      python3 -m pip install --no-deps "torchvision==${TV}" --index-url "https://download.pytorch.org/whl/${CU}"; \
    fi

COPY clef /opt/jevs/clef

ENV PYTHONUNBUFFERED=1
WORKDIR /opt/jevs
ENTRYPOINT ["python3", "-m", "clef"]
CMD ["serve"]
