# Образ нужен как готовый PyTorch с CUDA. Сервер внутри свой, SGLang не запускается.
FROM lmsysorg/sglang:dev-clef

# transformers без зависимостей, чтобы pip не заменил CUDA-сборку PyTorch.
# Версии ниже — минимум, который transformers 5.10.2 проверяет при импорте.
RUN python3 -m venv /opt/clef --system-site-packages \
    && /opt/clef/bin/pip install --no-deps "transformers==5.10.2" \
    && /opt/clef/bin/pip install \
        "huggingface_hub>=1.5.0,<2.0" \
        "safetensors>=0.4.3" \
        "accelerate>=1.1.0" \
        pillow \
        "regex>=2025.10.22" \
        "tokenizers>=0.22.0,<=0.23.0" \
        "pyyaml>=5.1" \
        "tqdm>=4.27" \
        "packaging>=20.0" \
        filelock \
        "numpy>=1.17" \
        requests \
    && TV="$(/opt/clef/bin/python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"0.{int(v[1])+15}.{v[2]}")')" \
    && CU="$(/opt/clef/bin/python -c 'import torch; c=torch.version.cuda or ""; print(("cu"+c.replace(".","")) if c else "")')" \
    && /opt/clef/bin/pip install --no-deps "torchvision==${TV}" --index-url "https://download.pytorch.org/whl/${CU}"

COPY clef /opt/jevs/clef

ENV PATH=/opt/clef/bin:$PATH
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/jevs
ENTRYPOINT ["python", "-m", "clef"]
CMD ["serve"]
