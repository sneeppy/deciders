# Образ нужен как готовый PyTorch с CUDA. Сервер внутри свой, SGLang не запускается.
FROM lmsysorg/sglang:dev-clef

RUN python3 -m venv /opt/clef --system-site-packages \
    && /opt/clef/bin/pip install --no-deps "transformers==5.10.2" \
    && /opt/clef/bin/pip install "huggingface_hub>=0.26" "safetensors>=0.4" accelerate pillow

COPY clef /opt/jevs/clef

ENV PATH=/opt/clef/bin:$PATH
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/jevs
ENTRYPOINT ["python", "-m", "clef"]
CMD ["serve"]
