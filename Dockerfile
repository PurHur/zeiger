# Zeiger inference API.
#   CPU   : docker compose up zeiger
#   ROCm  : docker compose --profile rocm up zeiger-rocm
# The torch wheel is chosen at build time; nothing else differs between the two.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.10.0

WORKDIR /app
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==${TORCH_VERSION}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY zeiger/ ./zeiger/
COPY serve.py bench.py test_zeiger.py ./

ENV ZEIGER_MODEL=/models/zeiger-0.6b \
    ZEIGER_PORT=8173 \
    ZEIGER_DEVICE=auto \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1
EXPOSE 8173
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if 'ok' in urllib.request.urlopen('http://127.0.0.1:8173/', timeout=8).read().decode() else 1)"
CMD ["python", "serve.py"]
