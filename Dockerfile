FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data/kb ./data/kb
COPY data/eval ./data/eval

# Train at build time so the image is self-contained, and pre-download the embedding model.
RUN python scripts/prepare_data.py && python scripts/train_classifier.py \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && rm -rf data/raw data/processed

RUN useradd --create-home app && mkdir -p /app/var && chown -R app /app
USER app
ENV SP_DB_PATH=/app/var/supportpilot.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "supportpilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
