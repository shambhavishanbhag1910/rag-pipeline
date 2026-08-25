FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && \
    python -m pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ARG PRELOAD_EMBEDDING_MODEL=true
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 appuser && \
    mkdir -p /opt/models /app/artifacts/evaluations && \
    chown -R appuser:appuser /opt/models /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip && \
    python -m pip install /wheels/* && \
    rm -rf /wheels

COPY --chown=appuser:appuser . .

RUN if [ "$PRELOAD_EMBEDDING_MODEL" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"; \
    fi && \
    chown -R appuser:appuser /opt/models /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
