FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV COMPETITION_ATTACHMENTS_ROOT=data/competition/deploy/attachments
WORKDIR /app


# ============================================================
# Install uv
# ============================================================

RUN pip install --no-cache-dir uv


# ============================================================
# Install Python dependencies first for Docker cache
# ============================================================

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --no-dev


# ============================================================
# Copy application and competition runtime artifacts
# ============================================================

COPY app ./app
COPY main.py ./main.py

COPY data/competition/deploy/attachments \
    ./data/competition/deploy/attachments

COPY data/competition/processed/corpora/competition_corpus_8cbec682dfce4962 \
    ./data/competition/processed/corpora/competition_corpus_8cbec682dfce4962

COPY data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5 \
    ./data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5


# ============================================================
# Runtime directories
# ============================================================

RUN mkdir -p \
    /app/data/runtime/checkpoints


EXPOSE 8000


CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]