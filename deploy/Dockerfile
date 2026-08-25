FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# ============================================================
# Install Competition Runtime dependencies
# ============================================================

COPY deploy/requirements-competition.txt \
    /app/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r /app/requirements.txt


# ============================================================
# Copy application
# ============================================================

COPY app /app/app
COPY main.py /app/main.py


# ============================================================
# Copy Competition Runtime data
# ============================================================

COPY data/competition/deploy/attachments \
    /app/data/competition/deploy/attachments

COPY data/competition/processed/corpora/competition_corpus_8cbec682dfce4962 \
    /app/data/competition/processed/corpora/competition_corpus_8cbec682dfce4962

COPY data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5 \
    /app/data/competition/processed/indexes/bm25/competition_bm25_index_07bc5262330bc2a5


# ============================================================
# Runtime directories
# ============================================================

RUN mkdir -p \
    /app/data/runtime/checkpoints


# ============================================================
# Service
# ============================================================

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]