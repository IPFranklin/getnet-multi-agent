# ---------- builder: install dependencies into a virtualenv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optionally bake the multilingual embedding model into the image (no download at runtime).
ARG PRELOAD_EMBEDDINGS=true
ENV FASTEMBED_CACHE_DIR=/opt/models
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/opt/models')"; \
    else mkdir -p /opt/models; fi

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_DIR=/opt/models EMBEDDING_BACKEND=fastembed
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models
COPY --chown=app:app app ./app
COPY --chown=app:app data ./data
COPY --chown=app:app evals ./evals
COPY --chown=app:app tests ./tests
COPY --chown=app:app pytest.ini ./
RUN mkdir -p /app/data/index /app/data/runtime && chown -R app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
