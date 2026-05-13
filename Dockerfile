# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir "uv==0.6.14"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
RUN useradd --create-home --shell /bin/bash lexicon && chown -R lexicon:lexicon /app
USER lexicon
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
