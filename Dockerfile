FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt /app/
COPY src /app/src
COPY configs /app/configs

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/artifacts && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["edge-agent"]
CMD ["run", "--config", "configs/example_config.json"]
