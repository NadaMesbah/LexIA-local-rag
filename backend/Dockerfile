FROM python:3.11-slim

WORKDIR /app

# System deps:
#   gcc/g++  -> build native wheels if needed
#   curl     -> container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of torch first (smaller, no CUDA) so the reranker
# runs on CPU and leaves the GPU entirely for Ollama / the LLM.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user (defence in depth — relevant for the "bank" framing).
# Create the data dir and hand it to the app user BEFORE the volume mounts; a
# fresh named volume inherits this ownership so the app can write to it.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# No --reload in normal runs (it doubles RAM and reloads on volume writes).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
