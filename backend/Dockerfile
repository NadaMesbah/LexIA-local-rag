FROM python:3.11-slim

WORKDIR /app

# System deps:
#   gcc/g++  -> build native wheels if needed
#   curl     -> container healthcheck
#   gosu     -> drop from root to the app user after fixing volume ownership
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl gosu \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU build of torch first (smaller, no CUDA) so the reranker
# runs on CPU and leaves the GPU entirely for Ollama / the LLM.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create the non-root app user (defence in depth — relevant for the "bank"
# framing). The actual server runs as this user; see the entrypoint below.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

EXPOSE 8000

# /app/data is a mounted named volume. A volume first created by an older,
# root-running image stays root-owned, which would make SQLite/Chroma read-only
# for the non-root user. So we start as root, fix ownership of the data dir,
# then drop to `appuser` via gosu for the actual server process. Inlined (no
# separate .sh file) to avoid Windows CRLF / exec-bit issues on checkout.
ENTRYPOINT ["/bin/sh", "-c", "chown -R appuser:appuser /app/data 2>/dev/null || true; exec gosu appuser \"$@\"", "--"]

# No --reload in normal runs (it doubles RAM and reloads on volume writes).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]