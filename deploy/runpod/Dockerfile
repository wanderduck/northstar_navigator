# NorthStar Navigator — RunPod Deployment Dockerfile
#
# Backup/reproducibility image for RunPod GPU pods.
# Primary deployment uses bare CUDA + setup.sh scripts;
# this Dockerfile exists for custom image builds and CI/CD.
#
# Build (from project root):
#   docker build --platform=linux/amd64 -t wanderduck/northstar-navigator:v0.1.0 -f deploy/runpod/Dockerfile .
#
# IMPORTANT:
#   - Always build with --platform=linux/amd64 (RunPod runs x86_64 Linux)
#   - Always use a versioned tag (e.g. v0.1.0), never :latest
#   - GGUF model weights are NOT baked in (5.3GB) — downloaded at runtime from HF Hub

# CUDA 12.1 for broad RunPod driver compatibility (12.6 fails on older drivers)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.13 from deadsnakes PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl zstd ca-certificates software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    python3.13 -m ensurepip --upgrade && \
    python3.13 -m pip install --no-cache-dir --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies
RUN pip install --no-cache-dir \
    "gradio>=5.29.0" \
    "chromadb>=1.0.0" \
    "sentence-transformers>=4.1.0" \
    "rank-bm25>=0.2.2" \
    "textstat>=0.7.4" \
    "ollama>=0.5.1" \
    "pydantic>=2.11.3" \
    "httpx>=0.28.1" \
    "fastapi>=0.115.0" \
    "huggingface-hub>=0.30.0"

WORKDIR /app

# Application code
COPY src/ /app/src/
COPY data/programs/ /app/data/programs/
COPY data/raw/dhs_combined_manual/ /app/data/raw/dhs_combined_manual/

# Startup script
COPY deploy/runpod/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Runtime config
ENV NAVIGATOR_DATA_DIR=/app/data
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH=/app/src
ENV OLLAMA_HOST=0.0.0.0:11434

VOLUME /workspace
EXPOSE 7860 11434

ENTRYPOINT ["/app/start.sh"]
