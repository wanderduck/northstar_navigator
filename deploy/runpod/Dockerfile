# NorthStar Navigator — RunPod Deployment Dockerfile
#
# Backup/reproducibility image. Primary deployment uses setup.sh scripts.
#
# Build (from project root):
#   docker build --platform=linux/amd64 -t wanderduck/northstar-navigator:v0.1.0 -f deploy/runpod/Dockerfile .
#
# IMPORTANT:
#   - Always build with --platform=linux/amd64 (RunPod runs x86_64 Linux)
#   - Always use a versioned tag (e.g. v0.1.0), never :latest
#   - GGUF model weights NOT baked in (5.3GB) — downloaded at runtime via hf CLI

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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
    "huggingface_hub[cli]>=0.30.0"

WORKDIR /app

# Application code + icon
COPY src/ /app/src/
COPY data/programs/ /app/data/programs/
COPY data/raw/dhs_combined_manual/ /app/data/raw/dhs_combined_manual/
COPY notebook_images/ /app/notebook_images/

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
