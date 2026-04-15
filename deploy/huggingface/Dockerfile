FROM python:3.13-slim

# Build tools for llama-cpp-python C++ compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# Limit cmake parallelism to avoid OOM on cpu-basic (16GB)
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Stage 1: PyTorch CPU-only (~200MB vs ~2GB full)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Stage 2: llama-cpp-python (compiles llama.cpp — slow but low-memory with parallelism=1)
RUN pip install --no-cache-dir llama-cpp-python>=0.3.0

# Stage 3: remaining deps (gradio needed explicitly in Docker mode)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + data
COPY --chown=user:user . .

EXPOSE 7860
CMD ["python", "app.py"]
