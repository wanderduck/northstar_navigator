#!/usr/bin/env bash
# deploy/runpod/setup.sh — First-run setup inside a RunPod pod
#
# Run this via SSH after pod creation:
#   bash /workspace/navigator/deploy/runpod/setup.sh
#
# Idempotent: safe to run multiple times. Skips steps already completed.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/wanderduck/northstar_navigator.git"
REPO_DIR="/workspace/navigator"
HF_REPO="wanderduck/northstar-navigator-gguf"
GGUF_FILE="model-q4_k_m.gguf"
GGUF_DIR="/workspace/gguf"
MODELFILE_PATH="$GGUF_DIR/Modelfile"
OLLAMA_MODEL="navigator"
export OLLAMA_MODELS="/workspace/.ollama/models"
export OLLAMA_HOST="0.0.0.0"

echo "============================================================"
echo " NorthStar Navigator — RunPod Setup"
echo "============================================================"
echo ""

# ── Step 1: System dependencies ───────────────────────────────────────────────
echo "[1/7] Installing system dependencies ..."
if command -v git &>/dev/null && command -v zstd &>/dev/null; then
    echo "      Already installed, skipping."
else
    apt-get update -qq
    apt-get install -y -qq curl zstd git python3-pip python3-venv >/dev/null 2>&1
    rm -rf /var/lib/apt/lists/*
    echo "      Done."
fi

# ── Step 2: Install Ollama ────────────────────────────────────────────────────
echo "[2/7] Installing Ollama ..."
if command -v ollama &>/dev/null; then
    echo "      Already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "      Done."
fi

# ── Step 3: Start Ollama and wait for readiness ──────────────────────────────
echo "[3/7] Starting Ollama server ..."
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "      Already running."
else
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1

    mkdir -p "$OLLAMA_MODELS"
    nohup ollama serve >/var/log/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "      PID: $OLLAMA_PID (log: /var/log/ollama.log)"

    echo "      Waiting for Ollama to be ready (up to 60s) ..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            echo "      Ready after ${i}s."
            break
        fi
        if [[ $i -eq 60 ]]; then
            echo "[error] Ollama failed to start within 60s. Check /var/log/ollama.log"
            exit 1
        fi
        sleep 1
    done
fi

# ── Step 4: Download GGUF from HuggingFace Hub ───────────────────────────────
echo "[4/7] Downloading GGUF model from HuggingFace Hub ..."
if [[ -f "$GGUF_DIR/$GGUF_FILE" ]]; then
    echo "      Already exists: $GGUF_DIR/$GGUF_FILE ($(du -h "$GGUF_DIR/$GGUF_FILE" | cut -f1))"
else
    if ! command -v hf &>/dev/null; then
        pip install -q "huggingface_hub[cli]"
    fi

    mkdir -p "$GGUF_DIR"
    echo "      Downloading $HF_REPO ..."
    hf download "$HF_REPO" "$GGUF_FILE" Modelfile --local-dir "$GGUF_DIR"
    echo "      Done: $(du -h "$GGUF_DIR/$GGUF_FILE" | cut -f1)"
fi

# Modelfile is downloaded from HF Hub alongside the GGUF
if [[ ! -f "$MODELFILE_PATH" ]]; then
    echo "[error] Modelfile not found at $MODELFILE_PATH — hf download may have failed"
    exit 1
fi

# ── Step 5: Create Ollama model ───────────────────────────────────────────────
echo "[5/7] Creating Ollama model '$OLLAMA_MODEL' ..."
if ollama show "$OLLAMA_MODEL" &>/dev/null; then
    echo "      Model already exists."
else
    echo "      Building from Modelfile ..."
    ollama create "$OLLAMA_MODEL" -f "$MODELFILE_PATH"
    echo "      Done."
fi

# Pre-warm: load model into VRAM
echo "      Pre-warming model (loading into VRAM) ..."
curl -sf http://localhost:11434/api/generate \
    -d '{"model":"'"$OLLAMA_MODEL"'","prompt":"Hello","stream":false,"options":{"num_predict":1}}' \
    >/dev/null 2>&1 || echo "      Warning: pre-warm request failed (non-fatal)"
echo "      Model warm."

# ── Step 6: Clone repo / install Python deps ─────────────────────────────────
echo "[6/7] Setting up application code ..."
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "      Repo exists, pulling latest ..."
    cd "$REPO_DIR" && git pull --ff-only 2>/dev/null || echo "      Warning: git pull failed. Continuing."
    cd /workspace
else
    echo "      Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[7/7] Installing Python dependencies ..."
pip install -q \
    "gradio>=5.29.0" \
    "chromadb>=1.0.0" \
    "sentence-transformers>=4.1.0" \
    "rank-bm25>=0.2.2" \
    "textstat>=0.7.4" \
    "ollama>=0.5.1" \
    "pydantic>=2.11.3" \
    "httpx>=0.28.1" \
    "fastapi>=0.115.0"
echo "      Done."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " Ollama:     running on :11434 (model: $OLLAMA_MODEL)"
echo " GGUF:       $GGUF_DIR/$GGUF_FILE"
echo " Repo:       $REPO_DIR"
echo " Models dir: $OLLAMA_MODELS"
echo ""
echo " Start the Gradio server:"
echo "   bash $REPO_DIR/deploy/runpod/start.sh"
echo "============================================================"
