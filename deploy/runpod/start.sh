#!/usr/bin/env bash
# deploy/runpod/start.sh — Start NorthStar Navigator services on RunPod
#
# Idempotent: safe to run on every pod start. Checks state before acting.
# Can be set as the pod's startup command for automatic restarts.
#
# Usage (inside pod):
#   bash /workspace/navigator/deploy/runpod/start.sh

set -euo pipefail

REPO_DIR="/workspace/navigator"
GGUF_DIR="/workspace/gguf"
MODELFILE_PATH="$GGUF_DIR/Modelfile"
OLLAMA_MODEL="navigator"
export OLLAMA_MODELS="/workspace/.ollama/models"
export OLLAMA_HOST="0.0.0.0"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
export NAVIGATOR_DATA_DIR="$REPO_DIR/data"

echo "============================================================"
echo " NorthStar Navigator — Starting services"
echo "============================================================"
echo ""

# ── Preflight ─────────────────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/src/navigator" ]]; then
    echo "[error] Repo not found at $REPO_DIR"
    echo "        Run setup.sh first: bash /workspace/navigator/deploy/runpod/setup.sh"
    exit 1
fi

if [[ ! -f "$GGUF_DIR/model-q4_k_m.gguf" ]]; then
    echo "[error] GGUF model not found at $GGUF_DIR/model-q4_k_m.gguf"
    echo "        Run setup.sh first."
    exit 1
fi

# ── Step 1: Ensure Ollama is running ──────────────────────────────────────────
echo "[1/4] Checking Ollama ..."
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "      Already running."
else
    echo "      Starting Ollama server ..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1

    mkdir -p "$OLLAMA_MODELS"
    nohup ollama serve >/var/log/ollama.log 2>&1 &
    echo "      PID: $! (log: /var/log/ollama.log)"

    echo "      Waiting for readiness (up to 60s) ..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            echo "      Ready after ${i}s."
            break
        fi
        if [[ $i -eq 60 ]]; then
            echo "[error] Ollama failed to start. Check /var/log/ollama.log"
            exit 1
        fi
        sleep 1
    done
fi

# ── Step 2: Verify model ─────────────────────────────────────────────────────
echo "[2/4] Checking model '$OLLAMA_MODEL' ..."
if ollama show "$OLLAMA_MODEL" &>/dev/null; then
    echo "      Model available."
else
    echo "      Model not found, creating from Modelfile ..."
    if [[ ! -f "$MODELFILE_PATH" ]]; then
        echo "[error] Modelfile not found at $MODELFILE_PATH. Run setup.sh first."
        exit 1
    fi
    ollama create "$OLLAMA_MODEL" -f "$MODELFILE_PATH"
    echo "      Done."
fi

# ── Step 3: Pre-warm model ───────────────────────────────────────────────────
echo "[3/4] Pre-warming model (loading into VRAM) ..."
curl -sf http://localhost:11434/api/generate \
    -d '{"model":"'"$OLLAMA_MODEL"'","prompt":"Hello","stream":false,"options":{"num_predict":1}}' \
    >/dev/null 2>&1 || echo "      Warning: pre-warm failed (non-fatal)"
echo "      Model warm."

# ── Step 4: Start Cloudflare Tunnel (if token saved) ─────────────────────────
CF_TOKEN_FILE="/workspace/.cloudflare_tunnel_token"
if [[ -f "$CF_TOKEN_FILE" ]] && command -v cloudflared &>/dev/null; then
    echo "[4/5] Starting Cloudflare Tunnel..."
    CF_TOKEN="$(cat "$CF_TOKEN_FILE")"
    nohup cloudflared tunnel --no-autoupdate run --token "$CF_TOKEN" \
        >/var/log/cloudflared.log 2>&1 &
    echo "      Tunnel PID: $! (log: /var/log/cloudflared.log)"
    echo "      URL: https://navigator.wanderduck.dev"
else
    echo "[4/5] Cloudflare Tunnel: not configured (optional)"
fi

# ── Step 5: Launch Gradio ────────────────────────────────────────────────────
POD_ID="${RUNPOD_POD_ID:-unknown}"
echo "[5/5] Launching Gradio UI on port 7860 ..."
echo ""
echo "============================================================"
echo " NorthStar Navigator is starting!"
echo ""
echo "   Gradio:  http://0.0.0.0:7860"
if [[ "$POD_ID" != "unknown" ]]; then
echo "   RunPod:  https://${POD_ID}-7860.proxy.runpod.net"
fi
if [[ -f "$CF_TOKEN_FILE" ]]; then
echo "   Custom:  https://navigator.wanderduck.dev"
fi
echo "   Ollama:  http://0.0.0.0:11434"
echo ""
echo " Press Ctrl+C to stop."
echo "============================================================"
echo ""

cd "$REPO_DIR"
exec python src/app.py
