#!/usr/bin/env bash
# deploy/runpod/deploy.sh — Create or manage a RunPod GPU pod for NorthStar Navigator
#
# Usage:
#   bash deploy/runpod/deploy.sh              # Create pod with RTX 4090
#   RUNPOD_GPU="NVIDIA L4" bash deploy/runpod/deploy.sh  # Override GPU type
#   bash deploy/runpod/deploy.sh --help       # Show help
#
# Requires: runpodctl (https://github.com/runpod/runpodctl)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
POD_NAME="northstar-navigator"
GPU_TYPE="${RUNPOD_GPU:-NVIDIA GeForce RTX 4090}"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB=30
VOLUME_DISK_GB=50
VOLUME_MOUNT="/workspace"
PORTS="7860/http,11434/http,22/tcp"

# ── Resolve project root (two levels up from this script) ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Help ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
NorthStar Navigator — RunPod Pod Deployment

USAGE
  bash deploy/runpod/deploy.sh [--help]

ENVIRONMENT VARIABLES
  RUNPOD_API_KEY    RunPod API key (loaded from .env if not set)
  RUNPOD_GPU        GPU type override (default: "NVIDIA GeForce RTX 4090")
                    Examples: "NVIDIA L4", "NVIDIA GeForce RTX 4090"

WHAT THIS DOES
  1. Checks for existing pod named "northstar-navigator"
  2. If found, offers to start or delete it
  3. If not found, creates a new pod with:
     - GPU: RTX 4090 (24GB VRAM, ~$0.44/hr) or override
     - Image: nvidia/cuda:12.6.3-runtime-ubuntu22.04
     - Container disk: 30GB, Volume: 50GB at /workspace
     - Ports: 7860 (Gradio), 11434 (Ollama), 22 (SSH)
  4. Prints pod URL and next steps

AFTER CREATION
  SSH into the pod and run:
    bash setup.sh   # First-time setup (in /workspace/navigator/deploy/runpod/)
    bash start.sh   # Start services

COST
  RTX 4090: ~$0.44/hr (~$10.50/day)
  Stop the pod when not in use to avoid charges.
HELP
    exit 0
fi

# ── Load API key from .env if not already set ─────────────────────────────────
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env"
    if [[ -f "$ENV_FILE" ]]; then
        RUNPOD_API_KEY="$(grep -E '^RUNPOD_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '[:space:]"'"'")"
        if [[ -n "$RUNPOD_API_KEY" ]]; then
            export RUNPOD_API_KEY
            echo "[ok] Loaded RUNPOD_API_KEY from $ENV_FILE"
        fi
    fi
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "[error] RUNPOD_API_KEY not found."
    echo "        Set it in your environment or add RUNPOD_API_KEY=<key> to $PROJECT_ROOT/.env"
    exit 1
fi

# ── Verify runpodctl is installed ─────────────────────────────────────────────
if ! command -v runpodctl &>/dev/null; then
    echo "[error] runpodctl not found. Install it:"
    echo "        bash <(curl -sL cli.runpod.io)"
    exit 1
fi

# Configure runpodctl with API key
runpodctl config --apiKey "$RUNPOD_API_KEY" &>/dev/null || true

# ── Check for existing pod ────────────────────────────────────────────────────
echo "[...] Checking for existing pod '$POD_NAME' ..."

EXISTING_ID=$(runpodctl pod list --all --name "$POD_NAME" 2>/dev/null | awk 'NR>1 {print $1; exit}' || true)

if [[ -n "$EXISTING_ID" ]]; then
    EXISTING_STATUS=$(runpodctl pod list --all --name "$POD_NAME" 2>/dev/null | awk 'NR>1 {print $3; exit}' || echo "unknown")
    echo "[found] Pod '$POD_NAME' exists (ID: $EXISTING_ID, Status: $EXISTING_STATUS)"
    echo ""
    echo "  1) Start pod (if stopped)"
    echo "  2) Delete pod and create new"
    echo "  3) Show pod info"
    echo "  4) Cancel"
    echo ""
    read -rp "Choice [1/2/3/4]: " choice

    case "$choice" in
        1)
            echo "[...] Starting pod $EXISTING_ID ..."
            runpodctl pod start "$EXISTING_ID"
            echo "[ok] Pod starting. Check: runpodctl pod get $EXISTING_ID"
            echo "     Gradio: https://$EXISTING_ID-7860.proxy.runpod.net"
            exit 0
            ;;
        2)
            echo "[...] Deleting pod $EXISTING_ID ..."
            runpodctl pod delete "$EXISTING_ID"
            echo "[ok] Pod deleted. Creating new pod..."
            sleep 2
            ;;
        3)
            runpodctl pod get "$EXISTING_ID"
            exit 0
            ;;
        4)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# ── Create pod ────────────────────────────────────────────────────────────────
echo ""
echo "[...] Creating RunPod pod:"
echo "      Name:           $POD_NAME"
echo "      GPU:            $GPU_TYPE"
echo "      Image:          $IMAGE"
echo "      Container disk: ${CONTAINER_DISK_GB}GB"
echo "      Volume:         ${VOLUME_DISK_GB}GB at $VOLUME_MOUNT"
echo "      Ports:          $PORTS"
echo ""

runpodctl pod create \
    --name "$POD_NAME" \
    --gpu-id "$GPU_TYPE" \
    --image "$IMAGE" \
    --container-disk-in-gb "$CONTAINER_DISK_GB" \
    --volume-in-gb "$VOLUME_DISK_GB" \
    --volume-mount-path "$VOLUME_MOUNT" \
    --ports "$PORTS" \
    --env '{"OLLAMA_HOST":"0.0.0.0","PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION":"python","OLLAMA_MODELS":"/workspace/.ollama/models"}'

echo ""
echo "============================================================"
echo " Pod creation requested!"
echo "============================================================"
echo ""
echo " Check status:   runpodctl pod list"
echo " Get SSH info:   runpodctl pod get <POD_ID>"
echo ""
echo " Next steps after pod is running:"
echo "   1. SSH into the pod"
echo "   2. git clone your repo to /workspace/navigator"
echo "   3. bash /workspace/navigator/deploy/runpod/setup.sh"
echo "   4. bash /workspace/navigator/deploy/runpod/start.sh"
echo ""
echo " Stop when idle:  runpodctl pod stop <POD_ID>"
echo " Delete:          runpodctl pod delete <POD_ID>"
echo "============================================================"
