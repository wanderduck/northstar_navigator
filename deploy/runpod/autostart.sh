#!/usr/bin/env bash
# deploy/runpod/autostart.sh — Auto-start NorthStar Navigator on pod boot
#
# Install this script to run automatically when the pod starts:
#   cp deploy/runpod/autostart.sh /workspace/autostart.sh
#   chmod +x /workspace/autostart.sh
#
# RunPod's base images check for /workspace/autostart.sh or /root/autostart.sh
# on container startup and execute it if present. Alternatively, set this as
# the pod's Docker start command in the RunPod console.
#
# This script is a thin wrapper around start.sh that:
# 1. Waits for the filesystem to be ready
# 2. Runs start.sh in the background with logging
# 3. Keeps the container alive

set -euo pipefail

WORKSPACE="/workspace"
NAVIGATOR_DIR="$WORKSPACE/navigator"
START_SCRIPT="$NAVIGATOR_DIR/deploy/runpod/start.sh"
LOG_FILE="$WORKSPACE/autostart.log"

echo "[autostart] $(date) — Pod starting" | tee -a "$LOG_FILE"

# Wait for workspace to be mounted
for i in $(seq 1 30); do
    if [[ -d "$NAVIGATOR_DIR" ]]; then
        break
    fi
    echo "[autostart] Waiting for workspace... (${i}s)" | tee -a "$LOG_FILE"
    sleep 1
done

if [[ ! -f "$START_SCRIPT" ]]; then
    echo "[autostart] start.sh not found at $START_SCRIPT" | tee -a "$LOG_FILE"
    echo "[autostart] Run setup.sh first, then copy this file to /workspace/autostart.sh" | tee -a "$LOG_FILE"
    # Keep container alive for SSH access
    sleep infinity
fi

echo "[autostart] Launching start.sh..." | tee -a "$LOG_FILE"
exec bash "$START_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
