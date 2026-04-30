#!/usr/bin/env bash
# deploy/runpod/deploy_a5000.sh — Create a cheaper RTX A5000 pod for testing
#
# Wrapper around deploy.sh that creates a SEPARATE pod on RTX A5000 hardware
# for development and testing purposes, leaving the primary pod
# (RTX 4090, used for the live competition demo) untouched.
#
# Pod naming (so both can coexist):
#   "northstar-navigator"      — RTX 4090 live-demo pod (created by deploy.sh)
#   "northstar-navigator-test" — RTX A5000 testing pod  (created by THIS script)
#
# GPU comparison (Secure Cloud, approximate):
#   RTX A5000 (this script):  ~$0.36/hr — Ampere, 24GB, 768 GB/s bandwidth
#   RTX 4090   (deploy.sh):   ~$0.69/hr — Ada Lovelace, 24GB, 1008 GB/s bandwidth
#   ≈48% cost saving for testing; ~25-30% slower tokens/sec on the A5000
#   (still well above human reading speed for the Gemma 4 E4B q4_k_m model).
#
# Usage:
#   bash deploy/runpod/deploy_a5000.sh           # Create / manage A5000 test pod
#   bash deploy/runpod/deploy_a5000.sh --help    # Help (forwarded to deploy.sh)
#
# All other behavior, flags, and env vars (RUNPOD_API_KEY, RUNPOD_CLOUD_TYPE, etc.)
# are inherited from deploy.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUNPOD_GPU="NVIDIA RTX A5000"
export POD_NAME="northstar-navigator-test"

exec bash "$SCRIPT_DIR/deploy.sh" "$@"
