#!/usr/bin/env bash
# deploy/runpod/cloudflare_tunnel.sh — Set up Cloudflare Tunnel on RunPod pod
#
# This creates a persistent tunnel from navigator.wanderduck.dev to localhost:7860
# on the RunPod pod, bypassing RunPod's proxy entirely.
#
# Prerequisites:
#   1. Cloudflare account with wanderduck.dev domain
#   2. Create a tunnel in Cloudflare Zero Trust dashboard:
#      - Go to: https://one.dash.cloudflare.com/ -> Networks -> Tunnels
#      - Click "Create a tunnel" -> Select "Cloudflared"
#      - Name it "northstar-navigator"
#      - Copy the tunnel token (long string starting with eyJ...)
#      - Add a public hostname: navigator.wanderduck.dev -> http://localhost:7860
#   3. Set CLOUDFLARE_TUNNEL_TOKEN env var or pass as argument
#
# Usage (inside pod):
#   CLOUDFLARE_TUNNEL_TOKEN="eyJ..." bash cloudflare_tunnel.sh
#   # or
#   bash cloudflare_tunnel.sh <token>

set -euo pipefail

TOKEN="${1:-${CLOUDFLARE_TUNNEL_TOKEN:-}}"

if [[ -z "$TOKEN" ]]; then
    cat <<'USAGE'
Cloudflare Tunnel Setup for NorthStar Navigator

Usage:
  CLOUDFLARE_TUNNEL_TOKEN="eyJ..." bash cloudflare_tunnel.sh
  bash cloudflare_tunnel.sh <tunnel-token>

To get a tunnel token:
  1. Go to https://one.dash.cloudflare.com/
  2. Networks -> Tunnels -> Create a tunnel
  3. Select "Cloudflared" connector
  4. Name: "northstar-navigator"
  5. Copy the token
  6. Add public hostname:
     - Subdomain: navigator
     - Domain: wanderduck.dev
     - Service: http://localhost:7860

USAGE
    exit 1
fi

# Install cloudflared if not present
if ! command -v cloudflared &>/dev/null; then
    echo "[1/3] Installing cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
    echo "      Installed: $(cloudflared --version)"
else
    echo "[1/3] cloudflared already installed: $(cloudflared --version)"
fi

# Save token for persistence across restarts
TOKEN_FILE="/workspace/.cloudflare_tunnel_token"
echo "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
echo "[2/3] Token saved to $TOKEN_FILE"

# Start tunnel
echo "[3/3] Starting Cloudflare Tunnel..."
echo "      navigator.wanderduck.dev -> http://localhost:7860"
echo ""
echo "      Press Ctrl+C to stop the tunnel."
echo ""

exec cloudflared tunnel --no-autoupdate run --token "$TOKEN"
