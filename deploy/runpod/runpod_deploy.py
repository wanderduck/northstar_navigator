#!/usr/bin/env python3
"""RunPod Pod lifecycle management for NorthStar Navigator.

Automates pod creation, start/stop, status checks, and deletion
via the RunPod REST API.

Usage:
    python deploy/runpod/runpod_deploy.py create
    python deploy/runpod/runpod_deploy.py status <pod_id>
    python deploy/runpod/runpod_deploy.py start <pod_id>
    python deploy/runpod/runpod_deploy.py stop <pod_id>
    python deploy/runpod/runpod_deploy.py delete <pod_id>
    python deploy/runpod/runpod_deploy.py list
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

API_BASE = "https://rest.runpod.io/v1"

# GPU fallback chain: try each in order until one succeeds
GPU_FALLBACK = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA L4",
]

TEMPLATE_CONFIG = {
    "name": "northstar-navigator",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "containerDiskInGb": 30,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "ports": "7860/http,11434/http,22/tcp",
    "env": [
        {"key": "OLLAMA_HOST", "value": "0.0.0.0"},
        {"key": "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "value": "python"},
        {"key": "OLLAMA_MODELS", "value": "/workspace/.ollama/models"},
    ],
    "isPublic": False,
    "isServerless": False,
}


def _load_api_key() -> str:
    """Load RunPod API key from environment or .env file."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if key:
        return key

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                if key:
                    return key

    print("ERROR: RUNPOD_API_KEY not found.")
    print(f"       Set it in environment or in {env_path}")
    sys.exit(1)


def _api_request(
    method: str,
    path: str,
    api_key: str,
    body: dict | None = None,
) -> dict | list:
    """Make an authenticated RunPod REST API request."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode() if exc.fp else ""
        print(f"API error {exc.code}: {error_body}")
        sys.exit(1)


def cmd_create(api_key: str) -> None:
    """Create a new Navigator pod with GPU fallback."""
    print("Creating NorthStar Navigator pod...")

    for gpu_type in GPU_FALLBACK:
        print(f"  Trying GPU: {gpu_type}")
        pod_config = {
            **TEMPLATE_CONFIG,
            "gpuTypeId": gpu_type,
            "gpuCount": 1,
            "cloudType": "SECURE",
        }

        try:
            result = _api_request("POST", "/pods", api_key, pod_config)
        except SystemExit:
            print(f"  {gpu_type} unavailable, trying next...")
            continue

        pod_id = result.get("id", "unknown")
        print(f"\nPod created: {pod_id}")
        print(f"  GPU: {gpu_type}")
        print(f"  Gradio: https://{pod_id}-7860.proxy.runpod.net")
        print(f"  Ollama: https://{pod_id}-11434.proxy.runpod.net")
        print(f"\nNext: SSH in, run setup.sh, then start.sh")
        return

    print("\nERROR: No GPU available in fallback chain.")
    print("  Try again later or use a different GPU type.")
    sys.exit(1)


def cmd_status(api_key: str, pod_id: str) -> None:
    """Show pod status and connection info."""
    result = _api_request("GET", f"/pods/{pod_id}", api_key)
    print(f"Pod: {pod_id}")
    print(f"  Name:   {result.get('name', '?')}")
    print(f"  Status: {result.get('desiredStatus', '?')}")
    print(f"  GPU:    {result.get('machine', {}).get('gpuDisplayName', '?')}")
    print(f"  Gradio: https://{pod_id}-7860.proxy.runpod.net")

    runtime = result.get("runtime", {})
    if runtime and runtime.get("ports"):
        for port_info in runtime["ports"]:
            print(f"  Port {port_info.get('privatePort')}: {port_info.get('ip')}:{port_info.get('publicPort')}")


def cmd_start(api_key: str, pod_id: str) -> None:
    """Start a stopped pod."""
    _api_request("POST", f"/pods/{pod_id}/start", api_key)
    print(f"Pod {pod_id} starting...")
    print(f"  Gradio: https://{pod_id}-7860.proxy.runpod.net")


def cmd_stop(api_key: str, pod_id: str) -> None:
    """Stop a running pod (preserves /workspace)."""
    _api_request("POST", f"/pods/{pod_id}/stop", api_key)
    print(f"Pod {pod_id} stopping. Volume data preserved.")


def cmd_delete(api_key: str, pod_id: str) -> None:
    """Terminate a pod permanently."""
    confirm = input(f"Delete pod {pod_id}? This destroys all data. [y/N]: ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return
    _api_request("DELETE", f"/pods/{pod_id}", api_key)
    print(f"Pod {pod_id} deleted.")


def cmd_list(api_key: str) -> None:
    """List all pods."""
    result = _api_request("GET", "/pods", api_key)
    if not result:
        print("No pods found.")
        return

    print(f"{'ID':<25} {'Name':<25} {'Status':<12} {'GPU':<20}")
    print("-" * 82)
    for pod in result:
        pod_id = pod.get("id", "?")
        name = pod.get("name", "?")
        status = pod.get("desiredStatus", "?")
        gpu = pod.get("machine", {}).get("gpuDisplayName", "?") if pod.get("machine") else "?"
        print(f"{pod_id:<25} {name:<25} {status:<12} {gpu:<20}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    api_key = _load_api_key()
    command = sys.argv[1]

    match command:
        case "create":
            cmd_create(api_key)
        case "status" if len(sys.argv) >= 3:
            cmd_status(api_key, sys.argv[2])
        case "start" if len(sys.argv) >= 3:
            cmd_start(api_key, sys.argv[2])
        case "stop" if len(sys.argv) >= 3:
            cmd_stop(api_key, sys.argv[2])
        case "delete" if len(sys.argv) >= 3:
            cmd_delete(api_key, sys.argv[2])
        case "list":
            cmd_list(api_key)
        case _:
            print(f"Unknown command: {command}")
            print("Commands: create, status, start, stop, delete, list")
            sys.exit(1)


if __name__ == "__main__":
    main()
