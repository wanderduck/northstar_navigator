#!/usr/bin/env python3
"""Health check for NorthStar Navigator services on RunPod.

Checks Ollama server, navigator model, and Gradio UI.
Returns exit code 0 if all healthy, 1 if any failing.

Usage:
    python deploy/runpod/health_check.py
    python deploy/runpod/health_check.py --json
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error


def check_ollama(base_url: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, f"running, {len(models)} model(s)"
    except Exception as exc:
        return False, f"offline ({exc.__class__.__name__})"


def check_model(
    model_name: str = "navigator",
    base_url: str = "http://localhost:11434",
) -> tuple[bool, str]:
    """Check if the navigator model is loaded in Ollama."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        return False, "cannot reach Ollama"

    model_names = [m.get("name", "") for m in data.get("models", [])]
    if any(n == model_name or n.startswith(f"{model_name}:") for n in model_names):
        return True, f"'{model_name}' loaded"

    available = ", ".join(model_names) if model_names else "(none)"
    return False, f"'{model_name}' not found. Available: {available}"


def check_gradio(port: int = 7860) -> tuple[bool, str]:
    """Check if Gradio is responding."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, f"responding on :{port}"
            return False, f"HTTP {resp.status}"
    except Exception as exc:
        return False, f"not reachable on :{port} ({exc.__class__.__name__})"


def check_gpu() -> tuple[bool, str]:
    """Check if GPU is available via nvidia-smi."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split("\n")[0]
            return True, gpu_info
        return False, "nvidia-smi failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    output_json = "--json" in sys.argv

    checks = {
        "ollama": check_ollama(),
        "model": check_model(),
        "gradio": check_gradio(),
        "gpu": check_gpu(),
    }

    all_ok = all(ok for ok, _ in checks.values())

    if output_json:
        result = {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}
        result["healthy"] = all_ok
        print(json.dumps(result, indent=2))
    else:
        print("NorthStar Navigator — Health Check")
        print("=" * 45)
        for name, (ok, detail) in checks.items():
            status = "OK" if ok else "FAIL"
            print(f"  [{status:>4}] {name:<10} {detail}")
        print("=" * 45)
        print(f"  Overall: {'HEALTHY' if all_ok else 'UNHEALTHY'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
