# NorthStar Navigator -- RunPod Deployment Guide

Deploy the NorthStar Navigator live demo on a RunPod GPU Pod.

```
  Browser
    |
    | HTTPS
    v
  RunPod Proxy (:7860)
    |
    v
  +-------------------------------+
  |     RunPod Pod (RTX 4090)     |
  |                               |
  |  +--------+    +-----------+  |
  |  | Ollama |<---| ChromaDB  |  |
  |  | :11434 |    | (on-disk) |  |
  |  +--------+    +-----------+  |
  |       ^                       |
  |       |                       |
  |  +--------+                   |
  |  | Gradio |                   |
  |  | :7860  |                   |
  |  +--------+                   |
  |                               |
  |  /workspace (volume disk)     |
  |    gguf/ navigator/ .ollama/  |
  +-------------------------------+
```

## Prerequisites

1. **RunPod account** with credits — [runpod.io](https://runpod.io)
2. **API key** — Settings > API Keys (All permission level)
3. **runpodctl CLI** — `bash <(curl -sL cli.runpod.io)`
4. **SSH key** — `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`

## Quick Start

```bash
# 1. Create pod (from project root) — RTX 4090 on Secure Cloud, live-demo pod
bash deploy/runpod/deploy.sh

# 2. SSH into the pod
runpodctl pod get <POD_ID>  # get SSH info
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# 3. First-run setup (inside pod, ~10 min)
bash /workspace/navigator/deploy/runpod/setup.sh

# 4. Start services
bash /workspace/navigator/deploy/runpod/start.sh

# 5. Open Gradio UI
#    https://<POD_ID>-7860.proxy.runpod.net
```

## Two Pods: Live Demo vs Testing

There are **two separate deploy entry points**, intentionally creating **two
separately-named pods** so they can coexist:

| Script | Pod name | GPU | Purpose | Cost (Secure) |
|---|---|---|---|---|
| `deploy.sh` | `northstar-navigator` | RTX 4090 | **Live competition demo** — keep stable for judging | ~$0.69/hr |
| `deploy_a5000.sh` | `northstar-navigator-test` | RTX A5000 | **Development/testing** — iterate on cheaper hardware | ~$0.36/hr |

`deploy_a5000.sh` is a thin wrapper that exports `RUNPOD_GPU` and `POD_NAME`
then delegates to `deploy.sh`, so they share all other deploy logic.

The A5000 has the same 24 GB of VRAM as the 4090, so the model fits identically.
The trade-off is ~25–30% slower token generation (memory bandwidth: 768 vs
1008 GB/s) — imperceptible for short single-user demo responses.

## Deployment Options

| Method | Command | Best for |
|---|---|---|
| Shell script (4090) | `bash deploy/runpod/deploy.sh` | Live-demo pod, quick interactive deploy |
| Shell script (A5000) | `bash deploy/runpod/deploy_a5000.sh` | Cheap testing pod, separate from live demo |
| Python API | `python deploy/runpod/runpod_deploy.py create` | Automation, GPU fallback chain |
| Console | runpod.io/console/pods | Manual setup |

## Daily Operations

```bash
runpodctl pod stop <POD_ID>     # Pause ($0.01/day storage only)
runpodctl pod start <POD_ID>    # Resume (re-run start.sh via SSH)
runpodctl pod list --all        # See all pods
```

After restarting a stopped pod, SSH in and run `start.sh` to relaunch services.

## Cost

Approximate Secure Cloud pricing — verify current rates at runpod.io/pricing.

| GPU | $/hr (Secure) | Used by | Notes |
|---|---|---|---|
| **RTX 4090** (24GB) | ~$0.69 | `deploy.sh` (live demo) | Fastest, 1008 GB/s bandwidth |
| **RTX A5000** (24GB) | ~$0.36 | `deploy_a5000.sh` (testing) | ~25–30% slower than 4090, ~48% cheaper |
| **L4** (24GB) | ~$0.43 | Fallback via `RUNPOD_GPU=` | Slowest of the three, 300 GB/s |

- Stopped pod: ~$0.01–0.03/day (volume storage only)
- Demo day budget: ~$1–5 for a few hours of active use
- Switching to Community Cloud (`RUNPOD_CLOUD_TYPE=COMMUNITY`) cuts prices
  further but reduces availability/reliability — not recommended for the
  judging-period live demo.

## Troubleshooting

| Problem | Fix |
|---|---|
| Blank Gradio page | SSH in, check `pgrep -f app.py`, run `start.sh` |
| Slow first response | Model warming up — send test query first |
| 100s timeout | Verify `demo.queue()` is in app.py (already added) |
| Lost data after restart | Keep data in `/workspace/` (setup.sh handles this) |
| GPU unavailable | Try `RUNPOD_GPU="NVIDIA L4" bash deploy.sh` or `RUNPOD_GPU="NVIDIA RTX A5000" bash deploy.sh` |
| Ollama won't start | Check `/var/log/ollama.log`, verify `OLLAMA_HOST=0.0.0.0` |

## Why RunPod Pod over Modal

| Factor | Modal | RunPod Pod |
|---|---|---|
| Gradio serving | ASGI mount (broken) | Native `demo.launch()` (works) |
| Cold start | 60-90s | None |
| Debugging | No SSH | Full SSH access |
| GPU cost | T4 $0.59/hr | RTX 4090 ~$0.69/hr Secure / RTX A5000 ~$0.36/hr Secure |

## File Reference

| File | Purpose |
|---|---|
| `deploy.sh` | Create live-demo pod (RTX 4090, `northstar-navigator`) via runpodctl |
| `deploy_a5000.sh` | Create testing pod (RTX A5000, `northstar-navigator-test`) — wrapper around `deploy.sh` |
| `setup.sh` | First-run setup (Ollama, GGUF, deps, source) |
| `start.sh` | Idempotent service launcher |
| `runpod_deploy.py` | Python API automation (create/start/stop/delete/list) |
| `health_check.py` | Service health verification |
| `Dockerfile` | Backup Docker image option |
| `docker-compose.yml` | Local dev testing |
| `build_and_push.sh` | Docker Hub push |
