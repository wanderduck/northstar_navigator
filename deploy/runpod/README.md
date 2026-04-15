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
# 1. Create pod (from project root)
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

## Deployment Options

| Method | Command | Best for |
|---|---|---|
| Shell script | `bash deploy/runpod/deploy.sh` | Quick interactive deploy |
| Python API | `python deploy/runpod/runpod_deploy.py create` | Automation, GPU fallback |
| Console | runpod.io/console/pods | Manual setup |

## Daily Operations

```bash
runpodctl pod stop <POD_ID>     # Pause ($0.01/day storage only)
runpodctl pod start <POD_ID>    # Resume (re-run start.sh via SSH)
runpodctl pod list --all        # See all pods
```

After restarting a stopped pod, SSH in and run `start.sh` to relaunch services.

## Cost

| GPU | $/hr | Model fit | Availability |
|---|---|---|---|
| **RTX 4090** (24GB) | $0.44 | 5.3GB GGUF + 18GB free | High (both clouds) |
| **L4** (24GB) | $0.24 | Same headroom, slower | Moderate (Secure only) |

- Stopped pod: ~$0.01-0.03/day (volume storage only)
- Demo day budget: ~$1-5 for a few hours of active use

## Troubleshooting

| Problem | Fix |
|---|---|
| Blank Gradio page | SSH in, check `pgrep -f app.py`, run `start.sh` |
| Slow first response | Model warming up — send test query first |
| 100s timeout | Verify `demo.queue()` is in app.py (already added) |
| Lost data after restart | Keep data in `/workspace/` (setup.sh handles this) |
| GPU unavailable | Try `RUNPOD_GPU="NVIDIA L4" bash deploy.sh` |
| Ollama won't start | Check `/var/log/ollama.log`, verify `OLLAMA_HOST=0.0.0.0` |

## Why RunPod Pod over Modal

| Factor | Modal | RunPod Pod |
|---|---|---|
| Gradio serving | ASGI mount (broken) | Native `demo.launch()` (works) |
| Cold start | 60-90s | None |
| Debugging | No SSH | Full SSH access |
| GPU cost | T4 $0.59/hr | RTX 4090 $0.44/hr |

## File Reference

| File | Purpose |
|---|---|
| `deploy.sh` | Create pod via runpodctl |
| `setup.sh` | First-run setup (Ollama, GGUF, deps, source) |
| `start.sh` | Idempotent service launcher |
| `runpod_deploy.py` | Python API automation (create/start/stop/delete/list) |
| `health_check.py` | Service health verification |
| `Dockerfile` | Backup Docker image option |
| `docker-compose.yml` | Local dev testing |
| `build_and_push.sh` | Docker Hub push |
