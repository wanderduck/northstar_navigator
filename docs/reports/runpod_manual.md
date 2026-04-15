# RunPod Platform Documentation Manual

> Compiled 2026-04-14 from comprehensive exploration of all RunPod documentation sections.
> Research source files preserved in `docs/reports/runpod_sections/` for reference.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Pods](#2-pods)
3. [Serverless](#3-serverless)
4. [Flash SDK](#4-flash-sdk)
5. [Flash CLI](#5-flash-cli)
6. [RunPod CLI (runpodctl)](#6-runpod-cli-runpodctl)
7. [API Reference](#7-api-reference)
8. [Storage & Network Volumes](#8-storage--network-volumes)
9. [GPU Types Reference](#9-gpu-types-reference)
10. [Public Endpoints](#10-public-endpoints)
11. [Hub & Templates](#11-hub--templates)
12. [Tutorials](#12-tutorials)
13. [Instant Clusters](#13-instant-clusters)
14. [Integrations](#14-integrations)
15. [Navigator Deployment Guide](#15-navigator-deployment-guide)
16. [Key Warnings & Gotchas](#16-key-warnings--gotchas)

---

## 1. Getting Started

### Account Setup

1. Create account at [runpod.io](https://runpod.io)
2. Add billing method (credit card or crypto)
3. Generate API key: **Settings > API Keys**

### API Key Permission Levels

| Level | Access |
|---|---|
| **All** | Full access to all operations |
| **Restricted** | Limited to specified operations |
| **Read Only** | View-only access |

**Warning:** Keys created before November 2024 are "legacy" keys with different behavior. Regenerate if issues arise.

### Deployment Types

| Type | Best For | Scaling | Billing |
|---|---|---|---|
| **Pods** | Persistent GPU instances, dev work, interactive | Manual | Per-hour |
| **Serverless** | Auto-scaling inference endpoints | Automatic (0 to N) | Per-second |
| **Flash** | Python-native serverless SDK | Automatic | Per-second |

### Pricing Quick Reference

- Stopped pod storage: $0.20/GB/month
- Network volumes: $0.07/GB/month (first 1TB), $0.05/GB/month (beyond)
- No ingress/egress fees
- Account spend limit: $80/hr default

---

## 2. Pods

Pods are persistent GPU instances — virtual machines with full root access, SSH, and JupyterLab. Most relevant for Navigator deployment.

### Pod Lifecycle

| State | Description | Billing |
|---|---|---|
| **Running** | GPU attached, full access | GPU + storage |
| **Stopped** | No GPU, data preserved in /workspace | Storage only ($0.10-$0.20/GB/month) |
| **Terminated** | Deleted, only network volume data survives | Nothing |

### Limitations

- **No Docker Compose** — single container per pod
- **No UDP** — TCP and HTTP only
- **No Windows** — Linux only
- **HTTP proxy has 100-second Cloudflare timeout** — long-running requests must use WebSocket or polling

### Pod URL Pattern

```
https://[POD_ID]-[PORT].proxy.runpod.net
```

Example: `https://abc123-7860.proxy.runpod.net` for Gradio on port 7860.

### Creating a Pod

**Via CLI:**
```bash
runpodctl pod create \
  --name "navigator" \
  --gpu-id "NVIDIA L4" \
  --image "nvidia/cuda:12.6.3-runtime-ubuntu22.04" \
  --container-disk-in-gb 50 \
  --volume-in-gb 100 \
  --volume-mount-path /workspace \
  --ports "7860/http,11434/http,22/tcp" \
  --env '{"OLLAMA_HOST":"0.0.0.0"}' \
  --cloud-type SECURE
```

**Via REST API:**
```bash
curl --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "name": "navigator",
    "gpuTypeId": "NVIDIA L4",
    "imageName": "nvidia/cuda:12.6.3-runtime-ubuntu22.04",
    "containerDiskInGb": 50,
    "volumeInGb": 100,
    "volumeMountPath": "/workspace",
    "ports": "7860/http,11434/http,22/tcp",
    "env": {"OLLAMA_HOST": "0.0.0.0"},
    "cloudType": "SECURE"
  }'
```

### Pod Create Flags (CLI)

| Flag | Default | Description |
|---|---|---|
| `--template-id` | — | Template ID |
| `--image` | — | Docker image (required if no template) |
| `--name` | — | Pod name |
| `--gpu-id` | — | GPU type string (e.g., "NVIDIA L4") |
| `--gpu-count` | 1 | Number of GPUs |
| `--compute-type` | GPU | GPU or CPU |
| `--container-disk-in-gb` | 20 | Ephemeral disk size |
| `--volume-in-gb` | — | Persistent volume size |
| `--volume-mount-path` | /workspace | Volume mount point |
| `--ports` | — | e.g., "8888/http,22/tcp" |
| `--env` | — | JSON string: '{"KEY":"value"}' |
| `--cloud-type` | SECURE | SECURE or COMMUNITY |
| `--data-center-ids` | — | Comma-separated datacenter IDs |
| `--network-volume-id` | — | Attach network volume |
| `--ssh` | true | Enable SSH |

### Pod Lifecycle Commands

```bash
runpodctl pod list              # List running pods
runpodctl pod list --all        # Include stopped pods
runpodctl pod get <pod-id>      # Detailed info + SSH connection
runpodctl pod start <pod-id>    # Start stopped pod
runpodctl pod stop <pod-id>     # Stop (preserve /workspace)
runpodctl pod restart <pod-id>  # Restart
runpodctl pod delete <pod-id>   # Terminate permanently
```

### Connecting to a Pod

| Method | Access |
|---|---|
| **Web Terminal** | Browser-based terminal in RunPod console |
| **SSH** | `ssh root@[PUBLIC_IP] -p [TCP_PORT] -i ~/.ssh/id_ed25519` |
| **JupyterLab** | `https://[POD_ID]-8888.proxy.runpod.net` |
| **VS Code/Cursor** | Remote-SSH extension with above SSH config |

### Port Exposure

- **HTTP ports**: Proxied through `https://[POD_ID]-[PORT].proxy.runpod.net` with automatic HTTPS
- **TCP ports**: Direct access via public IP: `[PUBLIC_IP]:[EXTERNAL_PORT]`
- External TCP ports found in: `RUNPOD_TCP_PORT_[INTERNAL_PORT]` env var

### Storage Types

| Type | Persistence | Mount Point | Notes |
|---|---|---|---|
| **Container Disk** | Cleared on stop/restart | / | Ephemeral, fast |
| **Volume Disk** | Survives stop, cleared on terminate | /workspace | Persistent within pod lifecycle |
| **Network Volume** | Permanent, survives termination | Configurable | Portable between pods, $0.07/GB/month |

**Critical:** Pods with network volumes can only be terminated, not stopped. Data persists in the network volume.

### Pod Pricing

| Type | Description |
|---|---|
| **On-demand** | Pay per hour, no commitment |
| **Savings Plans** | 3-month (moderate discount) or 6-month (larger discount) |
| **Spot** | Lowest price, can be preempted with 5-second SIGTERM warning |
| **Storage** | $0.10-$0.20/GB/month for container/volume disk |

### Environment Variables

- Max 50 per pod
- **WARNING:** Updating env vars restarts the pod, clearing data outside volume mount
- RunPod auto-provides: `RUNPOD_POD_ID`, `RUNPOD_DC_ID`, `RUNPOD_POD_HOSTNAME`, `RUNPOD_GPU_COUNT`, `RUNPOD_CPU_COUNT`, `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_VOLUME_ID`, `RUNPOD_API_KEY`, `PUBLIC_KEY`, `CUDA_VERSION`, `PYTORCH_VERSION`

### Secrets

- Encrypted storage: `{{ RUNPOD_SECRET_secret_name }}` syntax
- Values **cannot be viewed** after creation
- Deletion is **permanent** with no undo

### Troubleshooting

- **Pod migration** (beta): Pod may get new ID/IP during migration
- **Storage full**: `df -h` to diagnose, clean up container disk or expand volume
- **Restarting stopped pod may get zero GPUs** if datacenter capacity changed

---

## 3. Serverless

Serverless endpoints auto-scale GPU workers from 0 to N based on request queue depth. Pay only for active compute time.

### Architecture

- **Handler function**: `def handler(job)` receives `job["input"]`, returns result dict
- **Workers**: Containers running your handler, auto-provisioned/scaled
- **Endpoints**: HTTPS URLs routing requests to worker pools

### Handler Pattern

```python
import runpod

def handler(job):
    """Process a single job."""
    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    
    # Your inference logic here
    result = model.generate(prompt)
    
    return {"output": result}

runpod.serverless.start({"handler": handler})
```

### Request Modes

| Mode | Endpoint | Timeout | Result Retention |
|---|---|---|---|
| **Synchronous** | `POST /runsync` | 30 seconds | 1 minute |
| **Asynchronous** | `POST /run` | None (poll status) | 30 minutes |

```bash
# Synchronous request
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello"}}'

# Asynchronous request
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello"}}'

# Check status
curl "https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Endpoint Configuration

| Setting | Description |
|---|---|
| `workersMin` | Minimum warm workers (0 for scale-to-zero) |
| `workersMax` | Maximum concurrent workers |
| `idleTimeout` | Seconds before idle worker shuts down |
| `scalerType` | QUEUE_DELAY or REQUEST_COUNT |
| `gpuIds` | GPU type(s) for workers |

### Rate Limits

| Endpoint | Rate | Concurrent |
|---|---|---|
| `/runsync` | 2000/10s | 400 |
| `/run` | 1000/10s | 200 |
| `/cancel` | 100/10s | — |
| `/purge-queue` | 2/10s | — |

Exceeding limits returns HTTP 429. Dynamic scaling based on `WorkersMax * 500`.

### Pricing

Per-second GPU billing with two tiers:

| GPU | Flex ($/s) | Active Worker ($/s) | Discount |
|---|---|---|---|
| A4000 | $0.00016 | ~20-30% less | — |
| A100-80GB | — | — | — |
| H100 | $0.00116 | ~20-30% less | — |
| B200 | $0.00240 | ~20-30% less | — |

Active workers (always-warm) get 20-30% discount over flex pricing.

### Cold Starts

First request to a scaled-to-zero endpoint provisions a new worker. This can take:
- 30-90 seconds (simple containers)
- 3-10 minutes (GPU provisioning + model download)

Mitigate with `workersMin >= 1` (always-warm).

### SDKs

- **Python**: `pip install runpod`
- **JavaScript/TypeScript**: `npm install runpod`
- **Go**: `go get github.com/runpod/go-sdk`

### Dockerfile for Serverless

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install runpod torch transformers
COPY handler.py /handler.py
CMD ["python", "/handler.py"]
```

### vLLM Integration

RunPod has first-class vLLM support for LLM serving with OpenAI-compatible endpoints. Deploy via Hub or custom Docker image.

---

## 4. Flash SDK

Flash is RunPod's Python-native serverless SDK. Currently in **beta**.

### Requirements

- **Python 3.12** (strictly required)
- **macOS or Linux** only (Windows in development)
- Package: `pip install runpod-flash`

### Endpoint Types

| Type | Pattern | Use Case |
|---|---|---|
| **Queue-based** | `@Endpoint()` on function | Simple single-function endpoints |
| **Route-based** | `Endpoint` class + `@api.get/@api.post` | Multi-route REST APIs |
| **Docker-based** | Custom Docker image | Complex dependencies |
| **Hub-based** | Deploy from RunPod Hub | Pre-built solutions |

### Queue-Based Example

```python
from runpod_flash import Endpoint

@Endpoint(name="my-endpoint", gpu=GpuGroup.ADA_24)
async def handler(text: str) -> dict:
    return {"result": process(text)}
```

### Route-Based Example

```python
from runpod_flash import Endpoint, GpuGroup

api = Endpoint(
    name="text-api",
    cpu="cpu5c-4-8",      # 4 vCPU, 8GB RAM
    workers=(0, 3),       # Scale 0 to 3
    idle_timeout=600
)

@api.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}

@api.post("/analyze")
async def analyze(text: str) -> dict:
    return {"words": len(text.split())}
```

### GPU Groups (GpuGroup enum)

| Group | GPUs | VRAM |
|---|---|---|
| `ANY` | Any available GPU | Varies |
| `ADA_24` | RTX 4090, L4 | 24GB |
| `ADA_48` | RTX 6000 Ada, L40S | 48GB |
| `AMPERE_24` | A5000, RTX 3090 | 24GB |
| `AMPERE_48` | A40, RTX A6000 | 48GB |
| `AMPERE_80` | A100-80GB | 80GB |
| `ADA_80_PRO` | H100-80GB | 80GB |
| `HOPPER_141` | H200 | 141GB |

### CPU Types

| Type | vCPU | RAM |
|---|---|---|
| `cpu5c-4-8` | 4 | 8GB |
| `cpu5c-8-16` | 8 | 16GB |
| `cpu3c-4-8` | 4 | 8GB |
| `cpu3g-4-16` | 4 | 16GB |

**Note:** CPU endpoints restricted to EU-RO-1 datacenter.

### Key Limitations

- **1.5GB deployment artifact limit** (auto-excludes PyTorch/triton from bundle)
- Python 3.12 strict requirement
- macOS/Linux only
- Worker quota: 30 workers total across all endpoints (standard account)

### Worker States

| State | Description |
|---|---|
| **IDLE** | Warm, waiting for requests |
| **RUNNING** | Processing a request |
| **THROTTLED** | Exceeded concurrent request limit |

---

## 5. Flash CLI

Complete CLI for Flash SDK development and deployment. See `docs/reports/runpod_sections/agent05.md` for exhaustive detail.

### Commands

| Command | Description |
|---|---|
| `flash init` | Scaffold new project (creates handler.py, .flashignore, pyproject.toml) |
| `flash login` | Browser-based auth, stores credentials |
| `flash run` | Local dev server with auto-provisioning, Swagger UI at `/docs` |
| `flash build` | Package deployment artifact (1.5GB max) |
| `flash deploy` | Deploy to RunPod (or `--preview` for temporary preview) |
| `flash app list/create/get/delete` | Manage Flash apps |
| `flash env list/get/delete` | Manage environments |
| `flash undeploy` | Clean up endpoints with double-confirmation safety |
| `flash update` | Self-update with automatic background version checks |

### Dev Server (`flash run`)

- Auto-provisions GPU/CPU workers
- Lazy mode (default): provisions on first request
- Eager mode (`--eager`): provisions at startup
- Swagger UI at `http://localhost:PORT/docs`
- Hot-reload on code changes

### Deployment Workflow

```bash
flash init              # Scaffold project
# ... develop locally ...
flash run               # Test locally with dev server
flash deploy            # Deploy to production
flash deploy --preview  # Deploy preview (temporary URL)
flash undeploy          # Clean up when done
```

---

## 6. RunPod CLI (runpodctl)

### Installation

```bash
# Auto-detect OS/arch
bash <(wget -qO- cli.runpod.io)
# or
bash <(curl -sL cli.runpod.io)

# macOS Homebrew
brew install runpod/runpodctl/runpodctl

# conda/mamba
conda install conda-forge::runpodctl
```

### Configuration

```bash
runpodctl doctor                    # Interactive first-time setup
runpodctl config --apiKey YOUR_KEY  # Manual API key setup
```

Config stored at `~/.runpod/config.toml`.

### Command Groups

| Command | Alias | Description |
|---|---|---|
| `pod` | — | Pod lifecycle management |
| `serverless` | `sls` | Serverless endpoint management |
| `template` | `tpl` | Template CRUD |
| `hub` | — | Hub marketplace |
| `network-volume` | `nv` | Network volume management |
| `registry` | `reg` | Container registry auth |
| `gpu` | — | List available GPUs |
| `datacenter` | `dc` | List datacenters |
| `billing` | — | View billing history |
| `user` | `me` | Account info |
| `ssh` | — | SSH key management |
| `send` | — | P2P file send |
| `receive` | — | P2P file receive |
| `doctor` | — | Setup diagnostics |
| `update` | — | Self-update |

### File Transfer (P2P)

```bash
# On sender machine
runpodctl send myfile.gguf
# Output: code is 1234-word-phrase

# On receiver machine (e.g., inside Pod)
runpodctl receive 1234-word-phrase
```

Best for small-to-medium files. For large files, use SCP/rsync.

### Template Management

```bash
# List templates
runpodctl template list --type official
runpodctl template search "pytorch"

# Create Pod template
runpodctl template create \
  --name "navigator" \
  --image "nvidia/cuda:12.6.3-runtime-ubuntu22.04" \
  --container-disk-in-gb 50 \
  --volume-in-gb 100 \
  --ports "7860/http,11434/http,22/tcp" \
  --env '{"OLLAMA_HOST":"0.0.0.0"}'

# Create Serverless template (distinct from Pod templates!)
runpodctl template create \
  --name "navigator-serverless" \
  --image "your-registry/navigator:v1" \
  --serverless
```

**Important:** Pod templates and Serverless templates are distinct. Use `--serverless` flag for Serverless templates. Each Serverless template binds to ONE endpoint only.

### Network Volume Management

```bash
runpodctl nv list
runpodctl nv create --name "navigator-data" --size 100 --data-center-id "US-GA-1"
runpodctl nv update <id> --size 200  # Can only grow, never shrink
runpodctl nv delete <id>             # PERMANENT — all data lost
```

### SSH

```bash
runpodctl ssh list-keys                           # List account SSH keys
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub  # Add public key
runpodctl ssh info <pod-id>                       # Get SSH connection details (does NOT connect)
```

### Billing

```bash
runpodctl billing pods --bucket-size day --start-time 2026-04-01T00:00:00Z
runpodctl billing serverless --grouping endpointId
runpodctl billing network-volume
```

---

## 7. API Reference

RunPod exposes **5 distinct API surfaces**, all authenticated with Bearer token (except S3).

### API Surfaces

| API | Base URL | Protocol |
|---|---|---|
| Serverless Endpoints | `https://api.runpod.ai/v2/{ENDPOINT_ID}/` | REST |
| Pods REST | `https://rest.runpod.io/v1/pods` | REST |
| Network Volumes REST | `https://rest.runpod.io/v1/networkvolumes` | REST |
| GraphQL | `https://api.runpod.io/graphql` | GraphQL |
| S3 Storage | `https://s3api-{DATACENTER}.runpod.io/` | S3/AWS SigV4 |
| Public Endpoints | `https://api.runpod.ai/v2/{MODEL_SLUG}/` | REST |

### Serverless Endpoints API (8 operations)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/runsync` | Synchronous job (30s timeout) |
| POST | `/run` | Asynchronous job |
| GET | `/status/{JOB_ID}` | Check job status |
| GET | `/stream/{JOB_ID}` | Stream job output |
| POST | `/cancel/{JOB_ID}` | Cancel running job |
| POST | `/retry/{JOB_ID}` | Retry failed job |
| POST | `/purge-queue` | Clear all queued jobs |
| GET | `/health` | Endpoint health check |

Request body: `input` (required), optional `webhook`, `policy` (executionTimeout, lowPriority, ttl), `s3Config`.

### Pods REST API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/pods` | Create pod |
| POST | `/v1/pods/{ID}/start` | Start stopped pod |
| POST | `/v1/pods/{ID}/stop` | Stop running pod |
| PATCH | `/v1/pods/{ID}` | Update pod |
| DELETE | `/v1/pods/{ID}` | Terminate pod |

### S3-Compatible Storage

Endpoint: `https://s3api-{DATACENTER}.runpod.io/`

Uses **separate S3 API key** (not RunPod API key). Supports: CreateBucket, DeleteBucket, ListBuckets, PutObject, GetObject, DeleteObject, ListObjectsV2, HeadObject, multipart uploads.

**NOT supported:** bucket policies, ACLs, versioning, lifecycle rules, notifications.

### MCP Tools (28 available)

RunPod provides MCP server tools for programmatic access:
- **Pods**: create, get, list, start, stop, delete, update
- **Endpoints**: create, get, list, update, delete
- **Templates**: create, get, list, update, delete
- **Network Volumes**: create, get, list, update, delete
- **Registry Auth**: create, get, list, delete

---

## 8. Storage & Network Volumes

### Storage Comparison

| Feature | Container Disk | Volume Disk | Network Volume |
|---|---|---|---|
| **Persistence** | Lost on stop/restart | Survives stop, lost on terminate | Survives termination |
| **Mount** | / (root) | /workspace | Configurable |
| **Speed** | Fastest (local NVMe) | Fast (local NVMe) | 200-400 MB/s sustained |
| **Portability** | None | None | Attach to any pod/endpoint |
| **Pricing** | $0.10-$0.20/GB/month | $0.10-$0.20/GB/month | $0.07/GB/month (first 1TB) |
| **Max Size** | — | — | 4000 GB |

### Network Volume Details

- **Performance**: NVMe SSD-backed, 200-400 MB/s sustained, 10 GB/s peak burst
- **Pricing**: $0.07/GB/month (first 1TB), $0.05/GB/month (beyond 1TB)
- **Mount paths**: `/runpod-volume` for Serverless, `/workspace` for Pods
- **Can only grow**, never shrink
- Deletion is permanent
- REST API: `POST https://rest.runpod.io/v1/networkvolumes`

### File Transfer Methods

| Method | Best For | Speed |
|---|---|---|
| `runpodctl send/receive` | Quick P2P transfers | Moderate |
| SCP/rsync over SSH | Large files, incremental sync | Fast |
| Cloud Sync | AWS S3, GCS, Azure, Backblaze, Dropbox | Varies |
| S3 API | Programmatic access | Moderate |

---

## 9. GPU Types Reference

### Key GPUs for ML Workloads

| GPU | VRAM | Use Case | Approx Pod Price |
|---|---|---|---|
| RTX 4090 | 24 GB | Inference, small training | ~$0.44/hr |
| L4 | 24 GB | Inference, cost-effective | ~$0.24/hr |
| L40S | 48 GB | Medium models | ~$0.74/hr |
| A100-80GB | 80 GB | Large training/inference | ~$1.64/hr |
| H100-80GB | 80 GB | Fastest training | ~$2.49/hr |
| H200 | 141 GB | Largest models | ~$3.29/hr |

### Serverless GPU Pools

| Pool ID | Included GPUs |
|---|---|
| AMPERE_16 | A2, RTX 3070 |
| AMPERE_24 | A5000, RTX 3090, RTX 4090 |
| ADA_24 | RTX 4090, L4 |
| AMPERE_48 | A40, RTX A6000 |
| ADA_48_PRO | RTX 6000 Ada, L40S |
| AMPERE_80 | A100-80GB |
| ADA_80_PRO | H100-80GB |
| HOPPER_141 | H200-141GB |

### GPU Selection Guide

| Workload | Recommended GPU | VRAM Needed |
|---|---|---|
| Small LLM inference (< 8B params) | L4, RTX 4090 | 16-24 GB |
| Medium LLM inference (8-30B params) | A100-80GB | 48-80 GB |
| Large LLM inference (30B+ params) | H100, H200 | 80-141 GB |
| Fine-tuning (QLoRA) | A100-80GB | 40-80 GB |
| Image generation | RTX 4090, L4 | 16-24 GB |

---

## 10. Public Endpoints

Pre-deployed AI model APIs with instant access. No setup required — just API key and POST request.

### Authentication

```bash
curl -X POST "https://api.runpod.ai/v2/MODEL_SLUG/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {...}}'
```

### Request Modes

- **Sync** (`/runsync`): Immediate result, results expire in 1 minute
- **Async** (`/run`): Queue job, poll `/status/{JOB_ID}`, results expire in 30 minutes

### Response Format

```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "delayTime": 150,
  "executionTime": 2500,
  "output": { ... }
}
```

### Model Categories

- **Text**: GPT-OSS (120B), various LLMs with OpenAI-compatible endpoints
- **Image**: SDXL, Flux, Stable Diffusion variants ($0.0025-$0.027/image)
- **Video**: Various video generation models ($0.06-$0.15/second)
- **Audio**: Chatterbox TTS ($0.001/second)

### Rate Limits

- `/runsync`: 2000 requests/10s
- `/run`: 1000 requests/10s
- Cold start on first request: 30-60 seconds

### OpenAI Compatibility

LLM public endpoints support OpenAI-compatible API format — works with Cursor, Cline, Vercel AI SDK, and any OpenAI-compatible client.

**No charge for failed generations.**
**Output URLs expire after 7 days** — download immediately.

---

## 11. Hub & Templates

### RunPod Hub

Marketplace at `console.runpod.io/hub` for discovering and deploying pre-configured AI repos.

### Publishing to Hub

1. Prepare GitHub repo with `handler.py`, `Dockerfile`, `README.md`
2. Create `.runpod/hub.json` and `.runpod/tests.json`
3. Create GitHub release (Hub indexes **releases, not commits**)
4. RunPod builds, tests, then **manually reviews** before publication

### hub.json Schema

```json
{
  "title": "Your Tool Name",
  "description": "Brief description",
  "type": "serverless",
  "category": "language",
  "config": {
    "runsOn": "GPU",
    "containerDiskInGb": 20,
    "gpuCount": 1,
    "gpuIds": "ADA_24",
    "env": [
      {"key": "MODEL_NAME", "input": {"name": "Model", "type": "string", "default": "..."}}
    ]
  }
}
```

Categories: `audio`, `embedding`, `language`, `video`, `image`.

### tests.json Schema

```json
{
  "tests": [
    {"name": "test_basic", "input": {"prompt": "test"}, "timeout": 10000}
  ],
  "config": {
    "gpuTypeId": "NVIDIA GeForce RTX 4090",
    "gpuCount": 1
  }
}
```

Test passes if endpoint returns HTTP 200.

### Revenue Sharing

| Monthly Compute Hours | Revenue Share |
|---|---|
| 10,000+ | 7% |
| 5,000-9,999 | 5% |
| 1,000-4,999 | 3% |
| 100-999 | 1% |
| Below 100 | 0% |

Paid as **RunPod credits** (not cash), first week of each month. Tiers reset monthly.

### Pod Templates

| Type | Description | Support |
|---|---|---|
| **Official** | Curated by RunPod | Full support |
| **Community** | User-created | Discord only |
| **Custom** | Private, your own | Self-supported |

Create via REST API:
```bash
curl -X POST https://rest.runpod.io/v1/templates \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"navigator","imageName":"IMAGE","containerDiskInGb":50,"ports":["7860/http","11434/http","22/tcp"],"volumeInGb":100,"volumeMountPath":"/workspace","isPublic":false,"isServerless":false}'
```

### Docker Best Practices

- Extend RunPod base images: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **Always build with `--platform linux/amd64`** (required for RunPod)
- **Use versioned tags, NOT `:latest`** — RunPod caches images
- Three startup modes: default services, app + services (`/start.sh &`), app-only (`ENTRYPOINT []`)

---

## 12. Tutorials

### Container Basics

- Three base image tiers: bare CUDA, framework-specific (PyTorch), RunPod base (includes Jupyter/SSH)
- **Critical:** All images must be `--platform=linux/amd64`
- **Never use `:latest` tag** — RunPod caches images, causing stale deployments
- RunPod base images include `/start.sh` for Jupyter/SSH services
- Serverless Dockerfile pattern: install deps, copy handler.py, `CMD ["python", "/handler.py"]`

### Ollama on a Pod (CRITICAL for Navigator)

**Source:** https://docs.runpod.io/tutorials/pods/run-ollama

1. **Deploy Pod**: Use PyTorch template, expose port **11434**, set `OLLAMA_HOST=0.0.0.0`
2. **Install Ollama**:
   ```bash
   apt update && apt install -y lshw zstd
   curl -fsSL https://ollama.com/install.sh | sh
   (ollama serve > ollama.log 2>&1) &
   ```
3. **Pull/Create Model**:
   ```bash
   ollama pull gemma3:4b
   # OR for custom GGUF:
   ollama create navigator -f /workspace/Modelfile
   ```
4. **External Access URL**: `https://POD_ID-11434.proxy.runpod.net/api/generate`
5. **Test**:
   ```bash
   curl https://POD_ID-11434.proxy.runpod.net/api/generate \
     -d '{"model": "navigator", "prompt": "Hello", "stream": false}'
   ```

**Key notes:**
- Streaming is default; add `"stream": false` for non-streaming
- No Docker Compose on Pods — Ollama + Gradio must run as co-located processes
- Pods have **no cold start** (always running) — more reliable for live demos

### Flash REST API Tutorial

- Build multi-route REST API with CPU + GPU endpoints
- `Endpoint` class with `@api.get()` / `@api.post()` decorators
- Separate CPU (text processing) and GPU (ML inference) endpoints
- Local testing with `flash run` (Swagger UI at `/docs`)
- Deploy with `flash deploy` — separate endpoints scale independently

### Web App Integration (Serverless)

- Deploy model from Hub, get endpoint ID + API key
- Frontend `fetch()` with `Authorization: Bearer` header
- POST to `https://api.runpod.ai/v2/ENDPOINT_ID/runsync`
- Response: `delayTime`, `executionTime`, `status`, `output`

---

## 13. Instant Clusters

Multi-node GPU clusters with high-speed networking for distributed training and large-scale inference.

### Architecture

- Multiple GPU nodes provisioned in the **same data center**
- One node designated **primary** (`NODE_RANK=0`)
- Pre-configured static IPs and environment variables
- Network: `ens1`-`ens8` for inter-node (up to 3200 Gbps), `eth0` for external only

### Supported Hardware

| GPU | Network Speed | Node Range | GPU Range |
|---|---|---|---|
| **B200** | 3200 Gbps | 2-8 nodes | 16-64 GPUs |
| **H200** | 3200 Gbps | 2-8 nodes | 16-64 GPUs |
| **H100** | 3200 Gbps | 2-8 nodes | 16-64 GPUs |
| **A100** | 1600 Gbps | 2-8 nodes | 16-64 GPUs |

Each node has 8 GPUs. For >8 nodes (up to 512 GPUs), contact sales.

### Auto-Set Environment Variables

| Variable | Description |
|---|---|
| `PRIMARY_ADDR` / `MASTER_ADDR` | Primary node address (both equivalent) |
| `PRIMARY_PORT` / `MASTER_PORT` | Primary node port |
| `NODE_ADDR` | This node's static IP on cluster network |
| `NODE_RANK` | Global rank (0 = primary) |
| `NUM_NODES` | Total nodes in cluster |
| `NUM_TRAINERS` | GPUs per node |
| `WORLD_SIZE` | Total GPUs (`NUM_NODES * NUM_TRAINERS`) |
| `HOST_NODE_ADDR` | `PRIMARY_ADDR:PRIMARY_PORT` |

### Critical NCCL Configuration

```bash
export NCCL_SOCKET_IFNAME=ens1  # REQUIRED — use high-bandwidth internal network
export NCCL_DEBUG=INFO           # Optional — debug logging
```

**WARNING:** Without `NCCL_SOCKET_IFNAME=ens1`, NCCL defaults to `eth0` (172.xxx range) causing timeouts. This is the #1 failure mode.

### PyTorch Distributed Training

```bash
torchrun \
  --nproc_per_node=$NUM_TRAINERS \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  your_script.py
```

Minimal handler:
```python
import os, torch, torch.distributed as dist

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return local_rank, dist.get_rank(), dist.get_world_size(), device
```

### Axolotl Fine-Tuning

```bash
torchrun \
  --nnodes $NUM_NODES \
  --node_rank $NODE_RANK \
  --nproc_per_node $NUM_TRAINERS \
  --rdzv_id "myjob" \
  --rdzv_backend static \
  --rdzv_endpoint "$PRIMARY_ADDR:$PRIMARY_PORT" \
  -m axolotl.cli.train lora-1b.yml
```

**IMPORTANT:** Dynamic `c10d` rendezvous backend is NOT supported. Must use `--rdzv_backend static`.

### Managed Slurm Clusters

- Pre-installed Slurm + munge, zero configuration
- **Only supports official RunPod PyTorch images** — deploying other images means Slurm won't start
- Standard Slurm commands (`sinfo`, `sbatch`, `squeue`, `scontrol`) work out-of-the-box
- Config at `/etc/slurm/slurm.conf` and `/etc/slurm/gres.conf`

### Deployment

1. Go to `console.runpod.io/cluster`
2. Click **Create Cluster**
3. Select cluster type (Standard or Slurm), pod count, GPU type, region
4. Optional: attach network volume (region must match)
5. **Always delete clusters when done** — monitor via Billing Explorer > Cluster tab

### Key Warnings

1. **NCCL_SOCKET_IFNAME=ens1** — must be set for any distributed training
2. **Static rendezvous only** — `c10d` dynamic backend not supported
3. **Slurm requires PyTorch images** — managed Slurm only works with official RunPod PyTorch
4. **Default spend limits** — contact `help@runpod.io` for larger clusters
5. **Do NOT use `eth0`** for inter-node comms — 172.xxx IPs cause timeouts

---

## 14. Integrations

RunPod integrates with external tools via REST APIs, OpenAI-compatible endpoints, and orchestration frameworks.

### OpenAI-Compatible Endpoints (Key Pattern)

The most broadly useful integration pattern. Any tool that accepts a custom OpenAI base URL works with RunPod:

```
https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1
```

Authentication: RunPod API key as Bearer token. Works with LangChain, CrewAI, n8n, Cursor, Vercel AI SDK, and any OpenAI-compatible client.

**Available public endpoints:**
- Qwen3 32B AWQ: `https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1`
- IBM Granite-4.0-H-Small: `https://api.runpod.ai/v2/granite-4-0-h-small/openai/v1`

**vLLM tool calling** (required for agentic integrations):
```bash
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=hermes
REASONING_PARSER=qwen3
```
**Warning:** Not all models support tool calling.

### dstack (Pod Orchestration)

Open-source tool for automated Pod orchestration via YAML configs.

```bash
pip install -U "dstack[all]"
```

Config at `~/.dstack/server/config.yml`:
```yaml
projects:
  - name: main
    backends:
      - type: runpod
        creds:
          type: api_key
          api_key: YOUR_RUNPOD_API_KEY
```

Task config (`.dstack.yml`):
```yaml
type: task
name: vllm-inference
python: "3.10"
env:
  - MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
commands:
  - pip install vllm
  - vllm serve $MODEL_NAME --port 8000
ports:
  - 8000
spot_policy: on-demand
resources:
  gpu:
    name: "RTX4090"
    memory: "24GB"
```

```bash
dstack server    # Start server (provides ADMIN-TOKEN)
dstack init      # Initialize project
dstack apply     # Deploy task
dstack stop NAME # Stop and release resources
```

**Volumes** (persistent storage, region-locked):
```yaml
type: volume
name: my-volume
backend: runpod
region: EUR-IS-1
size: 100GB
```

### SkyPilot (Multi-Cloud Framework)

```bash
pip install "runpod>=1.6"
runpod config                          # Paste API key
pip install "skypilot-nightly[runpod]" # Note: nightly build
sky check                              # Verify setup
sky launch -c mycluster hello_sky.yaml # Deploy
```

Task config:
```yaml
resources:
  cloud: runpod
workdir: .
setup: |
  pip install -r requirements.txt
run: |
  python train.py
```

**Note:** Uses `skypilot-nightly` (not stable release), suggesting RunPod integration is newer.

### Mods (CLI Tool)

AI-powered command-line tool that integrates with Unix pipelines.

```yaml
# config_template.yml
runpod:
  base-url: https://api.runpod.ai/v2/${YOUR_ENDPOINT}/openai/v1
  api-key-env: RUNPOD_API_KEY
  models:
    openchat/openchat-3.5-1210:
      aliases: ["openchat"]
      max-input-chars: 8192
```

```bash
ls ~/Downloads | mods --api runpod --model openchat -f "describe these files" | glow
```

### n8n (Workflow Automation)

Connect RunPod vLLM endpoints to n8n workflows for AI-powered automation.

1. Deploy vLLM Serverless endpoint (or use public endpoint shortcut)
2. In n8n: Add "AI Agent" node > "OpenAI Chat Model" node
3. Configure credentials:
   - **API Key**: RunPod API key
   - **Base URL**: `https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1`
4. Select model: `qwen/qwen3-32b-awq`

**Warning:** Initial requests may take minutes as endpoint scales up. Connection test confirms reachability but NOT full compatibility.

### Transformer Lab (ML Research Environment)

Open-source research environment for training, fine-tuning, and evaluating models.

```bash
curl -fsSL https://lab.cloud/install.sh | bash -s -- multiuser_setup
cd ~/.transformerlab/src && ./run.sh  # Web UI at http://localhost:8338
```

Provider config in Team Settings:
```json
{
  "api_key": "YOUR_RUNPOD_API_KEY",
  "api_base_url": "https://rest.runpod.io/v1"
}
```

Task config:
```yaml
name: hello-runpod
resources:
  compute_provider: runpod-provider
  accelerators: "A40:1"
setup: |
  pip install torch
run: |
  python train.py
```

**Requires shared cloud storage** (S3, GCS, or Azure Blob) for remote task execution.

### Integration Limitations

- **vLLM tool calling**: Not all models support it — check per-model compatibility
- **dstack volumes are region-locked**: Volume region ties the Pod to that region
- **Transformer Lab needs cloud storage**: S3/GCS/Azure beyond RunPod itself
- **SkyPilot uses nightly builds**: `skypilot-nightly[runpod]`, not stable release
- **n8n connection test insufficient**: Confirms reachability, not format compatibility
- **No dedicated pages for LangChain, CrewAI, W&B, HuggingFace** — use OpenAI-compatible endpoint pattern

---

## 15. Navigator Deployment Guide

Based on all research, here is the recommended RunPod deployment strategy for NorthStar Navigator.

### Recommended Approach: Pod with L4 GPU

| Parameter | Value | Rationale |
|---|---|---|
| **GPU** | NVIDIA L4 (24GB) | 5.3GB GGUF fits easily, $0.24/hr |
| **Image** | `nvidia/cuda:12.6.3-runtime-ubuntu22.04` | Matches current Modal setup |
| **Container Disk** | 50 GB | Room for Ollama + model + deps |
| **Volume** | 100 GB at /workspace | Persist model + ChromaDB |
| **Ports** | 7860/http, 11434/http, 22/tcp | Gradio, Ollama, SSH |
| **Env** | `OLLAMA_HOST=0.0.0.0` | Required for external Ollama access |

### Setup Script (run inside Pod)

```bash
#!/bin/bash
# 1. Install system deps
apt-get update && apt-get install -y curl zstd python3-pip git && rm -rf /var/lib/apt/lists/*

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Start Ollama in background
export OLLAMA_HOST=0.0.0.0
(ollama serve > /workspace/ollama.log 2>&1) &
sleep 5

# 4. Download GGUF from HuggingFace Hub
pip install huggingface_hub
huggingface-cli download wanderduck/northstar-navigator-gguf \
  --local-dir /workspace/gguf \
  --include "*.gguf" "Modelfile"

# 5. Create Ollama model from GGUF
ollama create navigator -f /workspace/gguf/Modelfile

# 6. Install Navigator dependencies
pip install gradio chromadb sentence-transformers rank-bm25 \
  textstat ollama pydantic httpx fastapi

# 7. Clone/copy Navigator source
# (Use runpodctl send/receive or git clone)

# 8. Start Gradio
cd /workspace/navigator
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
NAVIGATOR_DATA_DIR=/workspace/navigator/data \
python src/app.py
```

### Access URLs

- **Gradio UI**: `https://[POD_ID]-7860.proxy.runpod.net`
- **Ollama API**: `https://[POD_ID]-11434.proxy.runpod.net`
- **SSH**: `ssh root@[PUBLIC_IP] -p [TCP_PORT]`

### Cost Estimate

- L4 GPU: ~$0.24/hr = ~$5.76/day
- With $60 RunPod credit: ~10 days continuous runtime
- For demo purposes: start pod when needed, stop when done (saves on GPU cost, pays storage only)

### Alternative: Serverless (if scale-to-zero needed)

If cost is more important than latency:
1. Create Docker image with Ollama + GGUF + Navigator baked in
2. Deploy as Serverless endpoint
3. Use `/runsync` for requests
4. Scale to zero when idle (no cost)
5. **Tradeoff**: Cold start of 3-10 minutes on first request

### Pod vs Serverless for Navigator

| Factor | Pod | Serverless |
|---|---|---|
| **Cold start** | None (always running) | 3-10 minutes |
| **Cost when idle** | $0.24/hr (GPU) or $0.10-0.20/GB/mo (stopped) | $0 |
| **Setup complexity** | Low (SSH in, install, run) | Medium (Dockerfile, handler) |
| **Reliability** | High | Medium (cold starts, timeouts) |
| **Best for** | Live demos, development | Production APIs |

**Recommendation for hackathon demo: Pod** — no cold start, immediate availability, easy to debug via SSH/JupyterLab.

---

## 16. Key Warnings & Gotchas

1. **Pods with network volumes cannot be stopped**, only terminated (data persists in volume)
2. **Stopping a pod preserves /workspace** but clears container disk; you still pay volume storage
3. **Restarting a stopped pod may get zero GPUs** if datacenter capacity changed
4. **Editing a running pod resets it completely** — save work first
5. **HTTP proxy has 100-second Cloudflare timeout** — use WebSocket/polling for long requests
6. **Updating env vars restarts the pod** and clears non-volume data
7. **All Docker images must be `--platform linux/amd64`** — critical for ARM Mac users
8. **Never use `:latest` tag** — RunPod caches images, causing stale deployments
9. **Secret values cannot be viewed after creation**, deletion is permanent
10. **Network volumes can only grow**, never shrink; deletion is permanent
11. **Serverless templates are distinct from Pod templates** — use `--serverless` flag
12. **Serverless result retention**: 1 minute (sync), 30 minutes (async) — fetch promptly
13. **Output URLs from Public Endpoints expire after 7 days**
14. **Flash requires Python 3.12 strictly**, has 1.5GB artifact limit
15. **Flash CPU endpoints restricted to EU-RO-1 datacenter**
16. **Hub indexes GitHub releases, not commits** — `.runpod/` dir takes precedence over root
17. **No Docker Compose, no UDP, no Windows** on Pods
18. **Config lives at `~/.runpod/config.toml`** (CLI), not environment variables
19. **`runpodctl ssh info` does NOT start an SSH session** — only returns connection details
20. **Account default spend limit is $80/hr** — adjust in console if needed
21. **Instant Clusters: `NCCL_SOCKET_IFNAME=ens1` is mandatory** for distributed training — without it, NCCL uses eth0 causing timeouts
22. **Instant Clusters: only `--rdzv_backend static`** — dynamic c10d rendezvous not supported
23. **Managed Slurm only works with official RunPod PyTorch images** — custom images prevent Slurm from starting
24. **OpenAI-compatible base URL**: `https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1` — works with LangChain, CrewAI, n8n, any OpenAI client
25. **vLLM tool calling not universal** — requires `ENABLE_AUTO_TOOL_CHOICE=true` + model-specific parsers; not all models support it
26. **dstack volumes are region-locked** — creating a volume ties the Pod to that specific region
27. **SkyPilot uses `skypilot-nightly[runpod]`** (nightly build, not stable release)
