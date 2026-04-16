# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle "Gemma 4 Good Hackathon" competition entry: **NorthStar Navigator** — a plain-language government benefits navigator for Minnesota. Deadline: May 18, 2026.

Deliverables: video demo (≤3 min), Kaggle writeup (≤1500 words), public code repo, live demo.

## Environment Setup

- **Python 3.13** managed with **uv** (lock file: `uv.lock`)
- Uses **direnv** — run `direnv allow` to set `LD_LIBRARY_PATH` for CUDA/cuDNN libs
- GPU stack: TensorFlow 2.21 (CUDA 12) + PyTorch (CUDA 13) + RAPIDS cuDF/cuML/cuGraph (CUDA 12)
- CUDA libs are preloaded via ctypes in the notebook's GPU cell (dual cu12/cu13 setup)
- Environment variables in `.env` (gitignored)

## Commands

```bash
uv sync                    # Install/sync dependencies
uv sync --extra navigator  # Install navigator-specific deps (chromadb, gradio, etc.)
uv add <package>           # Add a dependency
jupyter lab                # Launch JupyterLab
uv run pytest tests/ -v    # Run test suite (uv run required for correct env)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python src/app.py  # Launch Gradio UI (http://localhost:7860)
```

**Important:** Always prefix Python/pytest commands with `uv run` to use the correct virtualenv.

### RunPod Deployment (Primary)

```bash
bash deploy/runpod/deploy.sh                         # Create pod (RTX 4090, or RUNPOD_GPU="NVIDIA L4" for fallback)
python deploy/runpod/runpod_deploy.py create          # Alternative: Python API with GPU fallback chain
python deploy/runpod/runpod_deploy.py list             # List pods
python deploy/runpod/runpod_deploy.py stop <pod_id>    # Stop pod (preserves /workspace)
python deploy/runpod/runpod_deploy.py start <pod_id>   # Restart pod
# Inside pod via SSH:
bash deploy/runpod/setup.sh                           # First-run setup (~10 min)
bash deploy/runpod/start.sh                           # Start Ollama + Gradio (every pod start)
python deploy/runpod/health_check.py                  # Verify services healthy
```

### Ollama

```bash
ollama pull gemma3:4b              # Pull base model (fallback)
ollama create navigator -f output/gguf/Modelfile  # Load fine-tuned GGUF into Ollama
ollama list                        # Check installed models
```

Model tag configured in `src/navigator/config.py` → `OLLAMA_MODEL`. Currently `navigator` (fine-tuned Gemma 4 E4B GGUF via QLoRA).

### Scraping & Training

```bash
PYTHONPATH=src uv run python scripts/scrape_dhs_manual.py     # Scrape DHS Combined Manual
PYTHONPATH=src uv run python scripts/scrape_county_pages.py   # Scrape 5 county + 3 CAP agency sites
PYTHONPATH=src uv run python scripts/download_sam_gov.py      # Download SAM.gov assistance listings (needs SAM_GOV_API_KEY in .env)
PYTHONPATH=src PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python scripts/ingest_all.py  # Ingest scraped data into ChromaDB
```

Training scripts run on Kaggle/Colab only (need unsloth, trl, transformers, datasets):
```bash
python training/train_unsloth.py --dataset data/training/combined.jsonl  # QLoRA fine-tune
python training/export_gguf.py --model training/output/final             # Export to GGUF for Ollama
```

### Modal Fine-Tuning (Cloud GPU)

```bash
modal run deploy/modal_finetune.py        # Unsloth QLoRA on A100 (preferred)
modal run deploy/modal_finetune_plain.py  # Raw PEFT QLoRA on A100 (fallback)
modal run deploy/modal_finetune_plain.py::convert_gguf  # Re-merge LoRA + GGUF export without retraining
modal volume get navigator-finetune-output /gguf/ ./output/gguf/  # Download GGUF to local
modal run deploy/modal_generate_training.py --workers 4              # Generate English (Gemini) + translate (Cloud Translate NMT)
modal run deploy/modal_generate_training.py --count 500 --workers 2  # Smaller run
modal run deploy/modal_generate_training.py --review-only            # Review/improve existing translations with Gemini
modal secret create gcloud-translate GOOGLE_CLOUD_PROJECT=<project-id> GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat service-account.json)"  # Cloud Translate auth
modal run deploy/modal_finetune_multilingual.py --lang all  # Fine-tune Hmong + Somali LoRA adapters
```

### DHS Scraper Cookie Workflow

DHS site uses Radware CAPTCHA that blocks all headless browsers. Export cookies from a headed session first:
```bash
PYTHONPATH=src uv run python scripts/export_dhs_cookies.py  # Opens headed browser, solve CAPTCHA, saves cookies
PYTHONPATH=src uv run python scripts/scrape_dhs_manual.py   # Uses saved cookies automatically
```

## Project Status

- **NorthStar Navigator** (codename: L'Étoile Link) — implementation complete, 75 tests passing
  - Design spec: `docs/superpowers/specs/2026-04-05-plain-language-government-navigator-design.md`
  - Implementation plan: `docs/superpowers/plans/2026-04-05-plain-language-government-navigator.md`
  - Architecture: Three-stage pipeline (Intake → Eligibility → Response) with Gemma 4 E4B via Ollama
  - Prize targets: Main + Digital Equity + Safety & Trust + Ollama + Unsloth ($130K ceiling)
  - Fine-tuning: QLoRA on A100 complete, GGUF q4_k_m exported, loaded into Ollama as `navigator`
  - Data: 267 county programs, 515 SAM.gov federal listings, 614 DHS Combined Manual sections scraped; 1,938 documents in ChromaDB
  - Multilingual: Hmong/Somali training pipeline — Cloud Translate NMT for translation, Gemini for English generation (3-model fallback chain), per-language LoRA fine-tuning
  - Multilingual fine-tuning: Complete — 4x A100-80GB DDP, 3 epochs, ~10K examples (EN+ES+HMN+SO), GGUF q4_k_m (5.3GB) loaded into Ollama
  - Live demo: Modal deployment via `deploy/modal_app.py` (`@app.cls` + `@modal.enter()` + ASGI on T4); HuggingFace Space at `wanderduck/northstar-navigator` (Docker SDK, in progress)
  - HF Hub: GGUF model at `wanderduck/northstar-navigator-gguf` (q4_k_m, 5.3GB)
  - RunPod deployment: Complete and tested — Pod with RTX 4090 on RunPod, Gradio works natively via `demo.launch()` + `demo.queue()`. Deployment scripts in `deploy/runpod/`. Reference manual at `docs/reports/runpod_manual.md`
  - Kaggle writeup: Complete at `kaggle_writeup/kaggle_writeup.md` (~1,163 words, under 1,500 limit)
  - Kaggle notebook: Complete at `northstar_navigator.ipynb` (10 sections, runs on T4 GPU)
  - Remaining: Video demo, stable pod URL for submission, ChromaDB transfer to pod, Cloudflare Tunnel (optional)

## Project Structure

- `src/navigator/` — Core Navigator application
  - `models.py` — Pydantic data models (UserProfile, Program, EligibilityResult)
  - `intake.py` — Stage 1: situation parsing and profile extraction
  - `eligibility.py` — Stage 2: rule-based + RAG eligibility engine
  - `response.py` — Stage 3: plain language response generation
  - `ollama_client.py` — Ollama API wrapper
  - `prompts.py` — System prompts for all pipeline stages
  - `readability.py` — Flesch-Kincaid reading level checking
  - `config.py` — Settings, paths, model configuration
  - `rag/` — RAG pipeline (ChromaDB, BM25, hybrid retrieval, ingestion)
  - `tools/` — Function calling tools (FPL calc, benefits search, county programs, docs)
- `src/app.py` — Gradio UI entry point
- `scripts/` — Data scraping and ingestion scripts
- `training/` — Unsloth fine-tuning and GGUF export
- `deploy/runpod/` — **Primary deployment** (Pod with RTX 4090, setup.sh + start.sh scripts, Dockerfile backup)
- `deploy/` — Modal cloud GPU scripts (fine-tuning, training data generation, multilingual LoRA)
- `deploy/huggingface/` — HuggingFace Spaces deployment (Docker SDK, paused)
- `kaggle_writeup/` — Competition writeup (kaggle_writeup.md)
- `northstar_navigator.ipynb` — Kaggle submission notebook (Ollama + Gemma 4 demo, runs on T4)
- `tests/` — Full test suite (pytest, `pythonpath = ["src"]` in pyproject.toml)
- `data/` — Scraped data, ChromaDB store, training datasets
- `models/` — Gemma 4 model weights (31B + E4B)
- `docs/competition/` — competition rules and overview
- `docs/ideas/` — brainstormed competition ideas and plans
- `docs/research/` — domain research reports from sub-agent research
- `docs/superpowers/` — design specs and implementation plans
- `gemma4_goodhackathon_main.ipynb` — primary working notebook
- `notebook_images/` — images referenced by notebooks

## Navigator Dependencies

Navigator deps are in `[project.optional-dependencies] navigator` to avoid conflicts with the heavy RAPIDS/CUDA stack. Install with `uv sync --extra navigator`.

## Competition Notes

- Prize tracks: Main ($100K), Impact ($50K across 5 areas), Special Technology ($50K across 5 tools — Cactus, LiteRT, llama.cpp, Ollama, Unsloth)
- Projects can win both Main Track and Special Technology prizes simultaneously
- NEC, AWS D1.1, IPC/UPC trade standards are copyrighted — use educational summaries and inspection checklists for RAG, not full standard text
- Kaggle writeup: 1,500 word max (penalty for exceeding). Saved at `kaggle_writeup/kaggle_writeup.md`
- Kaggle notebook: `northstar_navigator.ipynb` — self-contained, installs Ollama + downloads GGUF, runs on T4 GPU
- Video script and action plan: `kaggle_writeup/video_script.md`
- Cover image for Kaggle Media Gallery: `docs/Styling/NorthStar_Navigator_banner.png`
- Branding assets (colors, logos, icons): `docs/Styling/`

## Key Technical Notes

- Gradio 6.x: `ChatInterface` examples must be lists-of-lists when `additional_inputs` used; `theme` passed to `launch()` not `Blocks()`
- **Gradio streaming**: Convert `fn` to generator yielding cumulative strings. Add `generate_stream()` on the service class. `ChatInterface` handles generators natively.
- **Gradio custom theme**: Use `gr.themes.Base()` with `gr.themes.Color(c50=..., c950=...)` for hue, then `.set()` for specific overrides. Wanderduck palette defined in `docs/Styling/Wanderduck_ColourDuck_Preview.html`.
- **Gradio header icon**: `gr.Image` adds container padding. Use `gr.HTML` with inline `<img style="height:56px">` for precise icon-beside-title alignment.
- **Pydantic models for LLM output**: Use `field: type | None = None` for any field the LLM might not extract. Strict `int`/`str` without defaults will crash on multilingual or incomplete input.
- Ollama must be running (`systemctl start ollama`) before launching the Gradio UI
- `cuml.accel` is disabled due to RAPIDS cu12 vs system CUDA 13.1 header mismatch; use cuML via direct imports instead
- The notebook preloads both cu12 and cu13 shared libraries to support TensorFlow and PyTorch simultaneously
- Plotting uses Plotly (not matplotlib for interactive charts)
- ChromaDB requires `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` env var (set in `tests/conftest.py`)
- ChromaDB 1.5.5 `EmbeddingFunction` is a Protocol, not an abstract base class — adapter needs `name()`, `get_config()`, `build_from_config()`
- Navigator implementation is in worktree `.claude/worktrees/navigator-implementation/` (branch: `worktree-navigator-implementation`)
- **Gemma 4 ClippableLinear**: PEFT can't find LoRA targets because `Gemma4ClippableLinear` extends `nn.Module` not `nn.Linear`. Fix: patch `__bases__` post-load + add property delegates for `weight/bias/in_features/out_features` → `self.linear.*`. Unsloth handles this internally. **Important**: save original `__bases__` before patching and restore before any subsequent `from_pretrained()` calls in the same process, or `nn.Linear.__init__()` will fail.
- **Gemma 4 E4B DDP**: Multimodal model has vision/audio tower params that don't participate in text-only loss. Must set `ddp_find_unused_parameters=True` in SFTConfig.
- **SFTConfig not TrainingArguments**: Current TRL uses `SFTConfig` for SFT-specific params (`max_length`, `dataset_text_field`, `packing`). `max_seq_length` was removed — use `max_length`. `SFTTrainer` renamed `tokenizer` to `processing_class`.
- **`torch_dtype` → `dtype`**: transformers git HEAD renamed the param in `from_pretrained()`. Use `dtype=torch.bfloat16`.
- **SAM.gov API v1**: Response key is `assistanceListingsData` (not `assistanceListings`), CFDA in `assistanceListingId`, title in `title`, objective in `overview.objective`
- **`apply_chat_template` return type**: With `return_tensors="pt"` it returns a `BatchEncoding` dict, not a tensor. Use `return_dict=True` and access `result["input_ids"]`.
- **Gemini Python SDK**: Use `google-genai` (not deprecated `google-generativeai`). Import: `from google import genai`, client: `genai.Client(api_key=...)`.
- **Cloud Translate API**: `google-cloud-translate` v3 (NMT). Language codes: `es`, `hmn` (Hmong), `so` (Somali). Auth via service account JSON in Modal secret.
- **Gemma 4 31B**: `google/gemma-4-31B-it` needs A100-80GB in bf16. E4B fits on L4 or RTX 4090.
- **Navigator import chain**: `intake.py` and `response.py` import `OllamaClient` → triggers `import ollama` at module level. Must be installed even with alternative clients.

### Modal (Legacy — training only, not used for live demo)

Modal is used for cloud GPU training scripts (`deploy/modal_finetune*.py`, `deploy/modal_generate_training.py`) but NOT for the live demo. Gradio static files failed behind Modal's ASGI proxy. Key gotchas for training scripts:
- `modal.gpu` module removed in SDK 1.4.x — use `gpu="A100-80GB:4"` string format
- `output_vol.commit()` between phases; uncommitted writes lost on crash
- `modal secret create` for gemini-api, huggingface, gcloud-translate
- `.starmap()` for parallel GPU workers; each writes own file to avoid Volume conflicts
- `git+https://` pip installs require `.apt_install("git")`

### RunPod Deployment

**IMPORTANT: ALWAYS read `docs/reports/runpod_manual.md` before writing ANY RunPod code.**

- **RunPod documentation manual**: `docs/reports/runpod_manual.md` — comprehensive reference covering all RunPod services (Pods, Serverless, Flash SDK, CLI, API, storage, GPU types, Hub, tutorials, and Navigator-specific deployment guide)
- **Recommended deployment**: Pod with RTX 4090 ($0.44/hr, 24GB VRAM), L4 fallback ($0.24/hr). Ports 7860/http + 11434/http + 22/tcp, `OLLAMA_HOST=0.0.0.0`
- **Pod URL pattern**: `https://[POD_ID]-[PORT].proxy.runpod.net`
- **No Docker Compose on Pods** — Ollama and Gradio run as co-located processes
- **HTTP proxy 100-second Cloudflare timeout** — long requests need WebSocket/polling
- **All Docker images must be `--platform linux/amd64`**, never use `:latest` tag (RunPod caches)
- **CLI**: `runpodctl pod create --image IMG --gpu-id "NVIDIA L4" --ports "7860/http,11434/http,22/tcp"`
- **File transfer to Pod**: `runpodctl send FILE` on local, `runpodctl receive CODE` inside Pod
- **Pods REST API**: `https://rest.runpod.io/v1/pods` with Bearer token auth
- **Network volumes**: $0.07/GB/month, mount at `/workspace`, can only grow never shrink
- **Pods with network volumes can only be terminated, not stopped** — data persists in volume
- **Deployment tooling**: `deploy/runpod/` directory:
  - `deploy.sh` — Pod creation via runpodctl (RTX 4090 primary, L4 fallback via RUNPOD_GPU env var)
  - `setup.sh` — First-run idempotent setup (Ollama, GGUF from HF Hub, Python deps, source code)
  - `start.sh` — Idempotent service launcher (Ollama bg + Gradio fg, run on every pod start)
  - `runpod_deploy.py` — Python REST API automation (create/start/stop/status/delete/list subcommands)
  - `health_check.py` — Service health verification (Ollama + model + Gradio + GPU)
  - `Dockerfile` — Backup Docker image option (primary path is setup.sh)
  - `docker-compose.yml` — Local dev environment for testing before deploying
  - `build_and_push.sh` — Docker image build/push (`--platform linux/amd64`, versioned tags)
- **Deployment workflow**: `deploy.sh` (create pod) -> SSH in -> `setup.sh` (first run) -> `start.sh` (every start)
- **Gradio on RunPod**: `demo.queue()` + `demo.launch(server_name="0.0.0.0")` — works natively, no ASGI mounting needed (unlike Modal)
- **Health check**: `check_health()` in `src/app.py` verifies Ollama + model status, shown in sidebar on page load
- **OLLAMA_MODELS persistence**: `config.py` detects `RUNPOD_POD_ID` and redirects to `/workspace/.ollama/models`
- **OLLAMA_BASE_URL**: Now reads from environment (supports Docker Compose where Ollama is a separate service)
- **RunPod base image**: Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (pre-cached on RunPod nodes, avoids Docker Hub rate limits and CUDA driver mismatches). Do NOT use bare `nvidia/cuda:*` images — 12.6 fails on older drivers, 12.1 is deprecated, 12.4 hits Docker Hub pull limits.
- **RunPod SSH**: Proxy-based — `ssh POD_ID-HASH@ssh.runpod.io` (not IP+port). Register keys first: `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`
- **HuggingFace CLI**: `huggingface-cli` is deprecated. Use `hf` instead: `hf download REPO FILE1 FILE2 --local-dir DIR`
- **Cloudflare Tunnel**: Service must be `http://localhost:7860` (NOT `https`). Wrong protocol causes 403. Token saved at `/workspace/.cloudflare_tunnel_token`, auto-started by `start.sh`.
- **Stable demo URL**: `https://navigator.wanderduck.dev` via Cloudflare Tunnel (survives pod recreation if same token used)

### Sub-Agent Patterns

- **Background agents cannot write files** — Write, Read, Bash, Edit, Grep all denied for background sub-agents. Always instruct agents to output content in their response text, then write files yourself as the orchestrator.
- **`runpodctl` CLI flags** differ from REST API params: `--gpu-id` (not `gpuTypeId`), `--image` (not `imageName`), `--container-disk-in-gb` (not `containerDiskInGb`), `--volume-in-gb` (not `volumeInGb`)
