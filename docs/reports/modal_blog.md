# From Local GPU to Cloud A100 in 10 Minutes: A Practical Guide to Modal for GPU Acceleration

*RTX 2080 Ti not cutting it. A100 needed. Zero YAML written.*

---

## The Problem Every ML Developer Knows

You've got a model to fine-tune. Your local GPU doesn't have enough VRAM. You could spin up a cloud VM, wrestle with CUDA drivers, configure Docker, set up SSH tunnels, and pray your spot instance doesn't get preempted mid-training. Or you could just read this post.

[Modal](https://modal.com) is a serverless GPU platform that lets you define your entire cloud environment — container image, GPU type, persistent storage, secrets — directly in Python. No YAML. No Kubernetes. No Dockerfiles (unless you want them). You write a decorated function, run `modal run app.py`, and your code executes on whatever hardware you specified. When it's done, the container disappears and you stop paying.

This post covers Modal from first principles through production deployment, using four progressively complex examples from a real project: [NorthStar Navigator](https://github.com/wanderduck), a Gemma 4 fine-tuning pipeline built for the Kaggle "Gemma 4 Good" Hackathon. Concepts, working code, and the gotchas the docs skip.

---

## Getting Started

Three commands. That's it.

```bash
pip install modal
modal setup    # Opens browser for authentication
modal run app.py
```

`modal setup` links your local CLI to your Modal account via browser-based auth. After that, every `modal run` command packages your function, builds a container image (cached after first run), ships it to Modal's infrastructure, and streams logs back to your terminal.

No SSH keys. No cloud console. No instance lifecycle management.

---

## Core Concepts

Five primitives. Everything in Modal composes from these:

| Concept | What It Does | Analogy |
|---------|-------------|---------|
| **App** | Groups related functions | A project namespace |
| **Image** | Defines the container environment | A Dockerfile, but in Python |
| **Function** | Your code, decorated with resource requirements | A Lambda function with a GPU |
| **Volume** | Persistent file storage across runs | A network drive that mounts into your container |
| **Secret** | Injects credentials as environment variables | `.env` but managed and encrypted |

Everything lives in your Python file. The `Image` builder is a method chain where each call adds a cached layer — order them from least to most frequently changing, and subsequent runs skip what hasn't changed.

---

## Example 1: Hello, GPU

The simplest possible Modal program. One function, one GPU, one result.

```python
"""
modal_hello_gpu.py — Run with: modal run modal_hello_gpu.py
"""
import modal

app = modal.App("hello-gpu")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(image=image, gpu="T4")
def hello_gpu() -> dict[str, str | int]:
    """Run on a T4 GPU and return device info."""
    import torch

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    # Trivial matmul to confirm CUDA is executing
    a = torch.ones(1024, 1024, device=device)
    b = torch.ones(1024, 1024, device=device)
    c = a @ b

    return {
        "device_name": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "matmul_sum": c.sum().item(),
    }


@app.local_entrypoint()
def main() -> None:
    info = hello_gpu.remote()
    print(f"GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)")
    print(f"CUDA: {info['cuda_version']}")
    print(f"1024x1024 matmul sum: {info['matmul_sum']:.0f}")Tier
```

The only difference from local execution: `hello_gpu.remote()`. The function body is identical. Modal handles CUDA drivers, container networking, and result serialization.

First run takes ~60 seconds for image build and pip install. After that, cached — starts in seconds.

### Available GPUs

| Tier | GPUs | Use Case |
|------|------|----------|
| Mid-range | T4, L4, A10, L40S | Inference, light training |
| High-end | A100 (40/80GB), H100, H200 | Serious fine-tuning |
| Next-gen | B200, B300 | Maximum throughput |

Request a specific variant with `gpu="A100-40GB"`, or let Modal upgrade you automatically (it may give you an 80GB A100 at the 40GB price). Pin with `gpu="A100-40GB!"` (exclamation mark) if you need exact hardware.

For multi-GPU workloads: `gpu="H100:8"` gives you 8 H100s in a single container.

---

## Example 2: Model Inference with Persistent State

Real inference workloads load a model once and serve many requests. Modal's `@app.cls` pattern holds the model in GPU memory across calls:

```python
"""
modal_inference.py — Run with: modal run modal_inference.py
"""
import modal

app = modal.App("gpu-inference")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.0",
        "accelerate>=1.0.0",
    )
)

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


@app.cls(image=image, gpu="T4")
class SentimentClassifier:
    """Model loads once per container, reused across all calls."""

    @modal.enter()
    def load_model(self) -> None:
        from transformers import pipeline

        print("Loading model (runs once per container)...")
        self.pipe = pipeline("text-classification", model=MODEL_ID, device=0)
        print("Model ready.")

    @modal.method()
    def classify(self, texts: list[str]) -> list[dict[str, str | float]]:
        results = self.pipe(texts, batch_size=32)
        return [
            {"text": t, "label": r["label"], "score": round(r["score"], 4)}
            for t, r in zip(texts, results)
        ]


@app.local_entrypoint()
def main() -> None:
    prompts = [
        "This GPU is incredibly fast.",
        "The container cold start was painfully slow.",
        "Modal makes deployment surprisingly straightforward.",
        "I can't get this CUDA driver to work at all.",
    ]

    classifier = SentimentClassifier()
    results = classifier.classify.remote(prompts)

    for r in results:
        print(f"[{r['label']:8}] {r['score']:.4f} | {r['text']}")
```

`@modal.enter()` runs exactly once when the container starts. The model stays loaded in VRAM for every subsequent `.remote()` call until the container scales down. That's the difference between a 200ms response and a 30-second one.

---

## Example 3: Fine-Tuning with Volumes and Secrets

This is the pattern we used for NorthStar Navigator — QLoRA fine-tuning Gemma 4 E4B on an A100-40GB. Two concepts unlock production use here: **Volumes** for persistent storage and **Secrets** for credential management.

```python
"""
modal_finetune.py — Run with: modal run modal_finetune.py

Prerequisites:
    modal secret create huggingface HF_TOKEN=hf_your_token_here
"""
import modal

app = modal.App("model-finetune")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    )
)

# Volume persists across runs — model weights survive container shutdown
vol = modal.Volume.from_name("finetune-output", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=180 * 60,  # 3 hours max
    volumes={"/output": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)

def finetune(dataset_path: str = "/data/training.jsonl"):
    """QLoRA fine-tune on A100. Saves adapters to persistent Volume."""
    import json
    import os
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ["HF_TOKEN"]  # Injected by Secret
    model_name = "google/gemma-3-4b-it"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model in 4-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    records = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    dataset = Dataset.from_dict({"messages": [r["messages"] for r in records]})

    # Train
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/output/checkpoints",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=5,
            bf16=True,
            optim="adamw_8bit",
            max_length=2048,
        ),
    )

    print("Training...")
    result = trainer.train()
    print(f"Loss: {result.metrics.get('train_loss', 'N/A'):.4f}")

    # Save and commit to Volume
    trainer.save_model("/output/lora")
    vol.commit()  # Critical: flush writes to persistent storage
    print("Adapters saved to /output/lora")


@app.local_entrypoint()
def main():
    finetune.remote()
```

### Volume Gotchas We Learned the Hard Way

1. **`vol.commit()` is not automatic.** If your container crashes or times out before the commit call, in-flight writes are lost. For long training jobs, commit after every checkpoint save — not just at the end.

2. **Paths must be absolute.** Writing to `output/model.bin` saves to the ephemeral container filesystem. Writing to `/output/model.bin` (the mounted path) saves to the Volume.

3. **CLI paths need leading slashes.** `modal volume get my-vol /gguf/ ./local/` works. `modal volume get my-vol gguf/ ./local/` doesn't. We burned 20 minutes on this one.

4. **Last-write-wins.** If two containers write to the same file concurrently, one write silently disappears. Design your output paths to be unique per run.

---

## Example 4: Serving a Web App on GPU

The `@modal.web_server()` decorator turns any function that binds a port into a public HTTPS endpoint. We used this to deploy NorthStar Navigator — Ollama as a background subprocess, Gradio on top:

```python
"""
modal_web_server.py
  Dev mode:   modal serve modal_web_server.py   (ephemeral URL)
  Production: modal deploy modal_web_server.py  (persistent URL)
"""
import subprocess
import time
import modal

app = modal.App("gpu-web-app")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("gradio>=5.25.0", "ollama>=0.4.8")
)

MODEL_TAG = "gemma3:4b"
PORT = 7860


def _wait_for_ollama(timeout: int = 60) -> None:
    """Poll Ollama's health endpoint until ready."""
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print("Ollama ready.")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Ollama failed to start within {timeout}s")


@app.function(image=image, gpu="T4")
@modal.web_server(port=PORT, startup_timeout=300)
def serve():
    """Start Ollama, pull model, launch Gradio."""
    import gradio as gr
    import ollama as ollama_client

    # Start Ollama as a background process
    subprocess.Popen(["ollama", "serve"])
    _wait_for_ollama()

    # Pull model (cached in container between requests if warm)
    print(f"Pulling {MODEL_TAG}...")
    ollama_client.pull(MODEL_TAG)
    print("Model ready.")

    def chat(message: str, history: list[list[str]]) -> str:
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        response = ollama_client.chat(model=MODEL_TAG, messages=messages)
        return response["message"]["content"]

    demo = gr.ChatInterface(
        fn=chat,
        title="NorthStar Navigator",
        description=f"Powered by {MODEL_TAG} on Modal T4 GPU",
    )
    
    demo.launch(
        server_name="0.0.0.0",  # Must bind 0.0.0.0, not localhost
        server_port=PORT,
        share=False,            # Modal provides the public URL
    )
```

Two details that will burn you if you miss them:

- **`server_name="0.0.0.0"`** — Gradio defaults to `127.0.0.1`, which Modal's routing layer can't reach from outside the container. Forget this and you get a silent timeout.
- **`share=False`** — Modal provides the public HTTPS URL. Gradio's own tunneling (`share=True`) is redundant and adds latency.

`modal serve` keeps the endpoint alive while your terminal is connected — perfect for development. `modal deploy` makes it persistent.

---

## Real-World Lessons from NorthStar Navigator

NorthStar Navigator is a plain-language government benefits navigator that helps Minnesota residents figure out what assistance programs they qualify for. The entire fine-tuning and deployment pipeline runs on Modal. Here's what the docs don't tell you.

### The ClippableLinear Problem

Gemma 4 uses a custom `Gemma4ClippableLinear` layer that extends `nn.Module` instead of `nn.Linear`. PEFT's LoRA injection finds target layers via `isinstance(module, nn.Linear)` — so it silently skips every attention and MLP layer. Zero trainable parameters. No error message.

The fix is ugly but works: load the model first, then monkey-patch `__bases__` on the `ClippableLinear` class to include `nn.Linear`, and add property delegates for `weight`, `bias`, `in_features`, and `out_features` pointing to `self.linear.*`. It has to happen *after* model loading — vision tower components call `ClippableLinear(config)` during construction, and `nn.Linear.__init__` expects `(in_features, out_features)`.

If you're using Unsloth, it handles this internally. If you're using raw PEFT + TRL, you need the patch.

### Timeouts Are Not Suggestions

Our first fine-tuning run hit a `FunctionTimeoutError` at 42% completion — 90 minutes wasn't enough for 3 epochs on the full dataset. Modal kills the container hard at the timeout. No graceful shutdown, no final `vol.commit()`. Everything written since the last commit is gone.

Fix: reduce epochs (3 to 2), increase batch size (1 to 2), set timeout to 180 minutes. The rule: benchmark throughput with a small `max_steps` first, then set production timeout to 1.5x expected runtime.

### The Volume Path Trap

`modal volume get navigator-finetune-output gguf/ ./output/gguf/` silently downloads nothing. `modal volume get navigator-finetune-output /gguf/ ./output/gguf/` works. The leading slash matters and the error message doesn't help.

### SFTConfig, Not TrainingArguments

Current TRL (0.15+) requires `SFTConfig`, not `TrainingArguments`. SFT-specific parameters like `max_length`, `packing`, and `max_seq_length` are rejected by `TrainingArguments`. The error message is clear enough — the problem is every tutorial from 2024 still uses the old API.

### Cost Reality Check

Modal gives you $30/month free credit (plus periodic promotional credits). Actual numbers from our workloads:

| Workload | GPU | Duration | Approximate Cost |
|----------|-----|----------|-----------------|
| Fine-tuning (2 epochs, Gemma 4 E4B) | A100-40GB | ~2 hours | ~$6 |
| GGUF conversion | A100-40GB | ~15 min | ~$0.75 |
| Gradio demo serving | T4 | per hour | ~$0.59/hr |
| Model pull (one-time) | T4 | ~5 min | ~$0.05 |

The T4 at $0.59/hr with scale-to-zero is cheap for demos — the Gradio endpoint costs nothing when idle, and cold-starts in about 60 seconds, dominated by Ollama startup and model load rather than Modal overhead.

---

## When to Use Modal (and When Not To)

**Modal is the right call when:**

- You need GPUs for hours, not weeks. Per-second billing and scale-to-zero make bursty workloads much cheaper than reserved instances.
- You're a small team that doesn't want to own infrastructure. Everything is code — version it, review it, hand it off.
- You're iterating fast. Cached image layers mean your second run starts in seconds.
- You need a demo endpoint now. `modal serve` hands you a public HTTPS URL in under a minute.

**Look elsewhere when:**

- You need GPUs 24/7 for weeks. Reserved instances on traditional clouds win at sustained utilization.
- You need hardware-level control — custom networking, specific NVLink topologies. Modal abstracts the hardware, which is usually the point but occasionally the problem.
- Your data can't leave your network. Modal runs on shared infrastructure; check their security docs before putting anything regulated in there.

---

## Quick Reference

```bash
# Setup
pip install modal && modal setup

# Run a script on cloud GPU
modal run app.py

# Serve a web endpoint (ephemeral, for development)
modal serve app.py

# Deploy a persistent web endpoint
modal deploy app.py

# Create a secret
modal secret create huggingface HF_TOKEN=hf_your_token

# Manage volumes
modal volume create my-volume
modal volume ls my-volume /
modal volume get my-volume /path/to/file ./local/destination/

# View logs
modal app logs my-app

# List running apps
modal app list
```

---

## The Infrastructure Tax Is Optional

Modal removes the infrastructure tax from GPU computing. You stop thinking about drivers, containers, and instance lifecycle, and start thinking about your actual problem. For our hackathon project, it turned a multi-day DevOps slog (provision VM, install CUDA, configure Docker, set up monitoring) into a single Python file we iterated on in an afternoon.

The four patterns here — GPU function, persistent inference class, volume-backed training, web endpoint — handle the vast majority of ML workloads. Start with Example 1, confirm it runs, build up from there.

When you're done, the meter stops. No zombie instances, no surprise bills. Just `modal run`, and then silence.

---

*This post was written while fine-tuning Gemma 4 E4B for [NorthStar Navigator](https://github.com/wanderduck), a plain-language government benefits navigator built for the Kaggle Gemma 4 Good Hackathon. The entire training and serving pipeline runs on Modal.*
