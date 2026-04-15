"""Fine-tune Gemma 4 E4B on 4x A100-80GB with DDP.

Runs multi-GPU distributed training on the combined multilingual dataset
(English, Spanish, Hmong, Somali). Exports merged GGUF for Ollama.

Architecture:
  - 4x A100-80GB with PyTorch DDP (via torchrun)
  - Model loaded in bf16 per GPU (~8GB each, no quantization for clean DDP)
  - LoRA adapters trained with gradient sync across GPUs
  - Effective batch size: 4 per GPU × 4 GPUs × 4 grad_accum = 64
  - After training: merge LoRA, export GGUF, create Modelfile

Usage:
    modal run deploy/modal_finetune.py        # Train on 4x A100-80GB
    modal run deploy/modal_finetune.py::list_outputs  # List output files

Download GGUF:
    modal volume get navigator-finetune-output /gguf/ ./output/gguf/
    ollama create navigator -f output/gguf/Modelfile
"""

import modal
import textwrap

MINUTES = 60

output_vol = modal.Volume.from_name("navigator-finetune-output", create_if_missing=True)

finetune_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential")
    .pip_install(
        "torch",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "sentencepiece",
        "protobuf",
        "scipy",
        "rich",
    )
    .pip_install(
        # Gemma 4 needs unreleased transformers
        "git+https://github.com/huggingface/transformers.git",
    )
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp /llama.cpp",
        "cd /llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
        "pip install /llama.cpp/gguf-py",
    )
    .add_local_dir("data/training", remote_path="/data/training", copy=True)
)

app = modal.App("navigator-finetune", image=finetune_image)

# Training script launched via torchrun across 4 GPUs
TRAIN_SCRIPT = textwrap.dedent(r'''
import json
import os
import random

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import modeling_gemma4
from trl import SFTConfig, SFTTrainer


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    if is_main:
        print(f"=== Training on {world_size} GPUs ===")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model in bf16 — no 4-bit for clean DDP
    model_name = "google/gemma-4-E4B-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if is_main:
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")

    # Patch ClippableLinear for PEFT compatibility
    _Orig = modeling_gemma4.Gemma4ClippableLinear
    _Orig.__bases__ = (nn.Linear,) + tuple(
        b for b in _Orig.__bases__ if b is not nn.Module and b is not object
    )
    _Orig.weight = property(lambda self: self.linear.weight)
    _Orig.bias = property(lambda self: self.linear.bias)
    _Orig.in_features = property(lambda self: self.linear.in_features)
    _Orig.out_features = property(lambda self: self.linear.out_features)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # Load combined multilingual training data
    TRAINING_FILES = [
        # New Gemini-generated + Cloud Translate data
        "/data/training/generated_training_data/english.jsonl",
        "/data/training/generated_training_data/spanish.jsonl",
        "/data/training/generated_training_data/hmong.jsonl",
        "/data/training/generated_training_data/somali.jsonl",
        # Prior hand-curated batches
        "/data/training/combined.jsonl",
    ]

    records = []
    for path in TRAINING_FILES:
        if not os.path.exists(path):
            continue
        count = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    count += 1
        if is_main and count > 0:
            print(f"  Loaded {count} from {os.path.basename(path)}")

    random.seed(42)
    random.shuffle(records)

    formatted_texts = []
    for ex in records:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        formatted_texts.append(text)

    dataset = Dataset.from_dict({"text": formatted_texts})
    if is_main:
        print(f"Total training examples: {len(dataset)}")

    # Training config — DDP across 4 GPUs
    training_args = SFTConfig(
        output_dir="/output/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 4 * 4 GPUs * 4 = 64
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
        ddp_find_unused_parameters=True,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    if is_main:
        print("Starting training...")

    result = trainer.train()

    if is_main:
        metrics = result.metrics
        print(f"\nTraining complete!")
        print(f"  Loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"  Runtime: {metrics.get('train_runtime', 0):.1f}s")

        # Save LoRA adapters (main process only)
        trainer.save_model("/output/lora")
        tokenizer.save_pretrained("/output/lora")
        print("LoRA saved to /output/lora")


if __name__ == "__main__":
    main()
''').strip()


SYSTEM_PROMPT = (
    "You are NorthStar Navigator, a plain-language government benefits navigator for Minnesota. "
    "You help people understand which government assistance programs they may "
    "be eligible for based on their situation. You speak English, Spanish, Hmong, and Somali. "
    "Respond in the same language the user writes in. Be warm, clear, and actionable. "
    "Always cite specific eligibility thresholds and application portals. "
    'Never say someone "qualifies" — say "may be eligible." '
    "End every response with a disclaimer that this is informational, not legal advice."
)


@app.function(
    gpu="A100-80GB:4",
    timeout=420 * MINUTES,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def finetune():
    """Run multi-GPU DDP training, then merge LoRA and export GGUF."""
    import os
    import subprocess
    import sys

    # --- Phase 1: DDP Training across 4 GPUs ---
    print("=" * 60)
    print("Phase 1: DDP Training (4x A100-80GB)")
    print("=" * 60)

    # Write training script to disk
    with open("/tmp/train.py", "w") as f:
        f.write(TRAIN_SCRIPT)

    # Launch with torchrun
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", "4",
            "--master_port", "29500",
            "/tmp/train.py",
        ],
        env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # --- Phase 2: Merge LoRA + Export GGUF (single GPU) ---
    print("\n" + "=" * 60)
    print("Phase 2: Merge LoRA + GGUF Export")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4 import modeling_gemma4
    from peft import PeftModel

    model_name = "google/gemma-4-E4B-it"
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Patch ClippableLinear again (fresh process)
    _Orig = modeling_gemma4.Gemma4ClippableLinear
    _orig_bases = _Orig.__bases__
    _Orig.__bases__ = (nn.Linear,) + tuple(
        b for b in _Orig.__bases__ if b is not nn.Module and b is not object
    )
    _Orig.weight = property(lambda self: self.linear.weight)
    _Orig.bias = property(lambda self: self.linear.bias)
    _Orig.in_features = property(lambda self: self.linear.in_features)
    _Orig.out_features = property(lambda self: self.linear.out_features)

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, "/output/lora")
    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()

    merged_dir = "/output/merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    # Free GPU memory before GGUF conversion
    del merged, model, base_model
    torch.cuda.empty_cache()

    # Convert to GGUF
    gguf_dir = "/output/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    print("Converting to GGUF (f16)...")

    convert_rc = os.system(
        f"python /llama.cpp/convert_hf_to_gguf.py {merged_dir} "
        f"--outfile {gguf_dir}/model-f16.gguf --outtype f16"
    )
    if convert_rc != 0:
        print("WARNING: GGUF conversion failed")
        output_vol.commit()
        return

    print("Quantizing to q4_k_m...")
    quant_rc = os.system(
        f"/llama.cpp/build/bin/llama-quantize "
        f"{gguf_dir}/model-f16.gguf {gguf_dir}/model-q4_k_m.gguf q4_k_m"
    )
    if quant_rc == 0 and os.path.exists(f"{gguf_dir}/model-q4_k_m.gguf"):
        os.remove(f"{gguf_dir}/model-f16.gguf")
        gguf_filename = "model-q4_k_m.gguf"
        print(f"GGUF exported: {gguf_dir}/{gguf_filename}")
    else:
        gguf_filename = "model-f16.gguf"
        print("WARNING: Quantization failed, keeping f16")

    # Create Ollama Modelfile
    modelfile = f'''FROM ./{gguf_filename}
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER num_ctx 2048
SYSTEM "{SYSTEM_PROMPT}"
'''
    with open(f"{gguf_dir}/Modelfile", "w") as f:
        f.write(modelfile)
    print("Modelfile written")

    # Commit volume so GGUF is safe even if Phase 3 crashes
    output_vol.commit()
    print("Volume committed — GGUF safe")

    # --- Phase 3: Test inference ---
    print("\n" + "=" * 60)
    print("Phase 3: Test Inference")
    print("=" * 60)

    # Undo ClippableLinear patch so from_pretrained can construct the model cleanly
    _Orig.__bases__ = _orig_bases
    for attr in ("weight", "bias", "in_features", "out_features"):
        if isinstance(getattr(_Orig, attr, None), property):
            delattr(_Orig, attr)

    test_model = AutoModelForCausalLM.from_pretrained(
        merged_dir, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
    )
    test_tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)

    test_prompts = [
        ("English", "I'm a single mom with two kids in Hennepin County. I just lost my job. What help is available?"),
        ("Spanish", "Soy madre soltera con dos hijos en el condado de Ramsey. Necesito ayuda con comida y renta."),
        ("Hmong", "Kuv yog ib leeg niam nrog ob tug menyuam. Kuv nyob hauv Ramsey County. Dab tsi pab tau kuv?"),
        ("Somali", "Waxaan ahay hooyo keligeed ah oo leh laba carruur ah. Waxaan ku noolahay Hennepin County. Maxaa caawimaad ah?"),
    ]

    for lang, prompt in test_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = test_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = inputs["input_ids"].to(test_model.device)

        with torch.no_grad():
            outputs = test_model.generate(
                input_ids=input_ids, max_new_tokens=512,
                temperature=1.0, top_p=0.95, do_sample=True,
            )
        response = test_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"\n[{lang}] {prompt[:60]}...")
        print(f"  → {response[:200]}...")

    # Commit volume
    output_vol.commit()

    # List outputs
    print("\n" + "=" * 60)
    print("Output files:")
    for root, dirs, files in os.walk("/output/gguf"):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} ({size:.1f} MB)")

    print("\nDone! Download GGUF:")
    print("  modal volume get navigator-finetune-output /gguf/ ./output/gguf/")
    print("  ollama create navigator -f output/gguf/Modelfile")


@app.function(
    volumes={"/output": output_vol},
    timeout=5 * MINUTES,
)
def list_outputs():
    """List files in the output volume."""
    import os
    print("Files in output volume:")
    for root, dirs, files in os.walk("/output"):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} ({size:.1f} MB)")


@app.local_entrypoint()
def main():
    finetune.remote()
