"""Fine-tune Gemma 4 E4B for Hmong and Somali on Modal A100-40GB.

Trains separate LoRA adapters per language. Each adapter is merged into the
base model and exported as a GGUF for Ollama.

Usage:
    modal run deploy/modal_finetune_multilingual.py --lang hmn    # Hmong
    modal run deploy/modal_finetune_multilingual.py --lang so     # Somali
    modal run deploy/modal_finetune_multilingual.py --lang all    # Both
"""

import modal

MINUTES = 60

output_vol = modal.Volume.from_name("navigator-finetune-output", create_if_missing=True)

finetune_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential")
    .pip_install(
        "torch",
        "torchvision",
        "peft",
        "accelerate",
        "bitsandbytes",
        "trl",
        "datasets",
        "sentencepiece",
        "protobuf",
        "rich",
        "scipy",
        "git+https://github.com/huggingface/transformers.git",
    )
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp /llama.cpp",
        "cd /llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
        "pip install /llama.cpp/gguf-py",
    )
    .add_local_dir("data/training", remote_path="/data/training", copy=True)
)

app = modal.App("navigator-finetune-multilingual", image=finetune_image)

LANG_CONFIG = {
    "hmn": {
        "name": "Hmong",
        "file": "hmong.jsonl",
        "ollama_tag": "navigator-hmn",
        "system_prompt": (
            "You are NorthStar Navigator, a plain-language government benefits navigator "
            "for Minnesota. You help Hmong-speaking residents understand which government "
            "assistance programs they may be eligible for. Respond in Hmong (Hmoob). "
            "Use simple, clear Hmong. Include English program names in parentheses "
            "(e.g., SNAP, MFIP, UI) since these are the official names. "
            "Be warm and actionable. Never say someone qualifies — say they may be eligible. "
            "End every response with a disclaimer that this is informational, not legal advice."
        ),
    },
    "so": {
        "name": "Somali",
        "file": "somali.jsonl",
        "ollama_tag": "navigator-so",
        "system_prompt": (
            "You are NorthStar Navigator, a plain-language government benefits navigator "
            "for Minnesota. You help Somali-speaking residents understand which government "
            "assistance programs they may be eligible for. Respond in Somali (Soomaali). "
            "Use simple, clear Somali. Include English program names in parentheses "
            "(e.g., SNAP, MFIP, UI) since these are the official names. "
            "Be warm and actionable. Never say someone qualifies — say they may be eligible. "
            "End every response with a disclaimer that this is informational, not legal advice."
        ),
    },
}


@app.function(
    gpu="A100-40GB",
    timeout=180 * MINUTES,
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def finetune(lang: str = "hmn"):
    """Train a language-specific LoRA adapter."""
    import json
    import os
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4 import modeling_gemma4
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    config = LANG_CONFIG[lang]
    data_file = f"/data/training/{config['file']}"
    output_dir = f"/output/{lang}"

    print(f"=== Fine-tuning for {config['name']} ({lang}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check training data exists
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found. Generate training data first.")
        return

    # Load model
    model_name = "google/gemma-4-E4B-it"
    print("Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model loaded: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Patch ClippableLinear for PEFT compatibility
    _Orig = modeling_gemma4.Gemma4ClippableLinear
    _Orig.__bases__ = (nn.Linear,) + tuple(
        b for b in _Orig.__bases__ if b is not nn.Module and b is not object
    )
    _Orig.weight = property(lambda self: self.linear.weight)
    _Orig.bias = property(lambda self: self.linear.bias)
    _Orig.in_features = property(lambda self: self.linear.in_features)
    _Orig.out_features = property(lambda self: self.linear.out_features)
    print("Patched Gemma4ClippableLinear")

    # Load training data
    records = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    dataset = Dataset.from_dict({"messages": [r["messages"] for r in records]})
    print(f"Training examples: {len(dataset)}")

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
    model.print_trainable_parameters()

    # Training config — slightly more epochs for smaller dataset
    num_epochs = 3 if len(dataset) < 300 else 2
    training_args = SFTConfig(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Training {config['name']} ({num_epochs} epochs)...")
    result = trainer.train()
    print(f"Loss: {result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Runtime: {result.metrics.get('train_runtime', 0):.1f}s")

    # Save LoRA adapters
    trainer.save_model(f"{output_dir}/lora")
    print(f"LoRA adapters saved to {output_dir}/lora")

    # Merge and save
    print("Merging LoRA into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}/merged", safe_serialization=True)
    trainer.processing_class.save_pretrained(f"{output_dir}/merged")

    # Convert to GGUF
    gguf_dir = f"{output_dir}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    print("Converting to GGUF (q4_k_m)...")
    convert_result = os.system(
        f"python /llama.cpp/convert_hf_to_gguf.py {output_dir}/merged "
        f"--outfile {gguf_dir}/model-f16.gguf --outtype f16"
    )
    if convert_result == 0:
        os.system(
            f"/llama.cpp/build/bin/llama-quantize "
            f"{gguf_dir}/model-f16.gguf {gguf_dir}/model-q4_k_m.gguf q4_k_m"
        )
        if os.path.exists(f"{gguf_dir}/model-q4_k_m.gguf"):
            os.remove(f"{gguf_dir}/model-f16.gguf")
            print(f"GGUF exported: {gguf_dir}/model-q4_k_m.gguf")
        else:
            print("WARNING: Quantization may have failed")
    else:
        print("WARNING: GGUF conversion failed")

    # Create Ollama Modelfile
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    if gguf_files:
        modelfile = f"""FROM ./{gguf_files[0]}
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER num_ctx 2048
SYSTEM "{config['system_prompt']}"
"""
        with open(f"{gguf_dir}/Modelfile", "w") as f:
            f.write(modelfile)
        print("Modelfile written.")

    # Test inference
    print(f"\n--- Test Inference ({config['name']}) ---")
    model = trainer.model
    tokenizer = trainer.processing_class
    model.eval()

    test_q = {
        "hmn": "Kuv yog ib leeg niam nrog ob tug menyuam. Kuv poob kuv txoj haujlwm. Kuv nyob hauv Ramsey County. Dab tsi pab tau kuv?",
        "so": "Waxaan ahay hooyo keligeed ah oo leh laba carruur ah. Waan ka lumay shaqadaydii. Waxaan ku noolahay Hennepin County. Maxaa caawimaad ah?",
    }

    test_messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": test_q.get(lang, test_q["hmn"])},
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=512,
            temperature=1.0, top_p=0.95, do_sample=True,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(response[:500])

    output_vol.commit()

    # List outputs
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} ({size:.1f} MB)")

    print(f"\nDone! Download GGUF with:")
    print(f"  modal volume get navigator-finetune-output /{lang}/gguf/ ./output/gguf-{lang}/")
    print(f"  ollama create {config['ollama_tag']} -f ./output/gguf-{lang}/Modelfile")


@app.function(
    volumes={"/output": output_vol},
    timeout=5 * MINUTES,
)
def list_outputs():
    """List all files in the output volume."""
    import os
    for root, dirs, files in os.walk("/output"):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  {path} ({size:.1f} MB)")


@app.local_entrypoint()
def main(lang: str = "hmn"):
    if lang == "all":
        for l in ["so", "hmn"]:
            print(f"\n{'='*60}")
            print(f"Training {LANG_CONFIG[l]['name']}...")
            print(f"{'='*60}\n")
            finetune.remote(lang=l)
    else:
        finetune.remote(lang=lang)
