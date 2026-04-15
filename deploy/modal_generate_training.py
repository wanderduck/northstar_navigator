"""Generate multilingual training data using Gemini API + Cloud Translate.

English generation uses Gemini API with model fallback chain:
  gemini-3.1-pro-preview -> gemini-2.5-pro -> gemini-3-flash-preview
Automatically switches on persistent 429/quota errors (per-container state).

Translation uses Google Cloud Translate v3 (NMT) for speed and reliability.
Much faster than Gemini-based translation (~30 API calls vs ~600 per language).

Pipeline:
  1. Generate English scenario+response pairs (Gemini, grounded in DHS content)
  2. Translate to Spanish, Hmong, Somali (Cloud Translate NMT)
  Output: 4 JSONL files, 3000 examples each

All phases fan out across --workers containers for parallel API calls.

Prerequisites:
  modal secret create gemini-api GEMINI_API_KEY=<key>
  modal secret create gcloud-translate \\
    GOOGLE_CLOUD_PROJECT=<project-id> \\
    GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat service-account.json)"

Usage:
    modal run deploy/modal_generate_training.py                          # Single worker, 3000/lang
    modal run deploy/modal_generate_training.py --workers 4              # 4 parallel workers
    modal run deploy/modal_generate_training.py --count 500 --workers 2  # Smaller run

Download results:
    modal volume get navigator-finetune-output /training/ ./data/training/
"""

import modal

MINUTES = 60
GEN_BATCH = 5         # Q+A pairs per Gemini generation call
DHS_PER_BATCH = 5     # DHS sections sampled per generation call
DHS_MAX_CHARS = 2500  # max chars per DHS section in prompt

# Gemini model fallback chain — auto-switches on persistent rate limits
MODEL_CHAIN = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
]

# Cloud Translate: max 1024 strings/request, 30K codepoints recommended
TRANSLATE_MAX_CODEPOINTS = 25000  # leave margin below 30K limit

output_vol = modal.Volume.from_name("navigator-finetune-output", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("google-genai", "google-cloud-translate")
    .add_local_dir("data/raw/dhs_combined_manual", remote_path="/data/dhs", copy=True)
    .add_local_dir("data/training", remote_path="/data/existing", copy=True)
)

app = modal.App("navigator-gen-training", image=cpu_image)

LANG_CONFIG = {
    "en": {
        "name": "English",
        "native_name": "English",
        "translate_code": "en",
        "output_file": "/output/training/english.jsonl",
        "system_prompt": (
            "You are NorthStar Navigator, a plain-language government benefits navigator "
            "for Minnesota. You help residents understand which government assistance "
            "programs they may be eligible for. Use simple, clear language at a 6th grade "
            "reading level. Be warm and actionable. Never say someone qualifies — say they "
            "may be eligible. End every response with a disclaimer that this is informational, "
            "not legal advice."
        ),
    },
    "es": {
        "name": "Spanish",
        "native_name": "Espanol",
        "translate_code": "es",
        "output_file": "/output/training/spanish.jsonl",
        "system_prompt": (
            "You are NorthStar Navigator, a plain-language government benefits navigator "
            "for Minnesota. You help Spanish-speaking residents understand which government "
            "assistance programs they may be eligible for. Respond in Spanish (Espanol). "
            "Use simple, clear Spanish. Include English program names in parentheses "
            "(e.g., SNAP, MFIP, UI) since these are the official names. "
            "Be warm and actionable. Never say someone qualifies — say they may be eligible. "
            "End every response with a disclaimer that this is informational, not legal advice."
        ),
    },
    "hmn": {
        "name": "Hmong",
        "native_name": "Hmoob",
        "translate_code": "hmn",
        "output_file": "/output/training/hmong.jsonl",
        "existing_file": "/data/existing/hmong.jsonl",
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
        "native_name": "Soomaali",
        "translate_code": "so",
        "output_file": "/output/training/somali.jsonl",
        "existing_file": "/data/existing/somali.jsonl",
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

COUNTIES = [
    "Ramsey", "Hennepin", "Dakota", "Scott", "Carver", "Washington", "Anoka",
    "Olmsted", "St. Louis", "Stearns", "Blue Earth", "Crow Wing", "Rice",
]


# ---------------------------------------------------------------------------
# Helpers — DHS data
# ---------------------------------------------------------------------------

def _load_dhs_sections() -> list[dict]:
    """Load DHS Combined Manual sections from the image."""
    from pathlib import Path

    sections = []
    dhs_dir = Path("/data/dhs")
    for f in sorted(dhs_dir.glob("section_*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")

        # Parse header
        title = f.stem
        for line in lines[:6]:
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
                break

        # Get content after header separator
        content_start = 0
        for i, line in enumerate(lines[:10]):
            if line.startswith("===="):
                content_start = i + 1
                break

        content = "\n".join(lines[content_start:]).strip()
        if len(content) > 50:
            sections.append({"title": title, "content": content, "file": f.name})

    return sections


# ---------------------------------------------------------------------------
# Helpers — Gemini with model fallback
# ---------------------------------------------------------------------------

# Per-container mutable state: index into MODEL_CHAIN.
# Each Modal container has its own Python process, so this is container-local.
_model_state = {"idx": 0}


def _gemini_call(client, prompt: str, max_retries: int = 4) -> str | None:
    """Call Gemini with exponential backoff and model fallback on rate limits.

    On persistent 429/quota errors, automatically switches to the next model
    in MODEL_CHAIN. The switch is sticky for the rest of this container's life.
    """
    import time

    while _model_state["idx"] < len(MODEL_CHAIN):
        model = MODEL_CHAIN[_model_state["idx"]]
        hit_rate_limit = False

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    hit_rate_limit = True
                    wait = 2 ** (attempt + 2)
                    print(f"  Rate limited on {model}, waiting {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                elif "block" in err or "safety" in err:
                    print(f"  Content filtered, skipping")
                    return None
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print(f"  API error after {max_retries} attempts: {e}")
                        return None

        # All retries exhausted — if rate-limited, try next model
        if hit_rate_limit:
            _model_state["idx"] += 1
            if _model_state["idx"] < len(MODEL_CHAIN):
                next_model = MODEL_CHAIN[_model_state["idx"]]
                print(f"  *** Falling back: {model} -> {next_model} ***")
                continue  # retry same prompt with next model
            else:
                print(f"  *** All models rate-limited, giving up ***")
                return None
        else:
            # Non-rate-limit failure after all retries
            return None

    return None


def _parse_json(text: str) -> list[dict]:
    """Extract JSON array from a Gemini response (handles code blocks)."""
    import json
    import re

    # Strip markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Helpers — Cloud Translate
# ---------------------------------------------------------------------------

def _init_translate_client():
    """Initialize Cloud Translate v3 client from Modal secret."""
    import json
    import os
    from google.cloud import translate_v3
    from google.oauth2 import service_account

    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if creds_json:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info)
        return translate_v3.TranslationServiceClient(credentials=creds)
    # Fall back to Application Default Credentials
    return translate_v3.TranslationServiceClient()


def _chunk_for_translate(texts: list[str], max_codepoints: int = TRANSLATE_MAX_CODEPOINTS) -> list[list[str]]:
    """Split texts into chunks that fit within Cloud Translate codepoint limits."""
    chunks = []
    current_chunk: list[str] = []
    current_size = 0

    for text in texts:
        text_len = len(text)
        # Start new chunk if adding this text would exceed limit
        if current_size + text_len > max_codepoints and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(text)
        current_size += text_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _translate_texts(
    client,
    parent: str,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    max_retries: int = 3,
) -> list[str]:
    """Translate a list of texts via Cloud Translate v3, with chunking and retry."""
    import time

    all_translated: list[str] = []
    chunks = _chunk_for_translate(texts)

    for chunk_idx, chunk in enumerate(chunks):
        for attempt in range(max_retries):
            try:
                response = client.translate_text(
                    contents=chunk,
                    target_language_code=target_lang,
                    source_language_code=source_lang,
                    parent=parent,
                    mime_type="text/plain",
                )
                all_translated.extend(
                    t.translated_text for t in response.translations
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Translate API error (chunk {chunk_idx}), "
                          f"retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"  Translate API failed after {max_retries} "
                          f"attempts (chunk {chunk_idx}): {e}")
                    # Fill with empty strings so indices stay aligned
                    all_translated.extend("" for _ in chunk)

    return all_translated


# ---------------------------------------------------------------------------
# Helpers — sharding
# ---------------------------------------------------------------------------

def _shard_output_path(lang_key: str, shard_idx: int, total_shards: int) -> str:
    """Return shard file path if sharded, or final output path if single worker."""
    if total_shards > 1:
        return f"/output/training/{lang_key}_shard_{shard_idx}.jsonl"
    return LANG_CONFIG[lang_key]["output_file"]


# ---------------------------------------------------------------------------
# Phase 1: Generate English training pairs (Gemini with fallback)
# ---------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    secrets=[modal.Secret.from_name("gemini-api")],
    volumes={"/output": output_vol},
    timeout=420 * MINUTES,
)
def generate_english(count: int, shard_idx: int = 0, total_shards: int = 1) -> int:
    """Generate English scenario+response pairs grounded in DHS content."""
    import json
    import math
    import os
    import random
    import time
    from google import genai

    # Calculate this shard's portion
    shard_count = count // total_shards
    if shard_idx < count % total_shards:
        shard_count += 1

    # Stagger start to spread initial API burst
    if total_shards > 1:
        time.sleep(shard_idx * 1.5)

    # Different random state per shard for diversity
    random.seed(42 + shard_idx * 1000)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    sections = _load_dhs_sections()
    tag = f"Shard {shard_idx}/{total_shards}" if total_shards > 1 else "English"
    print(f"{tag}: loaded {len(sections)} DHS sections")
    print(f"{tag}: model chain = {MODEL_CHAIN}")

    output_path = _shard_output_path("en", shard_idx, total_shards)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume: count existing
    existing = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())
        print(f"{tag}: resuming from {existing} existing examples")

    needed = shard_count - existing
    if needed <= 0:
        print(f"{tag}: already have {existing} examples, done")
        return existing

    system_prompt = LANG_CONFIG["en"]["system_prompt"]
    batches_needed = math.ceil(needed / GEN_BATCH)
    generated = 0

    print(f"{tag}: generating {needed} examples ({batches_needed} batches)...")

    with open(output_path, "a") as out_f:
        for batch_idx in range(batches_needed):
            if generated >= needed:
                break

            # Sample random DHS sections as context
            sampled = random.sample(sections, min(DHS_PER_BATCH, len(sections)))
            dhs_context = "\n\n".join(
                f"--- {s['title']} ---\n{s['content'][:DHS_MAX_CHARS]}"
                for s in sampled
            )
            county_sample = random.sample(COUNTIES, 4)

            prompt = f"""You are creating training data for NorthStar Navigator, a Minnesota government benefits chatbot.

Here are reference sections from the Minnesota DHS Combined Manual:

{dhs_context}

Generate exactly {GEN_BATCH} unique training examples. Each has a "question" from a Minnesota resident and a "response" from the Navigator chatbot.

Requirements for QUESTIONS:
- Each should mention a specific Minnesota county (use counties like {', '.join(county_sample)})
- Describe a specific household situation (age, family size, income, employment)
- Ask about specific needs (food, housing, medical, childcare, heating, disability, etc.)
- Vary demographics: single parents, elderly, veterans, refugees, disabled, working poor, etc.
- Be realistic and conversational, like a real person asking for help

Requirements for RESPONSES:
- Write at a 6th grade reading level using short sentences
- Reference specific programs by name (SNAP, MFIP, MA, MinnesotaCare, CCAP, GRH, MSA, GA, UI, WIC, EA, SSI, SSDI, Section 8, LIHEAP)
- Include eligibility hints based on the DHS reference content above
- Mention specific actions: "Call your county office", "Apply at ApplyMN.dhs.mn.gov", "Visit your local county human services office"
- Be warm and empathetic. Use "you may be eligible" never "you qualify"
- End with: "This information is for guidance only and is not legal advice. Contact your county human services office for help applying."
- Each response should be 150-300 words

Return ONLY a JSON array:
[{{"question": "...", "response": "..."}}]"""

            result = _gemini_call(client, prompt)
            if not result:
                continue

            items = _parse_json(result)
            for item in items:
                q = item.get("question", "").strip()
                a = item.get("response", "").strip()
                if len(q) < 20 or len(a) < 100:
                    continue

                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                generated += 1

            out_f.flush()

            if (batch_idx + 1) % 50 == 0:
                output_vol.commit()
                current_model = MODEL_CHAIN[min(_model_state["idx"], len(MODEL_CHAIN) - 1)]
                print(f"  {tag}: [{generated}/{needed}] generated (model: {current_model})")

            time.sleep(0.1)

    output_vol.commit()
    total = existing + generated
    current_model = MODEL_CHAIN[min(_model_state["idx"], len(MODEL_CHAIN) - 1)]
    print(f"{tag}: complete — {total} examples (final model: {current_model})")
    return total


# ---------------------------------------------------------------------------
# Phase 2: Translate to target language (Cloud Translate NMT)
# ---------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    secrets=[modal.Secret.from_name("gcloud-translate")],
    volumes={"/output": output_vol},
    timeout=30 * MINUTES,
)
def translate_to_language(lang: str, shard_idx: int = 0, total_shards: int = 1) -> int:
    """Translate English training pairs to a target language via Cloud Translate."""
    import json
    import os

    config = LANG_CONFIG[lang]
    lang_name = config["name"]
    target_code = config["translate_code"]
    system_prompt = config["system_prompt"]
    tag = f"{lang_name} shard {shard_idx}/{total_shards}" if total_shards > 1 else lang_name

    # Initialize Cloud Translate client
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = _init_translate_client()
    parent = f"projects/{project_id}/locations/global"

    # Read English source from volume
    output_vol.reload()
    english_path = LANG_CONFIG["en"]["output_file"]
    if not os.path.exists(english_path):
        print(f"ERROR: {english_path} not found. Run English generation first.")
        return 0

    all_english = []
    with open(english_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                all_english.append({
                    "question": ex["messages"][1]["content"],
                    "response": ex["messages"][2]["content"],
                })

    # Calculate this shard's slice of English examples
    total_examples = len(all_english)
    shard_size = total_examples // total_shards
    remainder = total_examples % total_shards
    start = shard_idx * shard_size + min(shard_idx, remainder)
    end = start + shard_size + (1 if shard_idx < remainder else 0)
    english_pairs = all_english[start:end]

    output_path = _shard_output_path(lang, shard_idx, total_shards)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check existing translations for resume
    existing = 0
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())

    remaining = english_pairs[existing:]
    if not remaining:
        print(f"{tag}: already translated {existing} examples, done")
        return existing

    print(f"{tag}: translating {len(remaining)} examples via Cloud Translate "
          f"(have {existing})...")

    # Translate questions and responses separately to maintain alignment
    questions = [p["question"] for p in remaining]
    responses = [p["response"] for p in remaining]

    print(f"  {tag}: translating {len(questions)} questions...")
    translated_qs = _translate_texts(client, parent, questions, "en", target_code)

    print(f"  {tag}: translating {len(responses)} responses...")
    translated_rs = _translate_texts(client, parent, responses, "en", target_code)

    # Write training examples
    translated = 0
    with open(output_path, "a") as out_f:
        for tq, tr in zip(translated_qs, translated_rs):
            if len(tq) < 10 or len(tr) < 50:
                continue

            example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": tq},
                    {"role": "assistant", "content": tr},
                ]
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            translated += 1

    output_vol.commit()
    total = existing + translated
    print(f"{tag}: translation complete — {total} total examples")
    return total


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    volumes={"/output": output_vol},
    timeout=10 * MINUTES,
)
def merge_shards(lang_key: str, num_shards: int) -> int:
    """Merge shard files into final output file."""
    import os

    output_vol.reload()
    output_path = LANG_CONFIG[lang_key]["output_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    with open(output_path, "w") as out_f:
        for i in range(num_shards):
            shard_path = f"/output/training/{lang_key}_shard_{i}.jsonl"
            if os.path.exists(shard_path):
                with open(shard_path) as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)
                            total += 1
                os.remove(shard_path)

    output_vol.commit()
    print(f"Merged {num_shards} {lang_key} shards -> {total} examples")
    return total


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(count: int = 3000, workers: int = 1):
    print(f"=== Generating {count} training examples per language ===")
    print(f"  Workers: {workers}")
    print(f"  Languages: English, Spanish, Hmong, Somali")
    print(f"  English gen: Gemini (fallback: {' -> '.join(MODEL_CHAIN)})")
    print(f"  Translation: Google Cloud Translate NMT")
    print(f"  Batch size: generate={GEN_BATCH}")
    print()

    # Phase 1: Generate English (Gemini, sharded across workers)
    print(f"Phase 1/2: Generating English examples "
          f"({workers} worker{'s' if workers > 1 else ''})...")
    if workers > 1:
        gen_args = [(count, i, workers) for i in range(workers)]
        shard_counts = list(generate_english.starmap(gen_args))
        print(f"  Shards complete: {shard_counts}")
        en_count = merge_shards.remote("en", workers)
    else:
        en_count = generate_english.remote(count)
    print(f"  English: {en_count} examples\n")

    # Phase 2: Translate via Cloud Translate (fast — sharding optional)
    print(f"Phase 2/2: Translating via Cloud Translate NMT "
          f"({workers} worker{'s' if workers > 1 else ''} per lang)...")
    if workers > 1:
        translate_args = []
        for lang in ["es", "hmn", "so"]:
            for i in range(workers):
                translate_args.append((lang, i, workers))
        list(translate_to_language.starmap(translate_args))

        # Merge per language
        merge_args = [("es", workers), ("hmn", workers), ("so", workers)]
        lang_counts = list(merge_shards.starmap(merge_args))
        for lang_name, c in zip(["Spanish", "Hmong", "Somali"], lang_counts):
            print(f"  {lang_name}: {c} examples")
    else:
        translate_args = [("es",), ("hmn",), ("so",)]
        lang_counts = list(translate_to_language.starmap(translate_args))
        for lang_name, c in zip(["Spanish", "Hmong", "Somali"], lang_counts):
            print(f"  {lang_name}: {c} examples")
    print()

    print("=" * 50)
    print("Complete! Download all training data:")
    print("  modal volume get navigator-finetune-output /training/ ./data/training/")
    print()
    print("Output files:")
    print(f"  english.jsonl  — {en_count} examples")
    print(f"  spanish.jsonl  — {lang_counts[0]} examples")
    print(f"  hmong.jsonl    — {lang_counts[1]} examples")
    print(f"  somali.jsonl   — {lang_counts[2]} examples")
