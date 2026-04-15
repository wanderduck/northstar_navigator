"""Review existing translations via Cloud Translate round-trip.

Reads prior Hmong/Somali translations (from a Gemma 4 E4B run), back-translates
them to English, then forward-translates back to the target language using
Cloud Translate NMT. This smooths out quality issues from the original
model-generated translations.

Prerequisites:
  modal secret create gcloud-translate \\
    GOOGLE_CLOUD_PROJECT=<project-id> \\
    GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat service-account.json)"

Usage:
    modal run deploy/modal_review_existing.py                # Review Hmong + Somali
    modal run deploy/modal_review_existing.py --lang hmn     # Hmong only
    modal run deploy/modal_review_existing.py --lang so      # Somali only

Download results:
    modal volume get navigator-finetune-output /training/ ./data/training/
"""

import modal

MINUTES = 60

# Cloud Translate: max 1024 strings/request, 30K codepoints recommended
TRANSLATE_MAX_CODEPOINTS = 25000

output_vol = modal.Volume.from_name("navigator-finetune-output", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("google-cloud-translate")
    .add_local_dir("data/training", remote_path="/data/existing", copy=True)
)

app = modal.App("navigator-review-translations", image=cpu_image)

LANG_CONFIG = {
    "hmn": {
        "name": "Hmong",
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


# ---------------------------------------------------------------------------
# Cloud Translate helpers
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
    return translate_v3.TranslationServiceClient()


def _chunk_for_translate(texts: list[str], max_codepoints: int = TRANSLATE_MAX_CODEPOINTS) -> list[list[str]]:
    """Split texts into chunks that fit within Cloud Translate codepoint limits."""
    chunks = []
    current_chunk: list[str] = []
    current_size = 0

    for text in texts:
        text_len = len(text)
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
                    all_translated.extend("" for _ in chunk)

    return all_translated


# ---------------------------------------------------------------------------
# Review function
# ---------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    secrets=[modal.Secret.from_name("gcloud-translate")],
    volumes={"/output": output_vol},
    timeout=30 * MINUTES,
)
def review_existing(lang: str) -> int:
    """Re-translate existing translations via Cloud Translate round-trip.

    Back-translates target-language content to English, then forward-translates
    back to the target language. This produces cleaner NMT output than the
    original Gemma 4 E4B model-generated translations.
    """
    import json
    import os

    config = LANG_CONFIG[lang]
    lang_name = config["name"]
    target_code = config["translate_code"]
    existing_path = config.get("existing_file")

    if not existing_path or not os.path.exists(existing_path):
        print(f"No existing {lang_name} translations to review")
        return 0

    # Check if already reviewed
    flag_path = f"/output/training/{lang}_reviewed.flag"
    output_vol.reload()
    if os.path.exists(flag_path):
        print(f"{lang_name}: already reviewed (flag exists), skipping")
        return 0

    # Initialize Cloud Translate client
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    client = _init_translate_client()
    parent = f"projects/{project_id}/locations/global"

    # Load existing translations from image (prior Gemma 4 E4B run)
    examples = []
    with open(existing_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if not examples:
        print(f"No examples found in {existing_path}")
        return 0

    print(f"Reviewing {len(examples)} existing {lang_name} translations "
          f"via Cloud Translate round-trip...")

    # Extract questions and responses in the target language
    questions = [ex["messages"][1]["content"] for ex in examples]
    responses = [ex["messages"][2]["content"] for ex in examples]

    # Back-translate to English
    print(f"  Back-translating {len(questions)} questions to English...")
    english_qs = _translate_texts(client, parent, questions, target_code, "en")
    print(f"  Back-translating {len(responses)} responses to English...")
    english_rs = _translate_texts(client, parent, responses, target_code, "en")

    # Forward-translate back to target language
    print(f"  Re-translating to {lang_name}...")
    reviewed_qs = _translate_texts(client, parent, english_qs, "en", target_code)
    reviewed_rs = _translate_texts(client, parent, english_rs, "en", target_code)

    # Write results
    output_path = config["output_file"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reviewed = 0
    improved = 0
    system_prompt = config["system_prompt"]

    with open(output_path, "a") as out_f:
        for rq, rr, orig in zip(reviewed_qs, reviewed_rs, examples):
            orig_q = orig["messages"][1]["content"]
            orig_a = orig["messages"][2]["content"]

            # Fall back to original if round-trip produced empty results
            if len(rq) < 10 or len(rr) < 50:
                rq, rr = orig_q, orig_a
            elif rq != orig_q or rr != orig_a:
                improved += 1

            example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rq},
                    {"role": "assistant", "content": rr},
                ]
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            reviewed += 1

    # Write flag to prevent re-review
    with open(flag_path, "w") as f:
        f.write(f"Reviewed {reviewed} examples, improved {improved}\n")

    output_vol.commit()
    print(f"{lang_name}: review complete — {reviewed} reviewed, {improved} improved")
    return reviewed


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(lang: str = "all"):
    if lang == "all":
        langs = ["hmn", "so"]
    elif lang in LANG_CONFIG:
        langs = [lang]
    else:
        print(f"ERROR: unknown language '{lang}'. Use: hmn, so, or all")
        return

    print(f"=== Reviewing existing translations via Cloud Translate ===")
    print(f"  Languages: {', '.join(LANG_CONFIG[l]['name'] for l in langs)}")
    print()

    review_args = [(l,) for l in langs]
    counts = list(review_existing.starmap(review_args))

    for l, c in zip(langs, counts):
        print(f"  {LANG_CONFIG[l]['name']}: {c} translations reviewed")

    print("\nDone! Download:")
    print("  modal volume get navigator-finetune-output /training/ ./data/training/")
