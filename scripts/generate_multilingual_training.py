S"""Generate Hmong and Somali training data by translating English examples.

Uses the navigator Ollama model to translate user questions and assistant
responses into target languages. Keeps the system prompt in English (the model
needs to understand instructions) but translates the conversation turns.

For Somali: the base model handles this well, so direct translation works.
For Hmong: the base model struggles, so we generate bilingual examples
(Hmong question → English+Hmong response) to teach the model the pattern.

Usage:
    PYTHONPATH=src uv run python scripts/generate_multilingual_training.py --lang hmn --count 500
    PYTHONPATH=src uv run python scripts/generate_multilingual_training.py --lang so --count 500
    PYTHONPATH=src uv run python scripts/generate_multilingual_training.py --lang all --count 500
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "navigator"
TRAINING_DIR = Path("data/training")

LANG_CONFIG = {
    "hmn": {
        "name": "Hmong",
        "native_name": "Hmoob",
        "output_file": "hmong.jsonl",
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
        "output_file": "somali.jsonl",
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

# Diverse user scenarios to translate — covers different programs, counties,
# household types, and situations
USER_SCENARIOS = [
    "I'm a single mom with two kids, ages 3 and 7. I just lost my job last week and I live in Ramsey County. What help is available for food and rent?",
    "I'm a 68-year-old veteran living alone in Hennepin County on Social Security. My heating bill is too high this winter. What programs can help?",
    "My husband and I both work minimum wage jobs. We have 4 kids and live in Dakota County. We can't afford childcare. What options do we have?",
    "I'm 22 and just aged out of foster care. I have no family support and I'm in Washington County. What benefits am I eligible for?",
    "I'm pregnant with my first child and I don't have health insurance. I live in Scott County and work part-time making $1,200 a month.",
    "I'm disabled and can't work. I live alone in Anoka County on $800 a month disability. I need help with groceries and medical costs.",
    "We are a refugee family that just arrived in Minnesota. We have 3 children and live in Hennepin County. We don't speak much English. What help is available?",
    "I'm a grandparent raising my two grandchildren in Ramsey County. Their parents are not in the picture. I'm 62 and retired. What programs can help us?",
    "I just got divorced and I have sole custody of my 3 kids. My ex doesn't pay child support. I live in Carver County and make $28,000 a year.",
    "I'm a college student working part-time. I'm 19 and live on my own in Hennepin County. I can barely afford food. Can I get SNAP benefits?",
    "My family of 5 was just evicted. We're staying in a shelter in Ramsey County. I need help finding permanent housing and getting food assistance.",
    "I'm a 45-year-old single man in Dakota County. I was just diagnosed with a serious illness and I can't work. I have no savings. What help exists?",
    "I work full time but my hours just got cut to 20 per week. I have two teenagers and we live in Hennepin County. Our income dropped from $3,000 to $1,500 a month.",
    "I'm a 70-year-old widow in Scott County. I own my home but can't afford property taxes and groceries on my $1,100 Social Security check.",
    "We have a newborn baby and my wife is recovering from a difficult birth. I'm the only one working and we're in Washington County. What programs help new parents?",
    "I'm 25, single, no kids, and I just got laid off from my construction job. I live in Anoka County. Can I get unemployment and food stamps?",
    "My family just fled domestic violence. I have 3 kids under 10 and we're in a shelter in Hennepin County. I need everything — food, housing, medical care.",
    "I'm a seasonal worker and my job ended for the winter. I have a family of 4 in Carver County. How do I get through the off-season?",
    "Both my parents are elderly and need in-home care. I'm their caregiver in Ramsey County but I can't afford to stop working. What programs help family caregivers?",
    "I'm undocumented and I have two US-citizen children. We live in Dakota County. Are there any programs that can help my children even if I'm not eligible?",
    "I have a 16-year-old with a disability who needs special services. We live in Hennepin County and I make about $40,000 a year. What's available?",
    "I'm a single dad with one kid in Ramsey County. I work two jobs but still can't make ends meet. My total income is about $2,800 a month.",
    "My apartment was damaged in a fire and I lost everything. I'm a single person in Washington County with no renter's insurance. Where do I start?",
    "I'm 60 years old, living alone in Scott County, and I just got diagnosed with diabetes. I don't have health insurance. What are my options?",
    "We're a family of 6 and both parents are working but our combined income is only $35,000. We live in Dakota County and need help with utilities and food.",
]


def translate_text(text: str, target_lang: str, lang_name: str) -> str | None:
    """Use Ollama to translate text to the target language."""
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Keep program names (SNAP, MFIP, UI, WIC, TANF, CCAP, MA, SSI, SSDI, EA) in English. "
        f"Keep phone numbers and URLs in their original form. "
        f"Output ONLY the translation, nothing else.\n\n"
        f"{text}"
    )

    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.error("Translation failed: %s", e)
        return None


def generate_response(user_msg: str, system_prompt: str) -> str | None:
    """Generate a navigator response in the target language."""
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        log.error("Response generation failed: %s", e)
        return None


def load_english_examples(path: Path, count: int) -> list[dict]:
    """Load a random subset of English training examples."""
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d["messages"]
            if len(msgs) >= 3:
                user_msg = msgs[1]["content"]
                # Filter to English-only examples (ASCII-heavy)
                non_ascii = sum(1 for c in user_msg if ord(c) > 127)
                if non_ascii < 5 and len(user_msg) > 20:
                    examples.append(d)

    random.shuffle(examples)
    return examples[:count]


def generate_for_language(lang_code: str, count: int, source_file: Path):
    """Generate training data for a specific language."""
    config = LANG_CONFIG[lang_code]
    lang_name = config["name"]
    output_path = TRAINING_DIR / config["output_file"]

    log.info("Generating %d %s training examples", count, lang_name)

    # Load existing to avoid duplicates and count progress
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for line in f if line.strip())
        log.info("Found %d existing examples in %s", existing, output_path.name)

    needed = count - existing
    if needed <= 0:
        log.info("Already have %d examples, nothing to do", existing)
        return

    # Strategy: mix two approaches
    # 1. Translate English scenarios → target language questions, generate responses
    # 2. Use curated scenarios directly with the target-language system prompt

    # Open output file in append mode
    with open(output_path, "a") as out_f:
        generated = 0

        # Phase 1: Curated diverse scenarios (high quality)
        scenarios = USER_SCENARIOS.copy()
        random.shuffle(scenarios)

        for scenario in scenarios:
            if generated >= needed:
                break

            log.info("[%d/%d] Translating scenario...", generated + 1, needed)

            # Translate the user question
            translated_q = translate_text(scenario, lang_code, lang_name)
            if not translated_q:
                continue

            # Generate response in target language
            response = generate_response(translated_q, config["system_prompt"])
            if not response or len(response) < 100:
                log.warning("  Response too short, skipping")
                continue

            example = {
                "messages": [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": translated_q},
                    {"role": "assistant", "content": response},
                ],
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_f.flush()
            generated += 1
            log.info("  Generated example %d/%d (%d chars)", generated, needed, len(response))

            time.sleep(1)

        # Phase 2: Translate from existing English training data
        if generated < needed:
            log.info("Phase 2: Translating from English training data (%d more needed)", needed - generated)
            english_examples = load_english_examples(source_file, needed - generated + 50)

            for ex in english_examples:
                if generated >= needed:
                    break

                user_msg = ex["messages"][1]["content"]
                log.info("[%d/%d] Translating existing example...", generated + 1, needed)

                translated_q = translate_text(user_msg, lang_code, lang_name)
                if not translated_q:
                    continue

                response = generate_response(translated_q, config["system_prompt"])
                if not response or len(response) < 100:
                    continue

                example = {
                    "messages": [
                        {"role": "system", "content": config["system_prompt"]},
                        {"role": "user", "content": translated_q},
                        {"role": "assistant", "content": response},
                    ],
                }
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                out_f.flush()
                generated += 1
                log.info("  Generated example %d/%d", generated, needed)

                time.sleep(1)

    total = existing + generated
    log.info("Done: %d total %s examples in %s", total, lang_name, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multilingual training data for NorthStar Navigator.",
    )
    parser.add_argument(
        "--lang", required=True, choices=["hmn", "so", "all"],
        help="Target language: hmn (Hmong), so (Somali), or all",
    )
    parser.add_argument(
        "--count", type=int, default=500,
        help="Number of examples to generate per language (default: 500)",
    )
    parser.add_argument(
        "--source", type=Path, default=TRAINING_DIR / "generated.jsonl",
        help="Source English training data file",
    )
    args = parser.parse_args()

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if args.lang == "all":
        for lang in ["so", "hmn"]:  # Somali first (higher quality base)
            generate_for_language(lang, args.count, args.source)
    else:
        generate_for_language(args.lang, args.count, args.source)


if __name__ == "__main__":
    main()
