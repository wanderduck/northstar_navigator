"""Gradio UI for NorthStar Navigator — HuggingFace Spaces version.

Uses llama-cpp-python instead of Ollama for inference.
Downloads the fine-tuned GGUF from HF Hub on startup.
"""

import logging
import os
import sys

# Add navigator source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gradio as gr
from huggingface_hub import hf_hub_download

from llama_client import LlamaCppClient
from navigator.models import UserProfile, Dependent, ReadingLevel
from navigator.intake import IntakeProcessor
from navigator.eligibility import EligibilityEngine
from navigator.response import ResponseGenerator
from navigator.prompts import RESPONSE_DISCLAIMER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download GGUF model from HF Hub
MODEL_REPO = "wanderduck/northstar-navigator-gguf"
MODEL_FILE = "model-q4_k_m.gguf"

logger.info("Downloading model from %s ...", MODEL_REPO)
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
logger.info("Model downloaded to %s", model_path)

# Initialize components with LlamaCppClient instead of OllamaClient
client = LlamaCppClient(model_path=model_path, n_ctx=2048)
intake = IntakeProcessor(client=client)
engine = EligibilityEngine()
generator = ResponseGenerator(client=client)


def process_message(
    message: str,
    history: list[dict],
    reading_level: str,
    language: str,
) -> str:
    """Process a user message through the Navigator pipeline."""
    try:
        # Stage 1: Extract profile
        profile, missing = intake.extract(message)

        # Override reading level and language from UI settings
        profile.reading_level = ReadingLevel(reading_level)
        profile.language = {"English": "en", "Spanish": "es", "Hmong": "hmn",
                           "Somali": "so", "Karen": "kar"}.get(language, "en")

        # If missing critical info, ask follow-up
        if missing:
            return intake.ask_followup(missing)

        # Stage 2: Determine eligibility
        benefits_response = engine.evaluate(profile)

        # Stage 3: Generate plain-language response
        response_text = generator.generate(benefits_response, profile)

        # Append sources section
        sources = _format_sources(benefits_response)
        if sources:
            response_text += f"\n\n---\n**Sources & Reasoning**\n{sources}"

        return response_text

    except Exception as e:
        logger.exception("Error processing message")
        return (
            "I'm sorry, I encountered an error processing your request. "
            "Please try rephrasing your situation, or contact 211 by dialing 2-1-1 "
            "for immediate assistance.\n\n"
            f"Error: {e}"
        )


def _format_sources(benefits_response) -> str:
    """Format the sources accordion content."""
    lines = []
    for r in benefits_response.eligible_programs:
        if r.source:
            lines.append(f"- **{r.program_name}**: {r.reason} (Source: {r.source})")
    return "\n".join(lines)


# Build the Gradio interface
with gr.Blocks(
    title="NorthStar Navigator",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# NorthStar Navigator\n"
        "*Powered by fine-tuned Gemma 4 E4B — A plain-language government benefits navigator for Minnesota*"
    )

    with gr.Row():
        # Left sidebar
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            reading_level = gr.Radio(
                choices=["simple", "standard", "detailed"],
                value="standard",
                label="Reading Level",
            )
            language = gr.Dropdown(
                choices=["English", "Spanish", "Hmong", "Somali", "Karen"],
                value="English",
                label="Language",
            )
            gr.Markdown(
                "---\n"
                "*Describe your situation in your own words. "
                "Include details like your income, household size, "
                "county, and what kind of help you need.*"
            )

        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=process_message,
                additional_inputs=[reading_level, language],
                examples=[
                    [
                        "I'm a single mom with two kids, ages 3 and 7. I just got laid off "
                        "from my warehouse job where I made $32,000. We're in Ramsey County "
                        "and I'm worried about paying rent and feeding my kids.",
                        "standard",
                        "English",
                    ],
                    [
                        "I'm a 68-year-old veteran in Hennepin County living on Social Security. "
                        "I'm having trouble paying my heating bill this winter.",
                        "standard",
                        "English",
                    ],
                    [
                        "Soy madre soltera con dos hijos. Perd\u00ed mi trabajo y necesito ayuda "
                        "con comida y alquiler. Vivo en el condado de Dakota.",
                        "standard",
                        "Spanish",
                    ],
                ],
            )

    gr.Markdown(
        f"---\n*{RESPONSE_DISCLAIMER}*\n\n"
        "Kaggle Gemma 4 Good Hackathon | Last updated: April 2026"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
