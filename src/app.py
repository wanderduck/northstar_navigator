"""Gradio UI for NorthStar Navigator."""

import logging
from collections.abc import Generator

import gradio as gr

from navigator.models import UserProfile, Dependent, ReadingLevel
from navigator.ollama_client import OllamaClient
from navigator.intake import IntakeProcessor
from navigator.eligibility import EligibilityEngine
from navigator.response import ResponseGenerator
from navigator.prompts import RESPONSE_DISCLAIMER
from navigator.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
client = OllamaClient()
intake = IntakeProcessor(client=client)
engine = EligibilityEngine()
generator = ResponseGenerator(client=client)


def check_health() -> str:
    """Check whether Ollama is running and the navigator model is loaded."""
    import urllib.request
    import json

    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return f"Ollama offline ({exc.__class__.__name__})"

    model_names = [m.get("name", "") for m in data.get("models", [])]
    if any(name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:") for name in model_names):
        return f"OK — model '{OLLAMA_MODEL}' loaded"

    available = ", ".join(model_names) if model_names else "(none)"
    return f"Model '{OLLAMA_MODEL}' not found. Available: {available}"


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
                           "Somali": "so"}.get(language, "en")

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
) as demo:
    gr.Markdown(
        "# NorthStar Navigator\n"
        "*Powered by Gemma 4 via Ollama - Your data never leaves this device*"
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
                choices=["English", "Spanish", "Hmong", "Somali"],
                value="English",
                label="Language",
            )

            gr.Markdown("---")
            health_status = gr.Textbox(
                label="System Status",
                interactive=False,
                value="Checking...",
            )
            health_btn = gr.Button("Refresh Status", size="sm")
            health_btn.click(fn=check_health, inputs=[], outputs=[health_status])

            gr.Markdown(
                "*Describe your situation in your own words. "
                "Include details like your income, household size, "
                "county, and what kind of help you need.*"
            )

        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=process_message,
                additional_inputs=[reading_level, language],
                # When additional_inputs are provided, examples must be lists:
                # [message, reading_level_value, language_value]
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
                        "Soy madre soltera con dos hijos. Perdí mi trabajo y necesito ayuda "
                        "con comida y alquiler. Vivo en el condado de Dakota.",
                        "standard",
                        "Spanish",
                    ],
                ],
            )

    gr.Markdown(
        f"---\n*{RESPONSE_DISCLAIMER}*\n\n"
        "Running locally via Ollama | Last updated: April 2026"
    )

    # Run health check on UI load
    demo.load(fn=check_health, inputs=[], outputs=[health_status])


def main():
    """Launch the Navigator Gradio app."""
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
