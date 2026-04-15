"""Gradio UI for NorthStar Navigator."""

import logging
from collections.abc import Generator
from pathlib import Path

import gradio as gr

from navigator.models import UserProfile, Dependent, ReadingLevel
from navigator.ollama_client import OllamaClient
from navigator.intake import IntakeProcessor
from navigator.eligibility import EligibilityEngine
from navigator.response import ResponseGenerator
from navigator.prompts import RESPONSE_DISCLAIMER
from navigator.config import OLLAMA_BASE_URL, OLLAMA_MODEL, PROJECT_ROOT

def _setup_logging():
    """Configure logging with file output when running on RunPod."""
    from navigator.config import IS_RUNPOD
    handlers = [logging.StreamHandler()]
    if IS_RUNPOD:
        from logging.handlers import RotatingFileHandler
        log_path = Path("/workspace/navigator.log")
        handlers.append(RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=3))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

_setup_logging()
logger = logging.getLogger(__name__)

ICON_PATH = PROJECT_ROOT / "docs" / "Styling" / "NorthStar_Navigator_icon.png"

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
) -> Generator[str, None, None]:
    """Process a user message through the Navigator pipeline (streaming)."""
    try:
        # Stage 1: Extract profile
        yield "Analyzing your situation..."
        profile, missing = intake.extract(message)

        # Override reading level and language from UI settings
        profile.reading_level = ReadingLevel(reading_level)
        profile.language = {"English": "en", "Spanish": "es", "Hmong": "hmn",
                           "Somali": "so"}.get(language, "en")

        # If missing critical info, ask follow-up
        if missing:
            yield intake.ask_followup(missing)
            return

        # Stage 2: Determine eligibility
        yield "Finding programs you may be eligible for..."
        benefits_response = engine.evaluate(profile)

        # Stage 3: Stream plain-language response
        sources = _format_sources(benefits_response)
        suffix = f"\n\n---\n**Sources & Reasoning**\n{sources}" if sources else ""

        for partial in generator.generate_stream(benefits_response, profile):
            yield partial

        # Append sources after streaming completes
        if suffix:
            yield partial + suffix

    except Exception as e:
        logger.exception("Error processing message")
        yield (
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
    gr.HTML(
        value=f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">'
        f'<img src="/file={ICON_PATH}" style="height:56px;width:auto;" alt="NorthStar Navigator">'
        f'<div><h1 style="margin:0;font-size:1.8em;">NorthStar Navigator</h1>'
        f'<p style="margin:2px 0 0;opacity:0.7;font-style:italic;">Powered by Gemma 4 via Ollama &mdash; Your data never leaves this device</p></div>'
        f'</div>',
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


def _build_theme() -> gr.themes.Base:
    """Build a custom Gradio theme using the Wanderduck color palette."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e8f5f0", c100="#c6e8dd", c200="#a0d9c8",
            c300="#8ACBBA", c400="#64AD9A", c500="#4d9683",
            c600="#3C6B58", c700="#2d5043", c800="#1e362d", c900="#0f1b17",
            c950="#080e0c",
        ),
        secondary_hue=gr.themes.Color(
            c50="#e6eef4", c100="#c0d4e3", c200="#97b7d0",
            c300="#6d9abd", c400="#4a7ea8", c500="#264660",
            c600="#1A3A50", c700="#142d3e", c800="#0e202c", c900="#08131a",
            c950="#040a0d",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f5f5", c100="#DFDFDF", c200="#bfbfbf",
            c300="#9F9F9F", c400="#9B8F83", c500="#7f7f7f",
            c600="#5f5f5f", c700="#3F3F3F", c800="#2F2F2F", c900="#1f1f1f",
            c950="#0F0F0F",
        ),
        font=["Inter", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0F0F0F",
        body_background_fill_dark="#0F0F0F",
        body_text_color="#DFDFDF",
        body_text_color_dark="#DFDFDF",
        block_background_fill="#1f1f1f",
        block_background_fill_dark="#1f1f1f",
        block_border_color="#264660",
        block_border_color_dark="#264660",
        button_primary_background_fill="#64AD9A",
        button_primary_background_fill_dark="#64AD9A",
        button_primary_text_color="#0F0F0F",
        button_primary_text_color_dark="#0F0F0F",
        input_background_fill="#2F2F2F",
        input_background_fill_dark="#2F2F2F",
        input_border_color="#3F3F3F",
        input_border_color_dark="#3F3F3F",
    )


def main():
    """Launch the Navigator Gradio app."""
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=_build_theme(),
    )


if __name__ == "__main__":
    main()
