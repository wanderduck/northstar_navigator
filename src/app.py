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
from navigator.translations import TRANSLATIONS, LANG_MAP, get_text


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

# Improvement #9: Embed icon as base64 data URI (avoids Gradio /file= path issues)
import base64
_icon_path = PROJECT_ROOT / "docs" / "Styling" / "NorthStar_Navigator_icon_small.png"
ICON_DATA_URI = ""
if _icon_path.exists():
    _b64 = base64.b64encode(_icon_path.read_bytes()).decode()
    ICON_DATA_URI = f"data:image/png;base64,{_b64}"

# Initialize components
client = OllamaClient()
intake = IntakeProcessor(client=client)
engine = EligibilityEngine()
generator = ResponseGenerator(client=client)

# ── Persistent text (never changes with language selection) ─────────────────

# Improvement #2: Prompt instruction tip in all 4 languages, persistent
PROMPT_TIP_HTML = (
    '<div style="padding:8px 0;">'
    '<p style="font-size:1.15em;font-weight:bold;font-style:italic;color:#EC7E78;line-height:1.5;">'
    'Describe your situation in your own words. '
    'Include details like your income, household size, '
    'county, and what kind of help you need.'
    '</p>'
    '<p style="font-size:1.15em;font-weight:bold;font-style:italic;color:#EC7E78;line-height:1.5;">'
    'Describa su situación con sus propias palabras. '
    'Incluya detalles como sus ingresos, el tamaño de su hogar, '
    'su condado y qué tipo de ayuda necesita.'
    '</p>'
    '<p style="font-size:1.15em;font-weight:bold;font-style:italic;color:#EC7E78;line-height:1.5;">'
    'Piav qhia koj qhov xwm txheej ntawm koj tus kheej cov lus. '
    'Suav nrog cov ntsiab lus zoo li koj cov nyiaj tau los, '
    'tsev neeg loj npaum li cas, lub nroog, thiab koj xav tau kev pab dab tsi.'
    '</p>'
    '<p style="font-size:1.15em;font-weight:bold;font-style:italic;color:#EC7E78;line-height:1.5;">'
    'Ku sharax xaaladaada oo isticmaal erayadaada. '
    'Ku dar faahfaahin sida dakhligaaga, tirada qoyskaaga, '
    'degmada, iyo nooca caawimada aad u baahan tahay.'
    '</p>'
    '</div>'
)

# Improvement #3: Four examples, one per language (persistent prompt text)
EXAMPLES = [
    [
        "I am 27 years old, female, single mother with three children under 12 "
        "years old. I live in Hennepin county, I make $1200 a month in income. "
        "What programs are there available to help me feed my children and "
        "myself and to help with my utility bills?",
        "standard",
        "English",
    ],
    [
        "Tengo 27 años, soy mujer y madre soltera con tres hijos menores de 12 "
        "años. Vivo en el condado de Hennepin y mis ingresos mensuales "
        "ascienden a $1,200. ¿Qué programas hay disponibles para ayudarme a "
        "alimentar a mis hijos y a mí misma, y para ayudarme con el pago de "
        "mis facturas de servicios públicos?",
        "standard",
        "Spanish",
    ],
    [
        "Kuv muaj hnub nyoog 27 xyoo, poj niam, ib leej niam uas muaj peb tug "
        "menyuam hnub nyoog qis dua 12 xyoos. Kuv nyob hauv Hennepin county, "
        "kuv khwv tau $1200 ib hlis. Muaj cov kev pab cuam twg los pab kuv pub "
        "kuv cov menyuam thiab kuv tus kheej noj thiab pab kuv them kuv cov "
        "nqi hluav taws xob?",
        "standard",
        "Hmong",
    ],
    [
        "Waxaan ahay 27 jir, dumar ah, hooyo keli ah oo leh saddex carruur ah "
        "oo ka yar 12 sano. Waxaan ku noolahay degmada Hennepin, waxaan "
        "sameeyaa $1200 bishii dakhli ahaan. Barnaamijyo noocee ah ayaa "
        "diyaar ah oo iga caawinaya inaan quudiyo carruurtayda iyo naftayda "
        "iyo inaan ka caawiyo biilasha korontada?",
        "standard",
        "Somali",
    ],
]


# ── Health check ────────────────────────────────────────────────────────────

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


# ── Chat pipeline ───────────────────────────────────────────────────────────

def process_message(
    message: str,
    history: list[dict],
    reading_level: str,
    language: str,
) -> Generator[str, None, None]:
    """Process a user message through the Navigator pipeline (streaming)."""
    try:
        yield "Analyzing your situation..."
        profile, missing = intake.extract(message)

        profile.reading_level = ReadingLevel(reading_level)
        # Use the dropdown language if explicitly set to non-English;
        # otherwise trust the language the model detected from the input text.
        # This ensures non-English examples respond in the correct language
        # even when the dropdown hasn't been updated by Gradio.
        dropdown_lang = {"English": "en", "Spanish": "es", "Hmong": "hmn",
                        "Somali": "so"}.get(language, "en")
        if dropdown_lang != "en":
            profile.language = dropdown_lang
        elif not profile.language or profile.language == "en":
            profile.language = dropdown_lang

        if missing:
            yield intake.ask_followup(missing)
            return

        yield "Finding programs you may be eligible for..."
        benefits_response = engine.evaluate(profile)

        sources_heading = get_text("sources_heading", profile.language)
        sources = _format_sources(benefits_response, profile.language)
        suffix = f"\n\n---\n**{sources_heading}**\n{sources}" if sources else ""

        for partial in generator.generate_stream(benefits_response, profile):
            yield partial

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


def _format_sources(benefits_response, lang_code: str = "en") -> str:
    """Format the sources accordion content in the appropriate language."""
    source_label = get_text("source_label", lang_code)
    lines = []
    for r in benefits_response.eligible_programs:
        if r.source:
            lines.append(f"- **{r.program_name}**: {r.reason} ({source_label}: {r.source})")
    return "\n".join(lines)


# ── Language change handler ─────────────────────────────────────────────────

def _build_title_html(lang_code: str) -> str:
    """Build the header HTML with conditional translated subtitle."""
    translation_line = get_text("title_translation", lang_code)
    subtitle = f'<p style="margin:0;font-size:1.1em;opacity:0.8;">{translation_line}</p>' if translation_line else ""
    security_text = get_text("data_security", lang_code)
    powered_text = get_text("powered_by", lang_code)

    return (
        f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">'
        f'<img src="{ICON_DATA_URI}" style="height:56px;width:auto;" alt="NorthStar Navigator">'
        f'<div>'
        f'<h1 style="margin:0;font-size:1.8em;color:#3E5E80;">NorthStar Navigator</h1>'
        f'{subtitle}'
        f'<p style="margin:2px 0 0;opacity:0.7;font-style:italic;">'
        f'{powered_text} &mdash; {security_text}</p>'
        f'</div></div>'
    )


def _build_settings_heading(lang_code: str) -> str:
    return f'<h3 style="color:#64AD9A;font-size:1.3em;margin:0;">{get_text("settings", lang_code)}</h3>'


def _build_reading_level_heading(lang_code: str) -> str:
    return f'<p style="color:#64AD9A;font-weight:bold;font-size:1.08em;margin:4px 0;">{get_text("reading_level_heading", lang_code)}</p>'


def _build_language_heading(_lang_code: str = "en") -> str:
    # Persistent — always shows all four languages
    return '<p style="color:#64AD9A;font-weight:bold;font-size:1.08em;margin:4px 0;">Language/Idioma/Lus/Luqadda</p>'


def _build_system_status_heading(lang_code: str) -> str:
    return f'<p style="color:#64AD9A;font-weight:bold;font-size:1.08em;margin:4px 0;">{get_text("system_status", lang_code)}</p>'


def _build_examples_heading(lang_code: str) -> str:
    return f'<h3 style="color:#64AD9A;font-size:1.3em;margin:8px 0 4px;">{get_text("examples_heading", lang_code)}</h3>'


def _build_footer(lang_code: str) -> str:
    running = get_text("footer_running", lang_code)
    updated = get_text("footer_updated", lang_code)
    return f"---\n*{RESPONSE_DISCLAIMER}*\n\n{running} | {updated}"


def on_language_change(language: str):
    """Update all translatable UI components when language changes."""
    lang_code = LANG_MAP.get(language, "en")
    return (
        _build_title_html(lang_code),
        _build_settings_heading(lang_code),
        _build_reading_level_heading(lang_code),
        _build_language_heading(lang_code),
        _build_system_status_heading(lang_code),
        get_text("refresh_status", lang_code),
        _build_examples_heading(lang_code),
        _build_footer(lang_code),
    )


# ── Build the Gradio interface ──────────────────────────────────────────────

with gr.Blocks(
    title="NorthStar Navigator",
    css="""
        .chatbot-container { height: 777px !important; }
        .chatbot-container .messages { height: 100% !important; }
    """,
) as demo:

    # Improvement #4 + #5 + #9: Title with icon, color, conditional translation
    title_html = gr.HTML(value=_build_title_html("en"))

    with gr.Row():
        # ── Left sidebar ────────────────────────────────────────────────
        with gr.Column(scale=1):
            # Improvement #8: Headings in Duck Green
            settings_heading = gr.HTML(value=_build_settings_heading("en"))

            # Improvement #6 (B approach): Markdown heading replaces component label
            reading_level_heading = gr.HTML(value=_build_reading_level_heading("en"))
            reading_level = gr.Radio(
                choices=["simple", "standard", "detailed"],
                value="standard",
                label="",  # Hidden — replaced by heading above
                show_label=False,
            )

            language_heading = gr.HTML(value=_build_language_heading("en"))
            language = gr.Dropdown(
                choices=["English", "Spanish", "Hmong", "Somali"],
                value="English",
                label="",
                show_label=False,
            )

            gr.HTML(value='<hr style="border-color:#264660;margin:8px 0;">')

            system_status_heading = gr.HTML(value=_build_system_status_heading("en"))
            health_status = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                value="Checking...",
            )
            refresh_btn = gr.Button("Refresh Status", size="sm")
            refresh_btn.click(fn=check_health, inputs=[], outputs=[health_status])

            # Improvement #2: Persistent prompt tip in 4 languages
            gr.HTML(value=PROMPT_TIP_HTML)

            # Improvement #3: Examples heading (translatable, below tips)
            examples_heading = gr.HTML(value=_build_examples_heading("en"))

        # ── Main chat area ──────────────────────────────────────────────
        with gr.Column(scale=3):
            # Improvement #7: Chatbot height 777px via elem_classes
            chatbot = gr.ChatInterface(
                fn=process_message,
                additional_inputs=[reading_level, language],
                chatbot=gr.Chatbot(height=777, elem_classes=["chatbot-container"]),
                examples=EXAMPLES,
            )

    # Footer (translatable)
    footer_md = gr.Markdown(value=_build_footer("en"))

    # ── Wire language change to update all translatable components ───────
    language.change(
        fn=on_language_change,
        inputs=[language],
        outputs=[
            title_html,
            settings_heading,
            reading_level_heading,
            language_heading,
            system_status_heading,
            refresh_btn,
            examples_heading,
            footer_md,
        ],
    )

    # Health check on page load
    demo.load(fn=check_health, inputs=[], outputs=[health_status])


# ── Theme ───────────────────────────────────────────────────────────────────

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
