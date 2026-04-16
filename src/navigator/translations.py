"""Hardcoded UI translations for NorthStar Navigator.

All non-persistent UI strings in English, Spanish, Hmong, and Somali.
Persistent text (example prompts, instruction tip, app title) is NOT here —
it lives directly in app.py since it never changes with language selection.
"""

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Sidebar headings
        "settings": "Settings",
        "reading_level_heading": "Navigator Answers Reading Level and Output Complexity",
        "language_label": "Language",
        "system_status": "System Status",
        "refresh_status": "Refresh Status",

        # Reading level choices
        "simple": "simple",
        "standard": "standard",
        "detailed": "detailed",

        # Data security message (below title)
        "data_security": (
            "Your data/information is not saved nor is sent anywhere "
            "inside or outside this application and exists only while "
            "this window is open"
        ),

        # Title translation line (empty for English — no subtitle shown)
        "title_translation": "",

        # Powered by line
        "powered_by": "Powered by Gemma 4 via Ollama",

        # Examples section
        "examples_heading": (
            "Examples: Click an example below to add it to the message "
            "input, then run it to see how the Navigator works and responds."
        ),
        "example_prompt_header": "Example Prompt",

        # Footer
        "footer_running": "Running locally via Ollama",
        "footer_updated": "Last updated: April 2026",
    },
    "es": {
        "settings": "Configuraciones",
        "reading_level_heading": "Nivel de lectura y complejidad de las respuestas del Navegador",
        "language_label": "Idioma",
        "system_status": "Estado del sistema",
        "refresh_status": "Actualizar estado",

        "simple": "simple",
        "standard": "estándar",
        "detailed": "detallado",

        "data_security": (
            "Su información/datos no se guardan ni se envían a ningún "
            "lugar dentro o fuera de esta aplicación y solo existen "
            "mientras esta ventana esté abierta"
        ),

        "title_translation": "(Navegador NorthStar)",

        "powered_by": "Impulsado por Gemma 4 a través de Ollama",

        "examples_heading": (
            "Ejemplos: Haga clic en un ejemplo a continuación para agregarlo "
            "al campo de mensaje, luego ejecútelo para ver cómo funciona "
            "y responde el Navegador."
        ),
        "example_prompt_header": "Ejemplo de consulta",

        "footer_running": "Ejecutando localmente a través de Ollama",
        "footer_updated": "Última actualización: abril 2026",
    },
    "hmn": {
        "settings": "Teeb Tsa",
        "reading_level_heading": "Tus Coj Qhia Cov Lus Teb Qib Nyeem thiab Qhov Nyuaj",
        "language_label": "Lus",
        "system_status": "Xwm Txheej Ntawm Lub Tshuab",
        "refresh_status": "Tshiab Dua Xwm Txheej",

        "simple": "yooj yim",
        "standard": "nruab nrab",
        "detailed": "nthuav dav",

        "data_security": (
            "Koj cov ntaub ntawv/xov xwm tsis raug khaws cia thiab tsis "
            "raug xa mus rau qhov twg hauv lossis sab nraud ntawm qhov "
            "kev pab cuam no thiab tsuas muaj thaum lub qhov rais no qhib xwb"
        ),

        "title_translation": "(NorthStar Tus Coj Qhia)",

        "powered_by": "Siv Gemma 4 los ntawm Ollama",

        "examples_heading": (
            "Piv Txwv: Nyem ib qho piv txwv hauv qab no ntxiv rau "
            "hauv qhov chaw xa xov, ces khiav nws saib seb tus Coj "
            "Qhia ua haujlwm thiab teb li cas."
        ),
        "example_prompt_header": "Piv Txwv Lus Nug",

        "footer_running": "Khiav hauv zos los ntawm Ollama",
        "footer_updated": "Hloov tshiab zaum kawg: Plaub Hlis 2026",
    },
    "so": {
        "settings": "Dejinta",
        "reading_level_heading": "Heerka Akhriska Jawaabaha Hagaha iyo Adag-ahaanshaha Wax-soo-saarka",
        "language_label": "Luuqadda",
        "system_status": "Xaaladda Nidaamka",
        "refresh_status": "Cusboonaysii Xaaladda",

        "simple": "fudud",
        "standard": "caadi",
        "detailed": "faahfaahsan",

        "data_security": (
            "Xogtaada/macluumaadkaaga lama kaydin karina laguma dirin "
            "meel gudaha ah ama dibadda ah ee codsigan waxayna jiraan "
            "kaliya inta daaqadani furan tahay"
        ),

        "title_translation": "(Hagaha NorthStar)",

        "powered_by": "Waxaa ku shaqeeya Gemma 4 iyada oo loo marayo Ollama",

        "examples_heading": (
            "Tusaalooyin: Guji tusaale hoose si aad ugu darto sanduuqa "
            "fariinta, ka dibna orod si aad u aragto sida Hagahu u "
            "shaqeeyo oo uu ugu jawaabo."
        ),
        "example_prompt_header": "Tusaale Fariimo",

        "footer_running": "Ku socda maxalli ahaan iyada oo loo marayo Ollama",
        "footer_updated": "Cusboonaysiintii ugu dambeysay: Abriil 2026",
    },
}


# Language code mapping (Gradio dropdown value -> dict key)
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "Hmong": "hmn",
    "Somali": "so",
}


def get_text(key: str, lang_code: str = "en") -> str:
    """Get a translated UI string. Falls back to English if key/lang missing."""
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )
