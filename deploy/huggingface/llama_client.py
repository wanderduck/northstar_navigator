"""Drop-in replacement for OllamaClient using llama-cpp-python.

Same interface as navigator.ollama_client.OllamaClient so it works
with IntakeProcessor, ResponseGenerator, etc. without changes.
"""

import json
import logging
from collections.abc import Generator

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """llama-cpp-python wrapper with the same API as OllamaClient."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        logger.info("Loading GGUF model from %s ...", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
        )
        logger.info("Model loaded")

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        messages = self._build_messages(user_message, system_prompt, history)
        output = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=1024,
        )
        return output["choices"][0]["message"]["content"]

    def chat_json(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        json_instruction = (
            "\n\nRespond ONLY with valid JSON. No markdown, no explanation, "
            "no code fences. Just the JSON object."
        )
        full_prompt = (system_prompt or "") + json_instruction
        text = self.chat(user_message, system_prompt=full_prompt, history=history)

        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response: %s\nRaw: %s", e, text)
            raise ValueError(f"Model did not return valid JSON: {e}") from e

    def chat_stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        messages = self._build_messages(user_message, system_prompt, history)
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
