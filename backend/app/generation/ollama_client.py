from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings

logger = logging.getLogger(__name__)


class OllamaServiceError(RuntimeError):
    """Raised when Ollama calls fail."""


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        self.timeout = self.settings.ollama_timeout_seconds

    def healthcheck(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            return True
        except Exception:
            return False

    @staticmethod
    def _sanitize_response(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned and "<think>" not in cleaned.lower():
            text = cleaned

        if text.lstrip().lower().startswith("<think>"):
            citation_match = re.search(r"(\[\d+\].*)", text, flags=re.DOTALL)
            refusal_match = re.search(
                r"(I don't know based on the provided documents\.)",
                text,
                flags=re.IGNORECASE,
            )
            if refusal_match:
                return refusal_match.group(1).strip()
            if citation_match:
                text = citation_match.group(1).strip()
            else:
                return "I don't know based on the provided documents."

        answer_line_match = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if answer_line_match:
            text = answer_line_match.group(1).strip()

        text = re.sub(
            r"^\[\d+\]\s+and\s+\[\d+\]\s+(mention|are|state|talk about|refer to).*?(?=[A-Z]|\d)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(
            r'^(The key sentence here is:|The relevant part says that|Both mention under.*?that)\s*',
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r'^"(.+)"\s+That seems$', r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        if text.lower().startswith("answer:"):
            text = text[7:].strip()

        if text:
            return text
        return "I don't know based on the provided documents."

    @retry(
        stop=stop_after_attempt(get_settings().ollama_max_retries),
        wait=wait_fixed(2),
        retry=retry_if_exception_type((httpx.HTTPError, OllamaServiceError)),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        model: str | None = None,
        num_predict: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.settings.ollama_llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict or self.settings.ollama_num_predict,
            },
        }
        if system:
            payload["system"] = system

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            raise OllamaServiceError(
                "Ollama is not reachable. Start Ollama and pull the required models."
            ) from exc

        data = response.json()
        text = str(data.get("response", "")).strip()
        if not text:
            raise OllamaServiceError("Ollama returned an empty response.")
        return self._sanitize_response(text)

    @retry(
        stop=stop_after_attempt(get_settings().ollama_max_retries),
        wait=wait_fixed(2),
        retry=retry_if_exception_type((httpx.HTTPError, OllamaServiceError)),
        reraise=True,
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for text in texts:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.settings.ollama_embedding_model, "prompt": text},
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = data.get("embedding")
                    if not embedding:
                        raise OllamaServiceError("Ollama returned an empty embedding.")
                    embeddings.append(embedding)
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            raise OllamaServiceError(
                "Embedding request failed. Ensure Ollama is running with nomic-embed-text pulled."
            ) from exc
        return embeddings
