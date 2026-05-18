import logging

from app.generation.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self.ollama_client = ollama_client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Generating embeddings", extra={"count": len(texts)})
        return self.ollama_client.embed(texts)
