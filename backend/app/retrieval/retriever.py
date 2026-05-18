from app.config import get_settings
from app.models.schemas import RetrievedChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import QdrantVectorStore


class Retriever:
    def __init__(self, embeddings: EmbeddingService, vector_store: QdrantVectorStore) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        vector = self.embeddings.embed_texts([query])[0]
        return self.vector_store.search(vector, limit=top_k or self.settings.retrieval_top_k)
