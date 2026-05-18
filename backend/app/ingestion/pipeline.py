from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from qdrant_client.http import models

from app.config import get_settings
from app.ingestion.chunking import split_text
from app.ingestion.loaders import load_document_bytes
from app.models.schemas import DocumentMetadata
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import QdrantVectorStore
from app.utils.ids import generate_chunk_id, generate_document_id, utc_now


class IngestionPipeline:
    def __init__(self, embeddings: EmbeddingService, vector_store: QdrantVectorStore) -> None:
        self.settings = get_settings()
        self.embeddings = embeddings
        self.vector_store = vector_store

    def ingest_file(self, filename: str, raw_bytes: bytes) -> DocumentMetadata:
        pages = load_document_bytes(filename, raw_bytes)
        if not pages:
            raise ValueError(f"No readable text found in '{filename}'.")

        document_id = generate_document_id(filename)
        created_at = utc_now()
        payloads: list[dict[str, object]] = []
        texts: list[str] = []

        for page in pages:
            for local_index, chunk_text in enumerate(
                split_text(page.text, self.settings.chunk_size, self.settings.chunk_overlap)
            ):
                global_index = len(texts) + 1
                chunk_id = generate_chunk_id(document_id, global_index)
                texts.append(chunk_text)
                payloads.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "created_at": created_at.isoformat(),
                        "order": local_index,
                    }
                )

        embeddings = self.embeddings.embed_texts(texts)
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, str(payload["chunk_id"]))),
                vector=vector,
                payload=payload,
            )
            for payload, vector in zip(payloads, embeddings, strict=True)
        ]
        self.vector_store.upsert_chunks(points)

        return DocumentMetadata(
            document_id=document_id,
            filename=filename,
            chunk_count=len(points),
            created_at=created_at,
            page_count=sum(1 for page in pages if page.page_number is not None) or None,
        )
