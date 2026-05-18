from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings
from app.models.schemas import DocumentMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when Qdrant is unavailable or misconfigured."""


class QdrantVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            timeout=self.settings.qdrant_timeout_seconds,
        )
        self.collection_name = self.settings.qdrant_collection_name

    def ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.settings.qdrant_vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:  # pragma: no cover - network failure
            raise VectorStoreError("Qdrant is not reachable. Is Docker Compose running?") from exc

    def healthcheck(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        self.ensure_collection()
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # pragma: no cover - network failure
            raise VectorStoreError(f"Failed to write vectors to Qdrant: {exc}") from exc

    def search(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        self.ensure_collection()
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=self.settings.retrieval_score_threshold,
            )
        except Exception as exc:  # pragma: no cover - network failure
            raise VectorStoreError("Failed to search Qdrant.") from exc

        chunks: list[RetrievedChunk] = []
        for index, result in enumerate(results, start=1):
            payload = result.payload or {}
            chunks.append(
                RetrievedChunk(
                    index=index,
                    score=float(result.score),
                    document_id=str(payload["document_id"]),
                    filename=str(payload["filename"]),
                    page_number=payload.get("page_number"),
                    chunk_id=str(payload["chunk_id"]),
                    text=str(payload["text"]),
                )
            )
        return chunks

    def list_documents(self) -> list[DocumentMetadata]:
        self.ensure_collection()
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        grouped: dict[str, dict[str, object]] = defaultdict(
            lambda: {"filename": "", "chunk_count": 0, "created_at": None, "pages": set()}
        )
        for record in records:
            payload = record.payload or {}
            document_id = str(payload["document_id"])
            grouped[document_id]["filename"] = str(payload["filename"])
            grouped[document_id]["chunk_count"] = int(grouped[document_id]["chunk_count"]) + 1
            grouped[document_id]["created_at"] = payload.get("created_at")
            if payload.get("page_number") is not None:
                grouped[document_id]["pages"].add(payload["page_number"])

        documents: list[DocumentMetadata] = []
        for document_id, item in grouped.items():
            created_at_raw = item["created_at"]
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if isinstance(created_at_raw, str)
                else datetime.now()
            )
            pages = item["pages"]
            documents.append(
                DocumentMetadata(
                    document_id=document_id,
                    filename=str(item["filename"]),
                    chunk_count=int(item["chunk_count"]),
                    created_at=created_at,
                    page_count=len(pages) if pages else None,
                )
            )
        return sorted(documents, key=lambda doc: doc.created_at, reverse=True)

    def delete_document(self, document_id: str) -> bool:
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        return True

    def get_document_chunks(self, document_id: str) -> list[RetrievedChunk]:
        self.ensure_collection()
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        chunks: list[RetrievedChunk] = []
        for index, record in enumerate(records, start=1):
            payload = record.payload or {}
            chunks.append(
                RetrievedChunk(
                    index=index,
                    score=1.0,
                    document_id=str(payload["document_id"]),
                    filename=str(payload["filename"]),
                    page_number=payload.get("page_number"),
                    chunk_id=str(payload["chunk_id"]),
                    text=str(payload["text"]),
                )
            )
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)
