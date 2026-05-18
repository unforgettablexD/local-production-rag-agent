from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    snippet: str


class RetrievedChunk(BaseModel):
    index: int
    score: float
    document_id: str
    filename: str
    page_number: int | None = None
    chunk_id: str
    text: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app_name: str
    environment: str
    ollama_reachable: bool
    qdrant_reachable: bool


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    created_at: datetime
    page_count: int | None = None


class UploadResponse(BaseModel):
    indexed_documents: list[DocumentMetadata]
    message: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentMetadata]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_retrieval: bool
    rewritten_query: str | None = None
    refusal: bool = False


class SummarizeResponse(BaseModel):
    document_id: str
    summary: str
    citations: list[Citation]


class CompareDocumentsRequest(BaseModel):
    document_id_a: str
    document_id_b: str


class CompareDocumentsResponse(BaseModel):
    document_id_a: str
    document_id_b: str
    comparison: str
    citations: list[Citation]


class EvaluationRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    max_questions: int | None = Field(default=None, ge=1, le=100)
    fast_mode: bool = False


class EvaluationMetrics(BaseModel):
    total_questions: int
    retrieval_hit_rate: float
    citation_presence_rate: float
    refusal_accuracy: float
    groundedness_rate: float


class EvaluationCaseResult(BaseModel):
    question: str
    expected_behavior: str
    expected_source_document: str | None = None
    actual_answer: str
    refusal: bool
    citation_present: bool
    retrieval_hit: bool
    grounded: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResponse(BaseModel):
    metrics: EvaluationMetrics
    results: list[EvaluationCaseResult]


class APIError(BaseModel):
    detail: str
