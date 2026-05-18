import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.dependencies import get_agent_tools, get_ingestion_pipeline, get_vector_store
from app.models.schemas import (
    CompareDocumentsRequest,
    CompareDocumentsResponse,
    DeleteDocumentResponse,
    DocumentsResponse,
    SummarizeResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    pipeline = get_ingestion_pipeline()
    indexed_documents = []
    for file in files:
        raw_bytes = await file.read()
        try:
            indexed_documents.append(pipeline.ingest_file(file.filename, raw_bytes))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Indexed documents", extra={"count": len(indexed_documents)})
    return UploadResponse(
        indexed_documents=indexed_documents,
        message=f"Indexed {len(indexed_documents)} document(s).",
    )


@router.get("", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    return DocumentsResponse(documents=get_vector_store().list_documents())


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    deleted = get_vector_store().delete_document(document_id)
    return DeleteDocumentResponse(document_id=document_id, deleted=deleted)


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
def summarize_document(document_id: str) -> SummarizeResponse:
    try:
        summary, citations = get_agent_tools().summarize_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SummarizeResponse(document_id=document_id, summary=summary, citations=citations)


@router.post("/compare", response_model=CompareDocumentsResponse)
def compare_documents(payload: CompareDocumentsRequest) -> CompareDocumentsResponse:
    try:
        comparison, citations = get_agent_tools().compare_documents(
            payload.document_id_a, payload.document_id_b
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CompareDocumentsResponse(
        document_id_a=payload.document_id_a,
        document_id_b=payload.document_id_b,
        comparison=comparison,
        citations=citations,
    )
