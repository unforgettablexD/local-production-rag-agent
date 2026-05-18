import re

from app.models.schemas import Citation, RetrievedChunk


REFUSAL_MESSAGE = "I don't know based on the provided documents."


def extract_citation_indices(answer: str) -> list[int]:
    return [int(match) for match in re.findall(r"\[(\d+)\]", answer)]


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    indices = sorted(set(extract_citation_indices(answer)))
    citations: list[Citation] = []
    chunk_map = {chunk.index: chunk for chunk in chunks}
    for index in indices:
        chunk = chunk_map.get(index)
        if not chunk:
            continue
        citations.append(
            Citation(
                index=index,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                snippet=chunk.text[:240],
            )
        )
    return citations


def evaluate_answer_grounding(answer: str, retrieved_chunks: list[RetrievedChunk]) -> bool:
    if answer.strip() == REFUSAL_MESSAGE:
        return True
    citation_indices = extract_citation_indices(answer)
    if not citation_indices:
        return False
    valid_indices = {chunk.index for chunk in retrieved_chunks}
    if not set(citation_indices).issubset(valid_indices):
        return False
    answer_terms = {term.lower() for term in re.findall(r"\b[a-zA-Z]{4,}\b", answer)}
    context_terms = {
        term.lower()
        for chunk in retrieved_chunks
        for term in re.findall(r"\b[a-zA-Z]{4,}\b", chunk.text)
    }
    overlap = len(answer_terms & context_terms)
    return overlap >= max(2, min(6, len(answer_terms) // 6 or 1))
