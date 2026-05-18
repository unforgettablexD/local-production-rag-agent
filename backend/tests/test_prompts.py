from app.generation.prompts import build_context_block, build_grounded_answer_prompt
from app.models.schemas import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        index=1,
        score=0.9,
        document_id="doc-1",
        filename="policy.md",
        page_number=2,
        chunk_id="doc-1-chunk-0001",
        text="Employees must submit expenses within 30 days.",
    )


def test_build_context_block_includes_metadata() -> None:
    context = build_context_block([_chunk()])
    assert "policy.md" in context
    assert "page 2" in context
    assert "doc-1-chunk-0001" in context


def test_grounded_prompt_includes_question() -> None:
    prompt = build_grounded_answer_prompt("What is the deadline?", [_chunk()])
    assert "What is the deadline?" in prompt
    assert "Include citations" in prompt
