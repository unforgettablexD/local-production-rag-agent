from datetime import UTC, datetime

from app.models.schemas import ChatResponse, Citation


def test_chat_response_accepts_valid_payload() -> None:
    payload = ChatResponse(
        answer="Expenses are due within 30 days. [1]",
        citations=[
            Citation(
                index=1,
                document_id="doc-1",
                filename="expense_policy.md",
                chunk_id="doc-1-chunk-0001",
                page_number=None,
                snippet="Expenses must be submitted within 30 days.",
            )
        ],
        retrieved_chunks=[],
        confidence=0.8,
        needs_retrieval=True,
        rewritten_query="expense policy deadline",
        refusal=False,
    )
    assert payload.citations[0].filename == "expense_policy.md"


def test_datetime_roundtrip_example() -> None:
    created_at = datetime.now(UTC)
    assert created_at.tzinfo is not None
