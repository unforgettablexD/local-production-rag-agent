from datetime import UTC, datetime
from uuid import uuid4


def generate_document_id(filename: str) -> str:
    stem = filename.rsplit(".", maxsplit=1)[0].lower().replace(" ", "-")
    return f"{stem}-{uuid4().hex[:8]}"


def generate_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-chunk-{index:04d}"


def utc_now() -> datetime:
    return datetime.now(UTC)
