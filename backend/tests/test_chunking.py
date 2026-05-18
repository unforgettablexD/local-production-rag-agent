from app.ingestion.chunking import split_text


def test_split_text_returns_overlapping_chunks() -> None:
    text = "a" * 1000
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 4
    assert len(chunks[0]) == 300
    assert len(chunks[1]) == 300


def test_split_text_validates_arguments() -> None:
    try:
        split_text("hello", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
