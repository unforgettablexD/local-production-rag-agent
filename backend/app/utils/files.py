from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def ensure_supported_file(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: PDF, DOCX, TXT, MD."
        )


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
