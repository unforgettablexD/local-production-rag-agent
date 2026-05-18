from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.utils.files import ensure_supported_file, normalize_whitespace


@dataclass
class LoadedPage:
    page_number: int | None
    text: str


def load_document_bytes(filename: str, raw_bytes: bytes) -> list[LoadedPage]:
    ensure_supported_file(filename)
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(raw_bytes)
    if suffix == ".docx":
        return _load_docx(raw_bytes)
    return _load_text(raw_bytes)


def _load_pdf(raw_bytes: bytes) -> list[LoadedPage]:
    reader = PdfReader(BytesIO(raw_bytes))
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize_whitespace(page.extract_text() or "")
        if text:
            pages.append(LoadedPage(page_number=index, text=text))
    return pages


def _load_docx(raw_bytes: bytes) -> list[LoadedPage]:
    document = DocxDocument(BytesIO(raw_bytes))
    text = normalize_whitespace(
        "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    )
    return [LoadedPage(page_number=None, text=text)] if text else []


def _load_text(raw_bytes: bytes) -> list[LoadedPage]:
    text = raw_bytes.decode("utf-8", errors="ignore")
    normalized = normalize_whitespace(text)
    return [LoadedPage(page_number=None, text=normalized)] if normalized else []
