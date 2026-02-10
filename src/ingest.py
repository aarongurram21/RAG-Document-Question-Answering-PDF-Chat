import io
import pathlib
from typing import Iterable, List, Tuple

import pypdf


def extract_text_from_pdf(file_like: io.BytesIO | pathlib.Path) -> str:
    """Extract raw text from a PDF file-like object or path."""
    reader = pypdf.PdfReader(file_like)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    chunk_size = max(chunk_size, 1)
    chunk_overlap = max(min(chunk_overlap, chunk_size - 1), 0)
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start <= 0:
            break
    return chunks


def prepare_documents(chunks: Iterable[str]) -> Tuple[List[str], List[dict]]:
    """Attach simple metadata for citation tracking."""
    texts: List[str] = []
    metadatas: List[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        texts.append(chunk)
        metadatas.append({"source": f"chunk-{idx}"})
    return texts, metadatas
