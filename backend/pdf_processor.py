"""
pdf_processor.py — PDF text extraction and chunking for HeyPDF 2.0

Uses pdfplumber for accurate multi-column text extraction.
Returns page-aware chunks so we can cite {pdf_name, page_number} in answers.

Chunk metadata format:
    {pdf_name, pdf_id, page_number, chunk_index, text}
"""

import io
import pdfplumber

# ── Chunking configuration ──────────────────────────────────────────────────
CHUNK_SIZE = 800     # characters per chunk (roughly 150-200 words)
CHUNK_OVERLAP = 100  # characters of overlap between consecutive chunks


def extract_text_with_pages(pdf_bytes_io: io.BytesIO, pdf_name: str) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Args:
        pdf_bytes_io: BytesIO containing the PDF data
        pdf_name: Original filename (for error messages)

    Returns:
        List of dicts: [{page_number: int, text: str}, ...]

    Raises:
        ValueError: If the PDF appears to be scanned (no extractable text)
    """
    pages = []

    with pdfplumber.open(pdf_bytes_io) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
            })

    # If every page is empty, it's likely a scanned/image PDF
    total_text = " ".join(p["text"] for p in pages).strip()
    if not total_text:
        raise ValueError(
            "This PDF appears to be scanned or image-only. "
            "Text extraction is not supported yet. "
            "Please upload a text-based PDF."
        )

    return pages


def chunk_pages(pages: list[dict], pdf_name: str, pdf_id: str) -> list[dict]:
    """
    Split page texts into overlapping fixed-size chunks with page metadata.

    Each chunk carries: {pdf_name, pdf_id, page_number, chunk_index, text}
    so we can cite the source in AI responses.

    Args:
        pages: Output of extract_text_with_pages()
        pdf_name: Original filename
        pdf_id: Unique UUID for this PDF

    Returns:
        Flat list of chunk dicts across all pages
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]

        if not text.strip():
            continue  # Skip blank pages

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            # Skip tiny trailing fragments (< 80 chars) — append to previous instead
            if len(chunk_text) < 80 and chunks:
                chunks[-1]["text"] += " " + chunk_text
                break

            chunks.append({
                "pdf_name": pdf_name,
                "pdf_id": pdf_id,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })

            chunk_index += 1
            start = end - CHUNK_OVERLAP  # Slide window with overlap

    return chunks


def process_pdf(pdf_bytes_io: io.BytesIO, pdf_name: str, pdf_id: str) -> tuple[list[dict], int]:
    """
    Full pipeline: extract text → chunk with metadata.

    Args:
        pdf_bytes_io: PDF content as BytesIO
        pdf_name: Original filename
        pdf_id: Unique UUID

    Returns:
        (chunks, page_count)

    Raises:
        ValueError: For scanned PDFs
    """
    pages = extract_text_with_pages(pdf_bytes_io, pdf_name)
    page_count = len(pages)
    chunks = chunk_pages(pages, pdf_name, pdf_id)
    return chunks, page_count


def get_full_text(pdf_bytes_io: io.BytesIO) -> str:
    """
    Get concatenated text from all pages (used for summary generation).
    Truncates at 8000 chars to stay within AI token limits.
    """
    with pdfplumber.open(pdf_bytes_io) as pdf:
        texts = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    full = "\n\n".join(texts)
    return full[:8000]  # Truncate for AI summary call
