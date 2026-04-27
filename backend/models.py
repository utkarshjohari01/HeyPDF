"""
models.py — Pydantic request/response schemas for HeyPDF 2.0

All API endpoints use these models for validation and serialization.
"""

from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    question: str
    answer: str


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    question: str
    active_pdf_ids: list[str]        # IDs of PDFs to search
    chat_history: list[ChatMessage] = []  # Previous conversation turns


class SourceCitation(BaseModel):
    """Identifies where an answer came from in a PDF."""
    pdf_name: str
    pdf_id: str
    page_number: int


class ChatResponse(BaseModel):
    """Response from the /chat endpoint."""
    answer: str
    sources: list[SourceCitation]


class PDFInfo(BaseModel):
    """Metadata for an uploaded PDF, returned from /upload and /pdfs."""
    pdf_id: str
    filename: str
    page_count: int
    summary: list[str]              # 3-5 bullet point summary
    suggested_questions: list[str]  # 3 starter questions the user might ask


class ExportRequest(BaseModel):
    """Request body for the /export endpoint."""
    chat_history: list[ChatMessage]
    session_name: Optional[str] = "HeyPDF Session"
