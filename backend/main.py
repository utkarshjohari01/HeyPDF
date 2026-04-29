"""
main.py — FastAPI backend for HeyPDF 2.0

Endpoints:
  GET    /               → Health check
  POST   /upload         → Upload + process a PDF; returns summary + suggested questions
  POST   /chat           → Semantic Q&A with conversation memory + source citations
  GET    /pdfs           → List all PDFs in the current session
  DELETE /pdfs/{pdf_id}  → Remove a PDF and its embeddings
  POST   /export         → Download full chat history as .txt

CORS is configured for the React dev server (localhost:5173).
All state is in-memory (session-based, no database).
"""

import io
import uuid
import logging
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import (
    ChatRequest, ChatResponse, SourceCitation,
    PDFInfo, ExportRequest, ChatMessage,
)
from pdf_processor import process_pdf, get_full_text
from embeddings import embed_and_store, search_chunks, delete_pdf_index
from key_manager import key_manager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HeyPDF 2.0",
    description="AI-powered PDF chat backend with multi-provider key rotation",
    version="2.0.0",
)

# CORS: allow any origin for easy local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: {pdf_id: PDFInfo-dict}
# Reset when the server restarts (session-based by design)
_pdf_store: dict[str, dict] = {}


# ── Helper: generate summary + suggested questions ────────────────────────────

def _generate_summary_and_questions(
    full_text: str, pdf_name: str
) -> tuple[list[str], list[str]]:
    """
    Call the AI (via key_manager) to generate:
      - 4 bullet-point summary of the PDF
      - 3 suggested starter questions

    Parses the structured AI response using a '---QUESTIONS---' separator.
    Falls back to defaults if the AI response can't be parsed.
    """
    summary_prompt = (
        f"You are summarizing a PDF document named '{pdf_name}'.\n\n"
        "Provide your response in EXACTLY this format:\n"
        "• First key point here\n"
        "• Second key point here\n"
        "• Third key point here\n"
        "• Fourth key point here\n"
        "---QUESTIONS---\n"
        "What is the main topic of this document?\n"
        "What are the key findings?\n"
        "What conclusions does the document reach?\n\n"
        "Replace the example lines above with actual content from the PDF below. "
        "Keep bullets concise (1 sentence each). Keep questions specific to this PDF.\n\n"
        f"PDF content:\n{full_text}"
    )

    response = key_manager.generate(summary_prompt, context="", chat_history=[])

    summary_bullets: list[str] = []
    suggested_questions: list[str] = []

    try:
        if "---QUESTIONS---" in response:
            summary_part, questions_part = response.split("---QUESTIONS---", 1)
        else:
            # Try to split on a blank line roughly at the midpoint
            summary_part = response
            questions_part = ""

        # Parse bullets
        for line in summary_part.split("\n"):
            line = line.strip().lstrip("•-*1234567890. ").strip()
            if len(line) > 15:
                summary_bullets.append(line)

        # Parse questions
        for line in questions_part.split("\n"):
            line = line.strip().lstrip("?123. ").strip()
            if len(line) > 10:
                suggested_questions.append(line)

    except Exception as e:
        logger.warning(f"[Upload] Failed to parse summary response: {e}")

    # Ensure we always return something useful
    if not summary_bullets:
        summary_bullets = [
            "Document processed successfully.",
            "Ask a question below to explore the content.",
        ]
    if not suggested_questions:
        suggested_questions = [
            "What is the main topic of this document?",
            "What are the key findings or conclusions?",
            "Can you summarize the most important points?",
        ]

    return summary_bullets[:5], suggested_questions[:3]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "HeyPDF 2.0 API is running", "version": "2.0.0"}


@app.get("/health")
def health():
    """Lightweight pre-warming endpoint for the landing page server status indicator."""
    from datetime import timezone
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/upload", response_model=PDFInfo)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a single PDF file.

    Pipeline:
      1. Read uploaded bytes
      2. Extract text by page via pdfplumber
      3. Chunk with page metadata
      4. Embed chunks and store FAISS index
      5. Generate AI summary + suggested questions

    Returns:
      PDFInfo with pdf_id, filename, page_count, summary, suggested_questions
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_id = str(uuid.uuid4())
    pdf_name = file.filename

    try:
        pdf_bytes = await file.read()

        logger.info(f"[Upload] Processing: {pdf_name} (pdf_id={pdf_id})")

        # 1. Extract and chunk
        chunks, page_count = process_pdf(BytesIO(pdf_bytes), pdf_name, pdf_id)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this PDF. It may be scanned or image-only.",
            )

        # 2. Embed and store
        embed_and_store(chunks, pdf_id)

        # 3. Generate summary + questions (uses AI)
        full_text = get_full_text(BytesIO(pdf_bytes))
        summary, suggested_questions = _generate_summary_and_questions(full_text, pdf_name)

        # 4. Store in session
        pdf_info = {
            "pdf_id": pdf_id,
            "filename": pdf_name,
            "page_count": page_count,
            "summary": summary,
            "suggested_questions": suggested_questions,
        }
        _pdf_store[pdf_id] = pdf_info

        logger.info(
            f"[Upload] Done: {pdf_name} — {len(chunks)} chunks across {page_count} pages"
        )
        return pdf_info

    except ValueError as e:
        # Scanned PDF error from pdf_processor
        logger.warning(f"[Upload] Scanned PDF: {pdf_name}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Unexpected error for {pdf_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a user question using semantic retrieval + AI generation.

    Pipeline:
      1. Embed the question
      2. Search FAISS indexes of active PDFs for top-5 relevant chunks
      3. Build context string with source annotations
      4. Call AI (via key_manager) with context + conversation history
      5. Return answer + deduplicated source citations

    Maintains conversation memory via chat_history (last 8 turns injected into prompt).
    """
    if not request.active_pdf_ids:
        raise HTTPException(
            status_code=400,
            detail="No active PDFs selected. Please toggle at least one PDF active in the sidebar.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Semantic retrieval
    relevant_chunks = search_chunks(
        query=request.question,
        pdf_ids=request.active_pdf_ids,
        top_k=5,
    )

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the selected PDFs for this question.",
        )

    # 2. Build context string (annotated with source for AI to reference)
    context_parts = []
    for chunk in relevant_chunks:
        source_tag = f"[Source: {chunk['pdf_name']}, Page {chunk['page_number']}]"
        context_parts.append(f"{source_tag}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Build chat history list for key_manager
    history = [
        {"question": m.question, "answer": m.answer}
        for m in request.chat_history
    ]

    # 4. Generate answer
    answer = key_manager.generate(
        prompt=request.question,
        context=context,
        chat_history=history,
    )

    # 5. Build deduplicated source citations
    seen: set[tuple] = set()
    sources: list[SourceCitation] = []
    for chunk in relevant_chunks:
        key = (chunk["pdf_id"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(SourceCitation(
                pdf_name=chunk["pdf_name"],
                pdf_id=chunk["pdf_id"],
                page_number=chunk["page_number"],
            ))

    return ChatResponse(answer=answer, sources=sources)


@app.get("/pdfs")
async def list_pdfs():
    """Return all PDFs currently in the session."""
    return list(_pdf_store.values())


@app.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """
    Remove a PDF from the session and delete its FAISS index from disk.
    """
    if pdf_id not in _pdf_store:
        raise HTTPException(status_code=404, detail="PDF not found in session.")

    filename = _pdf_store[pdf_id]["filename"]
    del _pdf_store[pdf_id]
    delete_pdf_index(pdf_id)

    logger.info(f"[Delete] Removed: {filename} (pdf_id={pdf_id})")
    return {"message": f"'{filename}' removed successfully."}


@app.post("/export")
async def export_chat(request: ExportRequest):
    """
    Generate and return a plain-text file of the full chat history.

    Format:
      HeyPDF Chat Export
      Generated: YYYY-MM-DD HH:MM:SS
      ============================================================

      Q1: <question>
      A1: <answer>

      Q2: ...
    """
    lines = [
        "HeyPDF 2.0 — Chat Export",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Session: {request.session_name}",
        "=" * 60,
        "",
    ]

    for i, turn in enumerate(request.chat_history, start=1):
        lines.append(f"Q{i}: {turn.question}")
        lines.append(f"A{i}: {turn.answer}")
        lines.append("")

    content = "\n".join(lines)

    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="heypdf_chat.txt"'},
    )
