/**
 * api.js — Axios API client for HeyPDF 2.0
 *
 * All backend calls go through this module.
 * Base URL: http://localhost:8000 (FastAPI dev server)
 */

import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 minutes — AI calls can be slow on first load
});

// ── PDF endpoints ──────────────────────────────────────────────────────────────

/**
 * Upload a single PDF file.
 * @param {File} file - The PDF File object from input or drop
 * @param {function} onProgress - Optional upload progress callback (0-100)
 * @returns {Promise<PDFInfo>} - {pdf_id, filename, page_count, summary, suggested_questions}
 */
export async function uploadPDF(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(pct);
      }
    },
  });

  return response.data;
}

/**
 * List all uploaded PDFs in the current session.
 * @returns {Promise<PDFInfo[]>}
 */
export async function listPDFs() {
  const response = await api.get('/pdfs');
  return response.data;
}

/**
 * Delete a PDF and its embeddings.
 * @param {string} pdfId
 * @returns {Promise<{message: string}>}
 */
export async function deletePDF(pdfId) {
  const response = await api.delete(`/pdfs/${pdfId}`);
  return response.data;
}

// ── Chat endpoint ──────────────────────────────────────────────────────────────

/**
 * Send a chat message and get an AI answer with source citations.
 * @param {string} question - User's question
 * @param {string[]} activePdfIds - IDs of PDFs to search
 * @param {Array<{question:string, answer:string}>} chatHistory - Recent conversation
 * @returns {Promise<{answer: string, sources: SourceCitation[]}>}
 */
export async function sendChat(question, activePdfIds, chatHistory) {
  const response = await api.post('/chat', {
    question,
    active_pdf_ids: activePdfIds,
    chat_history: chatHistory,
  });
  return response.data;
}

// ── Export endpoint ────────────────────────────────────────────────────────────

/**
 * Export chat history as a .txt file (triggers browser download).
 * @param {Array<{question:string, answer:string}>} chatHistory
 * @param {string} sessionName
 */
export async function exportChat(chatHistory, sessionName = 'HeyPDF Session') {
  const response = await api.post(
    '/export',
    { chat_history: chatHistory, session_name: sessionName },
    { responseType: 'blob' }
  );

  // Trigger browser download
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', 'heypdf_chat.txt');
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
