/**
 * PDFSidebar.jsx — Sidebar showing uploaded PDFs with toggle, summary, and delete
 *
 * Desktop: fixed left sidebar (300px wide)
 * Mobile: shown as a bottom sheet / drawer
 *
 * Props:
 *   pdfs           — array of PDFInfo objects
 *   activePdfIds   — Set of active PDF IDs
 *   onToggle(id)   — toggle a PDF active/inactive
 *   onDelete(id)   — delete a PDF
 *   onFilesSelected(files) — pass-through for upload button
 *   uploadLoading  — bool
 */

import { useState } from 'react';

// ── PDF list item ──────────────────────────────────────────────────────────────
function PDFItem({ pdf, isActive, onToggle, onDelete }) {
  const [summaryOpen, setSummaryOpen] = useState(false);

  return (
    <div className={`rounded-xl border transition-all duration-200 overflow-hidden ${
      isActive
        ? 'border-accent-500/40 bg-accent-500/8 dark:bg-accent-500/8'
        : 'border-gray-200 dark:border-navy-600 bg-white dark:bg-navy-800'
    }`}>
      {/* Main row */}
      <div className="flex items-center gap-2 p-3">
        {/* Active toggle checkbox */}
        <button
          onClick={() => onToggle(pdf.pdf_id)}
          className={`flex-shrink-0 w-5 h-5 rounded-md border-2 transition-all duration-200 flex items-center justify-center ${
            isActive
              ? 'bg-accent-500 border-accent-500 shadow-sm shadow-accent-500/30'
              : 'border-gray-300 dark:border-navy-500 hover:border-accent-400'
          }`}
          title={isActive ? 'Deactivate this PDF' : 'Activate this PDF for Q&A'}
          aria-label={isActive ? 'Deactivate PDF' : 'Activate PDF'}
        >
          {isActive && (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={3} stroke="white" className="w-3 h-3">
              <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
          )}
        </button>

        {/* PDF icon */}
        <div className="flex-shrink-0 w-7 h-7 rounded-md bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
          <span className="text-xs font-bold text-red-500">PDF</span>
        </div>

        {/* File info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate leading-tight"
            title={pdf.filename}>
            {pdf.filename}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            {pdf.page_count} {pdf.page_count === 1 ? 'page' : 'pages'}
          </p>
        </div>

        {/* Summary toggle */}
        <button
          onClick={() => setSummaryOpen((o) => !o)}
          className="flex-shrink-0 btn-icon p-1.5"
          title={summaryOpen ? 'Hide summary' : 'Show summary'}
          aria-label="Toggle summary"
          aria-expanded={summaryOpen}
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
            strokeWidth={2} stroke="currentColor"
            className={`w-4 h-4 transition-transform duration-200 ${summaryOpen ? 'rotate-180' : ''}`}>
            <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
          </svg>
        </button>

        {/* Delete */}
        <button
          onClick={() => onDelete(pdf.pdf_id)}
          className="flex-shrink-0 p-1.5 rounded-lg text-gray-400 hover:text-red-500
                     hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200"
          title="Delete PDF"
          aria-label="Delete PDF"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
            strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
          </svg>
        </button>
      </div>

      {/* Collapsible summary panel */}
      {summaryOpen && pdf.summary && pdf.summary.length > 0 && (
        <div className="px-3 pb-3 border-t border-gray-100 dark:border-navy-700 pt-2 animate-fade-in">
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
            Summary
          </p>
          <ul className="space-y-1.5">
            {pdf.summary.map((point, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-gray-600 dark:text-gray-300">
                <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-accent-500 mt-1.5" />
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Sidebar ────────────────────────────────────────────────────────────────────
export default function PDFSidebar({
  pdfs,
  activePdfIds,
  onToggle,
  onDelete,
  onFilesSelected,
  uploadLoading,
}) {
  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.name.toLowerCase().endsWith('.pdf')
    );
    if (files.length > 0) onFilesSelected(files);
  };

  return (
    <aside
      id="pdf-sidebar"
      className="sidebar-width flex-shrink-0 h-full flex flex-col
                 bg-gray-50/50 dark:bg-navy-950/50
                 border-r border-gray-200/60 dark:border-navy-700/60"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-5 pb-3">
        <div>
          <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200">Your PDFs</h2>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            {pdfs.length === 0 ? 'No PDFs yet' : `${pdfs.length} uploaded · ${activePdfIds.size} active`}
          </p>
        </div>

        {/* Small upload button in sidebar */}
        <label
          htmlFor="sidebar-upload-input"
          className={`btn-icon cursor-pointer ${uploadLoading ? 'opacity-50 pointer-events-none' : ''}`}
          title="Upload more PDFs"
          aria-label="Upload PDF"
        >
          {uploadLoading ? (
            <svg className="w-4 h-4 animate-spin text-accent-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={2} stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
          )}
          <input
            id="sidebar-upload-input"
            type="file" accept=".pdf" multiple className="hidden"
            onChange={(e) => {
              const files = Array.from(e.target.files || []);
              if (files.length > 0) onFilesSelected(files);
              e.target.value = '';
            }}
            disabled={uploadLoading}
          />
        </label>
      </div>

      {/* PDF list */}
      <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-2">
        {pdfs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-center px-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={1.2} stroke="currentColor"
              className="w-10 h-10 text-gray-300 dark:text-navy-600 mb-2">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
            <p className="text-xs text-gray-400 dark:text-gray-600">
              Upload a PDF to get started
            </p>
          </div>
        ) : (
          pdfs.map((pdf) => (
            <PDFItem
              key={pdf.pdf_id}
              pdf={pdf}
              isActive={activePdfIds.has(pdf.pdf_id)}
              onToggle={onToggle}
              onDelete={onDelete}
            />
          ))
        )}
      </div>

      {/* Bottom: drop hint */}
      {pdfs.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-200/60 dark:border-navy-700/60">
          <p className="text-xs text-center text-gray-400 dark:text-gray-600">
            Drop more PDFs here to add them
          </p>
        </div>
      )}
    </aside>
  );
}
