/**
 * UploadZone.jsx — Drag-and-drop + click-to-upload PDF zone
 *
 * Shown as a centered welcome screen when no PDFs have been uploaded yet.
 * Also reused as a floating drop overlay when dragging files over the window.
 *
 * Props:
 *   onFilesSelected(files: File[]) — called when user drops or picks files
 *   uploadLoading: bool
 */

import { useState, useCallback } from 'react';

export default function UploadZone({ onFilesSelected, uploadLoading }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
    );
    if (files.length > 0) onFilesSelected(files);
  }, [onFilesSelected]);

  const handleInputChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) onFilesSelected(files);
    e.target.value = '';
  };

  return (
    <div
      className="flex flex-col items-center justify-center h-full min-h-[60vh] px-4"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drop zone */}
      <label
        htmlFor="drop-zone-input"
        className={`
          group relative w-full max-w-md cursor-pointer
          border-2 border-dashed rounded-2xl p-10
          flex flex-col items-center gap-4 text-center
          transition-all duration-300 ease-out
          ${isDragging
            ? 'border-accent-500 bg-accent-500/10 scale-[1.02] shadow-xl shadow-accent-500/20'
            : 'border-gray-300 dark:border-navy-600 hover:border-accent-400 dark:hover:border-accent-500 bg-gray-50/50 dark:bg-navy-800/50 hover:bg-accent-500/5'
          }
          ${uploadLoading ? 'pointer-events-none opacity-70' : ''}
        `}
      >
        {/* Icon */}
        <div className={`
          w-16 h-16 rounded-2xl flex items-center justify-center
          transition-all duration-300
          ${isDragging
            ? 'bg-accent-500 shadow-lg shadow-accent-500/40'
            : 'bg-accent-500/10 dark:bg-accent-500/15 group-hover:bg-accent-500/20'
          }
        `}>
          {uploadLoading ? (
            <svg className="w-8 h-8 text-accent-500 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={1.5} stroke="currentColor"
              className={`w-8 h-8 transition-colors duration-300 ${
                isDragging ? 'text-white' : 'text-accent-500'
              }`}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
          )}
        </div>

        {/* Text */}
        <div className="space-y-1">
          <p className="text-base font-semibold text-gray-800 dark:text-gray-200">
            {uploadLoading
              ? 'Processing your PDF…'
              : isDragging
                ? 'Drop your PDFs here!'
                : 'Drop PDFs here or click to browse'
            }
          </p>
          {!uploadLoading && (
            <p className="text-sm text-gray-400 dark:text-gray-500">
              Supports multiple PDF files · Text-based PDFs only
            </p>
          )}
        </div>

        {/* Browse button (visual only — label handles click) */}
        {!uploadLoading && !isDragging && (
          <div className="btn-primary text-sm px-5 py-2">
            Browse Files
          </div>
        )}

        <input
          id="drop-zone-input"
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={handleInputChange}
          disabled={uploadLoading}
          aria-label="Upload PDF files"
        />
      </label>

      {/* Helper text below */}
      {!uploadLoading && (
        <div className="mt-8 max-w-sm text-center space-y-2">
          <p className="text-xs text-gray-400 dark:text-gray-600 font-medium uppercase tracking-wider">
            How it works
          </p>
          <div className="flex items-start gap-4 text-left">
            {[
              { step: '1', label: 'Upload', desc: 'Drop one or more PDF files' },
              { step: '2', label: 'Ask', desc: 'Type any question about your PDFs' },
              { step: '3', label: 'Cite', desc: 'Get answers with page references' },
            ].map((s) => (
              <div key={s.step} className="flex flex-col items-center gap-1 flex-1 text-center">
                <div className="w-7 h-7 rounded-full bg-accent-500/15 dark:bg-accent-500/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-accent-500">{s.step}</span>
                </div>
                <p className="text-xs font-semibold text-gray-700 dark:text-gray-300">{s.label}</p>
                <p className="text-xs text-gray-400 dark:text-gray-500">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
