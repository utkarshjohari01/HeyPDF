/**
 * TopBar.jsx — Fixed top navigation bar for HeyPDF 2.0
 *
 * Contains:
 *   - App branding (logo + name)
 *   - Upload button (triggers hidden file input)
 *   - Clear chat button
 *   - Export chat button (only if chatHistory is non-empty)
 *   - About modal trigger
 *   - Dark/Light theme toggle
 */

import { useRef } from 'react';

// ── Sun icon ──────────────────────────────────────────────────────────────────
function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
      strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z" />
    </svg>
  );
}

// ── Moon icon ─────────────────────────────────────────────────────────────────
function MoonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
      strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z" />
    </svg>
  );
}

// ── Upload icon ───────────────────────────────────────────────────────────────
function UploadIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
      strokeWidth={2} stroke="currentColor" className="w-4 h-4">
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
    </svg>
  );
}

export default function TopBar({
  darkMode,
  onToggleTheme,
  onFilesSelected,
  onClearChat,
  onExportChat,
  onOpenAbout,
  chatHistoryLength,
  uploadLoading,
}) {
  const fileInputRef = useRef(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset so re-selecting same file triggers onChange
    e.target.value = '';
  };

  return (
    <header
      id="top-bar"
      className="fixed top-0 left-0 right-0 z-40 h-16
                 bg-white/80 dark:bg-navy-900/80 backdrop-blur-md
                 border-b border-gray-200/60 dark:border-navy-700/60
                 flex items-center justify-between px-4 md:px-6
                 shadow-sm"
    >
      {/* ── Brand ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center shadow-md">
          <span className="text-white text-sm font-bold leading-none">H</span>
        </div>
        <div className="flex flex-col">
          <span className="text-base font-bold text-gray-900 dark:text-white tracking-tight leading-none">
            HeyPDF
          </span>
          <span className="text-[10px] text-gray-400 dark:text-gray-500 font-medium leading-none">
            AI PDF Chat
          </span>
        </div>
      </div>

      {/* ── Actions ───────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-1.5 md:gap-2">

        {/* Upload button */}
        <button
          id="upload-button"
          onClick={handleUploadClick}
          disabled={uploadLoading}
          className="btn-primary flex items-center gap-2 text-sm hidden sm:flex"
          title="Upload PDF(s)"
        >
          {uploadLoading ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <UploadIcon />
          )}
          <span className="hidden md:inline">{uploadLoading ? 'Processing…' : 'Upload PDF'}</span>
        </button>

        {/* Mobile upload (icon only) */}
        <button
          onClick={handleUploadClick}
          disabled={uploadLoading}
          className="btn-icon sm:hidden"
          title="Upload PDF"
        >
          <UploadIcon />
        </button>

        {/* Clear chat */}
        {chatHistoryLength > 0 && (
          <button
            id="clear-chat-button"
            onClick={onClearChat}
            className="btn-ghost text-sm flex items-center gap-1.5"
            title="Clear chat history"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9.75 14.25 12m0 0 2.25 2.25M14.25 12l2.25-2.25M14.25 12 12 14.25m-2.58 4.92-6.374-6.375a1.125 1.125 0 0 1 0-1.59L9.42 4.83c.21-.211.497-.33.795-.33H19.5a2.25 2.25 0 0 1 2.25 2.25v10.5a2.25 2.25 0 0 1-2.25 2.25h-9.284c-.298 0-.585-.119-.795-.33Z" />
            </svg>
            <span className="hidden md:inline">Clear</span>
          </button>
        )}

        {/* Export chat */}
        {chatHistoryLength > 0 && (
          <button
            id="export-chat-button"
            onClick={onExportChat}
            className="btn-ghost text-sm flex items-center gap-1.5"
            title="Export chat as .txt"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
            <span className="hidden md:inline">Export</span>
          </button>
        )}

        {/* About */}
        <button
          id="about-button"
          onClick={onOpenAbout}
          className="btn-icon"
          title="About HeyPDF"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
            strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
          </svg>
        </button>

        {/* Theme toggle */}
        <button
          id="theme-toggle-button"
          onClick={onToggleTheme}
          className="btn-icon"
          title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label="Toggle theme"
        >
          {darkMode ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      {/* Hidden file input — accepts multiple PDFs */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={handleFileChange}
        aria-hidden="true"
      />
    </header>
  );
}
