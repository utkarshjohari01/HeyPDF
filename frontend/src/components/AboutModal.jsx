/**
 * AboutModal.jsx — "About HeyPDF" information modal
 *
 * Accessible modal dialog explaining what the app does and
 * highlighting the multi-provider AI rotation as a tech feature.
 */

import { useEffect } from 'react';

export default function AboutModal({ isOpen, onClose }) {
  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4
                 bg-black/50 backdrop-blur-sm animate-fade-in"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="about-modal-title"
    >
      {/* Modal panel */}
      <div className="glass-card w-full max-w-lg p-6 md:p-8 animate-fade-in-up
                      dark:bg-navy-800/95 relative">

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 btn-icon"
          aria-label="Close about modal"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
            strokeWidth={2} stroke="currentColor" className="w-5 h-5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-500 to-accent-700
                          flex items-center justify-center shadow-lg">
            <span className="text-white text-lg font-bold">H</span>
          </div>
          <div>
            <h2 id="about-modal-title" className="text-xl font-bold text-gray-900 dark:text-white">
              HeyPDF 2.0
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">
              AI-Powered PDF Chat Assistant
            </p>
          </div>
        </div>

        {/* Description */}
        <div className="space-y-4 text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
          <p>
            <strong className="text-gray-900 dark:text-white">HeyPDF</strong> lets you upload
            one or more PDF documents and have a natural conversation with them using AI.
            Ask any question and get accurate, page-cited answers drawn directly from your documents.
          </p>

          {/* Features list */}
          <div className="space-y-2">
            <p className="font-semibold text-gray-800 dark:text-gray-200">Key Features:</p>
            <ul className="space-y-1.5 ml-1">
              {[
                '📄 Multi-PDF upload with semantic Q&A',
                '🔍 FAISS vector search for relevant context retrieval',
                '📍 Page-level source citations on every answer',
                '✨ Auto-generated summaries on upload',
                '💡 Suggested starter questions',
                '🧠 Conversation memory (last 8 turns)',
                '📥 Export full chat history as .txt',
              ].map((f, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="flex-shrink-0">{f.split(' ')[0]}</span>
                  <span>{f.split(' ').slice(1).join(' ')}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Tech highlight: key rotation */}
          <div className="mt-4 p-4 rounded-xl bg-accent-500/10 dark:bg-accent-500/15
                          border border-accent-500/25">
            <p className="font-semibold text-accent-600 dark:text-accent-400 mb-1.5">
              ⚡ Smart API Key Rotation
            </p>
            <p className="text-xs leading-relaxed">
              HeyPDF uses a <strong>multi-provider rotation system</strong> across four free AI
              providers — <strong>Groq → Gemini → OpenRouter → HuggingFace</strong>.
              If one provider's quota is exceeded, the app automatically falls back to the next,
              completely transparently. This means the app stays responsive even under heavy use
              with free-tier API keys.
            </p>
          </div>

          {/* Tech stack */}
          <div className="flex flex-wrap gap-2 pt-1">
            {['FastAPI', 'React', 'FAISS', 'sentence-transformers', 'pdfplumber', 'Tailwind CSS'].map(
              (tech) => (
                <span key={tech}
                  className="px-2.5 py-1 text-xs font-medium rounded-full
                             bg-gray-100 dark:bg-navy-700
                             text-gray-600 dark:text-gray-300
                             border border-gray-200 dark:border-navy-600">
                  {tech}
                </span>
              )
            )}
          </div>

          <p className="text-xs text-gray-400 dark:text-gray-500 pt-1">
            Built as a final year project. Runs entirely locally — no account, no cloud, no data leaves your machine.
          </p>
        </div>

        {/* Footer */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="btn-primary text-sm"
          >
            Got it!
          </button>
        </div>
      </div>
    </div>
  );
}
