/**
 * ChatWindow.jsx — Main chat interface
 *
 * Displays message bubbles, typing indicator, and the input bar.
 * Auto-scrolls to bottom on new messages.
 *
 * Props:
 *   messages         — array of {id, role, content, sources?}
 *   onSendMessage(q) — called when user submits a question
 *   isLoading        — bool (show typing indicator)
 *   activePdfCount   — number of active PDFs (for placeholder text)
 *   suggestedQuestions — string[]
 *   onSuggestedSelect(q) — handler for chips
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import MessageBubble from './MessageBubble';
import SuggestedQuestions from './SuggestedQuestions';
import UploadZone from './UploadZone';

// ── Typing indicator ───────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex items-center gap-2.5 animate-fade-in-up mb-4">
      {/* Mini avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-accent-500 to-accent-700 flex items-center justify-center shadow-sm">
        <span className="text-white text-xs font-bold leading-none">H</span>
      </div>
      {/* Dots */}
      <div className="bubble-ai flex items-center gap-1.5 py-3 px-4">
        <span className="typing-dot" style={{ animationDelay: '0ms' }} />
        <span className="typing-dot" style={{ animationDelay: '200ms' }} />
        <span className="typing-dot" style={{ animationDelay: '400ms' }} />
      </div>
    </div>
  );
}

// ── Empty state ────────────────────────────────────────────────────────────────
function EmptyChat({ activePdfCount }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-accent-500/10 dark:bg-accent-500/15 flex items-center justify-center mb-1">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
          strokeWidth={1.3} stroke="currentColor" className="w-7 h-7 text-accent-500">
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
        </svg>
      </div>
      <div>
        <p className="text-base font-semibold text-gray-700 dark:text-gray-300">
          Ready to answer your questions
        </p>
        <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
          {activePdfCount > 0
            ? `${activePdfCount} PDF${activePdfCount > 1 ? 's' : ''} active — ask anything!`
            : 'Activate a PDF in the sidebar to start chatting'}
        </p>
      </div>
    </div>
  );
}

// ── ChatWindow ─────────────────────────────────────────────────────────────────
export default function ChatWindow({
  messages,
  onSendMessage,
  isLoading,
  activePdfCount,
  hasPdfs,
  suggestedQuestions,
  onFilesSelected,
  uploadLoading,
}) {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize textarea
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    // Reset height then set to scroll height
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  };

  const handleSubmit = useCallback((e) => {
    e?.preventDefault();
    const q = inputValue.trim();
    if (!q || isLoading || activePdfCount === 0) return;
    onSendMessage(q);
    setInputValue('');
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [inputValue, isLoading, activePdfCount, onSendMessage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestedSelect = (question) => {
    setInputValue(question);
    setTimeout(() => {
      onSendMessage(question);
      setInputValue('');
    }, 50);
  };

  // If no PDFs uploaded at all, show the upload zone
  if (!hasPdfs) {
    return (
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <UploadZone onFilesSelected={onFilesSelected} uploadLoading={uploadLoading} />
        </div>
      </div>
    );
  }

  const canSend = inputValue.trim().length > 0 && !isLoading && activePdfCount > 0;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* ── Message list ─────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4">
        {messages.length === 0 ? (
          <EmptyChat activePdfCount={activePdfCount} />
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* ── Suggested questions ──────────────────────────────────────────── */}
      {messages.length === 0 && suggestedQuestions?.length > 0 && (
        <SuggestedQuestions
          questions={suggestedQuestions}
          onSelect={handleSuggestedSelect}
          disabled={isLoading || activePdfCount === 0}
        />
      )}

      {/* ── Input bar ────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-4 md:px-6 pb-4 pt-2
                      border-t border-gray-200/60 dark:border-navy-700/60">
        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2 glass-card p-2 shadow-lg
                     dark:bg-navy-800/90 focus-within:ring-2 focus-within:ring-accent-500/40
                     transition-shadow duration-200"
        >
          <textarea
            ref={textareaRef}
            id="chat-input"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={
              activePdfCount === 0
                ? 'Activate a PDF in the sidebar first…'
                : 'Ask a question about your PDFs… (Enter to send, Shift+Enter for newline)'
            }
            disabled={isLoading || activePdfCount === 0}
            rows={1}
            className="flex-1 bg-transparent border-none outline-none resize-none
                       text-sm text-gray-800 dark:text-gray-100
                       placeholder-gray-400 dark:placeholder-gray-600
                       py-2 px-2 leading-relaxed
                       disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Chat input"
            style={{ minHeight: '40px', maxHeight: '120px' }}
          />

          {/* Send button */}
          <button
            type="submit"
            disabled={!canSend}
            className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center
                       transition-all duration-200 ease-in-out
                       ${canSend
                         ? 'bg-accent-500 hover:bg-accent-600 shadow-md hover:shadow-accent-500/30 hover:shadow-lg'
                         : 'bg-gray-200 dark:bg-navy-700'
                       }`}
            aria-label="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
              strokeWidth={2} stroke="currentColor"
              className={`w-4 h-4 ${canSend ? 'text-white' : 'text-gray-400 dark:text-gray-500'}`}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
            </svg>
          </button>
        </form>

        <p className="text-xs text-center text-gray-400 dark:text-gray-600 mt-2">
          HeyPDF answers from your PDFs only · Powered by AI
        </p>
      </div>
    </div>
  );
}
