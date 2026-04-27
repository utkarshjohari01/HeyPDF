/**
 * Toast.jsx — In-app toast notification system for HeyPDF 2.0
 *
 * Usage (from parent via ref or context):
 *   <Toast toasts={toasts} />
 *
 * Each toast: { id, type: 'success'|'error'|'info', message }
 */

import { useEffect, useState } from 'react';

// ── Individual Toast Item ──────────────────────────────────────────────────────
function ToastItem({ toast, onRemove }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Fade in
    const showTimer = setTimeout(() => setVisible(true), 10);

    // Auto-dismiss after 3.5s
    const hideTimer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onRemove(toast.id), 300); // wait for fade-out
    }, 3500);

    return () => {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
    };
  }, [toast.id, onRemove]);

  const styles = {
    success: {
      border: 'border-emerald-500/40',
      bg: 'bg-emerald-500/10 dark:bg-emerald-500/20',
      icon: '✅',
      text: 'text-emerald-700 dark:text-emerald-400',
    },
    error: {
      border: 'border-red-500/40',
      bg: 'bg-red-500/10 dark:bg-red-500/20',
      icon: '❌',
      text: 'text-red-700 dark:text-red-400',
    },
    info: {
      border: 'border-accent-500/40',
      bg: 'bg-accent-500/10 dark:bg-accent-500/20',
      icon: 'ℹ️',
      text: 'text-accent-700 dark:text-accent-400',
    },
  };

  const s = styles[toast.type] || styles.info;

  return (
    <div
      className={`
        flex items-start gap-3 px-4 py-3 rounded-xl shadow-lg backdrop-blur-sm
        border ${s.border} ${s.bg}
        transition-all duration-300 ease-out
        ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}
        max-w-sm w-full
      `}
    >
      <span className="text-base leading-none mt-0.5 flex-shrink-0">{s.icon}</span>
      <p className={`text-sm font-medium ${s.text} flex-1`}>{toast.message}</p>
      <button
        onClick={() => onRemove(toast.id)}
        className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors ml-1 flex-shrink-0"
        aria-label="Dismiss notification"
      >
        ✕
      </button>
    </div>
  );
}

// ── Toast Container ────────────────────────────────────────────────────────────
export default function Toast({ toasts, onRemove }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div
      id="toast-container"
      className="fixed top-20 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} onRemove={onRemove} />
        </div>
      ))}
    </div>
  );
}
