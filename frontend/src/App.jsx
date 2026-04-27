/**
 * App.jsx — Root component for HeyPDF 2.0
 *
 * Manages all top-level state:
 *   - Dark/light theme (persisted in localStorage)
 *   - PDF list and active PDF IDs
 *   - Chat history
 *   - AI loading state
 *   - Toast notifications
 *   - Suggested questions (from most recently uploaded PDF)
 *   - About modal
 *
 * Layout:
 *   TopBar (fixed, full width)
 *   └── Content row (below top bar)
 *       ├── PDFSidebar (left, fixed width, hidden on mobile if no PDFs)
 *       └── ChatWindow (flex-1, fills remaining space)
 */

import { useState, useEffect, useCallback } from 'react';
import TopBar from './components/TopBar';
import PDFSidebar from './components/PDFSidebar';
import ChatWindow from './components/ChatWindow';
import AboutModal from './components/AboutModal';
import Toast from './components/Toast';
import { uploadPDF, deletePDF, sendChat, exportChat } from './api';

// ── Unique ID helper ───────────────────────────────────────────────────────────
let _idCounter = 0;
function uid() { return `msg_${++_idCounter}_${Date.now()}`; }

// ── Toast helper ───────────────────────────────────────────────────────────────
function createToast(type, message) {
  return { id: `toast_${Date.now()}_${Math.random()}`, type, message };
}

export default function App() {
  // ── Theme ──────────────────────────────────────────────────────────────────
  const [darkMode, setDarkMode] = useState(() => {
    const stored = localStorage.getItem('heypdf_theme');
    if (stored) return stored === 'dark';
    // Default: system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('heypdf_theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const toggleTheme = () => setDarkMode((d) => !d);

  // ── PDFs ───────────────────────────────────────────────────────────────────
  const [pdfs, setPdfs] = useState([]);                  // PDFInfo[]
  const [activePdfIds, setActivePdfIds] = useState(new Set()); // Set<string>
  const [uploadLoading, setUploadLoading] = useState(false);

  // ── Chat ───────────────────────────────────────────────────────────────────
  // messages: [{id, role:'user'|'ai', content, sources?}]
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // ── Suggested questions (from latest upload) ───────────────────────────────
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);

  // ── Toasts ─────────────────────────────────────────────────────────────────
  const [toasts, setToasts] = useState([]);
  const addToast = useCallback((type, message) => {
    setToasts((t) => [...t, createToast(type, message)]);
  }, []);
  const removeToast = useCallback((id) => {
    setToasts((t) => t.filter((toast) => toast.id !== id));
  }, []);

  // ── About modal ────────────────────────────────────────────────────────────
  const [aboutOpen, setAboutOpen] = useState(false);

  // ── Upload handler ─────────────────────────────────────────────────────────
  const handleFilesSelected = useCallback(async (files) => {
    if (uploadLoading) return;
    setUploadLoading(true);

    let uploadedCount = 0;
    let lastSuggestedQuestions = [];

    for (const file of files) {
      try {
        const pdfInfo = await uploadPDF(file);

        setPdfs((prev) => {
          // Avoid duplicates if same filename already exists
          const exists = prev.some((p) => p.filename === pdfInfo.filename);
          if (exists) return prev;
          return [...prev, pdfInfo];
        });

        // Auto-activate newly uploaded PDF
        setActivePdfIds((prev) => {
          const next = new Set(prev);
          next.add(pdfInfo.pdf_id);
          return next;
        });

        lastSuggestedQuestions = pdfInfo.suggested_questions || [];
        uploadedCount++;
      } catch (err) {
        const detail = err?.response?.data?.detail || err.message || 'Upload failed';
        addToast('error', `${file.name}: ${detail}`);
      }
    }

    if (uploadedCount > 0) {
      addToast('success', `${uploadedCount} PDF${uploadedCount > 1 ? 's' : ''} uploaded successfully!`);
      // Show suggested questions from the last uploaded PDF
      if (lastSuggestedQuestions.length > 0) {
        setSuggestedQuestions(lastSuggestedQuestions);
      }
    }

    setUploadLoading(false);
  }, [uploadLoading, addToast]);

  // ── Toggle PDF active/inactive ─────────────────────────────────────────────
  const handleTogglePdf = useCallback((pdfId) => {
    setActivePdfIds((prev) => {
      const next = new Set(prev);
      if (next.has(pdfId)) {
        next.delete(pdfId);
      } else {
        next.add(pdfId);
      }
      return next;
    });
  }, []);

  // ── Delete PDF ─────────────────────────────────────────────────────────────
  const handleDeletePdf = useCallback(async (pdfId) => {
    const pdf = pdfs.find((p) => p.pdf_id === pdfId);
    try {
      await deletePDF(pdfId);
      setPdfs((prev) => prev.filter((p) => p.pdf_id !== pdfId));
      setActivePdfIds((prev) => {
        const next = new Set(prev);
        next.delete(pdfId);
        return next;
      });
      addToast('info', `"${pdf?.filename}" removed.`);
    } catch {
      addToast('error', 'Failed to delete PDF. Please try again.');
    }
  }, [pdfs, addToast]);

  // ── Send chat message ──────────────────────────────────────────────────────
  const handleSendMessage = useCallback(async (question) => {
    if (!question.trim() || isLoading || activePdfIds.size === 0) return;

    // Hide suggested questions after first message
    setSuggestedQuestions([]);

    // Add user message immediately
    const userMsg = { id: uid(), role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      // Build chat_history from existing messages (up to last 8 AI turns)
      const chatHistory = [];
      const msgList = [...messages, userMsg];
      for (let i = 0; i < msgList.length - 1; i++) {
        if (msgList[i].role === 'user' && msgList[i + 1]?.role === 'ai') {
          chatHistory.push({
            question: msgList[i].content,
            answer: msgList[i + 1].content,
          });
        }
      }

      const response = await sendChat(
        question,
        Array.from(activePdfIds),
        chatHistory.slice(-8),
      );

      const aiMsg = {
        id: uid(),
        role: 'ai',
        content: response.answer,
        sources: response.sources || [],
      };
      setMessages((prev) => [...prev, aiMsg]);

    } catch (err) {
      const detail = err?.response?.data?.detail || err.message || 'Something went wrong';
      const errMsg = {
        id: uid(),
        role: 'ai',
        content: `❌ ${detail}`,
        sources: [],
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, activePdfIds, messages]);

  // ── Clear chat ─────────────────────────────────────────────────────────────
  const handleClearChat = useCallback(() => {
    setMessages([]);
    // Restore suggested questions from most recently active PDF
    const activePdf = pdfs.find((p) => activePdfIds.has(p.pdf_id));
    if (activePdf?.suggested_questions?.length > 0) {
      setSuggestedQuestions(activePdf.suggested_questions);
    }
  }, [pdfs, activePdfIds]);

  // ── Export chat ────────────────────────────────────────────────────────────
  const handleExportChat = useCallback(async () => {
    try {
      const chatHistory = [];
      for (let i = 0; i < messages.length - 1; i++) {
        if (messages[i].role === 'user' && messages[i + 1]?.role === 'ai') {
          chatHistory.push({
            question: messages[i].content,
            answer: messages[i + 1].content,
          });
        }
      }
      await exportChat(chatHistory);
      addToast('success', 'Chat exported as heypdf_chat.txt');
    } catch {
      addToast('error', 'Export failed. Please try again.');
    }
  }, [messages, addToast]);

  // ── Render ─────────────────────────────────────────────────────────────────
  const hasPdfs = pdfs.length > 0;

  return (
    <div className="h-screen flex flex-col bg-surface-50 dark:bg-navy-900 overflow-hidden">

      {/* Fixed top bar */}
      <TopBar
        darkMode={darkMode}
        onToggleTheme={toggleTheme}
        onFilesSelected={handleFilesSelected}
        onClearChat={handleClearChat}
        onExportChat={handleExportChat}
        onOpenAbout={() => setAboutOpen(true)}
        chatHistoryLength={messages.length}
        uploadLoading={uploadLoading}
      />

      {/* Content below top bar */}
      <div className="flex flex-1 overflow-hidden pt-16">

        {/* Sidebar — only visible when PDFs are uploaded */}
        {hasPdfs && (
          <PDFSidebar
            pdfs={pdfs}
            activePdfIds={activePdfIds}
            onToggle={handleTogglePdf}
            onDelete={handleDeletePdf}
            onFilesSelected={handleFilesSelected}
            uploadLoading={uploadLoading}
          />
        )}

        {/* Main chat area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <ChatWindow
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            activePdfCount={activePdfIds.size}
            hasPdfs={hasPdfs}
            suggestedQuestions={suggestedQuestions}
            onFilesSelected={handleFilesSelected}
            uploadLoading={uploadLoading}
          />
        </main>
      </div>

      {/* About modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />

      {/* Toast notifications */}
      <Toast toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
