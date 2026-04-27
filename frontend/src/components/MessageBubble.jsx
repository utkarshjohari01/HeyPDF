/**
 * MessageBubble.jsx — Individual chat message with source citations
 *
 * User messages: right-aligned, accent color
 * AI messages: left-aligned, glass card, with source citation badges
 *
 * Highlight mode: bolds and colors terms that appear in quotes
 * (any phrase wrapped in "double quotes" or **double stars** gets highlighted)
 */

// ── Text highlighter ─────────────────────────────────────────────────────────
// Converts **bold** and "quoted terms" to accent-colored spans
function parseHighlights(text) {
  if (!text) return text;

  // Split on **...**  or "..."
  const parts = [];
  const regex = /\*\*(.+?)\*\*|"([^"]{3,60})"/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Text before match
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.slice(lastIndex, match.index) });
    }
    // Highlighted term (group 1 for **bold**, group 2 for "quotes")
    const term = match[1] || match[2];
    parts.push({ type: 'highlight', content: term });
    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIndex) });
  }

  return parts;
}

// ── Render rich text ──────────────────────────────────────────────────────────
function RichText({ text }) {
  const parts = parseHighlights(text);

  if (!Array.isArray(parts)) {
    return <span>{text}</span>;
  }

  return (
    <>
      {parts.map((part, i) =>
        part.type === 'highlight' ? (
          <strong key={i} className="highlight-term">
            {part.content}
          </strong>
        ) : (
          <span key={i}>{part.content}</span>
        )
      )}
    </>
  );
}

// ── Render AI response with paragraph breaks ───────────────────────────────────
function AIResponseText({ text }) {
  const paragraphs = text.split(/\n\n+/);

  return (
    <div className="space-y-2 text-sm leading-relaxed">
      {paragraphs.map((para, i) => {
        // Detect bullet list lines
        const lines = para.split('\n');
        const isBulletList = lines.every((l) => l.trim().match(/^[-•*]\s+/) || !l.trim());

        if (isBulletList && lines.some((l) => l.trim())) {
          return (
            <ul key={i} className="list-none space-y-1 pl-0">
              {lines
                .filter((l) => l.trim())
                .map((line, j) => (
                  <li key={j} className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-accent-400 mt-2" />
                    <RichText text={line.replace(/^[-•*]\s+/, '')} />
                  </li>
                ))}
            </ul>
          );
        }

        return (
          <p key={i} className="whitespace-pre-wrap">
            <RichText text={para} />
          </p>
        );
      })}
    </div>
  );
}

// ── Source citation badge ─────────────────────────────────────────────────────
function SourceBadge({ source }) {
  return (
    <span className="source-badge">
      📄 {source.pdf_name.length > 25
        ? source.pdf_name.slice(0, 22) + '…'
        : source.pdf_name}{' '}
      · Page {source.page_number}
    </span>
  );
}

// ── AI avatar ─────────────────────────────────────────────────────────────────
function AIAvatar() {
  return (
    <div className="flex-shrink-0 w-7 h-7 rounded-full
                    bg-gradient-to-br from-accent-500 to-accent-700
                    flex items-center justify-center shadow-sm">
      <span className="text-white text-xs font-bold leading-none">H</span>
    </div>
  );
}

// ── MessageBubble ─────────────────────────────────────────────────────────────
export default function MessageBubble({ message }) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in-up mb-4">
        <div className="bubble-user">
          <p className="text-sm leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  // AI message
  return (
    <div className="flex items-start gap-2.5 animate-fade-in-up mb-4">
      <AIAvatar />
      <div className="flex flex-col gap-2 flex-1">
        <div className="bubble-ai">
          <AIResponseText text={message.content} />
        </div>

        {/* Source citations */}
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 ml-1">
            {message.sources.map((source, i) => (
              <SourceBadge key={i} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
