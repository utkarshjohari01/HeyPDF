/**
 * SuggestedQuestions.jsx — Clickable question chip buttons
 *
 * Shown below the chat input after a PDF is uploaded.
 * Clicking a chip auto-fills and submits the question.
 *
 * Props:
 *   questions: string[] — list of suggested questions
 *   onSelect(question: string) — called when a chip is clicked
 *   disabled: bool — disable chips while AI is responding
 */

export default function SuggestedQuestions({ questions, onSelect, disabled }) {
  if (!questions || questions.length === 0) return null;

  return (
    <div
      id="suggested-questions"
      className="flex flex-wrap gap-2 px-4 pb-1"
      aria-label="Suggested questions"
    >
      <span className="w-full text-xs text-gray-400 dark:text-gray-500 font-medium mb-0.5">
        💡 Try asking:
      </span>
      {questions.map((q, i) => (
        <button
          key={i}
          onClick={() => !disabled && onSelect(q)}
          disabled={disabled}
          className={`chip text-left ${
            disabled ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''
          }`}
          title={q}
          aria-label={`Ask: ${q}`}
        >
          {q.length > 60 ? q.slice(0, 57) + '…' : q}
        </button>
      ))}
    </div>
  );
}
