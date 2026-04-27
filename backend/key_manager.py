"""
key_manager.py — Multi-provider AI key rotation for HeyPDF 2.0

Provider priority order (hardcoded):
  1. Groq       — llama3-8b-8192              (fastest, best free tier)
  2. Gemini     — gemini-1.5-flash            (Google's free flash model)
  3. OpenRouter — mistralai/mistral-7b-instruct:free  (free-tier only)
  4. HuggingFace — mistralai/Mistral-7B-Instruct-v0.1 (HF Inference API)

Rotation logic:
  - Try current key of current provider
  - On quota/rate-limit (429, ResourceExhausted, RateLimitError):
      → Rotate to next key within same provider first
      → If all keys of that provider exhausted → move to next provider
  - If all providers exhausted → return friendly error message

All rotation is transparent to the caller (main.py). Only a final
exhaustion message is surfaced to the user.

Console logging format:
  [KeyManager] Groq key 1 failed. Trying Groq key 2...
  [KeyManager] All Groq keys exhausted. Switching to Gemini...
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Custom exception ─────────────────────────────────────────────────────────

class QuotaExceeded(Exception):
    """Raised by provider adapters when a key hits its quota or rate limit."""
    pass


# ── Key loading ──────────────────────────────────────────────────────────────

def _load_keys(env_prefix: str) -> list[str]:
    """
    Load all API keys for a provider from environment variables.

    Scans for {env_prefix}_1, {env_prefix}_2, ... until a key is missing.
    Example: env_prefix="GROQ_API_KEY" → reads GROQ_API_KEY_1, GROQ_API_KEY_2 ...
    """
    keys = []
    i = 1
    while True:
        key = os.getenv(f"{env_prefix}_{i}", "").strip()
        if not key:
            break
        keys.append(key)
        i += 1
    return keys


# ── Shared prompt builders ────────────────────────────────────────────────────

def _build_messages(prompt: str, context: str, chat_history: list[dict]) -> list[dict]:
    """
    Build an OpenAI-style messages list (works for Groq and OpenRouter).

    Includes a system prompt with PDF context, up to 8 prior conversation
    turns, then the current user question.
    """
    system_content = (
        "You are HeyPDF, an expert AI assistant that answers questions strictly "
        "based on the provided PDF context. "
        "Always be accurate and cite the source PDF and page when possible. "
        "If the answer is not in the provided context, clearly say so — do not hallucinate. "
        "Format your answers in clear, readable prose.\n\n"
        f"=== PDF Context ===\n{context}\n==================="
    )

    messages = [{"role": "system", "content": system_content}]

    # Inject up to the last 8 conversation turns (16 messages)
    for turn in chat_history[-8:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({"role": "user", "content": prompt})
    return messages


def _build_flat_prompt(prompt: str, context: str, chat_history: list[dict]) -> str:
    """
    Build a single-string prompt for providers that don't support chat format
    (e.g., HuggingFace text_generation endpoint).
    """
    history_str = ""
    for turn in chat_history[-6:]:
        history_str += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    return (
        "<s>[INST] You are HeyPDF, an AI that answers questions only from the given PDF context. "
        "Do not make up information outside the context.\n\n"
        f"=== PDF Context ===\n{context}\n===================\n\n"
        f"{history_str}"
        f"User: {prompt} [/INST]"
    )


# ── Provider adapter functions ────────────────────────────────────────────────
# Each takes (api_key, prompt, context, chat_history) → str
# Raises QuotaExceeded on rate limit / quota errors.
# Lets all other exceptions propagate (will be caught by KeyManager).

def _call_groq(api_key: str, prompt: str, context: str, chat_history: list) -> str:
    """Groq adapter using the official `groq` Python SDK."""
    from groq import Groq, RateLimitError

    client = Groq(api_key=api_key)
    messages = _build_messages(prompt, context, chat_history)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError as e:
        raise QuotaExceeded(f"Groq RateLimitError: {e}")
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ["rate", "quota", "429", "limit exceeded"]):
            raise QuotaExceeded(str(e))
        raise


def _call_gemini(api_key: str, prompt: str, context: str, chat_history: list) -> str:
    """Gemini adapter using the `google-generativeai` SDK."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build a single-turn prompt (Gemini SDK handles history differently)
    system = (
        "You are HeyPDF, an AI that answers questions strictly from PDF context. "
        "Do not hallucinate. Cite PDF name and page when possible.\n\n"
        f"=== PDF Context ===\n{context}\n===================\n\n"
    )
    history_str = ""
    for turn in chat_history[-6:]:
        history_str += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

    full_prompt = system + history_str + f"User: {prompt}\nAssistant:"

    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ["quota", "rate", "429", "resource_exhausted", "resourceexhausted"]):
            raise QuotaExceeded(str(e))
        raise


def _call_openrouter(api_key: str, prompt: str, context: str, chat_history: list) -> str:
    """OpenRouter adapter using OpenAI-compatible REST API (free models only)."""
    from openai import OpenAI, RateLimitError

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://heypdf.local",
            "X-Title": "HeyPDF 2.0",
        },
    )
    messages = _build_messages(prompt, context, chat_history)

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError as e:
        raise QuotaExceeded(f"OpenRouter RateLimitError: {e}")
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ["rate", "quota", "429", "limit"]):
            raise QuotaExceeded(str(e))
        raise


def _call_huggingface(api_key: str, prompt: str, context: str, chat_history: list) -> str:
    """HuggingFace adapter using the InferenceClient."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=api_key)
    flat_prompt = _build_flat_prompt(prompt, context, chat_history)

    try:
        # text_generation returns a string directly
        result = client.text_generation(
            prompt=flat_prompt,
            model="mistralai/Mistral-7B-Instruct-v0.1",
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        return result.strip()
    except Exception as e:
        err = str(e).lower()
        if any(kw in err for kw in ["quota", "rate", "429", "too many", "limit"]):
            raise QuotaExceeded(str(e))
        raise


# ── KeyManager class ──────────────────────────────────────────────────────────

# Maps provider name → (env_prefix, adapter_function)
_PROVIDER_CONFIGS = [
    ("groq",        "GROQ_API_KEY",        _call_groq),
    ("gemini",      "GEMINI_API_KEY",      _call_gemini),
    ("openrouter",  "OPENROUTER_API_KEY",  _call_openrouter),
    ("huggingface", "HUGGINGFACE_API_KEY", _call_huggingface),
]


class KeyManager:
    """
    Orchestrates multi-provider, multi-key rotation for AI calls.

    State is maintained across calls within a server session.
    Reset with .reset() if you want to restart from Groq on a new session.
    """

    def __init__(self):
        # Build provider list: [{name, keys, caller, current_key}, ...]
        self.providers = []
        for name, prefix, caller in _PROVIDER_CONFIGS:
            keys = _load_keys(prefix)
            self.providers.append({
                "name": name,
                "keys": keys,
                "caller": caller,
                "current_key": 0,
            })

        self.provider_index = 0  # Which provider we're currently on

        # Log loaded keys summary (don't print actual key values)
        for p in self.providers:
            count = len(p["keys"])
            status = f"{count} key(s) loaded" if count else "⚠ no keys configured"
            logger.info(f"[KeyManager] {p['name'].capitalize()}: {status}")

    def generate(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        """
        Generate an AI response, rotating through providers/keys as needed.

        Args:
            prompt: The user's question
            context: Relevant PDF text chunks as a single string
            chat_history: List of {question, answer} dicts (recent turns)

        Returns:
            AI response string, or an exhaustion message if all providers fail.
        """
        # Iterate through remaining providers starting at current_index
        while self.provider_index < len(self.providers):
            provider = self.providers[self.provider_index]

            # Skip providers with no keys configured
            if not provider["keys"]:
                logger.warning(
                    f"[KeyManager] {provider['name'].capitalize()}: no keys — skipping."
                )
                self.provider_index += 1
                continue

            key_idx = provider["current_key"]

            # All keys within this provider exhausted — move to next
            if key_idx >= len(provider["keys"]):
                next_name = (
                    self.providers[self.provider_index + 1]["name"].capitalize()
                    if self.provider_index + 1 < len(self.providers)
                    else "none"
                )
                logger.warning(
                    f"[KeyManager] All {provider['name'].capitalize()} keys exhausted. "
                    f"Switching to {next_name}..."
                )
                self.provider_index += 1
                continue

            api_key = provider["keys"][key_idx]
            key_display = key_idx + 1  # 1-indexed for human-readable logs

            try:
                logger.info(
                    f"[KeyManager] Trying {provider['name'].capitalize()} key {key_display}..."
                )
                result = provider["caller"](api_key, prompt, context, chat_history)
                # ✅ Success
                return result

            except QuotaExceeded:
                # Try next key within same provider
                next_key_idx = key_idx + 1
                if next_key_idx < len(provider["keys"]):
                    logger.warning(
                        f"[KeyManager] {provider['name'].capitalize()} key {key_display} failed. "
                        f"Trying {provider['name'].capitalize()} key {next_key_idx + 1}..."
                    )
                    provider["current_key"] = next_key_idx
                else:
                    # All keys for this provider done
                    provider["current_key"] = next_key_idx  # Mark as exhausted
                    logger.warning(
                        f"[KeyManager] All {provider['name'].capitalize()} keys exhausted. "
                        f"Switching to next provider..."
                    )
                    self.provider_index += 1

            except Exception as e:
                # Non-quota error (network, invalid response, etc.) — also rotate
                logger.error(
                    f"[KeyManager] {provider['name'].capitalize()} key {key_display} "
                    f"unexpected error: {type(e).__name__}: {e}"
                )
                next_key_idx = key_idx + 1
                if next_key_idx < len(provider["keys"]):
                    provider["current_key"] = next_key_idx
                else:
                    self.provider_index += 1

        # ❌ Everything exhausted
        logger.error("[KeyManager] All providers and keys exhausted.")
        return (
            "⚠️ All free API quotas are currently exceeded. "
            "Please try again in a few minutes or add more API keys to your .env file."
        )

    def reset(self) -> None:
        """Reset rotation back to the first provider and first key."""
        self.provider_index = 0
        for p in self.providers:
            p["current_key"] = 0
        logger.info("[KeyManager] Rotation reset to Groq key 1.")


# Singleton — imported and used by main.py
key_manager = KeyManager()
