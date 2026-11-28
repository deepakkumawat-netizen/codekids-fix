"""AI client: Groq primary, Gemini rotation fallback.

Request order
-------------
  1. GROQ_MODEL        openai/gpt-oss-120b     (best quality)
  2. GROQ_MODEL_FAST   openai/gpt-oss-20b      (lighter, separate headroom)
  3. GEMINI_MODEL      gemini-flash-lite-latest
  4. GEMINI_MODEL_ALT  gemini-3-flash-preview

Every id is env-driven so it can be changed from the Render dashboard
without a code deploy.

Do NOT reintroduce these — all decommissioned, they return 404:
  llama-3.3-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it,
  meta-llama/llama-4-scout-17b-16e-instruct, gemini-1.5-pro,
  gemini-2.0-flash, gemini-2.5-flash (blocked for newer API keys).

Gemini is called over plain REST (stdlib urllib) rather than the
google-genai SDK so there is no extra dependency to keep in sync.
"""
import json
import os
import urllib.error
import urllib.request

from openai import OpenAI

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL        = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MODEL_FAST   = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

# ── Gemini (rotation fallback) ────────────────────────────────────────────────
GEMINI_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_MODEL_ALT = os.getenv("GEMINI_MODEL_ALT", "gemini-3-flash-preview")
GEMINI_URL       = "https://generativelanguage.googleapis.com/v1beta/models"

GROQ_CHAIN   = [m for m in (GROQ_MODEL, GROQ_MODEL_FAST) if m]
GEMINI_CHAIN = [m for m in (GEMINI_MODEL, GEMINI_MODEL_ALT) if m]

_groq = OpenAI(api_key=GROQ_KEY, base_url=GROQ_URL) if GROQ_KEY else None

print(f"[ai_client] Groq={'ready' if _groq else 'NO KEY'} {GROQ_CHAIN} | "
      f"Gemini={'ready' if GEMINI_KEY else 'NO KEY'} {GEMINI_CHAIN}")


# openai/gpt-oss-* are REASONING models: they spend hidden tokens thinking
# before emitting any content, and that thinking is charged against
# max_tokens. The old llama-3.3-70b was not a reasoning model, so call sites
# written against it pass budgets as low as 200-400 — small enough that the
# reasoning alone exhausts them and `content` comes back as an empty string
# with finish_reason="length". We therefore ask for the shortest reasoning
# available and add headroom on top of whatever the caller asked for.
REASONING_RESERVE = 512


def _prep_kwargs(model: str, kwargs: dict) -> dict:
    """Add reasoning headroom for gpt-oss models. No-op for other models."""
    if not model.startswith("openai/gpt-oss"):
        return kwargs
    out = dict(kwargs)
    out.setdefault("reasoning_effort", "low")
    for key in ("max_tokens", "max_completion_tokens"):
        if out.get(key):
            out[key] = int(out[key]) + REASONING_RESERVE
    return out


def _empty_from_truncation(resp) -> bool:
    """True when the model used its whole budget thinking and said nothing."""
    try:
        choice = resp.choices[0]
        return (not (choice.message.content or "").strip()
                and choice.finish_reason == "length")
    except Exception:
        return False


def _should_rotate(err: Exception) -> bool:
    """Rate limit, dead model id, or a transient server blip is worth trying
    the next model. A bad key or a malformed request is not."""
    msg = str(err).lower()
    return any(s in msg for s in (
        "rate_limit", "rate limit", "429", "quota", "tokens per day", "tpd",
        "model_not_found", "does not exist", "no longer available",
        "not available to new users", "decommission", "resource_exhausted",
        "500", "502", "503", "504", "overloaded", "high demand",
        "timeout", "timed out", "unavailable",
    ))


# ── OpenAI-shaped response wrappers so callers never branch on provider ───────
class _Msg:
    __slots__ = ("content", "role")

    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role


class _Choice:
    __slots__ = ("message", "index", "finish_reason")

    def __init__(self, content):
        self.message = _Msg(content)
        self.index = 0
        self.finish_reason = "stop"


class _Response:
    def __init__(self, content, model):
        self.choices = [_Choice(content)]
        self.model = model


# ── Gemini REST call ──────────────────────────────────────────────────────────
def _gemini_generate(model, messages, timeout=60, **kwargs):
    """Call Gemini generateContent. Converts OpenAI-style messages to Gemini's
    contents + system_instruction shape and returns an OpenAI-shaped object."""
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    system_parts, contents = [], []
    for m in messages or []:
        role = m.get("role", "user")
        text = m.get("content", "")
        if not isinstance(text, str):
            continue
        if role == "system":
            system_parts.append(text)
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": text}],
        })
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    body = {"contents": contents, "generationConfig": {}}
    if system_parts:
        body["system_instruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if "temperature" in kwargs:
        body["generationConfig"]["temperature"] = kwargs["temperature"]
    max_out = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_out:
        body["generationConfig"]["maxOutputTokens"] = max_out

    req = urllib.request.Request(
        f"{GEMINI_URL}/{model}:generateContent",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini {model} HTTP {e.code}: "
                           f"{e.read().decode('utf-8', 'replace')[:300]}") from e

    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini {model} returned no candidates: {str(payload)[:200]}")
    text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    return _Response(text, model)


# ── Drop-in client (client.chat.completions.create) ───────────────────────────
class _Completions:
    def create(self, *, model=None, messages=None, timeout=None, **kwargs):
        msgs = messages or []
        last_err = None

        # Honour an explicit caller model first, then the standard chain.
        groq_chain = ([model] if model and model not in GROQ_CHAIN else []) + GROQ_CHAIN

        if _groq:
            for m in groq_chain:
                try:
                    resp = _groq.chat.completions.create(
                        model=m, messages=msgs, timeout=timeout or 30,
                        **_prep_kwargs(m, kwargs)
                    )
                    if _empty_from_truncation(resp):
                        # Reasoning ate the whole budget. Retry once, generous.
                        print(f"[ai_client] {m} spent its budget reasoning — retrying bigger")
                        big = dict(kwargs)
                        for key in ("max_tokens", "max_completion_tokens"):
                            if big.get(key):
                                big[key] = int(big[key]) * 2 + REASONING_RESERVE
                        resp = _groq.chat.completions.create(
                            model=m, messages=msgs, timeout=timeout or 30,
                            **_prep_kwargs(m, big)
                        )
                        if _empty_from_truncation(resp):
                            raise RuntimeError(
                                f"{m} returned no content within the token budget")
                    return resp
                except Exception as e:
                    last_err = e
                    if not _should_rotate(e):
                        raise
                    print(f"[ai_client] Groq {m} failed ({type(e).__name__}) — rotating")

        if GEMINI_KEY:
            # Gemini has no concept of these OpenAI-only kwargs.
            gkw = {k: v for k, v in kwargs.items()
                   if k in ("temperature", "max_tokens", "max_completion_tokens")}
            for m in GEMINI_CHAIN:
                try:
                    print(f"[ai_client] rotating to Gemini {m}")
                    return _gemini_generate(m, msgs, timeout=timeout or 60, **gkw)
                except Exception as e:
                    last_err = e
                    if not _should_rotate(e):
                        raise
                    print(f"[ai_client] Gemini {m} failed — rotating")

        if last_err:
            raise last_err
        raise RuntimeError("No AI provider — set GROQ_API_KEY or GEMINI_API_KEY")


class _Chat:
    completions = _Completions()


class FallbackClient:
    """Drop-in OpenAI client replacement.

    Groq (gpt-oss-120b -> gpt-oss-20b), then Gemini rotation
    (gemini-flash-lite-latest -> gemini-3-flash-preview).
    """
    chat = _Chat()


# ── Stateful ChatSession ──────────────────────────────────────────────────────
class ChatSession:
    """Multi-turn conversation with automatic memory."""

    def __init__(self, system_prompt="You are a helpful AI assistant."):
        self.history = [{"role": "system", "content": system_prompt}]

    def send(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        resp = _Completions().create(messages=self.history, timeout=30)
        reply = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def clear(self):
        self.history = self.history[:1]


# ── Function calling (Groq tool calling, OpenAI-style schemas) ────────────────
def call_with_tools(prompt, tools, system=""):
    """Run a prompt with tool calling.

    NOTE: `tools` must be OpenAI/Groq-style JSON schemas
    ([{"type": "function", "function": {...}}]), not the bare Python
    callables the old google-genai path accepted. Tool calling is Groq-only;
    the Gemini rotation does not carry tool definitions.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    if not _groq:
        raise RuntimeError("Tool calling needs GROQ_API_KEY")

    last_err = None
    for m in GROQ_CHAIN:
        try:
            resp = _groq.chat.completions.create(
                model=m, messages=msgs, tools=tools, timeout=30,
                **_prep_kwargs(m, {})
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            if not _should_rotate(e):
                raise
    raise last_err if last_err else RuntimeError("No Groq model available")


# ── Vision / OCR helper ───────────────────────────────────────────────────────
def describe_image(image_data_url, prompt="Describe this image."):
    """Send an image (data: URI or public URL) to Groq's vision model."""
    if not _groq:
        raise RuntimeError("No AI provider — set GROQ_API_KEY")

    resp = _groq.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
        timeout=60,
    )
    return resp.choices[0].message.content


# Singleton — import this everywhere
client = FallbackClient()
