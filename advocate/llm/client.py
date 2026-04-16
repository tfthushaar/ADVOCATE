"""
client.py  —  Unified LLM abstraction layer
Routes chat completion requests to the correct provider based on model prefix.

Supported models:
  OpenAI   : gpt-4o, gpt-4o-mini, gpt-4-turbo, o1-*, o3-*
  Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
  Google   : gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
             (uses the native google-generativeai SDK)

Environment variables required per provider:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# ─── Provider routing ─────────────────────────────────────────────────────────

OPENAI_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
    "o1", "o1-mini", "o3", "o3-mini",
}

ANTHROPIC_PREFIX = "claude"
GEMINI_PREFIX = "gemini"


def _is_openai(model: str) -> bool:
    return model in OPENAI_MODELS or model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")


def _is_anthropic(model: str) -> bool:
    return model.startswith(ANTHROPIC_PREFIX)


def _is_gemini(model: str) -> bool:
    return model.startswith(GEMINI_PREFIX)


# ─── Public API ───────────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
) -> tuple[str, float]:
    """
    Unified chat completion interface.

    Args:
        messages:   List of {"role": "system"|"user"|"assistant", "content": str}.
        model:      Model identifier string (determines provider automatically).
        max_tokens: Max tokens in the response.

    Returns:
        (response_text, latency_seconds)

    Raises:
        ValueError: If the model prefix is unrecognised or the API key is missing.
    """
    t0 = time.perf_counter()

    if _is_anthropic(model):
        text = _anthropic_completion(messages, model, max_tokens)
    elif _is_gemini(model):
        text = _gemini_completion(messages, model, max_tokens)
    elif _is_openai(model):
        text = _openai_completion(messages, model, max_tokens)
    else:
        raise ValueError(
            f"Unrecognised model '{model}'. "
            "Prefix must be 'gpt-', 'claude-', or 'gemini-'."
        )

    latency = round(time.perf_counter() - t0, 2)
    return text, latency


def _openai_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def _anthropic_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    # Split system message from user/assistant turns
    system = ""
    turns = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=turns,
    )
    return response.content[0].text.strip()


def _gemini_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    """
    Uses the native google-generativeai SDK.
    Converts the standard {role, content} message list into Gemini's format:
      - "system" role → system_instruction on the model
      - "user" / "assistant" turns → chat history
    """
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)

    # Extract system instruction and build history + final user message
    system_instruction = None
    history = []
    last_user_msg = ""

    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            last_user_msg = content
            # Add previous user turns to history if there are assistant responses after them
        elif role == "assistant":
            # Pair the preceding user message into history as a completed turn
            if last_user_msg:
                history.append({"role": "user", "parts": [last_user_msg]})
                last_user_msg = ""
            history.append({"role": "model", "parts": [content]})

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config=GenerationConfig(max_output_tokens=max_tokens),
    )

    if history:
        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(last_user_msg)
    else:
        response = gemini_model.generate_content(last_user_msg)

    return response.text.strip()


# ─── Model catalogue (for UI) ─────────────────────────────────────────────────

AVAILABLE_MODELS: dict[str, dict] = {
    # OpenAI
    "gpt-4o": {
        "provider": "OpenAI",
        "display": "GPT-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI flagship multimodal model",
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "display": "GPT-4o Mini",
        "env_key": "OPENAI_API_KEY",
        "description": "Fast, cost-efficient OpenAI model",
    },
    "gpt-4-turbo": {
        "provider": "OpenAI",
        "display": "GPT-4 Turbo",
        "env_key": "OPENAI_API_KEY",
        "description": "High-intelligence OpenAI model",
    },
    # Anthropic
    "claude-sonnet-4-6": {
        "provider": "Anthropic",
        "display": "Claude Sonnet 4.6",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic balanced performance model",
    },
    "claude-opus-4-6": {
        "provider": "Anthropic",
        "display": "Claude Opus 4.6",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic most capable model",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "Anthropic",
        "display": "Claude Haiku 4.5",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic fastest, most compact model",
    },
    # Google Gemini
    "gemini-2.0-flash": {
        "provider": "Google",
        "display": "Gemini 2.0 Flash",
        "env_key": "GOOGLE_API_KEY",
        "description": "Google fast multimodal model",
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "display": "Gemini 1.5 Pro",
        "env_key": "GOOGLE_API_KEY",
        "description": "Google high-capability long-context model",
    },
    "gemini-1.5-flash": {
        "provider": "Google",
        "display": "Gemini 1.5 Flash",
        "env_key": "GOOGLE_API_KEY",
        "description": "Google fast, efficient model",
    },
}


def models_for_provider(provider: str) -> list[str]:
    return [k for k, v in AVAILABLE_MODELS.items() if v["provider"] == provider]


def is_model_available(model_id: str) -> bool:
    """Returns True if the required API key env var is set for this model."""
    info = AVAILABLE_MODELS.get(model_id)
    if not info:
        return False
    return bool(os.getenv(info["env_key"]))
