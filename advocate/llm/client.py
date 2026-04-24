"""Unified LLM abstraction layer for OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

import functools
import random
import time

from dotenv import load_dotenv

from advocate.settings import available_provider_env_key, get_setting

load_dotenv()

OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
}

ANTHROPIC_PREFIX = "claude"
GEMINI_PREFIX = "gemini"


def with_retry_and_backoff(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 10
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                error_text = str(exc).lower()
                is_retryable = any(
                    token in error_text
                    for token in (
                        "429",
                        "rate limit",
                        "resource exhausted",
                        "quota",
                        "too many requests",
                        "502",
                        "503",
                    )
                )
                if not is_retryable or attempt == max_retries - 1:
                    raise

                model_arg = kwargs.get("model")
                if not model_arg and len(args) > 1:
                    model_arg = args[1]
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"[Rate Limit] Retrying {model_arg} in {sleep_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})...",
                )
                time.sleep(sleep_time)

    return wrapper


def _is_openai(model: str) -> bool:
    return model in OPENAI_MODELS or model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")


def _is_anthropic(model: str) -> bool:
    return model.startswith(ANTHROPIC_PREFIX)


def _is_gemini(model: str) -> bool:
    return model.startswith(GEMINI_PREFIX)


def provider_env_key_for_model(model_id: str) -> str | None:
    info = AVAILABLE_MODELS.get(model_id)
    if info:
        return info["env_key"]
    return available_provider_env_key(model_id)


def chat_completion(messages: list[dict], model: str, max_tokens: int = 4096) -> tuple[str, float]:
    """Run a chat completion and return response text plus latency."""
    started_at = time.perf_counter()

    if _is_anthropic(model):
        text = _anthropic_completion(messages, model, max_tokens)
    elif _is_gemini(model):
        text = _gemini_completion(messages, model, max_tokens)
    elif _is_openai(model):
        text = _openai_completion(messages, model, max_tokens)
    else:
        raise ValueError(
            f"Unrecognised model '{model}'. Prefix must be 'gpt-', 'claude-', or 'gemini-'.",
        )

    return text, round(time.perf_counter() - started_at, 2)


@with_retry_and_backoff
def _openai_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    from openai import OpenAI

    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


@with_retry_and_backoff
def _anthropic_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    import anthropic

    api_key = get_setting("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    system = ""
    turns: list[dict] = []

    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        else:
            turns.append(message)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=turns,
    )
    return response.content[0].text.strip()


@with_retry_and_backoff
def _gemini_completion(messages: list[dict], model: str, max_tokens: int) -> str:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    api_key = get_setting("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)

    system_instruction = None
    history = []
    last_user_message = ""

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            last_user_message = content
        elif role == "assistant":
            if last_user_message:
                history.append({"role": "user", "parts": [last_user_message]})
                last_user_message = ""
            history.append({"role": "model", "parts": [content]})

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config=GenerationConfig(max_output_tokens=max_tokens),
    )

    if history:
        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(last_user_message)
    else:
        response = gemini_model.generate_content(last_user_message)

    return response.text.strip()


AVAILABLE_MODELS: dict[str, dict] = {
    "gpt-4o": {
        "provider": "OpenAI",
        "display": "GPT-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "High-capability OpenAI model",
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
        "description": "OpenAI large-context model",
    },
    "claude-sonnet-4-6": {
        "provider": "Anthropic",
        "display": "Claude Sonnet 4.6",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Balanced Anthropic model",
    },
    "claude-opus-4-6": {
        "provider": "Anthropic",
        "display": "Claude Opus 4.6",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Highest-capability Anthropic model",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "Anthropic",
        "display": "Claude Haiku 4.5",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Fast Anthropic model",
    },
    "gemini-2.0-flash": {
        "provider": "Google",
        "display": "Gemini 2.0 Flash",
        "env_key": "GOOGLE_API_KEY",
        "description": "Fast Gemini model",
    },
    "gemini-2.0-flash-lite": {
        "provider": "Google",
        "display": "Gemini 2.0 Flash Lite",
        "env_key": "GOOGLE_API_KEY",
        "description": "Lightweight Gemini model",
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "display": "Gemini 1.5 Pro",
        "env_key": "GOOGLE_API_KEY",
        "description": "Long-context Gemini model",
    },
    "gemini-1.5-flash-latest": {
        "provider": "Google",
        "display": "Gemini 1.5 Flash",
        "env_key": "GOOGLE_API_KEY",
        "description": "Efficient Gemini model",
    },
}


def models_for_provider(provider: str) -> list[str]:
    return [model_id for model_id, info in AVAILABLE_MODELS.items() if info["provider"] == provider]


def is_model_available(model_id: str) -> bool:
    env_key = provider_env_key_for_model(model_id)
    if not env_key:
        return False
    return bool(get_setting(env_key))


def list_gemini_models() -> list[str]:
    """Return Gemini models available to the configured API key."""
    try:
        import google.generativeai as genai

        api_key = get_setting("GOOGLE_API_KEY")
        if not api_key:
            return []

        genai.configure(api_key=api_key)
        return [
            model.name.replace("models/", "")
            for model in genai.list_models()
            if "generateContent" in model.supported_generation_methods
        ]
    except Exception:
        return []
