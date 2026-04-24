"""Application settings helpers for local and Streamlit deployments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_CHROMA_CANDIDATES = (
    Path("./advocate/data/chroma_db"),
    Path("./advocate/advocate/data/chroma_db"),
)


def _streamlit_session_value(name: str) -> Any | None:
    try:
        import streamlit as st

        if name in st.session_state:
            value = st.session_state[name]
            if value not in (None, ""):
                return value
    except Exception:
        return None
    return None


def _streamlit_secret(name: str) -> Any | None:
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        return None
    return None


def get_setting(name: str, default: Any | None = None) -> Any | None:
    """Read a setting from session state, env vars, then Streamlit secrets."""
    session_value = _streamlit_session_value(name)
    if session_value not in (None, ""):
        return session_value

    value = os.getenv(name)
    if value not in (None, ""):
        return value

    secret_value = _streamlit_secret(name)
    if secret_value not in (None, ""):
        return secret_value

    return default


def get_bool_setting(name: str, default: bool = False) -> bool:
    value = get_setting(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_default_model() -> str:
    return str(get_setting("ADVOCATE_MODEL", "gpt-4o-mini"))


def get_chroma_persist_path() -> str:
    configured = get_setting("CHROMA_PERSIST_PATH")
    if configured:
        return str(configured)

    for candidate in _CHROMA_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    return str(_CHROMA_CANDIDATES[0])


def supabase_is_configured() -> bool:
    return bool(get_setting("SUPABASE_URL") and get_setting("SUPABASE_SERVICE_ROLE_KEY"))


def available_provider_env_key(model_id: str) -> str | None:
    model = model_id.strip().lower()
    if model.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if model.startswith("gemini"):
        return "GOOGLE_API_KEY"
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return "OPENAI_API_KEY"
    return None


def provider_is_configured(model_id: str) -> bool:
    env_key = available_provider_env_key(model_id)
    if not env_key:
        return False
    return bool(get_setting(env_key))
