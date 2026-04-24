"""Supabase persistence helpers for user accounts and saved analyses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from advocate.auth import hash_password, normalize_username, validate_signup, verify_password
from advocate.settings import get_setting, supabase_is_configured


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass
class StoredUser:
    id: str
    username: str
    created_at: str | None = None
    last_login_at: str | None = None


class SupabaseStore:
    def __init__(self) -> None:
        if not supabase_is_configured():
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.",
            )

        from supabase import Client, create_client

        self.url = str(get_setting("SUPABASE_URL"))
        self.key = str(get_setting("SUPABASE_SERVICE_ROLE_KEY"))
        self.client: Client = create_client(self.url, self.key)

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self.client.table("app_users").select("id", count="exact").limit(1).execute()
            return True, "Connected"
        except Exception as exc:
            return False, str(exc)

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        normalized = normalize_username(username)
        response = (
            self.client.table("app_users")
            .select("*")
            .eq("username", normalized)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def create_user(self, username: str, password: str) -> StoredUser:
        validation = validate_signup(username, password)
        if not validation.ok:
            raise ValueError(validation.message)

        normalized = normalize_username(username)
        if self.get_user_by_username(normalized):
            raise ValueError("That username is already taken.")

        payload = {
            "username": normalized,
            "password_hash": hash_password(password),
        }
        response = self.client.table("app_users").insert(payload).execute()
        if not response.data:
            raise RuntimeError("Supabase did not return the created user record.")
        row = response.data[0]
        return StoredUser(
            id=row["id"],
            username=row["username"],
            created_at=row.get("created_at"),
            last_login_at=row.get("last_login_at"),
        )

    def authenticate_user(self, username: str, password: str) -> StoredUser:
        user = self.get_user_by_username(username)
        if not user or not verify_password(password, user.get("password_hash", "")):
            raise ValueError("Invalid username or password.")

        now = datetime.now(timezone.utc).isoformat()
        self.client.table("app_users").update({"last_login_at": now}).eq("id", user["id"]).execute()
        return StoredUser(
            id=user["id"],
            username=user["username"],
            created_at=user.get("created_at"),
            last_login_at=now,
        )

    def save_analysis(
        self,
        *,
        user_id: str,
        title: str,
        model: str,
        run_mode: str,
        status: str,
        case_brief: str,
        summary: dict[str, Any],
        result: dict[str, Any],
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "title": title.strip()[:160],
            "model": model,
            "run_mode": run_mode,
            "status": status,
            "case_brief": case_brief,
            "summary": _json_ready(summary),
            "result": _json_ready(result),
            "errors": _json_ready(errors or []),
        }
        response = self.client.table("analysis_runs").insert(payload).execute()
        if not response.data:
            raise RuntimeError("Supabase did not return the saved analysis.")
        return response.data[0]

    def list_analyses(self, user_id: str, limit: int = 25) -> list[dict[str, Any]]:
        response = (
            self.client.table("analysis_runs")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    def get_analysis(self, run_id: str, user_id: str) -> dict[str, Any] | None:
        response = (
            self.client.table("analysis_runs")
            .select("*")
            .eq("id", run_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
