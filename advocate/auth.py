"""Username/password authentication helpers for Streamlit + Supabase."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from dataclasses import dataclass

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")
PBKDF2_ITERATIONS = 390_000


@dataclass(frozen=True)
class AuthValidation:
    ok: bool
    message: str = ""


def normalize_username(username: str) -> str:
    return username.strip().lower()


def validate_signup(username: str, password: str) -> AuthValidation:
    normalized = normalize_username(username)
    if not USERNAME_RE.fullmatch(normalized):
        return AuthValidation(
            ok=False,
            message="Username must be 3-32 characters and use only letters, numbers, or underscores.",
        )
    if len(password) < 8:
        return AuthValidation(ok=False, message="Password must be at least 8 characters long.")
    if password.strip() != password:
        return AuthValidation(ok=False, message="Password cannot start or end with spaces.")
    return AuthValidation(ok=True)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False
