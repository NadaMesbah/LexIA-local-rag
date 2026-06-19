"""
Authentication & authorization.

- Passwords hashed with bcrypt (the `bcrypt` library directly — passlib is
  unmaintained and breaks against modern bcrypt wheels).
- Stateless JWT (HS256) issued on login; sent as `Authorization: Bearer <token>`.
- FastAPI dependencies: `current_user` (any authenticated user) and `admin_user`.
- A default admin is seeded from env on startup (ADMIN_USERNAME / ADMIN_PASSWORD).

This is intentionally simple (no OAuth/SSO) — sufficient for a single-org local
deployment. SSO/LDAP would be the production follow-up.
"""

import os
import time
from typing import Optional

import jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import db

JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-change-me")
JWT_ALG = "HS256"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_HOURS", "12")) * 3600

_bearer = HTTPBearer(auto_error=False)


def _to_bytes(password: str) -> bytes:
    # bcrypt only considers the first 72 bytes and raises on longer input, so we
    # truncate consistently for both hashing and verification.
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(password), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user: dict) -> str:
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "role": user["role"],
        "exp": int(time.time()) + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def ensure_default_admin() -> None:
    """Seed an admin account and adopt any pre-auth data (idempotent)."""
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin")
    existing = db.get_user_by_username(username)
    if existing:
        admin_id = existing["id"]
    else:
        admin = db.create_user(username, hash_password(password), role="admin",
                               first_name="Admin", last_name="")
        admin_id = admin["id"]
        print(f"[AUTH] Seeded admin user '{username}'.")
        if os.getenv("ADMIN_PASSWORD") is None:
            print("[AUTH] WARNING: default admin password 'admin' in use — set ADMIN_PASSWORD.")
    # Adopt legacy (pre-auth) sessions/documents so they remain reachable.
    db.assign_legacy_owner(admin_id)


def _public(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "first_name": user.get("first_name", ""),
        "last_name": user.get("last_name", ""),
        "role": user["role"],
    }


def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> dict:
    if not creds or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user = db.get_user(payload.get("sub"))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return _public(user)


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return user
