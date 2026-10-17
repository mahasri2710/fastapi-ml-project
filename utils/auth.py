"""JWT authentication with roles (admin and user).

For learning only: two demo users, credentials from environment variables.
Defaults: admin / admin123 (role admin) and student / student123 (role user).
Never use these defaults in real projects.
"""
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

logger = logging.getLogger("ml_api")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-secret-key-change-me-in-production-0123456789")
ALGORITHM = "HS256"
TOKEN_MINUTES = 30

API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")
STUDENT_USERNAME = os.getenv("STUDENT_USERNAME", "student")
STUDENT_PASSWORD = os.getenv("STUDENT_PASSWORD", "student123")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

_SALT = os.urandom(16)


def _hash(password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT, 100_000)


# role "admin": can train, upload datasets and use everything else
# role "user" : can predict, evaluate and see model info
_USERS = {
    API_USERNAME: {"hash": _hash(API_PASSWORD), "role": "admin"},
    STUDENT_USERNAME: {"hash": _hash(STUDENT_PASSWORD), "role": "user"},
}
_DUMMY_HASH = _hash("dummy-password")


def authenticate(username: str, password: str) -> Optional[str]:
    """Return the user's role if username and password are right, else None."""
    user = _USERS.get(username)
    expected = user["hash"] if user else _DUMMY_HASH
    password_ok = hmac.compare_digest(_hash(password), expected)
    return user["role"] if (user and password_ok) else None


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency: returns {"username", "role"} for a valid token, else 401."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except jwt.PyJWTError:
        username = None
    user = _USERS.get(username) if username else None
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"username": username, "role": user["role"]}


def require_role(*allowed_roles: str):
    """Dependency factory: only users with one of these roles get in (else 403)."""

    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            logger.warning("User '%s' (role %s) was denied", user["username"], user["role"])
            raise HTTPException(
                status_code=403,
                detail="You do not have permission for this action",
            )
        return user

    return checker
