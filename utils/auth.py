"""Simple JWT authentication.

For learning only: one demo user, credentials from environment variables.
Defaults are admin / admin123. Never use these defaults in real projects.
"""
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

logger = logging.getLogger("ml_api")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-secret-key-change-me-in-production-0123456789")
ALGORITHM = "HS256"
TOKEN_MINUTES = 30
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

_SALT = os.urandom(16)


def _hash(password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT, 100_000)


_USER_HASH = _hash(API_PASSWORD)


def verify_user(username: str, password: str) -> bool:
    user_ok = hmac.compare_digest(username.encode(), API_USERNAME.encode())
    pass_ok = hmac.compare_digest(_hash(password), _USER_HASH)
    return user_ok and pass_ok


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Dependency: returns the username if the token is valid, else 401."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except jwt.PyJWTError:
        username = None
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
