"""Login route (returns a JWT token) and a small protected route."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from utils import auth
from utils.schemas import MeResponse, TokenResponse

logger = logging.getLogger("ml_api")
router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Log in with username and password to get a token."""
    if not auth.verify_user(form.username, form.password):
        logger.warning("Failed login for user '%s'", form.username)
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info("User '%s' logged in", form.username)
    return {"access_token": auth.create_access_token(form.username), "token_type": "bearer"}


@router.get("/me", response_model=MeResponse)
def me(username: str = Depends(auth.get_current_user)):
    """Shows who you are logged in as (needs a token)."""
    return {"username": username}
