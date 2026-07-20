"""JWT auth helpers for the Manufacturing DT API — demo users only."""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.environ.get("JWT_SECRET", "dev-secret-manufacturing-dt-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

DEMO_USERS = {
    "admin":    {"password": "admin",    "role": "admin",    "plant_id": None},
    "operator": {"password": "operator", "role": "operator", "plant_id": "alpha"},
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def create_access_token(username: str, role: str, plant_id: Optional[str]) -> str:
    payload = {
        "sub": username,
        "role": role,
        "plant_id": plant_id,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(token: str = Security(oauth2_scheme)) -> Optional[dict]:
    if not token:
        return None
    return _decode(token)


def require_admin(token: str = Security(oauth2_scheme)) -> dict:
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = _decode(token)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
