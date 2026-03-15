"""
Password hashing and JWT utilities.

Used by the auth router and auth dependency.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import jwt

# passlib's bcrypt backend detection breaks with bcrypt 5.x (strict 72-byte
# limit).  Use the bcrypt library directly — it works fine for normal ops.


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def create_access_token(
    subject: str,
    extra: dict[str, Any],
    secret: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "exp": expire, **extra}
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str) -> dict[str, Any]:
    """
    Decode and verify a JWT.  Raises JWTError on any failure.
    """
    return jwt.decode(token, secret, algorithms=[algorithm])
