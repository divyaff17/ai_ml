"""
middleware/auth.py — API key authentication middleware.

Provides both:
  * APIKeyMiddleware  — Starlette middleware class for global protection
  * verify_api_key    — FastAPI dependency for per-route protection
"""

from __future__ import annotations

import secrets
import uuid
from typing import Callable

from fastapi import Header, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from loguru import logger

import config

# ── Paths that skip authentication ────────────────────────────────────────────
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that lack a valid X-API-Key header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Inject a unique request-id for tracing
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Skip auth for public endpoints and OPTIONS (CORS preflight)
        if request.url.path in _PUBLIC_PATHS or request.method == "OPTIONS":
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        api_key = request.headers.get("X-API-Key", "")
        if not api_key or not secrets.compare_digest(api_key, config.API_SECRET_KEY):
            logger.warning(
                "Rejected request — invalid API key (path={})", request.url.path
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
                headers={"X-Request-ID": request_id},
            )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """FastAPI dependency that validates the X-API-Key header (kept for backward compat)."""
    if x_api_key != config.API_SECRET_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
