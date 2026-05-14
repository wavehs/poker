"""
Poker Helper API — FastAPI application.

Endpoints:
  GET  /health              — Health check
  POST /api/v1/analyze-frame    — Analyze a single frame
  POST /api/v1/analyze-sequence — Analyze a frame sequence
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import router


def _parse_origins(value: str) -> list[str]:
    return [o.strip() for o in value.split(",") if o.strip()]


# Comma-separated list of allowed origins. Defaults to the local dev UI.
# A literal "*" disables credentials (browsers reject "*" + credentials).
_origins_env = os.environ.get("POKER_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS = _parse_origins(_origins_env)
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

app = FastAPI(
    title="Poker Helper API",
    description="Real-time external poker assistant — API backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)
