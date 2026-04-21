"""
Poker Helper API — FastAPI application.

Endpoints:
  GET  /health              — Health check
  POST /api/v1/analyze-frame    — Analyze a single frame
  POST /api/v1/analyze-sequence — Analyze a frame sequence
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import router

app = FastAPI(
    title="Poker Helper API",
    description="Real-time external poker assistant — API backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow UI access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes
app.include_router(router)
