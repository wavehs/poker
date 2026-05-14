"""
API Routes — /health, /analyze-frame, /analyze-sequence.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from apps.api.pipeline import get_pipeline
from libs.common.schemas import FrameAnalysis

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Configuration ───────────────────────────────────────────────────────────

# Hard cap on how much base64-decoded image data we will accept per request.
# Defaults to 10 MiB per frame, configurable via env. Protects against trivial
# memory-exhaustion DoS via the public /analyze-* endpoints.
MAX_IMAGE_BYTES = int(os.environ.get("POKER_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_SEQUENCE_LEN = int(os.environ.get("POKER_MAX_SEQUENCE_LEN", "16"))


def _decode_base64_image(image_base64: str, index: int = 0) -> np.ndarray:
    """Decode a base64 image string into a BGR numpy array.

    Raises HTTPException(400) on invalid input or oversized payloads.
    """
    if not image_base64:
        raise HTTPException(status_code=400, detail=f"Empty image at index {index}")

    # Budget cheaply via the encoded length (3/4 of base64 length == decoded bytes).
    max_b64_len = (MAX_IMAGE_BYTES * 4) // 3 + 8
    if len(image_base64) > max_b64_len:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image at index {index} exceeds the {MAX_IMAGE_BYTES}-byte limit "
                "(adjust POKER_MAX_IMAGE_BYTES if intentional)"
            ),
        )

    try:
        image_data = base64.b64decode(image_base64, validate=False)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 at index {index}: {e}") from e

    if len(image_data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Decoded image at index {index} exceeds {MAX_IMAGE_BYTES} bytes",
        )

    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image at index {index}")
    return frame


# ─── Health ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: float = Field(default_factory=lambda: time.time())


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


# ─── Analyze Frame ───────────────────────────────────────────────────────────


class AnalyzeFrameRequest(BaseModel):
    """Request body for base64-encoded frame analysis."""
    image_base64: str = Field(..., description="Base64-encoded PNG/JPG image")
    frame_idx: int = Field(default=0)


@router.post("/api/v1/analyze-frame", response_model=FrameAnalysis)
async def analyze_frame_base64(request: AnalyzeFrameRequest) -> FrameAnalysis:
    """Analyze a single frame from a base64-encoded image."""
    frame = _decode_base64_image(request.image_base64)
    pipeline = get_pipeline()
    # The pipeline is CPU-bound (YOLO/OCR/Monte Carlo) so run it off the event loop.
    return await asyncio.to_thread(pipeline.analyze_frame, frame, request.frame_idx)


@router.post("/api/v1/analyze-frame/upload", response_model=FrameAnalysis)
async def analyze_frame_upload(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency-injection idiom
    frame_idx: int = 0,
) -> FrameAnalysis:
    """Analyze a single frame from an uploaded file."""
    content = await file.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds {MAX_IMAGE_BYTES} bytes",
        )
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image")

    pipeline = get_pipeline()
    return await asyncio.to_thread(pipeline.analyze_frame, frame, frame_idx)


# ─── Analyze Synthetic (dev/test) ────────────────────────────────────────────


@router.post("/api/v1/analyze-synthetic", response_model=FrameAnalysis)
async def analyze_synthetic() -> FrameAnalysis:
    """Analyze a synthetic blank frame (for testing the full pipeline)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pipeline = get_pipeline()
    return await asyncio.to_thread(pipeline.analyze_frame, frame, 0)


# ─── Session History ─────────────────────────────────────────────────────────


@router.get("/api/v1/session/history")
async def get_session_history() -> dict:
    """Retrieve the history of completed hands for the current session."""
    pipeline = get_pipeline()
    session_file = getattr(pipeline, "session_file", None)
    if session_file is None or not session_file.exists():
        return {"history": []}

    history: list[dict] = []
    try:
        with open(session_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except OSError as e:
        logger.warning("Failed to read session history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read session history") from e
    except json.JSONDecodeError as e:
        logger.warning("Corrupt session history entry: %s", e)
        raise HTTPException(status_code=500, detail="Corrupt session history") from e

    return {"history": history}


# ─── Analyze Sequence ────────────────────────────────────────────────────────


class AnalyzeSequenceRequest(BaseModel):
    """Request body for sequence analysis."""
    images_base64: list[str] = Field(
        ..., description="List of base64-encoded images in order"
    )


class SequenceResponse(BaseModel):
    """Response for sequence analysis."""
    analyses: list[FrameAnalysis]
    total_frames: int
    total_processing_ms: float


@router.post("/api/v1/analyze-sequence", response_model=SequenceResponse)
async def analyze_sequence(request: AnalyzeSequenceRequest) -> SequenceResponse:
    """Analyze a sequence of frames for temporal analysis."""
    if len(request.images_base64) > MAX_SEQUENCE_LEN:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Sequence length {len(request.images_base64)} exceeds "
                f"the {MAX_SEQUENCE_LEN} frame limit"
            ),
        )

    t0 = time.perf_counter()
    frames = [_decode_base64_image(img, index=i) for i, img in enumerate(request.images_base64)]

    pipeline = get_pipeline()
    analyses = await asyncio.to_thread(pipeline.analyze_sequence, frames)
    total_time = (time.perf_counter() - t0) * 1000

    return SequenceResponse(
        analyses=analyses,
        total_frames=len(analyses),
        total_processing_ms=total_time,
    )
