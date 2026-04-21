"""
API Routes — /health, /analyze-frame, /analyze-sequence.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from apps.api.pipeline import Pipeline, get_pipeline
from libs.common.schemas import FrameAnalysis

router = APIRouter()


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
    """
    Analyze a single frame from base64-encoded image.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pipeline = get_pipeline()
    return pipeline.analyze_frame(frame, frame_idx=request.frame_idx)


@router.post("/api/v1/analyze-frame/upload", response_model=FrameAnalysis)
async def analyze_frame_upload(
    file: UploadFile = File(...),
    frame_idx: int = 0,
) -> FrameAnalysis:
    """
    Analyze a single frame from uploaded file.
    """
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image")

    pipeline = get_pipeline()
    return pipeline.analyze_frame(frame, frame_idx=frame_idx)


# ─── Analyze Synthetic (dev/test) ────────────────────────────────────────────


@router.post("/api/v1/analyze-synthetic", response_model=FrameAnalysis)
async def analyze_synthetic() -> FrameAnalysis:
    """
    Analyze a synthetic blank frame (for testing the full pipeline).
    """
    # Create a 1920x1080 blank frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pipeline = get_pipeline()
    return pipeline.analyze_frame(frame, frame_idx=0)


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
    """
    Analyze a sequence of frames for temporal analysis.
    """
    t0 = time.perf_counter()
    frames: list[np.ndarray] = []

    for i, img_b64 in enumerate(request.images_base64):
        try:
            image_data = base64.b64decode(img_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot decode image {i}")
            frames.append(frame)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image at index {i}: {e}"
            )

    pipeline = get_pipeline()
    analyses = pipeline.analyze_sequence(frames)
    total_time = (time.perf_counter() - t0) * 1000

    return SequenceResponse(
        analyses=analyses,
        total_frames=len(frames),
        total_processing_ms=total_time,
    )
