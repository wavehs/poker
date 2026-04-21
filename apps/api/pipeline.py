"""
Pipeline — End-to-end frame analysis orchestrator.

Coordinates all services in the correct order:
Capture → Vision → Tracking → OCR → State Engine → Solver → Policy → Explainer

Phase 2: Adds ObjectTracker integration, per-stage profiling, and StageTimings.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from libs.common.profiler import PipelineProfiler
from libs.common.schemas import (
    ConfidenceReport,
    FrameAnalysis,
    Recommendation,
    Action,
    ActionType,
)
from libs.common.schemas_ext import StageTimings
from services.capture_agent.capture import CaptureAgent
from services.explainer.explainer import Explainer
from services.ocr_core.ocr import OCREngine
from services.policy_layer.policy import PolicyEngine
from services.state_engine.engine import StateEngine
from services.vision_core.detector import VisionDetector
from services.vision_core.tracker import ObjectTracker


class Pipeline:
    """
    Main analysis pipeline.

    Processes frames through the full detection → recommendation stack.
    Phase 2: added ObjectTracker and per-stage profiling.
    """

    def __init__(
        self,
        capture: CaptureAgent | None = None,
        detector: VisionDetector | None = None,
        ocr: OCREngine | None = None,
        state_engine: StateEngine | None = None,
        policy: PolicyEngine | None = None,
        explainer: Explainer | None = None,
        tracker: ObjectTracker | None = None,
        enable_profiling: bool = False,
    ) -> None:
        self.capture = capture or CaptureAgent()
        self.detector = detector or VisionDetector()
        self.ocr = ocr or OCREngine()
        self.state_engine = state_engine or StateEngine()
        self.policy = policy or PolicyEngine()
        self.explainer = explainer or Explainer()
        self.tracker = tracker  # None = no tracking
        self.profiler = PipelineProfiler() if enable_profiling else None

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp_ms: float | None = None,
    ) -> FrameAnalysis:
        """
        Run full analysis pipeline on a single frame.

        Args:
            frame: BGR frame as numpy array.
            frame_idx: Frame index.
            timestamp_ms: Frame timestamp (auto-generated if None).

        Returns:
            Complete FrameAnalysis with all detections, state, and recommendation.
        """
        t0 = time.perf_counter()
        timings = StageTimings()

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000

        # ── Step 1: Vision detection
        t_stage = time.perf_counter()
        detections = self.detector.detect(frame, frame_idx=frame_idx)
        timings.vision_ms = (time.perf_counter() - t_stage) * 1000

        # ── Step 2: Object tracking (optional)
        tracked_objects = []
        if self.tracker is not None:
            t_stage = time.perf_counter()
            tracked_objects = self.tracker.update(detections, frame_idx=frame_idx)
            timings.tracking_ms = (time.perf_counter() - t_stage) * 1000

        # ── Step 3: OCR extraction
        t_stage = time.perf_counter()
        ocr_results = self.ocr.extract(frame, detections)
        timings.ocr_ms = (time.perf_counter() - t_stage) * 1000

        # ── Step 4: State reconstruction
        t_stage = time.perf_counter()
        table_state, state_tracked = self.state_engine.update(
            detections, ocr_results,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            tracked_objects=tracked_objects if self.tracker else None,
        )
        timings.state_ms = (time.perf_counter() - t_stage) * 1000

        # Use tracker-provided objects if available, else state engine's
        final_tracked = tracked_objects if self.tracker else state_tracked

        # ── Step 5: Compute confidence scores
        vision_conf = (
            sum(d.bbox.confidence for d in detections) / len(detections)
            if detections else 0.0
        )
        ocr_conf = (
            sum(o.confidence for o in ocr_results) / len(ocr_results)
            if ocr_results else 0.0
        )
        state_conf = self.state_engine.get_state_confidence(table_state)

        # ── Step 6: Policy recommendation
        t_stage = time.perf_counter()
        recommendation = self.policy.recommend(
            table_state,
            state_confidence=state_conf,
            vision_confidence=vision_conf,
            ocr_confidence=ocr_conf,
        )
        timings.policy_ms = (time.perf_counter() - t_stage) * 1000

        # ── Step 7: Explanation
        t_stage = time.perf_counter()
        explanation = self.explainer.explain(recommendation, table_state)
        recommendation.explanation = explanation
        timings.explainer_ms = (time.perf_counter() - t_stage) * 1000

        processing_time = (time.perf_counter() - t0) * 1000
        timings.total_ms = processing_time

        # Record profiling data
        if self.profiler:
            self.profiler.record("vision", timings.vision_ms)
            self.profiler.record("tracking", timings.tracking_ms)
            self.profiler.record("ocr", timings.ocr_ms)
            self.profiler.record("state", timings.state_ms)
            self.profiler.record("policy", timings.policy_ms)
            self.profiler.record("explainer", timings.explainer_ms)
            self.profiler.record("total", timings.total_ms)
            self.profiler.end_frame()

        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            detections=detections,
            ocr_results=ocr_results,
            tracked_objects=final_tracked,
            table_state=table_state,
            recommendation=recommendation,
            processing_time_ms=processing_time,
        )

    def analyze_sequence(
        self,
        frames: list[np.ndarray],
    ) -> list[FrameAnalysis]:
        """
        Analyze a sequence of frames.

        Each frame is processed in order, building temporal context.

        Args:
            frames: List of BGR frames.

        Returns:
            List of FrameAnalysis results.
        """
        results: list[FrameAnalysis] = []
        for i, frame in enumerate(frames):
            analysis = self.analyze_frame(frame, frame_idx=i)
            results.append(analysis)
        return results


# ─── Module-level singleton for convenience ──────────────────────────────────

_default_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Get or create the default pipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline
