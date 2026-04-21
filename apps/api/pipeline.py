"""
Pipeline — End-to-end frame analysis orchestrator.

Coordinates all services in the correct order:
Capture → Vision → OCR → State Engine → Solver → Policy → Explainer
"""

from __future__ import annotations

import time

import numpy as np

from libs.common.schemas import (
    ConfidenceReport,
    FrameAnalysis,
    Recommendation,
    Action,
    ActionType,
)
from services.capture_agent.capture import CaptureAgent
from services.explainer.explainer import Explainer
from services.ocr_core.ocr import OCREngine
from services.policy_layer.policy import PolicyEngine
from services.state_engine.engine import StateEngine
from services.vision_core.detector import VisionDetector


class Pipeline:
    """
    Main analysis pipeline.
    
    Processes frames through the full detection → recommendation stack.
    """

    def __init__(
        self,
        capture: CaptureAgent | None = None,
        detector: VisionDetector | None = None,
        ocr: OCREngine | None = None,
        state_engine: StateEngine | None = None,
        policy: PolicyEngine | None = None,
        explainer: Explainer | None = None,
    ) -> None:
        self.capture = capture or CaptureAgent()
        self.detector = detector or VisionDetector()
        self.ocr = ocr or OCREngine()
        self.state_engine = state_engine or StateEngine()
        self.policy = policy or PolicyEngine()
        self.explainer = explainer or Explainer()

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

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000

        # ── Step 1: Vision detection
        detections = self.detector.detect(frame, frame_idx=frame_idx)

        # ── Step 2: OCR extraction
        ocr_results = self.ocr.extract(frame, detections)

        # ── Step 3: State reconstruction
        table_state, tracked_objects = self.state_engine.update(
            detections, ocr_results,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
        )

        # ── Step 4: Compute confidence scores
        vision_conf = (
            sum(d.bbox.confidence for d in detections) / len(detections)
            if detections else 0.0
        )
        ocr_conf = (
            sum(o.confidence for o in ocr_results) / len(ocr_results)
            if ocr_results else 0.0
        )
        state_conf = self.state_engine.get_state_confidence(table_state)

        # ── Step 5: Policy recommendation
        recommendation = self.policy.recommend(
            table_state,
            state_confidence=state_conf,
            vision_confidence=vision_conf,
            ocr_confidence=ocr_conf,
        )

        # ── Step 6: Explanation
        explanation = self.explainer.explain(recommendation, table_state)
        recommendation.explanation = explanation

        processing_time = (time.perf_counter() - t0) * 1000

        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            detections=detections,
            ocr_results=ocr_results,
            tracked_objects=tracked_objects,
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
