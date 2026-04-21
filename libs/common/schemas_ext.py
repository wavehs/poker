"""
Extended Schemas — Phase 2 additions that extend the base schemas.

These are backward-compatible extensions (optional fields).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StageTimings(BaseModel):
    """Per-stage latency breakdown for a single frame."""

    capture_ms: float = Field(default=0.0, ge=0.0, description="Screen capture latency")
    vision_ms: float = Field(default=0.0, ge=0.0, description="Object detection latency")
    ocr_ms: float = Field(default=0.0, ge=0.0, description="OCR recognition latency")
    tracking_ms: float = Field(default=0.0, ge=0.0, description="Object tracking latency")
    state_ms: float = Field(default=0.0, ge=0.0, description="State reconstruction latency")
    solver_ms: float = Field(default=0.0, ge=0.0, description="Equity computation latency")
    policy_ms: float = Field(default=0.0, ge=0.0, description="Policy engine latency")
    explainer_ms: float = Field(default=0.0, ge=0.0, description="Explanation generation latency")
    total_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency")

    @property
    def perception_ms(self) -> float:
        """Total perception stack latency (capture + vision + ocr + tracking)."""
        return self.capture_ms + self.vision_ms + self.ocr_ms + self.tracking_ms

    @property
    def reasoning_ms(self) -> float:
        """Total reasoning stack latency (state + solver + policy + explainer)."""
        return self.state_ms + self.solver_ms + self.policy_ms + self.explainer_ms
