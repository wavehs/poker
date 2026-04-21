"""
Vision Core — Object detection for poker table elements.

Phase 1: Returns synthetic/mock detections.
Phase 2+: YOLO11 model inference with real weights.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from libs.common.schemas import BoundingBox, Detection, DetectionClass


class VisionDetector:
    """
    Detects poker table elements in a frame.
    
    Phase 1: Mock detector that generates plausible synthetic detections.
    Phase 2: Will wrap Ultralytics YOLO11 model.
    """

    # Card labels for mock generation
    RANKS = "23456789TJQKA"
    SUITS = "hdcs"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_path: Path to YOLO .pt weights (None = mock mode).
            confidence_threshold: Minimum confidence to keep a detection.
            device: Inference device ('cpu', 'cuda:0', etc.).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._mock_mode = model_path is None

        if not self._mock_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model from weights (Phase 2)."""
        # TODO: from ultralytics import YOLO; self._model = YOLO(self.model_path)
        pass

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR frame as numpy array (H, W, 3).
            frame_idx: Frame index for tracking.
            
        Returns:
            List of Detection objects.
        """
        if self._mock_mode:
            return self._mock_detect(frame, frame_idx)
        return self._real_detect(frame, frame_idx)

    def _real_detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Run real YOLO inference (Phase 2 stub).
        """
        # TODO: Implement real inference
        # results = self._model(frame, device=self.device)
        # return self._parse_results(results, frame_idx)
        return []

    def _mock_detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Generate plausible mock detections for development.
        
        Generates:
        - 2 hero hole cards
        - 0-5 community cards
        - 1 pot region
        - 1 dealer button
        - 2-6 player panels
        """
        h, w = frame.shape[:2] if frame.ndim >= 2 else (1080, 1920)
        detections: list[Detection] = []
        used_cards: set[str] = set()

        # ── Hero hole cards (bottom center)
        for i in range(2):
            card = self._random_card(used_cards)
            detections.append(Detection(
                detection_class=DetectionClass.CARD,
                bbox=BoundingBox(
                    x=w * 0.42 + i * w * 0.06,
                    y=h * 0.78,
                    w=w * 0.05,
                    h=h * 0.09,
                    confidence=random.uniform(0.82, 0.98),
                ),
                label=card,
                frame_idx=frame_idx,
            ))

        # ── Community cards (center)
        num_community = random.choice([0, 3, 4, 5])
        for i in range(num_community):
            card = self._random_card(used_cards)
            detections.append(Detection(
                detection_class=DetectionClass.CARD,
                bbox=BoundingBox(
                    x=w * 0.3 + i * w * 0.07,
                    y=h * 0.40,
                    w=w * 0.05,
                    h=h * 0.09,
                    confidence=random.uniform(0.80, 0.97),
                ),
                label=card,
                frame_idx=frame_idx,
            ))

        # ── Pot
        pot_value = random.choice([50, 100, 200, 500, 1000, 2500])
        detections.append(Detection(
            detection_class=DetectionClass.POT,
            bbox=BoundingBox(
                x=w * 0.43,
                y=h * 0.33,
                w=w * 0.14,
                h=h * 0.05,
                confidence=random.uniform(0.85, 0.96),
            ),
            label=str(pot_value),
            frame_idx=frame_idx,
        ))

        # ── Dealer button
        dealer_seat = random.randint(0, 5)
        button_positions = [
            (0.50, 0.70), (0.75, 0.55), (0.75, 0.30),
            (0.50, 0.18), (0.25, 0.30), (0.25, 0.55),
        ]
        bx, by = button_positions[dealer_seat]
        detections.append(Detection(
            detection_class=DetectionClass.DEALER_BUTTON,
            bbox=BoundingBox(
                x=w * bx,
                y=h * by,
                w=w * 0.02,
                h=h * 0.03,
                confidence=random.uniform(0.88, 0.99),
            ),
            label="D",
            frame_idx=frame_idx,
        ))

        # ── Player panels
        num_players = random.randint(2, 6)
        panel_positions = [
            (0.45, 0.82), (0.72, 0.65), (0.72, 0.25),
            (0.45, 0.10), (0.18, 0.25), (0.18, 0.65),
        ]
        for i in range(num_players):
            px, py = panel_positions[i]
            stack = random.choice([500, 1000, 1500, 2000, 3000, 5000, 10000])
            detections.append(Detection(
                detection_class=DetectionClass.PLAYER_PANEL,
                bbox=BoundingBox(
                    x=w * px,
                    y=h * py,
                    w=w * 0.12,
                    h=h * 0.08,
                    confidence=random.uniform(0.80, 0.95),
                ),
                label=str(stack),
                frame_idx=frame_idx,
            ))

        return [d for d in detections if d.bbox.confidence >= self.confidence_threshold]

    def _random_card(self, used: set[str]) -> str:
        """Generate a random unique card code."""
        attempts = 0
        while attempts < 100:
            r = random.choice(self.RANKS)
            s = random.choice(self.SUITS)
            code = f"{r}{s}"
            if code not in used:
                used.add(code)
                return code
            attempts += 1
        return "??"
