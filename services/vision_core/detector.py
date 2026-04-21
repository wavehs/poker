"""
Vision Core — Object detection for poker table elements.

Phase 2: Real YOLO11 inference via Ultralytics, with mock fallback.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

import numpy as np

from libs.common.schemas import BoundingBox, Detection, DetectionClass

logger = logging.getLogger(__name__)


# ─── Class mapping ──────────────────────────────────────────────────────────

# Maps YOLO class index → DetectionClass
# Must match data/dataset.py YOLO_CLASSES
_YOLO_INDEX_TO_CLASS: dict[int, DetectionClass] = {
    0: DetectionClass.CARD,
    1: DetectionClass.CHIP_STACK,
    2: DetectionClass.POT,
    3: DetectionClass.DEALER_BUTTON,
    4: DetectionClass.PLAYER_PANEL,
    5: DetectionClass.BET_AMOUNT,
    6: DetectionClass.ACTION_BUTTON,
    7: DetectionClass.BOARD_AREA,
}


class VisionDetector:
    """
    Detects poker table elements in a frame.

    Phase 2: Wraps Ultralytics YOLO11 for real inference.
    Falls back to mock detection when no model is loaded.
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

        # Metrics
        self._last_inference_ms: float = 0.0
        self._total_inferences: int = 0

        if not self._mock_mode:
            self._load_model()

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def _load_model(self) -> None:
        """Load YOLO model from weights."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._mock_mode = False
            logger.info("YOLO model loaded: %s (device=%s)", self.model_path, self.device)
        except ImportError:
            logger.warning("ultralytics not installed, falling back to mock mode")
            self._mock_mode = True
        except Exception as e:
            logger.warning("Failed to load YOLO model '%s': %s. Using mock mode.",
                          self.model_path, e)
            self._mock_mode = True

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR frame as numpy array (H, W, 3).
            frame_idx: Frame index for tracking.

        Returns:
            List of Detection objects.
        """
        t0 = time.perf_counter()

        if self._mock_mode:
            detections = self._mock_detect(frame, frame_idx)
        else:
            detections = self._real_detect(frame, frame_idx)

        self._last_inference_ms = (time.perf_counter() - t0) * 1000
        self._total_inferences += 1

        return detections

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_idx_start: int = 0,
    ) -> list[list[Detection]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of BGR frames.
            frame_idx_start: Starting frame index.

        Returns:
            List of detection lists, one per frame.
        """
        if self._mock_mode:
            return [
                self._mock_detect(f, frame_idx_start + i)
                for i, f in enumerate(frames)
            ]

        return self._real_detect_batch(frames, frame_idx_start)

    def _real_detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """Run real YOLO inference on a single frame."""
        if self._model is None:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
            return self._parse_results(results, frame_idx)
        except Exception as e:
            logger.warning("YOLO inference failed: %s", e)
            return []

    def _real_detect_batch(
        self,
        frames: list[np.ndarray],
        frame_idx_start: int,
    ) -> list[list[Detection]]:
        """Run real YOLO inference on a batch of frames."""
        if self._model is None:
            return [[] for _ in frames]

        try:
            results = self._model.predict(
                frames,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            all_detections: list[list[Detection]] = []
            for i, result in enumerate(results):
                detections = self._parse_results([result], frame_idx_start + i)
                all_detections.append(detections)

            return all_detections
        except Exception as e:
            logger.warning("YOLO batch inference failed: %s", e)
            return [[] for _ in frames]

    def _parse_results(self, results, frame_idx: int) -> list[Detection]:
        """
        Parse Ultralytics Results into Detection objects.

        Args:
            results: List of ultralytics.engine.results.Results objects.
            frame_idx: Frame index for the detection.

        Returns:
            List of Detection objects.
        """
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])

                # Map class
                det_class = _YOLO_INDEX_TO_CLASS.get(cls_id, DetectionClass.CARD)

                # Convert xyxy to xywh
                w = x2 - x1
                h = y2 - y1

                detection = Detection(
                    detection_class=det_class,
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        w=w,
                        h=h,
                        confidence=conf,
                    ),
                    label="",  # Label will be filled by OCR or post-processing
                    frame_idx=frame_idx,
                    timestamp_ms=time.time() * 1000,
                )
                detections.append(detection)

        return detections

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
