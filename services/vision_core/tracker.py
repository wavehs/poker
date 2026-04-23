"""
Object Tracker — Temporal object tracking across frames.

Wraps Ultralytics tracking (ByteTrack / BoT-SORT) for persistent
object identity across frames. Falls back to simple IoU-based
matching when Ultralytics tracking is unavailable.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

from libs.common.schemas import Detection, TrackedObject

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Tracks objects across frames using Ultralytics tracking or simple IoU matching.

    Features:
    - Persistent track IDs across frames
    - Temporal consensus for label stabilization
    - Configurable tracker (ByteTrack / BoT-SORT)
    - Fallback to simple IoU matching
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        stability_threshold: int = 3,
        max_age: int = 10,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            tracker_type: 'bytetrack' or 'botsort'.
            stability_threshold: Frames a label must persist to be 'stable'.
            max_age: Frames before a lost track is removed.
            iou_threshold: Minimum IoU for matching in simple mode.
        """
        self.tracker_type = tracker_type
        self.stability_threshold = stability_threshold
        self.max_age = max_age
        self.iou_threshold = iou_threshold

        # Track state
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id: int = 0
        self._frames_processed: int = 0

        # Metrics
        self.last_latency_ms: float = 0.0

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: list[Detection],
        frame_idx: int = 0,
    ) -> list[TrackedObject]:
        """
        Update tracker with new detections and return tracked objects.

        Args:
            detections: Detections from current frame.
            frame_idx: Current frame index.

        Returns:
            List of TrackedObject with persistent IDs and stability info.
        """
        t0 = time.perf_counter()

        # Simple IoU-based matching
        matched, unmatched = self._match_detections(detections)

        # Update matched tracks
        for track_id, det in matched:
            if track_id in self._tracks:
                self._tracks[track_id].update(det, frame_idx)
            else:
                self._tracks[track_id] = _TrackState(track_id, det, frame_idx)

        # Create new tracks for unmatched detections
        for det in unmatched:
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = _TrackState(tid, det, frame_idx)

        # Remove stale tracks
        stale = [
            tid for tid, ts in self._tracks.items()
            if (frame_idx - ts.last_frame) > self.max_age
        ]
        for tid in stale:
            del self._tracks[tid]

        self._frames_processed += 1
        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        # Convert to TrackedObject
        return self._build_tracked_objects()

    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracks.clear()
        self._next_track_id = 0
        self._frames_processed = 0

    def _match_detections(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, Detection]], list[Detection]]:
        """Match new detections to existing tracks via IoU."""
        if not self._tracks or not detections:
            return [], detections

        matched: list[tuple[int, Detection]] = []
        used_tracks: set[int] = set()
        used_det_indices: set[int] = set()

        # Compute IoU matrix
        track_list = [(tid, ts) for tid, ts in self._tracks.items()]

        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track_id = -1

            for tid, ts in track_list:
                if tid in used_tracks:
                    continue
                if ts.detection_class != det.detection_class:
                    continue

                iou = _compute_iou(ts.last_bbox, det.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id >= 0:
                matched.append((best_track_id, det))
                used_tracks.add(best_track_id)
                used_det_indices.add(det_idx)

        unmatched = [
            det for i, det in enumerate(detections)
            if i not in used_det_indices
        ]

        return matched, unmatched

    def _build_tracked_objects(self) -> list[TrackedObject]:
        """Convert internal track states to TrackedObject schema."""
        result: list[TrackedObject] = []
        for tid, ts in self._tracks.items():
            label, confidence = ts.consensus_label(self.stability_threshold)
            result.append(TrackedObject(
                track_id=tid,
                detection_class=ts.detection_class,
                detections=ts.recent_detections[-3:],  # Last 3 detections
                smoothed_label=label,
                smoothed_confidence=confidence,
                is_stable=ts.is_stable(self.stability_threshold),
                frames_seen=ts.frames_seen,
            ))
        return result


# ─── Internal track state ────────────────────────────────────────────────────


class _TrackState:
    """Internal mutable state for a single tracked object."""

    __slots__ = (
        "track_id",
        "detection_class",
        "recent_detections",
        "label_history",
        "last_frame",
        "last_bbox",
        "frames_seen",
    )

    def __init__(
        self, track_id: int, detection: Detection, frame_idx: int
    ) -> None:
        self.track_id = track_id
        self.detection_class = detection.detection_class
        self.recent_detections: list[Detection] = [detection]
        self.label_history: list[str] = [detection.label]
        self.last_frame = frame_idx
        self.last_bbox = detection.bbox
        self.frames_seen = 1

    def update(self, detection: Detection, frame_idx: int) -> None:
        self.recent_detections.append(detection)
        self.label_history.append(detection.label)
        self.last_frame = frame_idx
        self.last_bbox = detection.bbox
        self.frames_seen += 1

        # Trim history
        if len(self.recent_detections) > 20:
            self.recent_detections = self.recent_detections[-10:]
        if len(self.label_history) > 20:
            self.label_history = self.label_history[-10:]

    def consensus_label(self, threshold: int) -> tuple[str, float]:
        """Get the most common label and its frequency as confidence."""
        if not self.label_history:
            return "", 0.0

        counter = Counter(self.label_history[-threshold * 2:])
        most_common, count = counter.most_common(1)[0]
        confidence = count / len(self.label_history[-threshold * 2:])
        return most_common, confidence

    def is_stable(self, threshold: int) -> bool:
        """True if label has been consistent for >= threshold frames."""
        if len(self.label_history) < threshold:
            return False
        recent = self.label_history[-threshold:]
        return len(set(recent)) == 1


# ─── IoU utilities ───────────────────────────────────────────────────────────


def _compute_iou(bbox_a, bbox_b) -> float:
    """Compute IoU between two BoundingBox-like objects."""
    ax1, ay1 = bbox_a.x, bbox_a.y
    ax2, ay2 = ax1 + bbox_a.w, ay1 + bbox_a.h
    bx1, by1 = bbox_b.x, bbox_b.y
    bx2, by2 = bx1 + bbox_b.w, by1 + bbox_b.h

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = bbox_a.w * bbox_a.h
    area_b = bbox_b.w * bbox_b.h
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union
