"""Tests for Object Tracker."""


from libs.common.schemas import BoundingBox, Detection, DetectionClass
from services.vision_core.tracker import ObjectTracker, _compute_iou

# ─── IoU ─────────────────────────────────────────────────────────────────────


class TestIoU:
    def test_identical_boxes(self):
        a = BoundingBox(x=100, y=100, w=50, h=50, confidence=0.9)
        iou = _compute_iou(a, a)
        assert abs(iou - 1.0) < 1e-6

    def test_non_overlapping(self):
        a = BoundingBox(x=0, y=0, w=50, h=50, confidence=0.9)
        b = BoundingBox(x=200, y=200, w=50, h=50, confidence=0.9)
        iou = _compute_iou(a, b)
        assert iou == 0.0

    def test_partial_overlap(self):
        a = BoundingBox(x=0, y=0, w=100, h=100, confidence=0.9)
        b = BoundingBox(x=50, y=50, w=100, h=100, confidence=0.9)
        iou = _compute_iou(a, b)
        # Overlap: 50x50=2500, Union: 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 0.01


# ─── ObjectTracker ───────────────────────────────────────────────────────────


def _make_detections(positions: list[tuple[float, float]], cls=DetectionClass.CARD):
    """Helper to make detections at given positions."""
    return [
        Detection(
            detection_class=cls,
            bbox=BoundingBox(x=x, y=y, w=50, h=70, confidence=0.9),
            label=f"det_{i}",
            frame_idx=0,
        )
        for i, (x, y) in enumerate(positions)
    ]


class TestObjectTracker:
    def test_initial_state(self):
        tracker = ObjectTracker()
        assert tracker.active_tracks == 0

    def test_single_frame_creates_tracks(self):
        tracker = ObjectTracker()
        dets = _make_detections([(100, 200), (300, 400)])
        tracked = tracker.update(dets, frame_idx=0)
        assert len(tracked) == 2
        assert tracker.active_tracks == 2

    def test_tracks_persist_across_frames(self):
        tracker = ObjectTracker()

        # Frame 0
        dets0 = _make_detections([(100, 200)])
        t0 = tracker.update(dets0, frame_idx=0)
        track_id_0 = t0[0].track_id

        # Frame 1: same position → same track
        dets1 = _make_detections([(105, 205)])  # slight movement
        t1 = tracker.update(dets1, frame_idx=1)

        assert len(t1) == 1
        assert t1[0].track_id == track_id_0
        assert t1[0].frames_seen == 2

    def test_label_stability(self):
        tracker = ObjectTracker(stability_threshold=3)

        # Feed same label 5 times
        for i in range(5):
            dets = [Detection(
                detection_class=DetectionClass.CARD,
                bbox=BoundingBox(x=100, y=200, w=50, h=70, confidence=0.9),
                label="Ah",
                frame_idx=i,
            )]
            tracked = tracker.update(dets, frame_idx=i)

        assert len(tracked) == 1
        assert tracked[0].is_stable
        assert tracked[0].smoothed_label == "Ah"

    def test_unstable_labels(self):
        tracker = ObjectTracker(stability_threshold=3)

        labels = ["Ah", "Kh", "Ah"]  # oscillating
        for i, label in enumerate(labels):
            dets = [Detection(
                detection_class=DetectionClass.CARD,
                bbox=BoundingBox(x=100, y=200, w=50, h=70, confidence=0.9),
                label=label,
                frame_idx=i,
            )]
            tracked = tracker.update(dets, frame_idx=i)

        assert not tracked[0].is_stable

    def test_stale_tracks_removed(self):
        tracker = ObjectTracker(max_age=2)

        # Frame 0: create track
        dets = _make_detections([(100, 200)])
        tracker.update(dets, frame_idx=0)

        # Frame 1-3: no detections
        for i in range(1, 4):
            tracker.update([], frame_idx=i)

        # Track should be removed after max_age
        assert tracker.active_tracks == 0

    def test_multiple_classes_tracked_separately(self):
        tracker = ObjectTracker()

        card = Detection(
            detection_class=DetectionClass.CARD,
            bbox=BoundingBox(x=100, y=200, w=50, h=70, confidence=0.9),
            label="Ah",
        )
        pot = Detection(
            detection_class=DetectionClass.POT,
            bbox=BoundingBox(x=100, y=200, w=50, h=70, confidence=0.9),
            label="500",
        )

        tracked = tracker.update([card, pot], frame_idx=0)
        assert len(tracked) == 2

    def test_reset(self):
        tracker = ObjectTracker()
        dets = _make_detections([(100, 200)])
        tracker.update(dets, frame_idx=0)
        assert tracker.active_tracks == 1

        tracker.reset()
        assert tracker.active_tracks == 0

    def test_latency_tracked(self):
        tracker = ObjectTracker()
        dets = _make_detections([(100, 200)])
        tracker.update(dets, frame_idx=0)
        assert tracker.last_latency_ms >= 0
