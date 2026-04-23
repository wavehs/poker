"""
Benchmark for ObjectTracker performance.
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, ".")

from libs.common.schemas import BoundingBox, Detection, DetectionClass
from services.vision_core.tracker import ObjectTracker

def make_detections(n: int, frame_idx: int = 0):
    return [
        Detection(
            detection_class=DetectionClass.CARD,
            bbox=BoundingBox(x=i*10, y=i*10, w=50, h=70, confidence=0.9),
            label=f"card_{i}",
            frame_idx=frame_idx,
        )
        for i in range(n)
    ]

def bench_tracker_update(num_detections: int = 100, num_frames: int = 10):
    tracker = ObjectTracker()

    # Warm up and create initial tracks
    dets = make_detections(num_detections, 0)
    tracker.update(dets, 0)

    # Benchmarking
    total_time = 0
    for i in range(1, num_frames + 1):
        # Slightly move detections to maintain matches
        dets = [
            Detection(
                detection_class=d.detection_class,
                bbox=BoundingBox(x=d.bbox.x + 1, y=d.bbox.y + 1, w=d.bbox.w, h=d.bbox.h, confidence=d.bbox.confidence),
                label=d.label,
                frame_idx=i
            )
            for d in dets
        ]

        t0 = time.perf_counter()
        tracker.update(dets, i)
        total_time += (time.perf_counter() - t0)

    avg_time_ms = (total_time / num_frames) * 1000
    print(f"Average update time for {num_detections} detections: {avg_time_ms:.4f} ms")
    return avg_time_ms

def bench_new_tracks(num_detections: int = 100):
    tracker = ObjectTracker()
    dets = make_detections(num_detections, 0)

    t0 = time.perf_counter()
    tracker.update(dets, 0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Time to create {num_detections} new tracks: {elapsed_ms:.4f} ms")
    return elapsed_ms

if __name__ == "__main__":
    print("--- Tracker Benchmark ---")
    bench_new_tracks(100)
    bench_new_tracks(1000)
    bench_tracker_update(100, 50)
    bench_tracker_update(1000, 50)
