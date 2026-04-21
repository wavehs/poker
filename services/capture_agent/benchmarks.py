"""
Capture Agent — Benchmarks for DXcam vs MSS capture performance.

Usage:
    python -m services.capture_agent.benchmarks
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np


def benchmark_backend(backend: str, num_frames: int = 50) -> dict | None:
    """Benchmark a specific capture backend."""
    from services.capture_agent.capture import CaptureAgent

    try:
        agent = CaptureAgent(source=backend)
    except Exception as e:
        print(f"  ⚠ Cannot create agent with backend '{backend}': {e}")
        return None

    if agent.backend != backend and backend != "auto":
        print(f"  ⚠ Requested '{backend}', resolved to '{agent.backend}'")
        if agent.backend == "file":
            print(f"  ⚠ Skipping blank-frame benchmark for '{backend}'")
            return None

    latencies: list[float] = []
    frame_sizes: list[int] = []

    print(f"  Capturing {num_frames} frames with '{agent.backend}'...")

    with agent:
        for i in range(num_frames):
            t0 = time.perf_counter()
            frame, meta = agent.capture_frame()
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            frame_sizes.append(frame.nbytes)

    result = {
        "backend": agent.backend,
        "frames": num_frames,
        "avg_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(num_frames * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(num_frames * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "fps": round(1000.0 / statistics.mean(latencies), 1),
        "frame_bytes": frame_sizes[0] if frame_sizes else 0,
    }
    return result


def print_report(results: list[dict]) -> None:
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("  CAPTURE BENCHMARK REPORT")
    print("=" * 70)

    for r in results:
        print(f"\n  Backend: {r['backend']}")
        print(f"  Frames:  {r['frames']}")
        print(f"  ─────────────────────────────")
        print(f"  Avg:     {r['avg_ms']:>8.2f} ms")
        print(f"  Median:  {r['median_ms']:>8.2f} ms")
        print(f"  P95:     {r['p95_ms']:>8.2f} ms")
        print(f"  P99:     {r['p99_ms']:>8.2f} ms")
        print(f"  Min:     {r['min_ms']:>8.2f} ms")
        print(f"  Max:     {r['max_ms']:>8.2f} ms")
        print(f"  FPS:     {r['fps']:>8.1f}")
        print(f"  Frame:   {r['frame_bytes'] / 1024:.0f} KB")

    print("\n" + "=" * 70)


def main() -> None:
    print("Poker Helper — Capture Benchmark")
    print(f"Platform: {sys.platform}")
    print()

    backends_to_test = ["dxcam", "mss"]
    if sys.platform != "win32":
        backends_to_test = ["mss"]

    results: list[dict] = []

    for backend in backends_to_test:
        print(f"\n▶ Testing backend: {backend}")
        result = benchmark_backend(backend, num_frames=50)
        if result:
            results.append(result)

    if results:
        print_report(results)
    else:
        print("\n⚠ No backends available for benchmarking on this platform.")


if __name__ == "__main__":
    main()
