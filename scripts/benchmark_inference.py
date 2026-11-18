"""Measure five cold, end-to-end single-image CLI runs and save raw timings."""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path, default=ROOT / "measurement/marker_photos/fresh_batch/img_001.jpeg")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--marker-mm", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=ROOT / "measurement/benchmark")
    args = parser.parse_args()
    if args.runs < 5:
        raise SystemExit("Assessment timing requires at least five runs")
    image = args.image.resolve()
    if not image.is_file():
        raise SystemExit(f"Image not found: {image}")
    script = ROOT / "inference/measure.py"
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    records = []
    for run in range(1, args.runs + 1):
        overlay = args.output / f"timing_run_{run:02d}.png"
        command = [sys.executable, str(script), str(image), "--marker-mm", str(args.marker_mm), "--output", str(overlay)]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if completed.returncode:
            raise SystemExit(f"Run {run} failed:\n{completed.stderr or completed.stdout}")
        payload = json.loads(completed.stdout)
        records.append((run, elapsed, payload["width_mm"], payload["height_mm"], payload["confidence"], payload["undistortion_applied"]))
        print(f"run {run}: {elapsed:.3f}s | {payload['width_mm']:.2f} x {payload['height_mm']:.2f} mm")

    times = [row[1] for row in records]
    summary = {
        "runs": len(times),
        "scope": "cold end-to-end CLI; process/model load, preprocessing, inference, marker scale and overlay write",
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "spread_seconds": max(times) - min(times),
        "image": str(image),
    }
    csv_path = args.output / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "elapsed_seconds", "width_mm", "height_mm", "confidence", "undistortion_applied"])
        writer.writerows(records)
    json_path = args.output / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved raw timings: {csv_path}")


if __name__ == "__main__":
    main()
