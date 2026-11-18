"""Sweep increasing synthetic occlusion over the printed scale marker.

This isolates the marker-contour gate; it does not test object segmentation or
measurement accuracy.  The selected frame is never modified on disk.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from inspect_marker_batch import find_square


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "measurement/marker_photos/fresh_batch/img_001.jpeg")
    parser.add_argument("--output", type=Path, default=ROOT / "measurement/marker_results/failure_sweep.csv")
    parser.add_argument("--step", type=int, default=5, help="Occluded marker height per step, as percent of its side")
    parser.add_argument("--max", type=int, default=60, dest="maximum", help="Maximum occluded marker height in percent")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    baseline = find_square(image)
    if baseline is None:
        raise SystemExit("Baseline marker not detected; choose a clear marker photo")
    _, points, side_px = baseline
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    h, w = image.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    rows = []
    for severity in range(0, args.maximum + 1, args.step):
        sample = image.copy()
        cover_h = round((y1 - y0) * severity / 100)
        if cover_h:
            # Simulate a white obstruction crossing the marker's upper edge.
            cv2.rectangle(sample, (x0 - 2, y0 - 2), (x1 + 2, y0 + cover_h), (255, 255, 255), -1)
        detected = find_square(sample)
        rows.append([severity, "PASS" if detected else "REFUSE", f"{side_px:.2f}", f"{cover_h}px"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["marker_height_occluded_pct", "marker_gate", "baseline_marker_side_px", "synthetic_cover_height_px"])
        writer.writerows(rows)
    print(f"Image: {args.image.name}; baseline marker: {side_px:.2f}px")
    for severity, result, _, cover_h in rows:
        print(f"{int(severity):>2}% occlusion ({cover_h}): {result}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
