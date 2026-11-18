"""Inspect printable scale markers in a folder of measurement photos."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def find_square(image: np.ndarray):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.015 * width * height < area < 0.18 * width * height:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sides = [float(np.linalg.norm(points[i] - points[(i + 1) % 4])) for i in range(4)]
        if min(sides) <= 0 or max(sides) / min(sides) > 1.35:
            continue
        candidates.append((area, points, sides))
    if not candidates:
        return None
    area, points, sides = max(candidates, key=lambda item: item[0])
    return area, points, float(np.mean(sides))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--marker-mm", type=float, default=50.0)
    parser.add_argument("--csv", type=Path, default=Path("measurement/marker_results/fresh_batch/marker_check.csv"))
    args = parser.parse_args()
    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"No image files in {args.folder}")
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        hit = find_square(image) if image is not None else None
        if hit is None:
            rows.append([path.name, "FAILED", "", "", "", ""])
            print(f"{path.name}: marker square not found")
            continue
        area, points, side_px = hit
        mm_px = args.marker_mm / side_px
        rows.append([path.name, "OK", f"{side_px:.2f}", f"{mm_px:.6f}", f"{area:.0f}", f"{points.round().astype(int).tolist()}"])
        print(f"{path.name}: {side_px:.2f} px; {mm_px:.6f} mm/px")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "marker_status", "marker_side_px", "mm_per_px", "contour_area_px2", "corners_xy"])
        writer.writerows(rows)
    print(f"Wrote {args.csv} ({sum(row[1] == 'OK' for row in rows)}/{len(rows)} marker detections)")


if __name__ == "__main__":
    main()
