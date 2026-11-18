from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from image_ops import normalize_landscape

ap = argparse.ArgumentParser(description="Normalize orientation and undistort captured images using saved calibration.")
ap.add_argument("source", help="input image folder")
ap.add_argument("destination", help="output folder")
ap.add_argument("--calibration", default="calibration/intrinsics.json")
a = ap.parse_args()
p = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
k, d = np.array(p["camera_matrix"]), np.array(p["distortion_coefficients"])
expected_size = tuple(p["image_size_px"])
source, out = Path(a.source), Path(a.destination)
out.mkdir(parents=True, exist_ok=True)
written = 0
for f in sorted(source.rglob("*")):
    if not f.is_file():
        continue
    im = cv2.imread(str(f))
    if im is None: continue
    im = normalize_landscape(im)
    h, w = im.shape[:2]
    if (w, h) != expected_size:
        raise SystemExit(f"{f}: normalized size {(w, h)} does not match calibration {expected_size}")
    nk, roi = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 1, (w, h))
    fixed = cv2.undistort(im, k, d, None, nk)
    destination = out / f.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), fixed):
        raise SystemExit(f"Could not write {destination}")
    written += 1
print(f"Wrote {written} normalized, undistorted images to {out}")
