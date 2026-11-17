from pathlib import Path
import argparse
import cv2

ap = argparse.ArgumentParser(description="Undistort captured images using saved calibration.")
ap.add_argument("source", help="input image folder")
ap.add_argument("destination", help="output folder")
ap.add_argument("--calibration", default="calibration/intrinsics.json")
a = ap.parse_args()
import json
p = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
k, d = __import__("numpy").array(p["camera_matrix"]), __import__("numpy").array(p["distortion_coefficients"])
out = Path(a.destination); out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(a.source).glob("*")):
    im = cv2.imread(str(f))
    if im is None: continue
    h, w = im.shape[:2]
    nk, roi = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 1, (w, h))
    fixed = cv2.undistort(im, k, d, None, nk)
    cv2.imwrite(str(out / f.name), fixed)
print(f"Wrote undistorted images to {out}")
