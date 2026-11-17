import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Calibrate a fixed camera from checkerboard images.")
    ap.add_argument("--images", default="calibration/images")
    ap.add_argument("--columns", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=7, help="inner corners down")
    ap.add_argument("--square-mm", type=float, required=True, help="measured checker square side")
    ap.add_argument("--output", default="calibration/intrinsics.json")
    args = ap.parse_args()
    paths = sorted(Path(args.images).glob("*"))
    pattern = (args.columns, args.rows)
    obj_template = np.zeros((args.columns * args.rows, 3), np.float32)
    obj_template[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    object_points, image_points = [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            object_points.append(obj_template.copy())
            image_points.append(corners)
    if len(image_points) < 20:
        raise SystemExit(f"Need >=20 usable checkerboard views; found {len(image_points)}. No parameters saved.")
    rms, camera, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    errors = []
    for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, camera, distortion)
        errors.append(float(cv2.norm(img, projected, cv2.NORM_L2) / len(projected) ** 0.5))
    result = {"camera_matrix": camera.tolist(), "distortion_coefficients": distortion.ravel().tolist(),
              "image_size_px": list(image_size), "checkerboard_inner_corners": list(pattern),
              "square_size_mm": args.square_mm, "views_used": len(image_points),
              "rms_reprojection_error_px": float(rms), "mean_view_reprojection_error_px": float(np.mean(errors))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("views_used", "rms_reprojection_error_px", "mean_view_reprojection_error_px")}, indent=2))

if __name__ == "__main__": main()
