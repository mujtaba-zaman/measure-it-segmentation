import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from image_ops import normalize_landscape


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
    object_points, image_points = [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = normalize_landscape(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise SystemExit(f"Inconsistent normalized image size in {path}: {current_size} != {image_size}")
        found, corners = cv2.findChessboardCorners(gray, pattern)
        detected_pattern = pattern
        if not found and args.columns != args.rows:
            detected_pattern = (args.rows, args.columns)
            found, corners = cv2.findChessboardCorners(gray, detected_pattern)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_template = np.zeros((detected_pattern[0] * detected_pattern[1], 3), np.float32)
            obj_template[:, :2] = np.mgrid[0:detected_pattern[0], 0:detected_pattern[1]].T.reshape(-1, 2) * args.square_mm
            object_points.append(obj_template)
            image_points.append(corners)
    if len(image_points) < 20:
        raise SystemExit(f"Need >=20 usable checkerboard views; found {len(image_points)}. No parameters saved.")
    rms, camera, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    errors = []
    for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rv, tv, camera, distortion)
        # OpenCV 5 may expose observed and projected points with different
        # channel layouts. Compare normalized Nx2 arrays directly.
        residual = img.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.linalg.norm(residual) / len(projected) ** 0.5))
    result = {"camera_matrix": camera.tolist(), "distortion_coefficients": distortion.ravel().tolist(),
              "image_size_px": list(image_size), "checkerboard_inner_corners": list(pattern),
              "orientation_normalization": "portrait frames rotated 90 degrees clockwise to landscape; checkerboard axis order may be transposed per view",
              "square_size_mm": args.square_mm, "views_used": len(image_points),
              "rms_reprojection_error_px": float(rms), "mean_view_reprojection_error_px": float(np.mean(errors))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("views_used", "rms_reprojection_error_px", "mean_view_reprojection_error_px")}, indent=2))

if __name__ == "__main__": main()
