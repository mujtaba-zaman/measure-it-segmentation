"""Create reviewed binary phone masks by outlining each image in a local window.

Controls: A proposes a threshold-based outline (inspect it), left click adds
polygon points, right click removes the last point, F previews, S saves, R
resets, N skips, Q quits.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args()

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    paths = [p for split in splits for p in sorted((ROOT / "dataset/images" / split).glob("*"))
             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not paths:
        raise SystemExit("No split images found. Check dataset/images/<split>/.")

    mask_dir = ROOT / "dataset/labels"
    mask_dir.mkdir(parents=True, exist_ok=True)
    window = "Measure It - polygon mask annotator"
    state = {"image": None, "points": [], "finished": False, "scale": 1.0}

    def propose_outline(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        border = max(3, round(min(binary.shape) * 0.004))
        binary[:border] = binary[-border:] = 0
        binary[:, :border] = binary[:, -border:] = 0
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 0.005 * image.shape[0] * image.shape[1] < cv2.contourArea(c)
                    < 0.75 * image.shape[0] * image.shape[1]]
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, 0.006 * cv2.arcLength(contour, True), True)
        return [tuple(point[0]) for point in polygon] if len(polygon) >= 3 else []

    def render():
        image = state["image"]
        if image is None:
            return
        view = image.copy()
        points = state["points"]
        for point in points:
            cv2.circle(view, point, 5, (0, 180, 255), -1)
        if len(points) > 1:
            cv2.polylines(view, [np.asarray(points, np.int32)], state["finished"], (0, 180, 255), 2)
        if state["finished"] and len(points) >= 3:
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
            overlay = view.copy()
            overlay[mask > 0] = (0.35 * overlay[mask > 0] + 0.65 * np.array([40, 210, 40])).astype(np.uint8)
            view = overlay
        cv2.putText(view, "A:auto outline (inspect)  Left:add  Right:undo  F:preview  S:save  R:reset  N:skip  Q:quit",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 230), 2, cv2.LINE_AA)
        scale = state["scale"]
        if scale != 1:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(window, view)

    def mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and not state["finished"]:
            scale = state["scale"]
            state["points"].append((round(x / scale), round(y / scale)))
            render()
        elif event == cv2.EVENT_RBUTTONDOWN and state["points"] and not state["finished"]:
            state["points"].pop()
            render()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                print(f"SKIP unreadable: {path.name}")
                continue
            height, width = image.shape[:2]
            state["scale"] = min(1.0, args.max_width / width, args.max_height / height)
            state["image"] = image
            state["points"] = []
            state["finished"] = False
            mask_path = mask_dir / f"{path.stem}.png"
            if mask_path.exists():
                existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if existing is not None and existing.shape == (height, width):
                    ys, xs = np.where(existing > 0)
                    if len(xs):
                        contour = cv2.findContours(existing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if contour:
                            state["points"] = [tuple(p[0]) for p in max(contour, key=cv2.contourArea)]
                            state["finished"] = True
            render()
            print(f"[{index}/{len(paths)}] {path.name} | mask {'exists' if mask_path.exists() else 'needed'}")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("a"):
                    proposal = propose_outline(image)
                    if proposal:
                        state["points"] = proposal
                        state["finished"] = True
                        print("Auto outline is only a proposal: inspect the green edge before saving.")
                    else:
                        print("No safe auto outline found; draw the phone boundary manually.")
                    render()
                elif key == ord("r"):
                    state["points"] = []
                    state["finished"] = False
                    render()
                elif key == ord("f"):
                    if len(state["points"]) >= 3:
                        state["finished"] = True
                        render()
                    else:
                        print("Add at least three points first.")
                elif key == ord("s"):
                    if len(state["points"]) < 3:
                        print("Draw and finish the phone outline before saving.")
                        continue
                    mask = np.zeros((height, width), np.uint8)
                    cv2.fillPoly(mask, [np.asarray(state["points"], np.int32)], 255)
                    if cv2.imwrite(str(mask_path), mask):
                        print(f"Saved {mask_path.relative_to(ROOT)}")
                        break
                    print(f"ERROR writing {mask_path}")
                elif key == ord("n"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
