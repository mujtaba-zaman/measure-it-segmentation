"""Check mask coverage and image/mask sizes before training."""
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
errors = []
for split in ("train", "val", "test"):
    items = [line.strip() for line in (ROOT / f"dataset/splits/{split}.txt").read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.startswith("#")]
    missing = 0
    valid = 0
    for relative in items:
        image_path = ROOT / "dataset" / relative
        mask_path = ROOT / "dataset/labels" / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if image is None:
            errors.append(f"{split}: unreadable image {image_path}")
            continue
        if mask is None:
            missing += 1
            continue
        if image.shape[:2] != mask.shape[:2]:
            errors.append(f"{split}: size mismatch {image_path.name}: image={image.shape[1]}x{image.shape[0]}, mask={mask.shape[1]}x{mask.shape[0]}")
            continue
        ratio = float(np.count_nonzero(mask) / mask.size)
        if not 0.005 <= ratio <= 0.95:
            errors.append(f"{split}: implausible foreground fraction {ratio:.3f} in {mask_path.name}")
            continue
        valid += 1
    print(f"{split}: {valid}/{len(items)} valid masks, {missing} missing")

if errors:
    for error in errors:
        print("ERROR:", error)
    raise SystemExit(1)
print("All split masks are present, nonempty, and match their image sizes.")
