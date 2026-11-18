"""Evaluate the frozen checkpoint once on the reserved test split."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


ROOT = Path(__file__).resolve().parents[1]


def build_model(weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        model.roi_heads.box_predictor.bbox_pred.in_features, 8)
    old = model.roi_heads.mask_predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(old.conv5_mask.in_channels, 256, 2)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def interpolated_ap(scores, ious, threshold):
    order = np.argsort(-np.asarray(scores))
    matches = np.asarray(ious)[order] >= threshold
    tp = np.cumsum(matches.astype(np.float64))
    fp = np.cumsum((~matches).astype(np.float64))
    recall = tp / max(len(scores), 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    return float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0.0
                          for r in np.linspace(0, 1, 101)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/maskrcnn.pt")
    parser.add_argument("--split-file", default="dataset/splits/test.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="measurement/test_evaluation")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(ROOT / args.weights, device)
    items = [line.strip() for line in (ROOT / args.split_file).read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    if not items:
        raise SystemExit("Test split is empty.")

    rows = []
    with torch.no_grad():
        for relative in items:
            image_path = ROOT / "dataset" / relative
            mask_path = ROOT / "dataset/labels" / f"{image_path.stem}.png"
            image = cv2.imread(str(image_path))
            gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or gt is None:
                raise SystemExit(f"Missing/unreadable test image or mask: {image_path} / {mask_path}")
            if image.shape[:2] != gt.shape[:2]:
                raise SystemExit(f"Image/mask size mismatch: {image_path.name}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to(device)
            prediction = model([tensor])[0]
            if len(prediction["scores"]):
                best = int(torch.argmax(prediction["scores"]))
                score = float(prediction["scores"][best].cpu())
                pred_mask = prediction["masks"][best, 0].cpu().numpy() >= 0.5
            else:
                score, pred_mask = 0.0, np.zeros(gt.shape, dtype=bool)
            gt_mask = gt > 0
            iou = mask_iou(pred_mask, gt_mask)
            rows.append({"image": relative, "score": score, "mask_iou": iou,
                         "predicted": score >= args.score_threshold,
                         "true_positive_iou50": score >= args.score_threshold and iou >= 0.5})

    positives = len(rows)
    tp = sum(row["true_positive_iou50"] for row in rows)
    predicted = sum(row["predicted"] for row in rows)
    thresholds = np.arange(0.5, 0.951, 0.05)
    aps = {f"{threshold:.2f}": interpolated_ap([r["score"] for r in rows],
                                                 [r["mask_iou"] for r in rows], threshold)
           for threshold in thresholds}
    result = {
        "test_images": positives,
        "score_threshold": args.score_threshold,
        "mean_mask_iou": float(np.mean([r["mask_iou"] for r in rows])),
        "precision_at_iou50": float(tp / predicted) if predicted else 0.0,
        "recall_at_iou50": float(tp / positives) if positives else 0.0,
        "mAP50": aps["0.50"],
        "mAP50_95": float(np.mean(list(aps.values()))),
        "ap_by_iou_threshold": aps,
        "note": "101-point interpolated AP; one top-scoring instance prediction per image; test split used only for this final evaluation.",
    }
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
