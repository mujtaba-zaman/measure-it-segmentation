# Measure It - Segmentation and Metric Measurement

AI/CV assessment project for estimating a flat object's dimensions in millimetres. It calibrates a fixed camera with OpenCV, trains a torchvision Mask R-CNN on project-captured phone images, and uses a known-size printed square reference in the object's plane. The supplied marker is inverted relative to OpenCV's standard ArUco polarity, so inference falls back to its isolated outer square. YOLO and Roboflow segmentation models are not used.

## Dataset and current results

- Target: iPhone X photographed with a second phone, reported by the candidate as iPhone 12.
- 100 unmarked target photos are assigned once to 70 train, 20 validation and 10 test. The split lists and raw-image SHA-256 manifest are frozen in commit `975d786` before training. Test images must not be used for training, tuning or checkpoint selection.
- The 100 raw photos are preserved under `dataset/raw_images/`; normalized, undistorted copies used for modeling are under `dataset/images/{train,val,test}/`.
- 20 unique checkerboard views were detected and used. Portrait frames are rotated clockwise into landscape coordinates. Calibration saved to `calibration/intrinsics.json`: image size 1280x960, RMS reprojection error 1.565 px, mean per-view error 1.150 px.
- 10 target-plus-marker photos are in `measurement/marker_photos/`; `img_009.jpeg` and `img_010.jpeg` are exact duplicates, leaving 9 distinct frames.
- 100 binary phone masks are now present: 70 train, 20 validation and 10 reserved test. `scripts/check_labels.py` reports all 100 masks nonempty and dimension-matched. The held-out test labels have not been used for training or tuning.
- Mask R-CNN was trained for 10 epochs on a Colab Tesla T4. Best recorded validation pixel IoU was 0.9662. On the 10-image held-out segmentation test, mean mask IoU was 0.9680 and the evaluator reported mAP@0.5 and mAP@0.5:0.95 of 1.0. These are segmentation metrics, not physical measurement accuracy.
- The trained checkpoint, training history, held-out metrics, four successful measurement overlays, and the full 10-photo measurement attempt are saved in this project. Four frames produced estimates; six were safely refused because the 50 mm reference square was occluded or could not be isolated. The ten frames include one exact duplicate. No calliper ground-truth measurements were collected, so dimensional accuracy/error is not established.

## Setup

The host has Python 3.15.2. A local `.venv` has the core packages from `requirements.txt` installed. To recreate it:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Training needs PyTorch and Torchvision from the official PyTorch wheel index. For a CPU-only install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the matching index from the official PyTorch installer. A Google Colab GPU runtime is also suitable after uploading the project data and labels.

## Workflow

1. Checkerboard square size used: 20 mm. The printed reference marker's outside black edge is treated as 50 mm, as measured/reported for this run.
2. Calibration already ran on 20 checkerboard images:
   `\.venv\Scripts\python.exe calibration/calibrate.py --square-mm 20 --columns 9 --rows 7`
3. Annotate one accurate phone silhouette mask per split image. Run `\.venv\Scripts\python.exe scripts/annotate_masks.py --split train`, then `val`, then `test`. Press `A` for a threshold-based outline proposal, inspect the green edge, and press `S` only if it is correct; otherwise press `R` and draw the outline with left-clicks. Press `F` to preview and `S` to save. Masks are 0 background / 255 phone at `dataset/labels/<image-stem>.png`. Review every outline; do not use test results to adjust training.
4. Train only after masks and the fixed split are committed:
   `\.venv\Scripts\python.exe models/train.py --epochs 10 --seed 42`
5. After the checkpoint and confidence threshold are frozen, run the final held-out metrics once:
   `\.venv\Scripts\python.exe models/evaluate.py --weights models/maskrcnn.pt`
6. Run the measurement command on a photo with the full marker visible in the same plane:
   `\.venv\Scripts\python.exe inference/measure.py path\to\new.jpg --marker-mm <measured-size>`
   Add `--skip-undistortion` for the raw-image comparison and use separate `--output` paths to preserve both overlays.
7. Current measurement evidence is in `measurement/marker_results/`. Four of ten frames produced estimates in both raw and undistorted modes; six were refused because the reference could not be isolated. The images include one exact duplicate. These estimates have no calliper ground truth, so do not present them as validated dimensions or measurement-accuracy results.
8. Remaining assessment evidence: ten distinct placements with calliper ground truth, dimensional-error summaries, raw-versus-undistorted error comparison, timing, failure severity sweep, and the <=5-minute Loom.

## Assessment constraints and reproducibility

- Use images captured for this project only. Do not use downloaded object datasets, Ultralytics YOLO, or Roboflow segmentation models. Pretrained backbones are allowed, but the segmentation model must be trained or fine-tuned on the labelled project images.
- The split files and SHA-256 manifest are committed before training; do not alter the test split afterward.
- Report only metrics produced from saved raw outputs. AI assistance is allowed by the assessment; the candidate must explain the submitted code and results in the recording.
- Record OS, CPU/GPU, RAM, camera model, image resolution, focus/zoom settings, library versions, physical marker measurement and any deviation.

See `docs/Plan.md`, `docs/Objectives.md` and `docs/Definition-of-Done.md` for the decisions, targets and evidence checklist.
