# Measure It - Segmentation and Metric Measurement

AI/CV assessment project for measuring a flat object in millimetres. It calibrates a fixed camera with OpenCV, trains a torchvision Mask R-CNN on reviewed masks of project-captured photos, and uses a measured coplanar ArUco marker as the image scale reference. YOLO and Roboflow segmentation models are not used.

## Dataset and current results

- Target: iPhone X photographed with a second phone, reported by the candidate as iPhone 12.
- 100 unmarked target photos are assigned once to 70 train, 20 validation and 10 test. The split lists and raw-image SHA-256 manifest are frozen in commit `975d786` before training. Test images must not be used for training, tuning or checkpoint selection.
- The 100 raw photos are preserved under `dataset/raw_images/`; normalized, undistorted copies used for modeling are under `dataset/images/{train,val,test}/`.
- 20 unique checkerboard views were detected and used. Portrait frames are rotated clockwise into landscape coordinates. Calibration saved to `calibration/intrinsics.json`: image size 1280x960, RMS reprojection error 1.565 px, mean per-view error 1.150 px.
- 10 target-plus-marker photos are in `dataset/measurement_images/`, but `img_009.jpeg` and `img_010.jpeg` are exact duplicates; there are 9 distinct photos. Retake one before using this as a 10-instance accuracy set.
- Binary object masks are not yet present. Model training, held-out evaluation and physical accuracy measurement have not run. No model performance or measurement accuracy is claimed.

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

1. Verify checkerboard square size is 20 mm and measure the marker's black outer edge with a ruler. The PDF says nominally 50 mm, but use the physical measurement.
2. Calibration already ran on 20 checkerboard images:
   `\.venv\Scripts\python.exe calibration/calibrate.py --square-mm 20 --columns 9 --rows 7`
3. Annotate one accurate phone silhouette mask per split image. Run `\.venv\Scripts\python.exe scripts/annotate_masks.py --split train`, then `val`, then `test`. Press `A` for a threshold-based outline proposal, inspect the green edge, and press `S` only if it is correct; otherwise press `R` and draw the outline with left-clicks. Press `F` to preview and `S` to save. Masks are 0 background / 255 phone at `dataset/labels/<image-stem>.png`. Review every outline; do not use test results to adjust training.
4. Train only after masks and the fixed split are committed:
   `\.venv\Scripts\python.exe models/train.py --epochs 10 --seed 42`
5. After the checkpoint and confidence threshold are frozen, run the final held-out metrics once:
   `\.venv\Scripts\python.exe models/evaluate.py --weights models/maskrcnn.pt`
6. Run the measurement command on a new photo with the marker visible in the same plane:
   `\.venv\Scripts\python.exe inference/measure.py path\to\new.jpg --marker-mm <measured-size>`
   Add `--skip-undistortion` for the raw-image comparison and use separate `--output` paths to preserve both overlays.
7. Compare at least 10 separate placements against ruler/calliper ground truth; report per-instance and summary errors, raw-versus-undistorted performance, one failure condition, and five-run timing.
8. Complete `docs/Plan.md`, `docs/Objectives.md`, and `docs/Definition-of-Done.md` with evidence. Record a Loom of 5 minutes or less answering the four assessment questions.

## Assessment constraints and reproducibility

- Use images captured for this project only. Do not use downloaded object datasets, Ultralytics YOLO, or Roboflow segmentation models. Pretrained backbones are allowed, but the segmentation model must be trained or fine-tuned on the labelled project images.
- The split files and SHA-256 manifest are committed before training; do not alter the test split afterward.
- Report only metrics produced from saved raw outputs. AI assistance is allowed by the assessment; the candidate must explain the submitted code and results in the recording.
- Record OS, CPU/GPU, RAM, camera model, image resolution, focus/zoom settings, library versions, physical marker measurement and any deviation.

See `docs/Plan.md`, `docs/Objectives.md` and `docs/Definition-of-Done.md` for the decisions, targets and evidence checklist.
