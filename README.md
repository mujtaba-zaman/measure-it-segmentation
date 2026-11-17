# Measure It — Segmentation and Metric Measurement

An 8-hour AI/CV assessment starter. It calibrates a fixed camera, trains a torchvision Mask R-CNN on your own binary object masks, then measures a flat object against a coplanar ArUco square of measured size. The reference marker supplies scale; it is not the object segmenter. Ultralytics YOLO and Roboflow segmentation models are not used.

## Important before running

The committed split reserves 70 train / 20 validation / 10 test image names. Capture at least 100 images and rename/copy them to those names before annotation. These are placeholders, not captured data. Replace the reserved file lists once with your final dataset and commit them **before any training**; never move test images after inspecting results. The initial commit could not be authored automatically because this machine has no configured Git author identity.

The pipeline assumes one foreground object per image, a flat object and reference in the same plane, low camera tilt, and an ArUco DICT_4X4_50 marker. Pixel-to-mm conversion uses the marker's known side length divided by its observed pixel side. Perspective tilt or different object/reference planes invalidate this simple scale.

## Setup

Use Python 3.10–3.12 in a virtual environment. Install PyTorch/Torchvision using the official command matching your CPU/GPU, then install `requirements.txt`. Keep camera focus, zoom and resolution fixed after calibration. See `docs/Plan.md`, `docs/Objectives.md`, and `docs/Definition-of-Done.md` for the assessment plan, targets and evidence checklist.

## Stages

1. Print `calibration/printables/measure-it-printables.pdf` at 100% / actual size with fit-to-page disabled. Page 1 is a 9x7-inner-corner checkerboard (20 mm squares); page 2 is DICT_4X4_50 marker ID 0 (50 mm black outer edge). Check both dimensions with a ruler before use. Put >=20 varied checkerboard images in `calibration/images/`.
2. `python calibration/calibrate.py --square-mm 20 --columns 9 --rows 7`
3. Capture object photos with the measured ArUco marker in the same plane. Annotate each target with a binary PNG mask at `dataset/labels/<image-stem>.png`; object pixels nonzero. Place images in `dataset/images/` under the names in split files. Keep the marker visible in every final inference image.
4. `python models/train.py --epochs 10 --seed 42` (downloads pretrained torchvision backbone weights on first run if not cached). Test split is deliberately not used for training or validation.
5. `python inference/measure.py path/to/new.jpg --marker-mm 50` (replace 50 with the actual measured marker side).
6. Record calliper truth and predictions in `measurement/accuracy.csv`, compare raw and undistorted measurements, then run `python measurement/summarize.py measurement/accuracy.csv`.

## Dataset and reproducibility

Object: **second smartphone (planned)**, photographed with the iPhone 12 rear 1x camera. Capture: **0**. Split reservation: 70/20/10 placeholder names. Labelling tool: **not yet selected**. Random seed: 42. Record OS, CPU/GPU, RAM, camera make/model, resolution, lens/focus settings, Python/library versions, marker measurement, and every deviation. No assessment metrics are claimed until produced by the user's own captured images.

## Current status / known gaps

This is a functional pipeline scaffold, not an assessed result. You still need camera calibration images, a selected object, its labelled images, a physical marker and measured size, calliper ground truth for >=10 instances, actual train/test metrics, raw-vs-undistorted comparison, failure-condition experiment, confidence/refusal-cost analysis, and a <=5-minute Loom. The training code reports validation pixel IoU only; proper test-split mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall still need implementation/evidence before submission. Inference also needs empirical calibration of confidence and reference-visibility/tilt rejection thresholds. No numbers should be entered by hand without raw evidence.

## Submission still required

Create a public GitHub repository under your account, push branch `dev-test01`, make a PR to its main branch, publish weights or a release asset and SHA-256 hash, and send the PR and Loom links as the assessment requests. No public repository or external upload has been created here.


## Capture guide (iPhone 12)

A phone cannot photograph itself for this setup: use the iPhone 12 camera to photograph a **different** object, or borrow a second camera if the iPhone itself is the target. A phone is a glossy target, so put it on a matte, contrasting background, keep the rear face flat, and keep the ArUco marker flat in the same plane. Avoid reflections and do not let the raised camera island overlap the phone outline.

1. Install the listed dependencies, then run `python calibration/make_printables_pdf.py`. It makes a checkerboard with 9x7 inner corners and 20 mm squares, plus a 50 mm `DICT_4X4_50` marker (ID 0). Print SVGs at **100% / actual size**, with page-fit/shrink disabled. Measure a checker square and the marker's black outer edge with a ruler; if print scaling changed their sizes, use the measured values in the commands. Tape the checkerboard flat to rigid cardboard.
2. Use the iPhone rear **1x** camera; avoid switching lenses or digital zoom, and keep resolution/orientation constant. Tap-hold to lock focus/exposure where possible. Capture 20+ sharp calibration photos with the board filling different parts of the frame and held at varied tilts/distances. Run calibration with the measured square size, e.g. `python calibration/calibrate.py --square-mm 20`.
3. Select one matte, flat object with clear edges and a contrasting background. Capture 100 images with the same camera/lens/settings: 70 train, 20 validation, 10 test. Vary position, rotation, distance and ordinary lighting; avoid blur. Include the same measured marker beside the object in every measurement photo, in the same plane. Never use test images to train or choose settings.
4. Use CVAT or Label Studio to draw one accurate foreground mask per image; export/convert each mask to a PNG with background 0 and object nonzero. Save using the matching image filename stem under `dataset/labels/`. Keep a copy of the split lists and commit them before training.
5. For the accuracy comparison, measure the object's width and height with a ruler/calliper, then take 10+ separate placements/photos and record ground truth for each. A low-cost digital calliper is preferable; a ruler with millimetre markings is acceptable for a rough prototype, but report its resolution as a limitation. Keep raw and undistorted measurements paired by image.

If you want to measure the iPhone 12 itself, borrow another camera for the photos. Your target choice is set to the second smartphone. A Git commit display name and email are still needed before commits can be created.
