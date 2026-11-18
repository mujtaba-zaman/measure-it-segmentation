# Plan — Measure It

## Scope and time budget
This 8-hour assessment measures one rigid, flat object in the plane of a known-size reference marker. Work in this order; stop and document honestly if time expires.

| Elapsed | Work | Evidence |
|---|---|---|
| 0:00–0:30 | Confirm camera/object/reference, freeze camera settings; commit plan and reserved splits first | This plan, `dataset/splits/` |
| 0:30–1:15 | Capture >=20 checkerboard views; calibrate and inspect reprojection error | calibration images, `calibration/intrinsics.json` |
| 1:15–2:45 | Capture 100 object images with the same camera settings; undistort and label; preserve reserved split | images, masks, split lists |
| 2:45–4:15 | Train/fine-tune Mask R-CNN; keep test split untouched | config, logs, checkpoint hash |
| 4:15–5:15 | Implement one-image inference: undistort, segment, detect reference marker, convert scale, abstain when unreliable | command output and overlay |
| 5:15–6:30 | Compare >=10 instances to calliper; compare raw vs undistorted | per-instance CSV, error summaries |
| 6:30–7:15 | Measure one chosen failure condition and one speed/accuracy target | raw repeated measurements |
| 7:15–8:00 | Record limitations, complete Definition of Done, prepare <=5 minute Loom and PR | docs, recording, PR |

## Decisions made before implementation
- **Chosen approach:** OpenCV checkerboard calibration and undistortion; a fine-tuned torchvision Mask R-CNN with a ResNet-FPN backbone; an ArUco square marker of measured side length as the in-plane scale reference.
- **Rejected approach:** Ultralytics YOLO/Roboflow segmentation models are prohibited by the brief. A simple threshold/contour-only object detector is also rejected because it does not meet the trained-segmentation requirement. Marker detection is limited to reference scale, not object segmentation.
- **Cost of rejection:** Mask R-CNN training and data preparation take longer and need Torch/Torchvision; the implementation may not reach useful accuracy within eight hours. The marker approach requires a printed ArUco marker coplanar with the object and is unsuitable if that cannot be arranged.
- **Current data status:** 20 unique checkerboard photos, 100 unique unmarked iPhone X target photos, and 10 target-plus-marker photos are present. The 100 target photos are assigned once to 70 train / 20 validation / 10 test using seed 42; split lists and a SHA-256 manifest were committed as `975d786` before training. Raw target photos are preserved under `dataset/raw_images/`; model copies are normalized to landscape and undistorted under `dataset/images/`. The marker set has an exact duplicate (`img_009.jpeg` / `img_010.jpeg`), leaving 9 distinct examples. No test images have been used for training or tuning.
- **Calibration completed:** all 20 checkerboard views were detected after orientation normalization. With the measured 20 mm square size, reprojection RMS is 1.565 px and mean per-view error is 1.150 px; parameters are in `calibration/intrinsics.json`.
- **Not yet completed:** 100 target masks are not annotated. Model training, held-out evaluation, physical marker-size verification, ruler/calliper comparisons, failure-condition experiment, speed measurements and recording remain pending. No model or accuracy metrics are fabricated.

## Decision record
At scaffold time the approach above is selected; no alternative has yet been empirically evaluated. Update this section if the actual work changes the plan, and record what the change costs.
