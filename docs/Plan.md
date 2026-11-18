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
- **Chosen approach:** OpenCV checkerboard calibration and undistortion; a fine-tuned torchvision Mask R-CNN with a ResNet-FPN backbone; a known-size printed square as the in-plane scale reference. The supplied marker is inverted relative to OpenCV's standard ArUco polarity, so inference falls back to its outer square contour.
- **Rejected approach:** Ultralytics YOLO/Roboflow segmentation models are prohibited by the brief. A simple threshold/contour-only object detector is also rejected because it does not meet the trained-segmentation requirement. Marker detection is limited to reference scale, not object segmentation.
- **Cost of rejection:** Mask R-CNN training and data preparation take longer and need Torch/Torchvision; the implementation may not reach useful accuracy within eight hours. The marker approach requires a printed ArUco marker coplanar with the object and is unsuitable if that cannot be arranged.
- **Current data status:** 20 unique checkerboard photos and 100 unique unmarked iPhone X target photos are present. The 100 target photos are assigned once to 70 train / 20 validation / 10 test using seed 42; split lists and a SHA-256 manifest were committed as `975d786` before training. Raw target photos are preserved under `dataset/raw_images/`; model copies are normalized to landscape and undistorted under `dataset/images/`. The current marker batch is `measurement/marker_photos/fresh_batch/`; frames 001–004 show the iPhone X and frames 005–010 show a Tecno-branded phone. No test images have been used for training or tuning.
- **Calibration completed:** all 20 checkerboard views were detected after orientation normalization. With the measured 20 mm square size, reprojection RMS is 1.565 px and mean per-view error is 1.150 px; parameters are in `calibration/intrinsics.json`.
- **Annotation completed:** 100 target masks were saved across the frozen split (70 train / 20 validation / 10 test). `scripts/check_labels.py` reports 70/70, 20/20 and 10/10 masks present, nonempty and size-matched. This structural check does not replace visual review of each contour.
- **Training completed in Colab:** Mask R-CNN trained for 10 epochs on a Tesla T4. Best validation pixel IoU was 0.9662; held-out test mean mask IoU was 0.9680, mAP50/mAP50:95 were 1.0. These segmentation results do not establish physical measurement accuracy.
- **Measurement run completed:** all 10 current marker photos produced raw and undistorted estimates (20/20 runs), and the 50 mm square was detected in all 10. Results are saved in `measurement/marker_results/fresh_batch/`. Four frames show the target iPhone X; six show a Tecno-branded phone, so the two device groups must not be combined. No calliper ground truth is available.
- **Additional checks:** a synthetic marker-occlusion sweep passed through 15% obstruction and refused from 20% onward; a confidence-threshold coverage table records the observed refusal cost. These checks do not validate dimensional accuracy.
- **Still pending:** calliper ground truth and dimensional-error statistics; ground-truth raw-vs-undistorted comparison; five-run latency measurement in Colab; Loom; GitHub authentication/push and PR/submission. No dimensional accuracy is claimed.

## Decision record
At scaffold time the approach above is selected; no alternative has yet been empirically evaluated. Update this section if the actual work changes the plan, and record what the change costs.
