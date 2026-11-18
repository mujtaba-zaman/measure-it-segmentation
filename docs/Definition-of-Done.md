# Definition of Done

Evidence must be recorded beside each item. Empty means not done; do not tick without evidence.

- [x] Camera calibration uses >=20 captured checkerboard/ChArUco views; reprojection error and parameter file recorded. Evidence: 20 views, RMS 1.565 px, mean per-view 1.150 px in `calibration/intrinsics.json`.
- [x] >=70 own images captured with the calibrated camera, undistorted, and labelled; split list committed before training. Evidence: 100 raw images and undistorted copies; split committed as `975d786`; 70 train, 20 validation and 10 test masks present. `scripts/check_labels.py` reports all masks nonempty and size-matched; contour quality still needs human review.
- [x] Trained non-Ultralytics, non-Roboflow segmentation model evaluated only on held-out test data (IoU, mAP@0.5, mAP@0.5:0.95, precision, recall). Evidence: 10 epochs on Colab Tesla T4; test mean mask IoU 0.9680, precision/recall@IoU50 1.0, mAP50 1.0 and mAP50:95 1.0 in `measurement/test_evaluation/metrics.json`. These are segmentation scores, not dimensional accuracy.
- [x] One command accepts a new image and emits overlay, width/height in mm and confidence; object/reference missing cases handled. Evidence: the fresh ten-photo batch produced estimates in raw and undistorted modes for all 10 images (20/20 runs); marker-square detection passed 10/10. CSVs are in `measurement/marker_results/fresh_batch/`.
- [ ] >=10 instances compared with calliper; per-instance errors, mean/median/worst absolute error and mean percentage error reported. Evidence:
- [ ] Raw vs undistorted errors compared on same instances. Evidence:
- [ ] One measurable speed or accuracy objective evaluated (timings use 5 runs, median and spread). Evidence:
- [ ] One chosen failure condition tested at increasing severity. Evidence: marker occlusion caused six refusals; no controlled severity sweep was performed.
- [ ] Refusal threshold and refused-image cost documented. Evidence:
- [ ] <=5-minute Loom answers all four questions and shows the submitted code. Link:
- [ ] Public repository branch `dev-test01`, staged commits, all three documents, README, model weights/hash, and PR are submitted. Link:
- [x] Incomplete work and limitations listed honestly. Evidence: no calliper ground truth; only four photos show the iPhone X while six show a Tecno-branded phone. Dimensional error statistics, controlled raw-vs-undistorted error comparison, timing, failure severity sweep, Loom and PR remain outstanding.
