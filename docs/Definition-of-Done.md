# Definition of Done

Evidence must be recorded beside each item. Empty means not done; do not tick without evidence.

- [ ] Camera calibration uses >=20 captured checkerboard/ChArUco views; reprojection error and parameter file recorded. Evidence: 20 views, RMS 1.565 px, mean per-view 1.150 px in `calibration/intrinsics.json`.
- [x] >=70 own images captured with the calibrated camera, undistorted, and labelled; split list committed before training. Evidence: 100 raw images and undistorted copies; split committed as `975d786`; 70 train, 20 validation and 10 test masks present. `scripts/check_labels.py` reports all masks nonempty and size-matched; contour quality still needs human review.
- [ ] Trained non-Ultralytics, non-Roboflow segmentation model evaluated only on held-out test data (IoU, mAP@0.5, mAP@0.5:0.95, precision, recall). Evidence:
- [ ] One command accepts a new image and emits overlay, width/height in mm and confidence; object/reference missing cases handled. Evidence:
- [ ] >=10 instances compared with calliper; per-instance errors, mean/median/worst absolute error and mean percentage error reported. Evidence:
- [ ] Raw vs undistorted errors compared on same instances. Evidence:
- [ ] One measurable speed or accuracy objective evaluated (timings use 5 runs, median and spread). Evidence:
- [ ] One chosen failure condition tested at increasing severity. Evidence:
- [ ] Refusal threshold and refused-image cost documented. Evidence:
- [ ] <=5-minute Loom answers all four questions and shows the submitted code. Link:
- [ ] Public repository branch `dev-test01`, staged commits, all three documents, README, model weights/hash, and PR are submitted. Link:
- [ ] Incomplete work and limitations listed honestly. Evidence: local training is blocked because Windows Application Control blocks PyTorch's `shm.dll`; the Colab notebook and data bundle are prepared. Model, final test metrics, accuracy evidence, Loom and PR are still outstanding.
