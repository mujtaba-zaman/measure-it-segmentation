# Fresh marker measurement run

The current photo batch is `measurement/marker_photos/fresh_batch/`. It contains 10 photos: `img_001`–`img_004` show the iPhone X target; `img_005`–`img_010` show a Tecno-branded phone. The outside marker edge is 50 mm.

The model returned an estimate for every frame in raw and undistorted modes (20/20 runs). The marker square was detected in all 10 photos. Per-frame estimates, confidence scores and scale factors are in `measurement/marker_results/fresh_batch/measurement_results.csv`; marker detections are in `measurement/marker_results/fresh_batch/marker_check.csv`.

| Device group | Frames | Undistorted mean (mm) | Raw mean (mm) |
|---|---|---:|---:|
| iPhone X | 001–004 | 73.04 × 146.91 | 73.83 × 149.46 |
| Tecno-branded phone | 005–010 | 81.68 × 175.98 | 85.14 × 177.21 |

For the iPhone X frames, undistorted estimates ranged from 72.64–73.30 mm wide and 143.18–152.34 mm long; raw estimates ranged from 73.43–74.57 mm wide and 143.75–155.38 mm long. Confidence scores across the batch ranged from 0.8723 to 0.9989.

These are model estimates, not validated dimensions. No calliper ground truth was collected, and confidence is a segmentation score rather than an accuracy interval. The devices are reported separately; do not treat the combined mean as a measurement of one object. Dimensional error, a controlled raw-versus-undistorted accuracy comparison, timing and a failure-severity sweep remain unmeasured.
