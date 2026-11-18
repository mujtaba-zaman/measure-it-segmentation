# Objectives

Set targets before measuring. These are initial working targets and must be revised only with a reason before looking at test results.

## MO-1 — Measurement accuracy (required)
- **What is measured:** Absolute width/height error for at least 10 held-out object instances compared with calliper ground truth.
- **How:** Save one row per instance and dimension in `measurement/accuracy.csv`; compute MAE, median and worst-case absolute error (mm) and mean absolute percentage error.
- **Target:** MAE <= 2.0 mm for both width and height. Rationale: a useful first target for a phone-camera prototype; adjust if object size/use case makes it inappropriate, before inspecting test results.
- **Conditions:** Fixed camera, focus, zoom, resolution; rigid flat object and reference in the same plane; calibrated camera.
- **Not included:** Occluded objects, non-planar scenes and images where confidence gates refuse a result.
- **Status/evidence:** Not validated. The saved marker run contains four successful frame estimates and six marker-detection refusals, but no calliper ground truth. Raw results: `measurement/marker_results/measurement_results.csv`. Do not report the estimates as accuracy evidence.

## MO-2 — Inference speed
- **What is measured:** End-to-end inference time on one fixed-resolution image.
- **How:** Five timed runs on the declared machine; report median and range/spread.
- **Target:** <= 2.0 seconds median on the declared hardware. Rationale: interactive prototype response; revise if hardware makes this impractical.
- **Conditions:** Record CPU, GPU, RAM, OS, camera, resolution; same model/checkpoint and warm-up policy.
- **Not included:** Model training time and image capture time.
- **Status/evidence:** Not measured; no five-run timing data was saved.

Report all measured test instances and raw outputs. Do not tune thresholds or checkpoints on the test split.
