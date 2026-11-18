# Objectives

Set targets before measuring. These are initial working targets and must be revised only with a reason before looking at test results.

## MO-1 — Measurement accuracy (required)
- **What is measured:** Absolute width/height error for at least 10 held-out object instances compared with calliper ground truth.
- **How:** Save one row per instance and dimension in `measurement/accuracy.csv`; compute MAE, median and worst-case absolute error (mm) and mean absolute percentage error.
- **Target:** MAE <= 2.0 mm for both width and height. Rationale: a useful first target for a phone-camera prototype; adjust if object size/use case makes it inappropriate, before inspecting test results.
- **Conditions:** Fixed camera, focus, zoom, resolution; rigid flat object and reference in the same plane; calibrated camera.
- **Not included:** Occluded objects, non-planar scenes and images where confidence gates refuse a result.
- **Status/evidence:** Not validated. The current 10-photo run yielded 20/20 raw/undistorted estimates; only frames 001–004 show the target iPhone X, while 005–010 show a Tecno-branded phone. No calliper ground truth is available. Raw results: `measurement/marker_results/fresh_batch/measurement_results.csv`. Do not present estimates as accuracy evidence.

## MO-2 — Inference speed
- **What is measured:** End-to-end inference time on one fixed-resolution image.
- **How:** Five timed runs on the declared machine; report median and range/spread.
- **Target:** <= 2.0 seconds median on the declared hardware. Rationale: interactive prototype response; revise if hardware makes this impractical.
- **Conditions:** Record CPU, GPU, RAM, OS, camera, resolution; same model/checkpoint and warm-up policy.
- **Not included:** Model training time and image capture time.
- **Status/evidence:** Not measured yet. `scripts/benchmark_inference.py` runs five cold end-to-end CLI measurements and saves raw runs plus median/range in `measurement/benchmark/`. The laptop TorchVision C++ operator currently fails to load, so execute this script in the existing Colab GPU runtime. Do not substitute the notebook's total cell duration for per-run latency.

## Refusal behavior — marker occlusion
- **Test:** On `img_001.jpeg`, add an opaque white strip across the marker's upper edge in increasing 5% steps of marker height and rerun the marker contour gate.
- **Result:** Detection passed at 0–15% synthetic occlusion and refused at 20–60%. Raw outcomes are in `measurement/marker_results/failure_sweep.csv`.
- **Cost / limitation:** The configured 0.75 mask-confidence cutoff yielded 0 refused images in this 10-photo batch, in either raw or undistorted mode. This is the observed coverage cost on this small batch only. The sweep probes marker detection, not the model's full failure behavior or measurement accuracy.

Report all measured test instances and raw outputs. Do not tune thresholds or checkpoints on the test split.
