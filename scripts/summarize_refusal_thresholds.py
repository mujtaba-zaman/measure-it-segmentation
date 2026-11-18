"""Replay confidence cutoffs against saved fresh-batch predictions.

This reports coverage/refusal trade-offs; it does not establish measurement
accuracy or select a threshold using the training/test split.
"""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "measurement/marker_results/fresh_batch/measurement_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "measurement/marker_results/refusal_threshold_sweep.csv")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    thresholds = (0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
    output = []
    for mode in ("undistorted", "raw"):
        mode_rows = [row for row in rows if row["mode"] == mode and row["status"] == "OK"]
        for threshold in thresholds:
            accepted = sum(float(row["confidence"]) >= threshold for row in mode_rows)
            refused = len(mode_rows) - accepted
            percent = 100.0 * refused / len(mode_rows) if mode_rows else 0.0
            output.append([f"{threshold:.2f}", mode, len(mode_rows), accepted, refused, f"{percent:.1f}"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence_threshold", "mode", "measured_runs", "accepted", "refused", "refused_percent"])
        writer.writerows(output)
    for row in output:
        print(f"threshold {row[0]} | {row[1]:12s} | accepted {row[3]}/{row[2]} | refused {row[4]} ({row[5]}%)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
