# Marker measurement run

The saved Colab output is archived in `measurement_results.csv`; annotated overlays are beside it. The model and scale fallback ran on all ten uploaded filenames using the printed marker's reported 50 mm outer edge.

| Image | Undistorted estimate (mm) | Raw estimate (mm) |
|---|---:|---:|
| `img_001.jpeg` | 73.30 × 143.71 | 73.43 × 147.69 |
| `img_002.jpeg` | 73.05 × 143.18 | 73.87 × 143.75 |
| `img_003.jpeg` | 72.64 × 148.43 | 73.47 × 151.02 |
| `img_004.jpeg` | 73.19 × 152.34 | 74.57 × 155.38 |

Mean of the four successful frames: **73.04 × 146.91 mm undistorted**, **73.83 × 149.46 mm raw**. Six frames (`img_005.jpeg` through `img_010.jpeg`) were refused because the marker square could not be isolated. The saved set also contains an exact duplicate (`img_009.jpeg` / `img_010.jpeg`).

These are model estimates, not validated dimensional measurements: no calliper ground truth was recorded, and the marker was partly occluded in several frames. The detector's confidence is segmentation confidence; it is not a measurement-accuracy confidence interval. Do not claim dimensional MAE or accuracy from these results.
