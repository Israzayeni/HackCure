# ai/common — shared helpers for the AI stages

Changes go through a PR reviewed by M2, M3 and M4, because the pipeline depends on them.

- `crop.py`:
  - `crop_cell(image, cell, margin=0.25, size=224)`: square, replicate-padded crop of one detection, resized.
  - `export_crops(image, cells, labels, out_dir)`: writes an `ImageFolder` layout plus `labels.csv`.
  - `draw_labeled_overlay(image, cells, labels)`: draws boxes, green = normal, red = abnormal.

Import it from the repo root: `from ai.common.crop import crop_cell`.
