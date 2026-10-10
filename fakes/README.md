# fakes — work against these until H7

Everything here is generated, mutually consistent, and committed. You don't need to run anything to start.

| Path | What | Who uses it |
|---|---|---|
| `images/malassez_0{1,2,3}.png` | Synthetic 1024×1024 Malassez fields: 5×4 grid (20 rectangles of 150×120 px), round normal cells, darker elongated abnormal cells, some cells on grid lines | everyone |
| `images/malassez_blurry.png`, `images/malassez_dark.png` | Quality-rejection test images (blurred and darkened copies of `malassez_01`) | M1, M4 |
| `json/<name>.preprocess.json` | Rectangles exactly matching the drawn grid | M1 (expected output), M2 (input until H7) |
| `json/<name>.detections.json` | One box per drawn cell, in or out of the grid | M2 (expected output), M3 (crop source) |
| `json/<name>.classifications.json` | One label/score per detection, same order | M3 (expected output) |
| `json/<name>.result.json` | Full result JSON, with border rule applied and formula-consistent | M2, M4, M5 |
| `json/samples.json` | 5 cases-service records (`sample_detail` shape) covering pending, corrected, validated | M5 |
| `json/rejection_examples.json` | One quality rejection per reason | M4, M5, dashboard |
| `images/fake_cells.png`, `json/fake_cells.detections.json`, `json/fake_cells.labels.json`, `crops/`, `images/fake_cells_overlay.png` | Labelled cells and 224×224 crops in `ImageFolder` layout (see root README, "Fake data for M3"). Testing code only, never for training or metrics | M3 |
| `stubs.py` | `fake_preprocess`, `fake_detect_cells`, `fake_classify_cell`, `fake_compute_result`, `fake_analyze`: same signatures as `contracts/interfaces.py` | M4 (and anyone needing an upstream stage) |
| `mock_inference/` | FastAPI mock of the inference service, serving these JSONs | M5, dashboard |
| `generate_fake_cells.py` | Regenerates the `fake_cells` files and `crops/` (fixed seed) | whoever changes the M3 fakes |
| `generate_fakes.py` | Regenerates the `malassez_*` images and JSON, `samples.json`, `rejection_examples.json` (fixed seed, deterministic) | whoever changes the fakes |
| `validate_fakes.py` | Checks all fakes and contract examples against the schemas and the formula | everyone, before pushing a contract change |

```bash
pip install -r fakes/requirements.txt
python fakes/generate_fakes.py     # only if you changed the generator
python fakes/validate_fakes.py     # must print "0 failed"
```

Results use dilution 10 (`malassez_01`), 20 (`malassez_02`) and 5 (`malassez_03`), so dilution bugs show up.
In each image, 4 cells sit on the outer bottom/right lines (not counted), 4 on the outer top/left lines (counted)
and 6 on interior lines (counted once). Cells outside the grid are detected but not counted.
