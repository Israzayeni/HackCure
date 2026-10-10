# ai/detection — M2 (Imene)

**Branch:** `ai/detection`

**Deliverables:**
- `detect_cells(image) -> [{"x", "y", "w", "h", "score"}]` (trained YOLOv8 detector)
- `compute_result(cells, rectangles, dilution) -> result JSON`: applies the border rule, counts, and computes the concentration
- the counting error measured against `golden_set/`

Signatures and docstrings: [`contracts/interfaces.py`](../../contracts/interfaces.py).
Output schemas: [`contracts/detections.schema.json`](../../contracts/detections.schema.json),
[`contracts/result.schema.json`](../../contracts/result.schema.json).

**Until H7:** use the hand-written rectangles in `fakes/json/<name>.preprocess.json` in place of M1's output.
`fakes/json/<name>.detections.json` holds the ground-truth boxes for the drawn cells, and
`<name>.result.json` holds the expected `compute_result` output, so you can unit-test the border rule against it.

**At integration:** `detect_cells` replaces `fakes.stubs.fake_detect_cells` and `compute_result` replaces
`fakes.stubs.fake_compute_result`.
