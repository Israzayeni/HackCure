# contracts — the source of truth

Everything the five members exchange is defined here. **Never change a file in this folder without a PR
to `/contracts` and an announcement to the whole team.** `fakes/validate_fakes.py` checks every fake and every
example against these schemas.

- `*.schema.json`: JSON Schema, draft 2020-12. Schemas reference each other by relative `$ref`.
- `examples/<schema>.example.json`: one valid payload per schema (two for `post_images_response`).
- `interfaces.py`: Python signatures and docstrings of the four AI pipeline functions.
- `db_schema.sql`: PostgreSQL tables of the cases service.

Items marked **PROPOSED** fill a gap in the original spec. They are collected in the root README's
"Open questions" section until the team confirms them.

## AI pipeline (Python, in-process)

| Function | Owner | Output schema |
|---|---|---|
| `preprocess(image) -> {"image", "rectangles"}` | M1 | `preprocess_output.schema.json` (`rectangle.schema.json`) |
| `detect_cells(image) -> [detection]` | M2 | `detections.schema.json` (`detection.schema.json`) |
| `classify_cell(crop) -> {"label", "score"}` | M3 | `classification.schema.json`; per image: `classifications.schema.json` |
| `compute_result(cells, rectangles, dilution) -> result` | M2 | `result.schema.json`; input `cells`: `classified_cells.schema.json` |

Formula: `concentration_per_uL = total_count / rectangles_counted × 100 × dilution_factor` (Malassez: 100 rectangles = 1 µL).
`result.cells` contains only the cells counted after the border rule, so `total_count == len(cells)`. The example
in the original spec (1 cell, total 412) was abbreviated.

## Inference service (FastAPI, M4)

| Method & path | Request | 200 response | Errors |
|---|---|---|---|
| `POST /images` | multipart/form-data, field `file` (PROPOSED) | `post_images_response.schema.json`: `{"image_id"}` or a quality rejection `{"accepted": false, "reason", "detail"}` | 400 `invalid_request` |
| `POST /analyze/{image_id}` | `analyze_request.schema.json`: `{"dilution_factor"}` (PROPOSED) | `result.schema.json` | 404 `image_not_found`, 422 `invalid_request`, 500 `analysis_failed`, 503 `models_not_loaded` |
| `GET /health` | — | `health.schema.json` (PROPOSED) | — |

## Cases service (ASP.NET Core, M5)

| Method & path | Request | Response | Errors |
|---|---|---|---|
| `POST /samples` | `create_sample_request.schema.json` | 201, `sample.schema.json` (status `pending`) | 422 `invalid_request` |
| `GET /samples` | — | 200, `sample_list.schema.json` | — |
| `GET /samples/{id}` (PROPOSED, not in spec) | — | 200, `sample_detail.schema.json` | 404 `sample_not_found` |
| `POST /samples/{id}/result` | `post_result_request.schema.json` | 201, `sample_detail.schema.json` | 404 `sample_not_found`, 409 `invalid_status` (sample already validated), 422 `invalid_request` |
| `PATCH /samples/{id}/correct` | `correct_request.schema.json` | 200, `sample_detail.schema.json` (status `corrected`) | 404, 409 `no_result` / `invalid_status`, 422 |
| `POST /samples/{id}/validate` | `validate_request.schema.json` | 200, `sample_detail.schema.json` (status `validated`) | 404, 409 `no_result` / `invalid_status` |
| `GET /samples/{id}/report` | — | 200, `application/pdf` | 404, 409 `not_validated` |

Status flow: `pending → corrected → validated`, and `pending → validated` directly. `corrected → corrected` (a
re-correction) is allowed (PROPOSED). Nothing leaves `validated`.

The cases API uses the DB column names (`concentration`, `cells` for `cells_json`). The inference API uses
`concentration_per_uL`. M5 maps between them when storing a result.

## Errors (both services, PROPOSED)

Every 4xx/5xx body is `error.schema.json`: `{"error": {"code": "...", "message": "..."}}`. A failed analysis returns
this body, never a partial result. Known codes: `invalid_request`, `image_not_found`, `analysis_failed`,
`models_not_loaded`, `sample_not_found`, `invalid_status`, `no_result`, `not_validated`. Add new codes through a contract PR.
