# services/inference — M4 (Yasmine), FastAPI

**Branch:** `services/inference`

**Deliverable:** a FastAPI service exposing `POST /images`, `POST /analyze/{image_id}` and `GET /health`, which
chains M1 → M2 → M3. Contracts: see the "Inference service" section of [`contracts/README.md`](../../contracts/README.md).

**Until H7:** call `fakes.stubs.fake_analyze` (which chains the four fake stage functions) in your endpoint code.
`fakes/mock_inference/` is the reference for the HTTP shapes: your service must return exactly what it returns.

**At integration:** swap the imports from `fakes.stubs` to the members' real modules. Call sites stay the same.
