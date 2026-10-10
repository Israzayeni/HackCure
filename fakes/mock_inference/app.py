"""Mock of the inference service (M4) that serves the /fakes JSONs. No AI, no image processing.

Lets M5 and the dashboard make real HTTP calls before H7. Shapes: contracts/README.md.

Run from the repo root:
    python -m uvicorn --app-dir fakes/mock_inference app:app --port 8001

Behaviour:
- POST /images decides from the uploaded file name only:
    *blurry*, *dark*, *overexposed*  -> quality rejection (fakes/json/rejection_examples.json)
    malassez_01/02/03.png            -> image_id "img_malassez_0N"
    *fail*                           -> accepted, but /analyze on it returns 500 analysis_failed
    anything else                    -> new image_id served with the malassez_01 result
- img_malassez_01/02/03 are known at startup, so /analyze works without uploading first.
- POST /analyze/{image_id} returns <name>.result.json, with the concentration recomputed for the requested dilution.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

FAKE_JSON = Path(__file__).resolve().parent.parent / "json"
FAKE_IMAGES = ["malassez_01", "malassez_02", "malassez_03"]
FAIL = "__fail__"

app = FastAPI(title="HackCure mock inference", version="fake-0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

images: dict[str, str] = {f"img_{n}": n for n in FAKE_IMAGES}   # image_id -> fake name (or FAIL)
_upload_ids = itertools.count(1)


def _load(name: str):
    return json.loads((FAKE_JSON / name).read_text(encoding="utf-8"))


REJECTIONS = {r["reason"]: r for r in _load("rejection_examples.json")}


class AnalyzeRequest(BaseModel):
    """contracts/analyze_request.schema.json"""
    model_config = ConfigDict(extra="forbid")
    dilution_factor: float = Field(gt=0)


def error(status: int, code: str, message: str) -> JSONResponse:
    """contracts/error.schema.json"""
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    code = "not_found" if exc.status_code == 404 else "invalid_request"
    return error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []))
    return error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid request')}")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/images")
async def post_images(file: UploadFile = File(...)):
    stem = Path(file.filename or "").stem.lower()
    for key, reason in (("blurry", "blurry"), ("dark", "too_dark"), ("overexposed", "overexposed")):
        if key in stem:
            return REJECTIONS[reason]
    if stem in FAKE_IMAGES:
        return {"image_id": f"img_{stem}"}
    image_id = f"img_upload_{next(_upload_ids):04d}"
    images[image_id] = FAIL if "fail" in stem else FAKE_IMAGES[0]
    return {"image_id": image_id}


@app.post("/analyze/{image_id}")
def analyze(image_id: str, body: AnalyzeRequest):
    name = images.get(image_id)
    if name is None:
        return error(404, "image_not_found", f"No image with id {image_id}.")
    if name == FAIL:
        return error(500, "analysis_failed", "Simulated failure: no grid found (mock).")
    result = _load(f"{name}.result.json")
    result["dilution_factor"] = body.dilution_factor
    result["concentration_per_uL"] = round(
        result["total_count"] / result["rectangles_counted"] * 100 * body.dilution_factor, 2)
    return result
