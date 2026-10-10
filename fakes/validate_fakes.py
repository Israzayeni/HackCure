"""Check every fake and every contract example against the contracts.

Run from the repo root:  python fakes/validate_fakes.py      (exit code 0 = all good)

Checks:
- every contracts/*.schema.json is a valid draft 2020-12 schema, and every schema has an example;
- every contracts/examples/*.example.json and every fakes/json/*.json validates against its schema;
- every result: total_count == len(cells), abnormal_count == number of 'abnormal' labels,
  concentration_per_uL == total_count / rectangles_counted * 100 * dilution_factor,
  confidence == mean cell score (PROPOSED definition);
- per fake image: detections and classifications line up, boxes and rectangles lie inside the image,
  the result's cells are detection centres with the matching labels, and fakes.stubs.fake_analyze
  reproduces <name>.result.json;
- samples.json covers every status, and each record is consistent with its status and dilution;
- rejection_examples.json has exactly one example per rejection reason.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fakes.stubs import IMAGES, fake_analyze, load_image  # noqa: E402

CONTRACTS = ROOT / "contracts"
FAKE_JSON = ROOT / "fakes" / "json"
BASE = "https://github.com/Israzayeni/HackCure/contracts/"

failures: list[str] = []
passed = 0


def check(ok: bool, what: str) -> None:
    global passed
    if ok:
        passed += 1
    else:
        failures.append(what)
        print(f"  FAIL {what}")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


schemas = {p.name: load(p) for p in sorted(CONTRACTS.glob("*.schema.json"))}
registry = Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in schemas.values()])


def validate(data, schema_ref: str, what: str) -> bool:
    """schema_ref is a schema file name, optionally with a #/$defs/... fragment."""
    validator = Draft202012Validator({"$ref": BASE + schema_ref}, registry=registry, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    for e in errors[:5]:
        print(f"  FAIL {what}: at {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:200]}")
    check(not errors, f"{what} matches {schema_ref}")
    return not errors


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, abs_tol=tol)


def check_result(r: dict, what: str, *, count="total_count", conc="concentration_per_uL", dilution=None) -> None:
    """Arithmetic consistency of a result JSON or a cases-service result record."""
    cells = r["cells"]
    dil = r["dilution_factor"] if dilution is None else dilution
    check(r[count] == len(cells), f"{what}: {count} ({r[count]}) == len(cells) ({len(cells)})")
    n_abn = sum(c["label"] == "abnormal" for c in cells)
    check(r["abnormal_count"] == n_abn, f"{what}: abnormal_count ({r['abnormal_count']}) == abnormal labels ({n_abn})")
    expected = r[count] / r["rectangles_counted"] * 100 * dil
    check(close(r[conc], expected, 0.01), f"{what}: {conc} ({r[conc]}) == {r[count]}/{r['rectangles_counted']}*100*{dil} ({expected:.2f})")
    mean = round(sum(c["score"] for c in cells) / len(cells), 2) if cells else 0
    check(close(r["confidence"], mean, 0.005), f"{what}: confidence ({r['confidence']}) == mean cell score ({mean}) [PROPOSED]")


_section = {"title": None, "start": 0, "fails": 0}


def section(title: str | None) -> None:
    """Start a new output section (None = finish the last one), printing the previous section's tally."""
    if _section["title"]:
        print(f"  {passed - _section['start']} passed, {len(failures) - _section['fails']} failed")
    if title:
        print(f"\n== {title}")
    _section.update(title=title, start=passed, fails=len(failures))


# ---------------------------------------------------------------- contracts
section("contracts: schemas are valid draft 2020-12")
for name, s in schemas.items():
    try:
        Draft202012Validator.check_schema(s)
        check(s.get("$id") == BASE + name, f"{name}: $id matches file name")
    except Exception as e:  # noqa: BLE001
        check(False, f"{name}: invalid schema: {e}")

section("contracts: examples")
examples = sorted((CONTRACTS / "examples").glob("*.example.json"))
covered = set()
for ex in examples:
    stem = ex.name.split(".")[0]
    covered.add(stem)
    if f"{stem}.schema.json" not in schemas:
        check(False, f"{ex.name}: no schema {stem}.schema.json")
        continue
    data = load(ex)
    if validate(data, f"{stem}.schema.json", f"examples/{ex.name}"):
        if stem == "result":
            check_result(data, ex.name)
        elif stem == "post_result_request":
            check_result(data["result"], ex.name)
        elif stem == "sample_detail" and data["result"]:
            check_result(data["result"], ex.name, conc="concentration", dilution=data["sample"]["dilution_factor"])
        elif stem == "analysis_result_record":
            check_result(data, ex.name, conc="concentration", dilution=10)  # dilution of sample.example.json
for name in schemas:
    check(name.removesuffix(".schema.json") in covered, f"{name} has an example")

# ---------------------------------------------------------------- per-image fakes
for name in IMAGES:
    section(f"fakes: {name}")
    img_path = ROOT / "fakes" / "images" / f"{name}.png"
    check(img_path.exists(), f"{img_path.name} exists")
    image = load_image(img_path)
    H, W = image.shape[:2]
    pre = load(FAKE_JSON / f"{name}.preprocess.json")
    dets = load(FAKE_JSON / f"{name}.detections.json")
    clss = load(FAKE_JSON / f"{name}.classifications.json")
    res = load(FAKE_JSON / f"{name}.result.json")
    ok = all([validate(pre, "preprocess_output.schema.json", f"{name}.preprocess.json"),
              validate(dets, "detections.schema.json", f"{name}.detections.json"),
              validate(clss, "classifications.schema.json", f"{name}.classifications.json"),
              validate(res, "result.schema.json", f"{name}.result.json")])
    if not ok:
        continue
    check((ROOT / pre["image"]).exists(), f"{name}.preprocess.json: image path {pre['image']} exists")
    rects = pre["rectangles"]
    check(len({r["id"] for r in rects}) == len(rects), f"{name}: rectangle ids unique")
    check(all(r["x"] + r["w"] <= W and r["y"] + r["h"] <= H for r in rects), f"{name}: rectangles inside image")
    check(all(d["x"] + d["w"] <= W and d["y"] + d["h"] <= H for d in dets), f"{name}: detections inside image")
    check(len(dets) == len(clss), f"{name}: one classification per detection ({len(clss)} vs {len(dets)})")
    check(res["rectangles_counted"] == len(rects), f"{name}: rectangles_counted == len(rectangles)")
    check_result(res, f"{name}.result.json")
    available = Counter((d["x"] + d["w"] // 2, d["y"] + d["h"] // 2, c["label"], c["score"]) for d, c in zip(dets, clss))
    used = Counter((c["x"], c["y"], c["label"], c["score"]) for c in res["cells"])
    check(not (used - available), f"{name}: every result cell is a detection centre with its classification")
    check(len(res["cells"]) < len(dets), f"{name}: border rule excluded some detections ({len(dets) - len(res['cells'])} excluded)")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fallback inside the stubs would be a bug here
        check(fake_analyze(image, res["dilution_factor"]) == res, f"{name}: fake_analyze reproduces result.json")

# ---------------------------------------------------------------- samples
section("fakes: samples.json")
samples = load(FAKE_JSON / "samples.json")
all_ok = True
for i, rec in enumerate(samples):
    all_ok &= validate(rec, "sample_detail.schema.json", f"samples.json[{i}]")
if all_ok:
    statuses = Counter(r["sample"]["status"] for r in samples)
    check(set(statuses) == {"pending", "corrected", "validated"}, f"samples.json covers every status {dict(statuses)}")
    check(len({r["sample"]["code"] for r in samples}) == len(samples), "samples.json: sample codes unique")
    for r in samples:
        s, v, res = r["sample"], r["validation"], r["result"]
        tag = f"{s['code']} ({s['status']})"
        check(all(img["sample_id"] == s["id"] for img in r["images"]), f"{tag}: images belong to the sample")
        if res:
            check(res["image_id"] in {img["id"] for img in r["images"] if img["quality_ok"]},
                  f"{tag}: result.image_id is an accepted image of the sample")
            check_result(res, tag, conc="concentration", dilution=s["dilution_factor"])
        if s["status"] == "pending":
            check(v is None, f"{tag}: pending sample has no validation row")
        else:
            check(res is not None and v is not None and v["sample_id"] == s["id"], f"{tag}: has a result and a validation row")
            if v:
                check((v["validated_at"] is not None) == (s["status"] == "validated"),
                      f"{tag}: validated_at set iff status is validated")
                if s["status"] == "corrected":
                    check(v["corrected_count"] is not None, f"{tag}: corrected sample has corrected_count")
                if v["corrected_count"] is not None:
                    check(v["corrected_abnormal"] is not None and v["corrected_abnormal"] <= v["corrected_count"],
                          f"{tag}: corrected_abnormal <= corrected_count")

# ---------------------------------------------------------------- rejections
section("fakes: rejection_examples.json")
rejections = load(FAKE_JSON / "rejection_examples.json")
for i, rej in enumerate(rejections):
    validate(rej, "post_images_response.schema.json#/$defs/rejected", f"rejection_examples.json[{i}]")
    validate(rej, "post_images_response.schema.json", f"rejection_examples.json[{i}] (full response)")
reasons = Counter(r.get("reason") for r in rejections)
check(reasons == Counter({"blurry": 1, "too_dark": 1, "overexposed": 1}), f"one example per rejection reason {dict(reasons)}")
for img in ("malassez_blurry.png", "malassez_dark.png"):
    check((ROOT / "fakes" / "images" / img).exists(), f"quality test image {img} exists")

# ---------------------------------------------------------------- summary
section(None)
print(f"\n{passed} checks passed, {len(failures)} failed")
sys.exit(1 if failures else 0)
