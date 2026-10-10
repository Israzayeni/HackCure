# ai/preprocessing — M1 (Isra)

**Branch:** `ai/preprocessing`

**Deliverable:** `preprocess(image) -> {"image": np.ndarray, "rectangles": [{"id", "x", "y", "w", "h"}]}`
— a cleaned image plus the pixel coordinates of every Malassez rectangle to count.
Exact signature and docstring: [`contracts/interfaces.py`](../../contracts/interfaces.py).
JSON form: [`contracts/preprocess_output.schema.json`](../../contracts/preprocess_output.schema.json).

**Until H7:** develop against `fakes/images/malassez_0{1,2,3}.png`. The expected rectangles for
each image are in `fakes/json/<name>.preprocess.json`, so you can check your output against them.
`malassez_blurry.png` and `malassez_dark.png` exist to test quality rejection.

**At integration:** your `preprocess` replaces `fakes.stubs.fake_preprocess`.
