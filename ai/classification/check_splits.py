"""Look for images shared between the dataset splits: exact copies, and near copies up to a flip or a
rotation by 90 degrees (Roboflow can export augmented copies). A leak makes the validation and test
scores optimistic.

    python ai/classification/check_splits.py --data /content/Leukemia-Cells-1

Each image gets a 256-bit average hash (16x16 grayscale > mean) in its 8 flip/rotation variants. A
pair at most --max-distance bits apart is reported. Look at the reported pairs before removing anything:
similar-looking cells can be distinct images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms import FOLDER_TO_LABEL  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def variant_hashes(path: Path) -> np.ndarray:
    """(8, 32) packed hashes: rotations by 0/90/180/270 degrees, each also mirrored."""
    a = np.asarray(Image.open(path).convert("L").resize((16, 16), Image.BILINEAR), dtype=float)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        for v in (r, np.fliplr(r)):
            out.append(np.packbits((v > v.mean()).ravel()))
    return np.stack(out)


def load_split(folder: Path) -> list[dict]:
    items = []
    for f in sorted(folder.glob("*/*")):
        if f.suffix.lower() in IMAGE_EXT:
            items.append({"path": f, "label": FOLDER_TO_LABEL.get(f.parent.name, f.parent.name),
                          "md5": hashlib.md5(f.read_bytes()).hexdigest(), "hashes": variant_hashes(f)})
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset root containing the split folders")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--max-distance", type=int, default=8, help="bits out of 256")
    p.add_argument("--out", help="optional CSV of the reported pairs")
    args = p.parse_args()

    splits = {s: load_split(Path(args.data) / s) for s in args.splits}
    for s, items in splits.items():
        print(f"{s}: {len(items)} images")
    pairs = []
    for sa, sb in combinations(args.splits, 2):
        a, b = splits[sa], splits[sb]
        if not a or not b:
            continue
        base = np.stack([it["hashes"][0] for it in a])          # (N_a, 32)
        md5_a = {it["md5"]: it for it in a}
        n_exact = n_near = 0
        for it in b:
            if it["md5"] in md5_a:
                n_exact += 1
                pairs.append((sb, it, sa, md5_a[it["md5"]], "exact"))
                continue
            dist = POPCOUNT[np.bitwise_xor(it["hashes"][:, None, :], base[None])].sum(-1).min(0)   # (N_a,)
            j = int(dist.argmin())
            if dist[j] <= args.max_distance:
                n_near += 1
                pairs.append((sb, it, sa, a[j], int(dist[j])))
        print(f"{sb} vs {sa}: {n_exact} exact copies, {n_near} near copies (<= {args.max_distance} bits)")

    for sb, it, sa, other, d in pairs[:30]:
        flag = "" if it["label"] == other["label"] else "   LABEL DIFFERS"
        print(f"  [{d}] {sb}/{it['path'].parent.name}/{it['path'].name}  ~  "
              f"{sa}/{other['path'].parent.name}/{other['path'].name}{flag}")
    if len(pairs) > 30:
        print(f"  ... {len(pairs) - 30} more" + (f", see {args.out}" if args.out else ", use --out to list all"))
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "path", "other_split", "other_path", "distance", "same_label"])
            for sb, it, sa, other, d in pairs:
                w.writerow([sb, it["path"], sa, other["path"], d, it["label"] == other["label"]])
    if not pairs:
        print("no leak found between the splits")


if __name__ == "__main__":
    main()
