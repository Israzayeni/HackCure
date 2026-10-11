"""Run the M3 comparison grid with train.py, one experiment after another, then compare them on the
validation split. The test split is never read here.

    python ai/classification/experiments.py --data /content/Leukemia-Cells-1 --out /content/drive/MyDrive/hackcure/exp
    python ai/classification/experiments.py --data ... --out ... --only ft_all ft_all_sampler
    python ai/classification/experiments.py --data ... --out ... --seeds 0 1 2 --only ft_all ft_layer4
    python ai/classification/experiments.py --out ... --summary-only

Each experiment writes <out>/<name>/{best.pt, last.pt, config.json, history.json, curves.png, summary.json}.
An experiment whose summary.json exists is skipped, so after a Colab timeout rerun the same command.
All experiments pick best.pt on validation F1 macro, decided before running them. The comparison is
written to <out>/comparison.csv and <out>/comparison.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parent / "train.py"
COMMON = ["--select-metric", "f1_macro"]
# Reference fine-tuning recipe: head first, then the whole backbone with a 10x smaller learning rate
# than the head, AdamW, cosine schedule, early stopping. Each "ft_*" variant changes ONE thing.
REF = ["--head-epochs", "3", "--epochs", "12", "--unfreeze", "all", "--optimizer", "adamw",
       "--weight-decay", "1e-4", "--lr", "1e-4", "--head-lr-mult", "10", "--scheduler", "cosine",
       "--patience", "4"]
EXPERIMENTS = {   # later arguments override earlier ones
    "baseline_frozen": ["--head-epochs", "5", "--epochs", "0"],   # frozen backbone, head only
    "repo_default": [],                                            # train.py defaults (2 + 8 epochs, Adam)
    "ft_layer4": REF + ["--unfreeze", "last"],
    "ft_all": REF,
    "ft_all_lr3e-5": REF + ["--lr", "3e-5"],
    "ft_all_moderate_aug": REF + ["--aug", "moderate"],
    "ft_all_no_balance": REF + ["--balance", "none"],
    "ft_all_sampler": REF + ["--balance", "sampler"],
    "ft_all_label_smoothing": REF + ["--label-smoothing", "0.1"],
    "resnet34": REF + ["--arch", "resnet34"],
    "efficientnet_b0": REF + ["--arch", "efficientnet_b0"],
    "mobilenet_v3_large": REF + ["--arch", "mobilenet_v3_large"],
}
COLUMNS = ["experiment", "arch", "best_epoch", "epochs_run", "minutes", "val_loss", "accuracy", "f1_macro",
           "f1_normal", "f1_abnormal", "recall_normal", "recall_abnormal", "roc_auc"]


def summarize(root: Path) -> None:
    rows = []
    for f in sorted(root.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"experiment": f.parent.name, "arch": s["config"]["arch"], "best_epoch": s["best_epoch"],
                     "epochs_run": s["epochs_run"], "minutes": s["minutes"],
                     **{k: s["best_val"][k] for k in COLUMNS[5:]}})
    if not rows:
        print(f"no summary.json under {root}")
        return
    rows.sort(key=lambda r: r["f1_macro"], reverse=True)
    with open(root / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    table = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    table += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
    (root / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\nValidation split, threshold 0.5, best epoch of each experiment (sorted by F1 macro):\n")
    print("\n".join(table))
    best = rows[0]["experiment"]
    print(f"\nbest on validation: {best}. Next, on the validation split only, choose the threshold:\n"
          f"  python ai/classification/evaluate.py --weights {root / best / 'best.pt'} "
          f"--data <data>/valid --out {root / best / 'val_metrics.json'} --sweep\n"
          "then evaluate that model ONCE on test with the chosen --threshold.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="dataset root with train/ and valid/")
    p.add_argument("--out", required=True, help="one sub-folder per experiment is created here")
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="run only these experiments")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--workers", default="2")
    p.add_argument("--summary-only", action="store_true", help="only rebuild the comparison table")
    args = p.parse_args()

    root = Path(args.out)
    if not args.summary_only:
        if not args.data:
            p.error("--data is required unless --summary-only")
        root.mkdir(parents=True, exist_ok=True)
        failed = []
        for seed in args.seeds:
            for name in args.only or list(EXPERIMENTS):
                run = name if len(args.seeds) == 1 else f"{name}_s{seed}"
                if (root / run / "summary.json").exists():
                    print(f"skip {run}: already done")
                    continue
                cmd = [sys.executable, str(TRAIN), "--data", args.data, "--out", str(root / run), *COMMON,
                       *EXPERIMENTS[name], "--seed", str(seed), "--workers", args.workers]
                print(f"\n=== {run} ===\n{' '.join(cmd)}", flush=True)
                if subprocess.run(cmd).returncode != 0:
                    failed.append(run)
        if failed:
            print(f"\nFAILED: {failed}")
    summarize(root)


if __name__ == "__main__":
    main()
