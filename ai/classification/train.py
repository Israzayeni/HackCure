"""Fine-tune the normal / abnormal cell classifier (M3).

Expects an ImageFolder dataset with train/ and valid/ splits, e.g. the Roboflow Leukemia-Cells
export:  <data>/train/{effected-cells,normal-cells}/*.jpg,  <data>/valid/...,  <data>/test/...
Folder names are mapped to contract labels with transforms.FOLDER_TO_LABEL. The test split is
never read here: model selection uses the validation split only.

Two phases: the new head alone, then the backbone too (--unfreeze last: only its last stage,
e.g. layer4; --unfreeze all: the whole network). After every epoch the checkpoint with the best
validation --select-metric is written to <out>/best.pt, and the current one to <out>/last.pt, so
a GPU timeout loses at most one epoch. <out> also gets config.json, history.json (losses and
validation metrics per epoch), curves.png and summary.json (best epoch and its validation metrics).

The defaults reproduce the original recipe. experiments.py runs the comparison grid.

Colab / Kaggle:
    !python train.py --data /content/Leukemia-Cells-1 --out /content/drive/MyDrive/hackcure
Local smoke test on the fake crops (code check only, never report these metrics):
    python ai/classification/train.py --data fakes/crops --train-split . --val-split . --epochs 1 --head-epochs 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import classification_metrics  # noqa: E402
from model import ARCHS, build_model, head_parameters, save_checkpoint, set_trainable  # noqa: E402
from transforms import AUGMENTATIONS, FOLDER_TO_LABEL, eval_transform, train_transform  # noqa: E402

SELECT_METRICS = ("f1_abnormal", "f1_macro", "f1_normal", "roc_auc")
HISTORY_KEYS = ("accuracy", "f1_macro", "f1_normal", "f1_abnormal", "precision_normal", "recall_normal",
                "precision_abnormal", "recall_abnormal", "roc_auc")


def idx_to_label_of(ds: ImageFolder) -> dict[int, str]:
    unknown = [c for c in ds.classes if c not in FOLDER_TO_LABEL]
    if unknown:
        raise SystemExit(f"Folders {unknown} have no label. Add them to FOLDER_TO_LABEL in transforms.py.")
    mapping = {i: FOLDER_TO_LABEL[c] for c, i in ds.class_to_idx.items()}
    if sorted(mapping.values()) != ["abnormal", "normal"]:
        raise SystemExit(f"Need exactly one 'normal' and one 'abnormal' folder, got {mapping}")
    return mapping


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def predict(model, loader, device, abnormal_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(p_abnormal, y_true_is_abnormal, mean unweighted cross-entropy) over the loader."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    probs, ys, loss_sum = [], [], 0.0
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += ce(logits, y.to(device)).item()
        probs.append(torch.softmax(logits, dim=1)[:, abnormal_idx].cpu())
        ys.append(y)
    y = torch.cat(ys).numpy()
    return torch.cat(probs).numpy(), y == abnormal_idx, loss_sum / max(len(y), 1)


def plot_curves(history: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))
    # The train loss includes class weights / label smoothing; the val loss is plain cross-entropy.
    ax_loss.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax_loss.plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    ax_loss.set_title("loss")
    for key in ("f1_macro", "f1_normal", "f1_abnormal"):
        ax_f1.plot(ep, [h[key] for h in history], "o-", label=key)
    ax_f1.set_title("validation F1 (threshold 0.5)")
    for ax in (ax_loss, ax_f1):
        heads = [h["epoch"] for h in history if h["phase"] == "head"]
        if heads and len(heads) < len(history):
            ax.axvline(max(heads) + 0.5, color="grey", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset root containing the split folders")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="valid")
    p.add_argument("--out", default="ai/classification/weights")
    p.add_argument("--arch", default="resnet18", choices=ARCHS)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--aug", default="strong", choices=AUGMENTATIONS, help="training augmentations")
    p.add_argument("--head-epochs", type=int, default=2, help="phase 1: train the new head only")
    p.add_argument("--epochs", type=int, default=8, help="phase 2: fine-tune the backbone too")
    p.add_argument("--unfreeze", default="all", choices=("last", "all"),
                   help="phase 2: last stage of the backbone (layer4 for ResNets) or the whole network")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3, help="phase 1 learning rate")
    p.add_argument("--lr", type=float, default=1e-4, help="phase 2 learning rate of the backbone")
    p.add_argument("--head-lr-mult", type=float, default=1.0,
                   help="phase 2 learning rate of the head = lr * this (e.g. 10)")
    p.add_argument("--optimizer", default="adam", choices=("adam", "adamw"))
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="cosine", choices=("cosine", "plateau", "none"))
    p.add_argument("--balance", default="loss", choices=("loss", "sampler", "none"),
                   help="class imbalance: weighted loss, balanced sampler, or nothing (never both)")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--select-metric", default="f1_abnormal", choices=SELECT_METRICS,
                   help="validation metric that picks best.pt (threshold 0.5)")
    p.add_argument("--patience", type=int, default=0,
                   help="stop a phase after this many epochs without a better best.pt (0: never)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = Path(args.data)
    train_ds = ImageFolder(data / args.train_split, transform=train_transform(args.img_size, args.aug))
    val_ds = ImageFolder(data / args.val_split, transform=eval_transform(args.img_size))
    idx_to_label = idx_to_label_of(train_ds)
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise SystemExit(f"train and val folders differ: {train_ds.class_to_idx} vs {val_ds.class_to_idx}")
    abnormal_idx = next(i for i, lab in idx_to_label.items() if lab == "abnormal")

    targets = torch.tensor(train_ds.targets)
    counts = torch.bincount(targets, minlength=2).float()
    val_counts = torch.bincount(torch.tensor(val_ds.targets), minlength=2)
    print(f"classes: {idx_to_label}  train counts: {counts.tolist()}  val counts: {val_counts.tolist()}")
    # Class imbalance, one method only: weighting the loss and balancing the batches would correct twice.
    class_weights = (counts.sum() / (2 * counts.clamp(min=1))).to(device) if args.balance == "loss" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    pin = device == "cuda"
    gen = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(num_workers=args.workers, pin_memory=pin, worker_init_fn=seed_worker)
    if args.balance == "sampler":
        sampler = WeightedRandomSampler((1.0 / counts.clamp(min=1))[targets].double(), len(train_ds),
                                        replacement=True, generator=gen)
        train_dl = DataLoader(train_ds, args.batch_size, sampler=sampler, **loader_kw)
    else:
        train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, generator=gen, **loader_kw)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, **loader_kw)

    model = build_model(args.arch, pretrained=True).to(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "idx_to_label": idx_to_label, "train_counts": counts.int().tolist(),
              "val_counts": val_counts.tolist(), "device": device, "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    history: list[dict] = []
    best_score, best_record = -float("inf"), None
    t_start = time.time()
    phases = [("head", args.head_epochs, args.lr_head, "head"), ("full", args.epochs, args.lr, args.unfreeze)]
    epoch = 0
    for phase, n_epochs, lr, mode in phases:
        if n_epochs == 0:
            continue
        set_trainable(model, args.arch, mode)
        head = list(head_parameters(model, args.arch))
        head_ids = {id(prm) for prm in head}
        backbone = [prm for prm in model.parameters() if prm.requires_grad and id(prm) not in head_ids]
        groups = [{"params": head, "lr": lr if mode == "head" else lr * args.head_lr_mult}]
        if backbone:
            groups.append({"params": backbone, "lr": lr})
        opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
        opt = opt_cls(groups, weight_decay=args.weight_decay)
        if args.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        elif args.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
        else:
            sched = None

        since_best = 0
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            lrs = [g["lr"] for g in opt.param_groups]
            model.train()
            running, seen = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * len(y)
                seen += len(y)

            p_abn, y_abn, val_loss = predict(model, val_dl, device, abnormal_idx)
            m = classification_metrics(y_abn, p_abn, threshold=0.5)
            score = m[args.select_metric] if m[args.select_metric] is not None else -1.0
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(score)
            elif sched is not None:
                sched.step()

            record = {"epoch": epoch, "phase": phase, "lr": lrs, "train_loss": round(running / max(seen, 1), 4),
                      "val_loss": round(val_loss, 4), **{k: m[k] for k in HISTORY_KEYS},
                      "confusion_matrix": m["confusion_matrix"]["values"], "seconds": round(time.time() - t0, 1)}
            history.append(record)
            (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

            extra = {"epoch": epoch, "val_f1_abnormal": m["f1_abnormal"], "val_acc": m["accuracy"],
                     "val_metrics": m, "config": config}
            save_checkpoint(out / "last.pt", model, args.arch, args.img_size, idx_to_label, **extra)
            tag = ""
            if score > best_score:
                best_score, best_record, since_best = score, record, 0
                save_checkpoint(out / "best.pt", model, args.arch, args.img_size, idx_to_label, **extra)
                tag = "  <- best, saved"
            else:
                since_best += 1
            print(f"[{phase}] epoch {epoch}  loss {record['train_loss']:.4f}  val loss {val_loss:.4f}  "
                  f"acc {m['accuracy']:.3f}  F1 macro {m['f1_macro']:.3f}  "
                  f"F1 normal {m['f1_normal']:.3f}  F1 abn {m['f1_abnormal']:.3f}  "
                  f"({record['seconds']:.0f}s){tag}")
            if args.patience and since_best >= args.patience:
                print(f"early stop: no better {args.select_metric} for {args.patience} epochs")
                break

    if best_record is None:
        raise SystemExit("no epoch was run (--head-epochs and --epochs are both 0)")
    summary = {"experiment": out.name, "select_metric": args.select_metric, "best_epoch": best_record["epoch"],
               "epochs_run": epoch, "minutes": round((time.time() - t_start) / 60, 1),
               "best_val": best_record, "config": config}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_curves(history, out / "curves.png")
    print(f"done. best val {args.select_metric} = {best_score:.3f} (epoch {best_record['epoch']})"
          f"  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
