"""Fine-tune the normal / abnormal cell classifier (M3).

Expects an ImageFolder dataset with train/ and valid/ splits, e.g. the Roboflow Leukemia-Cells
export:  <data>/train/{effected-cells,normal-cells}/*.jpg,  <data>/valid/...,  <data>/test/...
Folder names are mapped to contract labels with transforms.FOLDER_TO_LABEL.

Two phases: the new head alone, then the whole network. After every epoch the checkpoint with
the best validation F1 on the abnormal class is written to <out>/best.pt (and <out>/last.pt),
so a GPU timeout loses at most one epoch.

Colab / Kaggle:
    !python train.py --data /content/Leukemia-Cells-1 --out /content/drive/MyDrive/hackcure
Local smoke test on the fake crops (code check only, never report these metrics):
    python ai/classification/train.py --data fakes/crops --train-split . --val-split . --epochs 1 --head-epochs 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model, head_parameters, save_checkpoint  # noqa: E402
from transforms import FOLDER_TO_LABEL, eval_transform, train_transform  # noqa: E402


def idx_to_label_of(ds: ImageFolder) -> dict[int, str]:
    unknown = [c for c in ds.classes if c not in FOLDER_TO_LABEL]
    if unknown:
        raise SystemExit(f"Folders {unknown} have no label. Add them to FOLDER_TO_LABEL in transforms.py.")
    mapping = {i: FOLDER_TO_LABEL[c] for c, i in ds.class_to_idx.items()}
    if sorted(mapping.values()) != ["abnormal", "normal"]:
        raise SystemExit(f"Need exactly one 'normal' and one 'abnormal' folder, got {mapping}")
    return mapping


@torch.no_grad()
def evaluate(model, loader, device, abnormal_idx: int) -> dict:
    model.eval()
    tp = fp = fn = correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
        tp += ((pred == abnormal_idx) & (y == abnormal_idx)).sum().item()
        fp += ((pred == abnormal_idx) & (y != abnormal_idx)).sum().item()
        fn += ((pred != abnormal_idx) & (y == abnormal_idx)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"acc": correct / max(total, 1), "f1_abnormal": f1, "precision": precision, "recall": recall}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset root containing the split folders")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="valid")
    p.add_argument("--out", default="ai/classification/weights")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--head-epochs", type=int, default=2, help="phase 1: train the new head only")
    p.add_argument("--epochs", type=int, default=8, help="phase 2: fine-tune the whole network")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data = Path(args.data)
    train_ds = ImageFolder(data / args.train_split, transform=train_transform(args.img_size))
    val_ds = ImageFolder(data / args.val_split, transform=eval_transform(args.img_size))
    idx_to_label = idx_to_label_of(train_ds)
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise SystemExit(f"train and val folders differ: {train_ds.class_to_idx} vs {val_ds.class_to_idx}")
    abnormal_idx = next(i for i, lab in idx_to_label.items() if lab == "abnormal")

    counts = torch.bincount(torch.tensor(train_ds.targets), minlength=2).float()
    print(f"classes: {idx_to_label}  train counts: {counts.tolist()}  val size: {len(val_ds)}")
    # Weighted loss so the smaller class counts as much as the bigger one.
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts.clamp(min=1))).to(device))

    pin = device == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(args.arch, pretrained=True).to(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    phases = [("head", args.head_epochs, args.lr_head, True), ("full", args.epochs, args.lr, False)]
    epoch = 0
    for phase, n_epochs, lr, head_only in phases:
        if n_epochs == 0:
            continue
        for prm in model.parameters():
            prm.requires_grad = not head_only
        for prm in head_parameters(model, args.arch):
            prm.requires_grad = True
        opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
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
            sched.step()

            m = evaluate(model, val_dl, device, abnormal_idx)
            extra = {"epoch": epoch, "val_f1_abnormal": m["f1_abnormal"], "val_acc": m["acc"]}
            save_checkpoint(out / "last.pt", model, args.arch, args.img_size, idx_to_label, **extra)
            tag = ""
            if m["f1_abnormal"] > best_f1:
                best_f1 = m["f1_abnormal"]
                save_checkpoint(out / "best.pt", model, args.arch, args.img_size, idx_to_label, **extra)
                tag = "  <- best, saved"
            print(f"[{phase}] epoch {epoch}  loss {running / max(seen, 1):.4f}  val acc {m['acc']:.3f}  "
                  f"F1(abn) {m['f1_abnormal']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"({time.time() - t0:.0f}s){tag}")

    print(f"done. best val F1(abnormal) = {best_f1:.3f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
