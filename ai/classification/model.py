"""Model building and checkpoint I/O, shared by train.py, evaluate.py and classifier.py.

Checkpoint format (a dict saved with torch.save):
    {"arch": "resnet18", "img_size": 224, "idx_to_label": {0: "abnormal", 1: "normal"},
     "state_dict": ..., "val_f1_abnormal": float, "epoch": int}
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a 2-class head."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model


def head_parameters(model: nn.Module, arch: str):
    return model.fc.parameters() if arch == "resnet18" else model.classifier.parameters()


def save_checkpoint(path, model: nn.Module, arch: str, img_size: int, idx_to_label: dict, **extra) -> None:
    torch.save({"arch": arch, "img_size": img_size, "idx_to_label": idx_to_label,
                "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Returns (model in eval mode on device, checkpoint dict without the weights)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt.pop("state_dict"))
    model.to(device).eval()
    ckpt["idx_to_label"] = {int(k): v for k, v in ckpt["idx_to_label"].items()}
    return model, ckpt
