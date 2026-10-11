"""Model building and checkpoint I/O, shared by train.py, evaluate.py and classifier.py.

Checkpoint format (a dict saved with torch.save):
    {"arch": "resnet18", "img_size": 224, "idx_to_label": {0: "abnormal", 1: "normal"},
     "state_dict": ..., "val_f1_abnormal": float, "epoch": int, "val_metrics": dict, "config": dict}
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ARCHS = ("resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_large")
UNFREEZE_MODES = ("head", "last", "all")   # what set_trainable leaves trainable


def build_model(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a 2-class head."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return model


def head_parameters(model: nn.Module, arch: str):
    return model.fc.parameters() if arch.startswith("resnet") else model.classifier.parameters()


def last_stage(model: nn.Module, arch: str) -> nn.Module:
    """The last stage of the backbone: layer4 for ResNets, the last three feature blocks otherwise."""
    return model.layer4 if arch.startswith("resnet") else model.features[-3:]


def set_trainable(model: nn.Module, arch: str, mode: str) -> None:
    """mode "head": classification head only; "last": head + last_stage; "all": the whole network."""
    if mode not in UNFREEZE_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    for prm in model.parameters():
        prm.requires_grad = mode == "all"
    if mode == "last":
        for prm in last_stage(model, arch).parameters():
            prm.requires_grad = True
    for prm in head_parameters(model, arch):
        prm.requires_grad = True


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
