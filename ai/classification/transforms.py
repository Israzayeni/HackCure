"""Image preprocessing shared by training (train.py) and inference (classifier.py).

Training and inference must see the same preprocessing, so both import it from here.
Inputs are PIL RGB images. At inference, the crop is turned into a PIL image first.
"""

from __future__ import annotations

from torchvision import transforms as T

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)   # ImageNet statistics, matching the pretrained weights
STD = (0.229, 0.224, 0.225)

# Folder name in the dataset -> contract label. Folder names not listed here make train.py stop.
FOLDER_TO_LABEL = {
    "effected-cells": "abnormal",   # Roboflow Leukemia-Cells
    "normal-cells": "normal",
    "abnormal": "abnormal",         # fakes/crops
    "normal": "normal",
}


AUGMENTATIONS = ("strong", "moderate", "none")


def train_transform(size: int = IMG_SIZE, aug: str = "strong") -> T.Compose:
    """"strong" (default) is aimed at the domain gap: training images are stained smears, Malassez
    images are unstained, so colour is randomized or removed. Cells look the same at any
    orientation, so flips and rotations are safe. RandomResizedCrop varies the framing, so
    crops from crop_cell (cell ~2/3 of the width) do not surprise the model.

    "moderate" keeps the stain colours and changes the geometry less. It may score higher on
    Leukemia-Cells, whose test split is stained too, while generalizing worse to unstained
    Malassez images. "none" is the eval preprocessing, for comparison."""
    if aug == "none":
        return eval_transform(size)
    if aug == "moderate":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    if aug != "strong":
        raise ValueError(f"unknown aug {aug!r}")
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.1),
        T.RandomGrayscale(p=0.3),
        T.RandomApply([T.GaussianBlur(3)], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
