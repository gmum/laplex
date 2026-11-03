"""Dataset helper dla DIV2K High-Resolution.

DIV2K (Agustsson & Timofte, CVPR 2017):
  - 800 obrazow train + 100 val, rozne rozdzielczosci (~2K)
  - download: http://data.vision.ee.ethz.ch/cvl/DIV2K/

Struktura katalogu:
    C:\\datasets\\div2k\\
        DIV2K_train_HR\\  (0001.png ... 0800.png)
        DIV2K_valid_HR\\  (0801.png ... 0900.png)

Uzywamy random crop 1024x1024 (train) / center crop 1024x1024 (val),
zeby n = 1024*1024*3 = 3145728 — identyczne jak MetFaces-1024.
"""
from __future__ import annotations
import os
from typing import Optional, Callable, List

import torch
from torch.utils.data import Dataset
from PIL import Image

_DIV2K_ROOT   = os.environ.get("DIV2K_ROOT", "data/div2k")
_TRAIN_SUBDIR = "DIV2K_train_HR"
_VALID_SUBDIR = "DIV2K_valid_HR"


class FlatImageFolder(Dataset):
    """Folder PNG/JPG -> (x_flat: Tensor[3*H*W], 0)."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg")
        self.files = sorted(
            f for f in os.listdir(root) if f.lower().endswith(exts)
        )
        if not self.files:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_div2k_datasets(train_transform: Callable,
                        val_transform: Callable,
                        root: Optional[str] = None):
    """Zwraca (train_ds, val_ds, train_root, val_root)."""
    if root is None:
        root = _DIV2K_ROOT
    train_root = os.path.join(root, _TRAIN_SUBDIR)
    valid_root = os.path.join(root, _VALID_SUBDIR)
    train_ds = FlatImageFolder(train_root, transform=train_transform)
    val_ds   = FlatImageFolder(valid_root, transform=val_transform)
    return train_ds, val_ds, train_root, valid_root


if __name__ == "__main__":
    train_ds, val_ds, tr, vr = make_div2k_datasets(None, None)
    print(f"train: {len(train_ds)} files  ({tr})")
    print(f"val:   {len(val_ds)} files  ({vr})")
