"""
ImageNet-10 Dataset — PyTorch-compatible dataset class
=======================================================
ImageNet-10 is a subset of ImageNet containing 10 classes.
Expected directory structure:

    root/
    ├── train/
    │   ├── n01440764/   ← synset (e.g. "tench" — a fish)
    │   │   ├── img1.JPEG
    │   │   └── ...
    │   └── n02102040/
    │       └── ...
    └── val/
        ├── n01440764/
        └── ...

ImageNet-10 classes (10 commonly used synsets):
    n01440764 — tench (fish)
    n01774384 — black widow (spider)
    n01882714 — koala
    n02102040 — English springer (dog)
    n02979186 — cassette player
    n03000684 — chain saw
    n03028079 — church
    n03788195 — mosque
    n03877845 — palace
    n03994562 — pinwheel
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET10_SYNSETS: List[str] = [
    "n01440764",  
    "n01774384",  
    "n01882714",  
    "n02102040",  
    "n02979186",  
    "n03000684",  
    "n03028079",  
    "n03788195",  
    "n03877845",  
    "n03944341",  
]


IMAGENET10_LABELS: Dict[str, str] = {
    "n01440764": "tench",
    "n01774384": "black widow",
    "n01882714": "koala",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03788195": "mosque",
    "n03877845": "palace",
    "n03944341": "pinwheel",
}


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)

DEFAULT_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)



class ImageNet10(Dataset):
    """
    PyTorch-compatible ImageNet-10 dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing ``train/`` and ``val/`` subdirectories.
    split : {"train", "val"}
        Which data split to load.
    transform : callable, optional
        Transformations applied to images (``PIL.Image``).
        Standard ImageNet transforms are used by default.
    target_transform : callable, optional
        Transformations applied to labels (int).
    synsets : list[str], optional
        List of synsets to load. Defaults to ``IMAGENET10_SYNSETS``.
    extensions : tuple[str], optional
        Allowed image file extensions.

    Attributes
    ----------
    classes : list[str]
        Class names (synsets) sorted alphabetically.
    class_to_idx : dict[str, int]
        Mapping from synset to class index.
    idx_to_label : dict[int, str]
        Mapping from class index to human-readable label.
    samples : list[tuple[str, int]]
        List of (file_path, class_index) pairs.

    Example
    -------
    >>> from imagenet10_dataset import ImageNet10
    >>> train_ds = ImageNet10(root="/data/imagenet10", split="train")
    >>> img, label = train_ds[0]
    >>> print(img.shape, label)               
    >>> print(train_ds.idx_to_label[label])   
    """

    EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        synsets: Optional[List[str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got: '{split}'")

        self.root = Path(root)
        self.split = split
        self.synsets = synsets or IMAGENET10_SYNSETS
        self.extensions = extensions or self.EXTENSIONS

        
        if transform is None:
            transform = (
                DEFAULT_TRAIN_TRANSFORM if split == "train" else DEFAULT_VAL_TRANSFORM
            )
        self.transform = transform
        self.target_transform = target_transform

        
        self.classes: List[str] = sorted(self.synsets)
        self.class_to_idx: Dict[str, int] = {
            cls: i for i, cls in enumerate(self.classes)
        }
        self.idx_to_label: Dict[int, str] = {
            i: IMAGENET10_LABELS.get(cls, cls) for cls, i in self.class_to_idx.items()
        }

        
        self.samples: List[Tuple[str, int]] = self._make_dataset()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under: {self.root / split}\n"
                "Check the directory structure and synset names."
            )

    

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Walk synset directories and collect (path, label) pairs."""
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        samples: List[Tuple[str, int]] = []

        for synset in self.classes:
            synset_dir = split_dir / synset
            if not synset_dir.is_dir():
                
                print(
                    f"[WARNING] Missing directory for synset '{synset}': {synset_dir}"
                )
                continue

            class_idx = self.class_to_idx[synset]

            for file_path in sorted(synset_dir.iterdir()):
                if file_path.suffix in self.extensions:
                    samples.append((str(file_path), class_idx))

        return samples

    def _load_image(self, path: str) -> Image.Image:
        """Load an image from disk and convert it to RGB."""
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        image = self._load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    

    def __repr__(self) -> str:
        lines = [
            f"{self.__class__.__name__}(",
            f"  root={self.root},",
            f"  split='{self.split}',",
            f"  num_samples={len(self.samples)},",
            f"  num_classes={len(self.classes)},",
            f"  classes={self.classes}",
            ")",
        ]
        return "\n".join(lines)

    

    def get_label_name(self, idx: int) -> str:
        """Return the human-readable class name for the given index."""
        if idx not in self.idx_to_label:
            raise IndexError(f"Index {idx} out of range [0, {len(self.classes)-1}]")
        return self.idx_to_label[idx]

    def class_counts(self) -> Dict[str, int]:
        """Return a dict mapping {synset: sample_count}."""
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts





def get_imagenet10_loaders(
    root: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for the training and validation splits.

    Parameters
    ----------
    root : str | Path
        Root directory of the dataset (must contain train/ and val/).
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses used for data loading.
    pin_memory : bool
        Whether to use pinned memory (speeds up CPU→GPU transfers).

    Returns
    -------
    train_loader, val_loader : tuple[DataLoader, DataLoader]

    Example
    -------
    >>> train_loader, val_loader = get_imagenet10_loaders("/data/imagenet10")
    >>> for images, labels in train_loader:
    ...     print(images.shape)   
    ...     break
    """
    train_ds = ImageNet10(root=root, split="train")
    val_ds = ImageNet10(root=root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader




if __name__ == "__main__":
    root = "/dataset/ImageNet"

    print("=== ImageNet10 class smoke test ===\n")

    train_ds = ImageNet10(root=root, split="train")
    val_ds = ImageNet10(root=root, split="val")

    print(train_ds)
    print()

    print("Training samples  :", len(train_ds))
    print("Validation samples:", len(val_ds))
    print()

    print("Class distribution (train):")
    for synset, count in train_ds.class_counts().items():
        label = IMAGENET10_LABELS.get(synset, synset)
        print(f"  {synset} ({label:<20}): {count:>5} images")
    print()

    img, label = train_ds[0]
    print(
        f"Sample item: image={img.shape}, label={label} ({train_ds.get_label_name(label)})"
    )
