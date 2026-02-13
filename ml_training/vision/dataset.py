"""Tomato ripeness dataset loader with augmentation pipeline.

Supports loading from a local directory of images organized by class,
or downloading from Kaggle. Provides train/val/test splits with
configurable augmentation for training.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class TomatoDataset(Dataset):
    """Dataset for tomato ripeness classification.

    Expects directory structure:
        root/
        ├── class_0/
        │   ├── img001.jpg
        │   └── ...
        ├── class_1/
        │   └── ...
        └── ...

    Args:
        root: Root directory containing class subdirectories.
        class_names: List of class directory names in label order.
        transform: Torchvision transforms to apply.
    """

    def __init__(
        self,
        root: str | Path,
        class_names: list[str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.class_names = class_names
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory tree for images."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                # Try case-insensitive match
                matches = [
                    d for d in self.root.iterdir()
                    if d.is_dir() and d.name.lower() == class_name.lower()
                ]
                if matches:
                    class_dir = matches[0]
                else:
                    print(f"Warning: class directory '{class_name}' not found in {self.root}")
                    continue

            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images across {len(self.class_names)} classes")
        for name in self.class_names:
            count = sum(1 for _, l in self.samples if l == self.class_to_idx[name])
            print(f"  {name}: {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transforms(config: dict, split: str = "train") -> transforms.Compose:
    """Build torchvision transforms from config.

    Args:
        config: Vision config dict from config.yaml.
        split: One of 'train', 'val', 'test'.

    Returns:
        Composed transforms.
    """
    input_size = config.get("input_size", 224)
    aug_cfg = config.get("augmentation", {})

    # ImageNet normalization (used with pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        # Training augmentation
        crop_cfg = aug_cfg.get("random_resized_crop", {})
        jitter_cfg = aug_cfg.get("color_jitter", {})

        transform_list = [
            transforms.RandomResizedCrop(
                input_size,
                scale=tuple(crop_cfg.get("scale", [0.8, 1.0])),
                ratio=tuple(crop_cfg.get("ratio", [0.9, 1.1])),
            ),
        ]

        if aug_cfg.get("random_horizontal_flip", 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=aug_cfg["random_horizontal_flip"])
            )

        if aug_cfg.get("random_vertical_flip", 0) > 0:
            transform_list.append(
                transforms.RandomVerticalFlip(p=aug_cfg["random_vertical_flip"])
            )

        rotation = aug_cfg.get("random_rotation", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        if jitter_cfg:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=jitter_cfg.get("brightness", 0),
                    contrast=jitter_cfg.get("contrast", 0),
                    saturation=jitter_cfg.get("saturation", 0),
                    hue=jitter_cfg.get("hue", 0),
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Validation / test: deterministic resize + center crop
        transform_list = [
            transforms.Resize(int(input_size * 1.14)),  # 256 for 224 input
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(
    config: dict,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from config.

    Args:
        config: Full config dict (containing 'vision' key).
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    vision_cfg = config["vision"]
    dataset_cfg = vision_cfg["dataset"]
    train_cfg = vision_cfg["training"]

    root = Path(dataset_cfg["root"])
    class_names = dataset_cfg["class_names"]

    # Build full dataset with train transforms first (for splitting)
    full_dataset = TomatoDataset(
        root=root,
        class_names=class_names,
        transform=None,  # We'll apply transforms per-split
    )

    # Split
    n = len(full_dataset)
    train_ratio = dataset_cfg.get("train_split", 0.70)
    val_ratio = dataset_cfg.get("val_split", 0.15)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Wrap subsets with proper transforms
    train_transform = build_transforms(vision_cfg, split="train")
    val_transform = build_transforms(vision_cfg, split="val")

    train_ds = TransformSubset(train_subset, train_transform)
    val_ds = TransformSubset(val_subset, val_transform)
    test_ds = TransformSubset(test_subset, val_transform)

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")
    return train_loader, val_loader, test_loader


class TransformSubset(Dataset):
    """Wraps a Subset and applies a transform to the raw PIL images."""

    def __init__(self, subset: Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # The underlying TomatoDataset stores (path, label); we need raw PIL
        sample_idx = self.subset.indices[idx]
        img_path, label = self.subset.dataset.samples[sample_idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
