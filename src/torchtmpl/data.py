# coding: utf-8

# Standard imports
import os
import random
import logging
from pathlib import Path

# External imports
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Local imports
from . import analysis

class InferenceImageDataset(torch.utils.data.Dataset):
    """Dataset for unlabeled inference that returns (image_tensor, filename)."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )

        if len(self.samples) == 0:
            raise ValueError(f"No image found under test path: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = default_loader(str(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, path.name


class PadToSquare(torch.nn.Module):
    """Pad a CHW image tensor to square using a constant fill color."""
    def __init__(self, fill=(255, 255, 255)):
        super().__init__()
        self.fill = fill

    def forward(self, x):
        _, h, w = x.shape
        if h == w:
            return x

        m = max(h, w)
        pad_left = (m - w) // 2
        pad_right = m - w - pad_left
        pad_top = (m - h) // 2
        pad_bottom = m - h - pad_top

        return v2.functional.pad(
            x,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )

    
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


def build_transform_blocks(img_size=128, to_rgb=True, pad_fill=255):
    """
    Return transform blocks (lists) so you can compose train/val pipelines.
    """
    # ImageNet normalization (useful for pretrained backbones)
    mean = [0.485, 0.456, 0.406] if to_rgb else [0.5]
    std  = [0.229, 0.224, 0.225] if to_rgb else [0.5]

    fill = (pad_fill, pad_fill, pad_fill) if to_rgb else pad_fill

    # Common: convert + enforce channels
    common_pre = [
        v2.ToImage(),
        v2.Grayscale(num_output_channels=3 if to_rgb else 1),
    ]

    # Pad to square, then Resize with aspect ratio preserved
    resize_pad = [
        PadToSquare(fill=fill),
        v2.Resize((img_size, img_size), antialias=True),
    ]

    # Geometry augmentation (TRAIN only)
    geo_aug_train = [
        v2.RandomHorizontalFlip(p=0.3),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=180, fill=fill),
        v2.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            fill=fill,
        ),
    ]

    # Photometric augmentation (TRAIN only)
    photometric_train = [
        v2.RandomAutocontrast(p=0.1),
        v2.RandomEqualize(p=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    ]

    # Final: float + normalize
    to_tensor_norm = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]

    return {
        "common_pre": common_pre,
        "resize_pad": resize_pad,
        "geo_aug_train": geo_aug_train,
        "photometric_train": photometric_train,
        "to_tensor_norm": to_tensor_norm,
    }


def build_train_val_transforms(img_size=128, to_rgb=True, pad_fill=255):
    blocks = build_transform_blocks(img_size=img_size, to_rgb=to_rgb, pad_fill=pad_fill)

    train_tf = v2.Compose(
        blocks["common_pre"]
        + blocks["resize_pad"]
        + blocks["geo_aug_train"]
        + blocks["photometric_train"]
        + blocks["to_tensor_norm"]
    )

    val_tf = v2.Compose(
        blocks["common_pre"]
        + blocks["resize_pad"]
        + blocks["to_tensor_norm"]
    )

    return train_tf, val_tf


def build_datasets(data_config):
    """
    Build datasets, this is useful because analysis can run right after dataset creation.
    Returns:
      train_base: ImageFolder (train, with transform=None by default)
      test_ds: InferenceImageDataset (test/imgs, returns filename)
    """
    
    logging.info("  - Dataset creation")
    
    root_train_dataset = ImageFolder(
        root=data_config["trainpath"],
        transform=None,
    )
    
    root_test_dataset = InferenceImageDataset(
        root=data_config["testpath"],
        transform=None,
    )
    
    logging.info(f"  - Loaded {len(root_train_dataset)} train samples")
    logging.info(f"  - Loaded {len(root_test_dataset)} test samples")
    
    return root_train_dataset, root_test_dataset
    

def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config.get("valid_ratio", 0.1)
    batch_size = data_config.get("batch_size", 128)
    num_workers = data_config.get("num_workers", 4)
    seed = data_config.get("seed", 0)
    
    img_size = data_config.get("img_size", 128)
    to_rgb = data_config.get("to_rgb", True)
    pad_fill = data_config.get("pad_fill", 255)
    
    # --- Analysis step (one-shot)
    analysis_cfg = data_config.get("analysis", {})  # optionally nested under data
    analysis_enabled = analysis_cfg.get("compute_analysis", False)
    
    if analysis_enabled:
        # Build base datasets without transforms
        train_base, test_dataset = build_datasets(data_config)
        
        sample_size = analysis_cfg.get("sample_size", 200000)
        save_dir = analysis_cfg.get("out_dir", "./analysis")
        compute_percentiles = analysis_cfg.get("compute_percentiles", True)

        counts_df, report = analysis.analyze_imagefolder(
            train_base,
            sample_size=sample_size,
            seed=seed,
            out_dir=save_dir,
        )
    else:
        logging.info("  - Dataset analysis disabled (set data.analysis.enabled=true to run it)")

    # TODO: Improve data augmentation with other v2 transforms
    # Baseline transforms 
    input_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    train_full = ImageFolder(root=data_config["trainpath"], transform=input_transform)
    
    test_dataset = InferenceImageDataset(
        root=data_config["testpath"],
        transform=input_transform,
    )
    
    labels = np.array([y for _, y in train_full.samples])
    indices = np.arange(len(labels))
    
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=valid_ratio,
        stratify=labels,
        random_state=seed,
    )
    
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    valid_dataset = torch.utils.data.Subset(train_full, valid_indices)

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = len(train_full.classes)
    input_size = tuple(train_full[0][0].shape)

    return train_loader, valid_loader, test_loader, input_size, num_classes
