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
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Local imports
from . import analysis

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
    
    # Build base datasets without transforms
    train_base, test_dataset = build_datasets(data_config)
    
    # --- Analysis step (one-shot)
    analysis_cfg = data_config.get("analysis", {})  # optionally nested under data
    sample_size = analysis_cfg.get("sample_size", 200000)
    save_dir = analysis_cfg.get("out_dir", "./analysis")

    counts_df, report = analysis.analyze_imagefolder(
        train_base,
        sample_size=sample_size,
        seed=seed,
        out_dir=save_dir,
    )

    # TODO: Improve data augmentation with other v2 transforms
    # Baseline transforms 
    input_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    labels = np.array([y for _, y in train_base.samples])
    indices = np.arange(len(labels))
    
    train_full = ImageFolder(root=data_config["trainpath"], transform=input_transform)
    
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

    num_classes = len(train_base.classes)
    input_size = tuple(train_full[0][0].shape)

    return train_loader, valid_loader, test_loader, input_size, num_classes
