# coding: utf-8

# Standard imports
import logging
import random
from pathlib import Path

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import torchvision.transforms.functional as F

import numpy as np
import matplotlib.pyplot as plt


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


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("  - Dataset creation")

    input_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    root_train_dataset = ImageFolder(
        root=data_config["trainpath"],
        transform=input_transform,
    )
    
    test_dataset = InferenceImageDataset(
        root=data_config["testpath"],
        transform=input_transform,
    )

    logging.info(f"  - I loaded {len(root_train_dataset)} train samples")
    logging.info(f"  - I loaded {len(test_dataset)} test samples")

    indices = list(range(len(root_train_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(root_train_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_dataset = torch.utils.data.Subset(root_train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(root_train_dataset, valid_indices)

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

    num_classes = len(root_train_dataset.classes)
    input_size = tuple(root_train_dataset[0][0].shape)

    return train_loader, valid_loader, test_loader, input_size, num_classes
