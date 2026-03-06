# coding: utf-8

# Standard imports
import logging
import random
from pathlib import Path

# External imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.transforms import v2

# Local imports
from . import analysis


_META_EPS = 1.0


def _compute_size_metadata(height, width):
    h = float(height)
    w = float(width)
    meta = [
        np.log1p(h),
        np.log1p(w),
        np.log1p(h * w),
        np.log((w + _META_EPS) / (h + _META_EPS)),
    ]
    return torch.tensor(meta, dtype=torch.float32)


class MetadataImageFolder(ImageFolder):
    """ImageFolder variant returning (image, metadata, label)."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        width, height = sample.size
        metadata = _compute_size_metadata(height=height, width=width)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, metadata, target


class InferenceImageDataset(torch.utils.data.Dataset):
    """Dataset for unlabeled inference returning filename with optional metadata."""

    def __init__(self, root, transform=None, return_metadata=False):
        self.root = Path(root)
        self.transform = transform
        self.return_metadata = bool(return_metadata)
        valid_exts = {ext.lower() for ext in IMG_EXTENSIONS}
        self.samples = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts
        )

        if len(self.samples) == 0:
            raise ValueError(f"No image found under test path: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = default_loader(str(path))

        if self.return_metadata:
            width, height = img.size
            metadata = _compute_size_metadata(height=height, width=width)
        else:
            metadata = None

        if self.transform is not None:
            img = self.transform(img)

        if self.return_metadata:
            return img, metadata, path.name
        return img, path.name


class EnsureNumChannels(torch.nn.Module):
    """Ensure CHW tensor has the expected number of channels."""

    def __init__(self, out_channels=3):
        super().__init__()
        if out_channels not in (1, 3):
            raise ValueError("out_channels must be 1 or 3")
        self.out_channels = out_channels

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")

        c = x.shape[0]
        if c == self.out_channels:
            return x

        if self.out_channels == 3:
            if c == 1:
                return x.repeat(3, 1, 1)
            return x[:3]

        if c == 3:
            return x.mean(dim=0, keepdim=True)
        return x[:1]


class PadToSquare(torch.nn.Module):
    """Pad a CHW image tensor to square using a constant fill color."""

    def __init__(self, fill=(0, 0, 0)):
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
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
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


def _resolve_norm_stats(normalize, to_rgb):
    normalize = str(normalize).lower()
    if normalize in {"none", "identity"}:
        return None, None

    if normalize == "imagenet":
        if to_rgb:
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        return [0.5], [0.5]

    raise ValueError("data.normalize must be one of: imagenet, none")


def build_transform_blocks(
    img_size=128,
    to_rgb=True,
    pad_fill=0,
    normalize="imagenet",
    train_augment=None,
    keep_aspect_ratio=True,
    pad_to_square=True,
):
    """Return transform blocks (lists) so you can compose train/val pipelines."""

    train_augment = dict(train_augment or {})

    out_channels = 3 if to_rgb else 1
    mean, std = _resolve_norm_stats(normalize=normalize, to_rgb=to_rgb)
    fill = (pad_fill, pad_fill, pad_fill) if to_rgb else pad_fill

    common_pre = [
        v2.ToImage(),
        EnsureNumChannels(out_channels=out_channels),
    ]

    resize_pad = []
    if bool(keep_aspect_ratio) and bool(pad_to_square):
        resize_pad.append(PadToSquare(fill=fill))
    resize_pad.append(v2.Resize((img_size, img_size), antialias=True))

    geo_aug_train = []
    hflip_p = float(train_augment.get("hflip_p", 0.5))
    vflip_p = float(train_augment.get("vflip_p", 0.5))
    if hflip_p > 0:
        geo_aug_train.append(v2.RandomHorizontalFlip(p=hflip_p))
    if vflip_p > 0:
        geo_aug_train.append(v2.RandomVerticalFlip(p=vflip_p))

    affine_p = float(train_augment.get("affine_p", 0.7))
    rotate_deg = float(train_augment.get("rotate_deg", 180.0))
    translate = tuple(train_augment.get("translate", [0.05, 0.05]))
    scale = tuple(train_augment.get("scale", [0.9, 1.1]))
    shear_deg = float(train_augment.get("shear_deg", 7.0))
    if affine_p > 0:
        geo_aug_train.append(
            v2.RandomApply(
                [
                    v2.RandomAffine(
                        degrees=rotate_deg,
                        translate=translate,
                        scale=scale,
                        shear=shear_deg,
                        fill=fill,
                    )
                ],
                p=affine_p,
            )
        )

    photometric_train = []
    brightness = float(train_augment.get("brightness", 0.10))
    contrast = float(train_augment.get("contrast", 0.10))
    if brightness > 0 or contrast > 0:
        photometric_train.append(v2.ColorJitter(brightness=brightness, contrast=contrast))

    to_tensor_norm = [v2.ToDtype(torch.float32, scale=True)]
    if mean is not None and std is not None:
        to_tensor_norm.append(v2.Normalize(mean=mean, std=std))

    tensor_aug_train = []
    random_erasing_p = float(train_augment.get("random_erasing_p", 0.15))
    if random_erasing_p > 0:
        tensor_aug_train.append(v2.RandomErasing(p=random_erasing_p))

    return {
        "common_pre": common_pre,
        "resize_pad": resize_pad,
        "geo_aug_train": geo_aug_train,
        "photometric_train": photometric_train,
        "to_tensor_norm": to_tensor_norm,
        "tensor_aug_train": tensor_aug_train,
    }


def build_train_val_transforms(
    img_size=128,
    to_rgb=True,
    pad_fill=0,
    normalize="imagenet",
    train_augment=None,
    keep_aspect_ratio=True,
    pad_to_square=True,
):
    blocks = build_transform_blocks(
        img_size=img_size,
        to_rgb=to_rgb,
        pad_fill=pad_fill,
        normalize=normalize,
        train_augment=train_augment,
        keep_aspect_ratio=keep_aspect_ratio,
        pad_to_square=pad_to_square,
    )

    train_tf = v2.Compose(
        blocks["common_pre"]
        + blocks["resize_pad"]
        + blocks["geo_aug_train"]
        + blocks["photometric_train"]
        + blocks["to_tensor_norm"]
        + blocks["tensor_aug_train"]
    )

    val_tf = v2.Compose(
        blocks["common_pre"]
        + blocks["resize_pad"]
        + blocks["to_tensor_norm"]
    )

    return train_tf, val_tf


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    *,
    seed,
    drop_last=False,
    sampler=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs["worker_init_fn"] = _seed_worker
        loader_kwargs["generator"] = generator
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(**loader_kwargs)


def _compute_balanced_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def _compute_sampling_class_weights(class_counts, formula):
    counts = np.clip(np.asarray(class_counts, dtype=np.float64), a_min=1.0, a_max=None)
    formula_norm = str(formula).lower().replace(" ", "")

    if formula_norm in {"balanced", "inverse_freq", "inverse", "1/n", "1/n_c"}:
        weights = 1.0 / counts
    elif formula_norm in {
        "inverse_sqrt",
        "1/sqrt(class_count)",
        "1/sqrt(n)",
        "1/sqrt(n_c)",
    }:
        weights = 1.0 / np.sqrt(counts)
    elif formula_norm in {"uniform", "natural", "none"}:
        weights = np.ones_like(counts)
    else:
        raise ValueError(
            "Unknown class_weight_formula. Supported values: "
            "balanced, inverse_sqrt, uniform"
        )

    return (weights / weights.mean()).astype(np.float64)


def _resolve_to_rgb(data_config):
    if "grayscale_to_rgb" in data_config:
        mode = str(data_config.get("grayscale_to_rgb", "repeat_3_channels")).lower()
        return mode == "repeat_3_channels"
    return bool(data_config.get("to_rgb", True))


def get_dataloaders(data_config, use_cuda, *, build_test=True):
    valid_ratio = data_config.get("valid_ratio", 0.1)
    batch_size = data_config.get("batch_size", 128)
    num_workers = data_config.get("num_workers", 4)
    seed = data_config.get("seed", 0)

    img_size = data_config.get("img_size", 128)
    to_rgb = _resolve_to_rgb(data_config)
    pad_fill = data_config.get("pad_value", data_config.get("pad_fill", 0))
    normalize = data_config.get("normalize", "imagenet")
    keep_aspect_ratio = bool(data_config.get("keep_aspect_ratio", True))
    pad_to_square = bool(data_config.get("pad_to_square", True))

    train_augment = data_config.get("train_augment", {})
    enable_train_aug = bool(data_config.get("enable_train_augment", True))
    return_metadata = bool(data_config.get("return_metadata", False))

    pin_memory = data_config.get("pin_memory", use_cuda)
    persistent_workers = data_config.get("persistent_workers", num_workers > 0)
    prefetch_factor = data_config.get("prefetch_factor", 2)
    drop_last_train = data_config.get("drop_last_train", False)

    logging.info("  - Dataset creation")
    train_base = ImageFolder(root=data_config["trainpath"], transform=None)
    if len(train_base) == 0:
        raise ValueError(f"Empty train dataset at {data_config['trainpath']}")

    analysis_cfg = data_config.get("analysis", {})
    if analysis_cfg.get("compute_analysis", False):
        analysis.analyze_imagefolder(
            train_base,
            sample_size=analysis_cfg.get("sample_size", 200000),
            seed=seed,
            out_dir=analysis_cfg.get("out_dir", "./analysis"),
            compute_percentiles=analysis_cfg.get("compute_percentiles", True),
        )
    else:
        logging.info(
            "  - Dataset analysis disabled (set data.analysis.compute_analysis=true to run it)"
        )

    train_tf, eval_tf = build_train_val_transforms(
        img_size=img_size,
        to_rgb=to_rgb,
        pad_fill=pad_fill,
        normalize=normalize,
        train_augment=train_augment,
        keep_aspect_ratio=keep_aspect_ratio,
        pad_to_square=pad_to_square,
    )

    if not enable_train_aug:
        train_tf = eval_tf

    dataset_cls = MetadataImageFolder if return_metadata else ImageFolder

    train_full = dataset_cls(root=data_config["trainpath"], transform=train_tf)
    valid_full = dataset_cls(root=data_config["trainpath"], transform=eval_tf)

    test_dataset = None
    if build_test:
        test_dataset = InferenceImageDataset(
            root=data_config["testpath"],
            transform=eval_tf,
            return_metadata=return_metadata,
        )

    labels = np.fromiter((y for _, y in train_base.samples), dtype=np.int64, count=len(train_base.samples))
    indices = np.arange(len(labels))

    train_indices, valid_indices = train_test_split(
        indices,
        test_size=valid_ratio,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_full, valid_indices)

    num_classes = len(train_base.classes)
    train_labels = labels[train_indices]

    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    class_priors = class_counts / np.clip(class_counts.sum(), a_min=1.0, a_max=None)
    class_weights = _compute_balanced_class_weights(train_labels, num_classes)

    imbalance_cfg = data_config.get("imbalance", {})
    sampler_mode = str(data_config.get("sampler_mode", "")).lower().strip()
    if not sampler_mode:
        sampler_mode = "weighted_random" if imbalance_cfg.get("use_weighted_sampler", False) else "natural"

    class_weight_formula = data_config.get(
        "class_weight_formula",
        imbalance_cfg.get("class_weight_formula", "balanced"),
    )
    compute_class_weights = data_config.get(
        "compute_class_weights",
        imbalance_cfg.get("compute_class_weights", True),
    )

    train_sampler = None
    if sampler_mode == "weighted_random":
        sampling_weights = _compute_sampling_class_weights(class_counts, class_weight_formula)
        sample_weights = sampling_weights[train_labels]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        logging.info(
            "  - Using WeightedRandomSampler with class_weight_formula=%s",
            class_weight_formula,
        )
    elif sampler_mode != "natural":
        raise ValueError("data.sampler_mode must be one of: natural, weighted_random")

    train_loader = _build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=drop_last_train,
        sampler=train_sampler,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    valid_loader = _build_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed + 1,
        drop_last=False,
        sampler=None,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = _build_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 2,
            drop_last=False,
            sampler=None,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    sample_entry = train_full[0]
    image_tensor = sample_entry[0] if isinstance(sample_entry, (tuple, list)) else sample_entry
    input_size = tuple(image_tensor.shape)

    if compute_class_weights:
        train_loader.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    train_loader.class_counts = torch.tensor(class_counts, dtype=torch.float32)
    train_loader.class_priors = torch.tensor(class_priors, dtype=torch.float32)
    train_loader.num_classes = num_classes

    valid_loader.num_classes = num_classes
    valid_loader.class_priors = train_loader.class_priors

    if test_loader is not None:
        test_loader.num_classes = num_classes

    if test_dataset is None:
        logging.info(f"  - Train/Val sizes: {len(train_dataset)}/{len(valid_dataset)}")
    else:
        logging.info(
            f"  - Train/Val/Test sizes: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}"
        )
    logging.info(
        "  - Input size: %s | Classes: %s | Metadata: %s | Img size: %s",
        input_size,
        num_classes,
        return_metadata,
        img_size,
    )

    return train_loader, valid_loader, test_loader, input_size, num_classes
