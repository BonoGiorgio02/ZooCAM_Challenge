# coding: utf-8

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder


@dataclass
class DatasetReport:
    num_images: int
    num_classes: int
    min_w: int
    max_w: int
    min_h: int
    max_h: int
    bad_images: int
    size_percentiles: Dict[str, float]


def _safe_image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return (w, h) or None if the image cannot be opened."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def analyze_imagefolder(
    ds: ImageFolder,
    sample_size: Optional[int] = None,
    seed: int = 0,
    out_dir: Optional[str] = "./analysis",
    compute_percentiles: bool = True,
) -> Tuple[pd.DataFrame, DatasetReport]:
    """
    Analyze a torchvision ImageFolder:
      - class counts (fast, no image open)
      - image size min/max (opens images; sampled by default)
      - optional size percentiles

    Args:
      ds: ImageFolder created with transform=None
      sample_size: number of images to scan for size stats (None = scan all)
      seed: random seed for sampling
      out_dir: if not None, saves class_counts.csv and size_summary.json
      compute_percentiles: compute width/height percentiles on scanned sample

    Returns:
      (class_counts_df, report)
    """
    num_images = len(ds.samples)
    num_classes = len(ds.classes)

    # --- Class counts
    # As np.array([y for _, y in ds.samples]) but more optimized
    labels = np.fromiter((y for _, y in ds.samples), dtype=np.int64, count=num_images)
    
    # For each class, count the number of occurrences
    counts = np.bincount(labels, minlength=num_classes)

    class_counts_df = pd.DataFrame({
        "class_idx": np.arange(num_classes, dtype=np.int64),
        "class_name": ds.classes,
        "count": counts,
    }).sort_values("count", ascending=False).reset_index(drop=True)

    # --- Sample indices for size scan
    rng = np.random.default_rng(seed)
    all_idx = np.arange(num_images)
    if sample_size is not None and sample_size < num_images:
        scan_idx = rng.choice(all_idx, size=sample_size, replace=False)
    else:
        scan_idx = all_idx

    min_w = min_h = 1e9
    max_w = max_h = 0
    bad = 0

    widths = []
    heights = []

    for i in tqdm(scan_idx, desc="Scanning image sizes"):
        path, _ = ds.samples[int(i)]
        wh = _safe_image_size(path)
        if wh is None:
            bad += 1
            continue
        w, h = wh
        min_w = min(min_w, w)
        max_w = max(max_w, w)
        
        min_h = min(min_h, h)
        max_h = max(max_h, h)
        
        if compute_percentiles:
            widths.append(w)
            heights.append(h)

    if min_w == 10**9 or min_h == 10**9:
        raise RuntimeError("No readable images found while scanning sizes.")

    size_percentiles = {}
    if compute_percentiles and len(widths) > 0:
        for p in [1, 5, 25, 50, 75, 95, 99]:
            size_percentiles[f"w_p{p}"] = float(np.percentile(widths, p))
            size_percentiles[f"h_p{p}"] = float(np.percentile(heights, p))

    report = DatasetReport(
        num_images=num_images,
        num_classes=num_classes,
        min_w=min_w,
        max_w=max_w,
        min_h=min_h,
        max_h=max_h,
        bad_images=bad,
        size_percentiles=size_percentiles,
    )

    # --- Optional save to disk
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        class_counts_df.to_csv(os.path.join(out_dir, "class_counts.csv"), index=False)

        with open(os.path.join(out_dir, "size_summary.json"), "w") as f:
            json.dump({
                "num_images": report.num_images,
                "num_classes": report.num_classes,
                "min_w": report.min_w,
                "max_w": report.max_w,
                "min_h": report.min_h,
                "max_h": report.max_h,
                "bad_images": report.bad_images,
                "size_percentiles": report.size_percentiles,
            }, f, indent=2)

    # --- Logging
    logging.info("= Dataset analysis")
    logging.info(f"Images: {report.num_images} | Classes: {report.num_classes}")
    logging.info(f"Width min/max: {report.min_w}/{report.max_w} | Height min/max: {report.min_h}/{report.max_h}")
    if compute_percentiles and report.size_percentiles:
        logging.info(f"Median size: {report.size_percentiles.get('w_p50', None)} x {report.size_percentiles.get('h_p50', None)}")
    logging.info("Top 10 classes by count:")
    logging.info("\n" + class_counts_df.head(10).to_string(index=False))
    logging.info("Last 10 classes by count:")
    logging.info("\n" + class_counts_df.tail(10).to_string(index=False))

    return class_counts_df, report

def compute_dataset_mean_std(
    ds: ImageFolder,
    sample_size: Optional[int] = None,
    seed: int = 0,
):
    """
    Compute dataset grayscale mean and std for an ImageFolder.

    Args:
        ds: ImageFolder created with transform=None
        sample_size: number of images to scan (None = scan all)
        seed: random seed for sampling
    """

    num_images = len(ds.samples)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(num_images)

    if sample_size is not None and sample_size < num_images:
        scan_idx = rng.choice(all_idx, size=sample_size, replace=False)
    else:
        scan_idx = all_idx

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0
    bad = 0

    for i in tqdm(scan_idx, desc="Computing mean/std"):
        path, _ = ds.samples[int(i)]

        try:
            with Image.open(path) as im:
                im = im.convert("L")
                arr = np.asarray(im, dtype=np.float32) / 255.0

                pixel_sum += arr.sum()
                pixel_sq_sum += (arr ** 2).sum()
                pixel_count += arr.size

        except Exception:
            bad += 1

    if pixel_count == 0:
        raise RuntimeError("No readable images found.")

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print("\nDataset grayscale statistics")
    print(f"Mean: {mean:.6f}")
    print(f"Std : {std:.6f}")
    print(f"Bad images skipped: {bad}")

    return mean, std
