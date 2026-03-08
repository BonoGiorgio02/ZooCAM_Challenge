# coding: utf-8

import argparse
import contextlib
import csv
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from . import data, models, utils
from .main import _load_model_weights


def _resolve_device(device_arg):
    """Resolve device."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint_path(config):
    """Resolve checkpoint path."""
    if "test" in config and "checkpoint" in config["test"]:
        return config["test"]["checkpoint"]
    return os.path.join(config["logging"]["logdir"], "best_model.pt")


def _resolve_inference_settings(config):
    """Resolve inference settings."""
    data_cfg = config.get("data", {})
    inference_cfg = dict(config.get("inference", {}))
    legacy_test_cfg = config.get("test", {})
    if "tta" not in inference_cfg and bool(legacy_test_cfg.get("use_tta", False)):
        inference_cfg["tta"] = legacy_test_cfg.get("tta_names", ["orig"])

    tta_modes = inference_cfg.get("tta", ["orig"])
    if isinstance(tta_modes, str):
        tta_modes = [tta_modes]
    if not tta_modes:
        tta_modes = ["orig"]

    tta_norm_mean = inference_cfg.get("tta_norm_mean", data_cfg.get("mean", None))
    tta_norm_std = inference_cfg.get("tta_norm_std", data_cfg.get("std", None))

    tau_grid = inference_cfg.get("logit_adjustment_tau_grid", [0.0])
    if isinstance(tau_grid, (int, float)):
        tau_grid = [float(tau_grid)]
    tau_grid = [float(t) for t in tau_grid]

    use_tau_sweep = bool(inference_cfg.get("sweep_logit_adjustment", True))
    fixed_tau = inference_cfg.get("selected_tau", None)
    if fixed_tau is not None:
        fixed_tau = float(fixed_tau)

    amp_enabled = bool(config.get("train", {}).get("amp", True))

    return {
        "tta_modes": tta_modes,
        "tta_norm_mean": tta_norm_mean,
        "tta_norm_std": tta_norm_std,
        "tau_grid": tau_grid,
        "use_tau_sweep": use_tau_sweep,
        "fixed_tau": fixed_tau,
        "amp_enabled": amp_enabled,
    }


def _predict_logits_with_tta(
    model,
    inputs,
    metadata,
    tta_modes,
    device,
    amp_enabled,
    tta_norm_mean=None,
    tta_norm_std=None,
):
    """Execute predict logits with tta."""
    tta_modes = list(tta_modes) if tta_modes is not None else ["orig"]
    if len(tta_modes) == 0:
        tta_modes = ["orig"]

    use_amp = bool(amp_enabled and device.type == "cuda")
    logits_sum = None
    for mode in tta_modes:
        aug_inputs = utils.apply_tta(
            inputs,
            mode,
            norm_mean=tta_norm_mean,
            norm_std=tta_norm_std,
        )
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_ctx:
            logits = utils.model_forward(model, aug_inputs, metadata)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(len(tta_modes))


def _extract_valid_names(valid_loader):
    """Execute extract valid names."""
    subset = valid_loader.dataset
    if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
        raise ValueError("Validation dataset must be a Subset with indices to align committee models.")
    base_dataset = subset.dataset
    if not hasattr(base_dataset, "samples"):
        raise ValueError("Underlying dataset must expose .samples to recover validation names.")

    names = []
    for idx in subset.indices:
        sample_path = base_dataset.samples[idx][0]
        names.append(os.path.basename(str(sample_path)))
    return names


def _ensure_same_items(reference, candidate, what):
    """Execute ensure same items."""
    if len(reference) != len(candidate):
        raise ValueError(f"{what} length mismatch: {len(reference)} vs {len(candidate)}")
    for i, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            raise ValueError(
                f"{what} mismatch at position {i}: '{left}' vs '{right}'. "
                "All models must produce predictions in exactly the same order."
            )


def _select_best_tau_from_logits(logits, targets, num_classes, tau_grid, log_prior):
    """Select best tau from logits."""
    if not tau_grid:
        return 0.0, {}

    best_tau = float(tau_grid[0])
    best_f1 = -1.0
    tau_scores = {}

    for tau in tau_grid:
        tau = float(tau)
        preds = torch.argmax(logits - tau * log_prior.unsqueeze(0), dim=1)
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        utils._update_confusion_matrix(confusion, preds, targets, num_classes)
        f1 = utils.macro_f1_from_confusion(confusion)
        tau_scores[tau] = float(f1)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = tau

    return best_tau, tau_scores


def _macro_f1_from_logits(logits, targets, num_classes):
    """Execute macro f1 from logits."""
    preds = torch.argmax(logits, dim=1)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    utils._update_confusion_matrix(confusion, preds, targets, num_classes)
    return float(utils.macro_f1_from_confusion(confusion))


def _run_model_pipeline(config_path, device):
    """Run model pipeline."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model_class = str(model_cfg.get("class", "unknown_model"))
    model_name = Path(config_path).stem
    model_id = f"{model_name}:{model_class}"

    logging.info("")
    logging.info("============================================================")
    logging.info("Model: %s", model_id)
    logging.info("Config: %s", config_path)

    use_cuda = device.type == "cuda"
    data_cfg = config["data"]
    train_loader, valid_loader, test_loader, input_size, num_classes = data.get_dataloaders(
        data_cfg,
        use_cuda,
        build_test=True,
    )

    model = models.build_model(model_cfg, input_size, num_classes).to(device)
    model.eval()

    ckpt_path = _resolve_checkpoint_path(config)
    logging.info("Checkpoint: %s", ckpt_path)
    _load_model_weights(model, ckpt_path, device=device, strict=True)

    infer = _resolve_inference_settings(config)
    logging.info("TTA modes: %s", infer["tta_modes"])

    class_priors = getattr(train_loader, "class_priors", None)
    if class_priors is None:
        class_priors = torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    log_prior = torch.log(class_priors.detach().cpu().clamp_min(1e-12))

    valid_names = _extract_valid_names(valid_loader)
    val_logits_cpu = []
    val_targets_cpu = []

    with torch.inference_mode():
        for batch in tqdm(valid_loader, desc=f"[{model_name}] validation inference"):
            inputs, metadata, targets = utils.unpack_supervised_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)

            logits = _predict_logits_with_tta(
                model=model,
                inputs=inputs,
                metadata=metadata,
                tta_modes=infer["tta_modes"],
                device=device,
                amp_enabled=infer["amp_enabled"],
                tta_norm_mean=infer["tta_norm_mean"],
                tta_norm_std=infer["tta_norm_std"],
            )
            val_logits_cpu.append(logits.detach().cpu())
            val_targets_cpu.append(targets.detach().cpu())

    if not val_logits_cpu:
        raise RuntimeError(f"No validation logits collected for model {model_id}")

    val_logits = torch.cat(val_logits_cpu, dim=0)
    val_targets = torch.cat(val_targets_cpu, dim=0)
    if len(valid_names) != int(val_logits.shape[0]):
        raise ValueError(
            f"Validation name/logit count mismatch for {model_id}: "
            f"{len(valid_names)} names vs {int(val_logits.shape[0])} logits"
        )

    if infer["fixed_tau"] is not None:
        best_tau = float(infer["fixed_tau"])
        tau_scores = {}
        logging.info("Using fixed tau: %.4f", best_tau)
    elif infer["use_tau_sweep"] and infer["tau_grid"]:
        best_tau, tau_scores = _select_best_tau_from_logits(
            logits=val_logits,
            targets=val_targets,
            num_classes=num_classes,
            tau_grid=infer["tau_grid"],
            log_prior=log_prior,
        )
        for tau, f1 in sorted(tau_scores.items(), key=lambda kv: kv[0]):
            logging.info("  tau=%.4f -> val_macro_f1=%.6f", float(tau), float(f1))
        logging.info("Best tau: %.4f", best_tau)
    else:
        best_tau = 0.0
        tau_scores = {}
        logging.info("Tau sweep disabled. Using tau=0.0")

    val_logits = val_logits - best_tau * log_prior.unsqueeze(0)
    val_macro_f1 = _macro_f1_from_logits(val_logits, val_targets, num_classes)
    logging.info("Single-model validation macro-F1 (%s): %.6f", model_id, val_macro_f1)

    test_logits_cpu = []
    test_names = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"[{model_name}] test inference"):
            inputs, metadata, names = utils.unpack_inference_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)

            logits = _predict_logits_with_tta(
                model=model,
                inputs=inputs,
                metadata=metadata,
                tta_modes=infer["tta_modes"],
                device=device,
                amp_enabled=infer["amp_enabled"],
                tta_norm_mean=infer["tta_norm_mean"],
                tta_norm_std=infer["tta_norm_std"],
            )
            test_logits_cpu.append(logits.detach().cpu())

            if torch.is_tensor(names):
                names = names.detach().cpu().tolist()
            test_names.extend([os.path.basename(str(n)) for n in names])

    if not test_logits_cpu:
        raise RuntimeError(f"No test logits collected for model {model_id}")

    test_logits = torch.cat(test_logits_cpu, dim=0) - best_tau * log_prior.unsqueeze(0)
    if len(test_names) != int(test_logits.shape[0]):
        raise ValueError(
            f"Test name/logit count mismatch for {model_id}: "
            f"{len(test_names)} names vs {int(test_logits.shape[0])} logits"
        )

    return {
        "model_name": model_name,
        "model_id": model_id,
        "num_classes": int(num_classes),
        "valid_names": valid_names,
        "valid_targets": val_targets,
        "valid_logits": val_logits,
        "test_names": test_names,
        "test_logits": test_logits,
        "best_tau": float(best_tau),
        "tau_scores": tau_scores,
        "single_val_macro_f1": float(val_macro_f1),
        "config_path": config_path,
        "checkpoint_path": ckpt_path,
    }


def _generate_simplex_weights(num_models, step):
    """Execute generate simplex weights."""
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-9:
        raise ValueError("weights_step must divide 1.0 exactly (e.g. 0.2, 0.1, 0.05, 0.02)")

    all_int_weights = []

    def _recurse(prefix, remaining, slots_left):
        """Execute recurse."""
        if slots_left == 1:
            all_int_weights.append(prefix + [remaining])
            return
        for w in range(remaining + 1):
            _recurse(prefix + [w], remaining - w, slots_left - 1)

    _recurse([], units, num_models)
    return [tuple(w / float(units) for w in weights) for weights in all_int_weights]


def _write_submission(path, names, labels):
    """Execute write submission."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imgname", "label"])
        for name, label in zip(names, labels):
            writer.writerow([os.path.basename(str(name)), int(label)])


def run_committee(config_paths, out_dir, submission_name, weights_step, top_k, device_arg):
    """Run committee."""
    device = _resolve_device(device_arg)
    logging.info("Device: %s", device)
    logging.info("Committee models: %s", ", ".join(config_paths))

    model_outputs = []
    for config_path in config_paths:
        model_outputs.append(_run_model_pipeline(config_path, device))

    num_models = len(model_outputs)
    if num_models < 2:
        raise ValueError("Need at least two models for committee.")

    ref_valid_names = model_outputs[0]["valid_names"]
    ref_test_names = model_outputs[0]["test_names"]
    ref_valid_targets = model_outputs[0]["valid_targets"]
    num_classes = model_outputs[0]["num_classes"]

    for m in model_outputs[1:]:
        _ensure_same_items(ref_valid_names, m["valid_names"], "Validation names")
        _ensure_same_items(ref_test_names, m["test_names"], "Test names")
        if m["num_classes"] != num_classes:
            raise ValueError(
                f"num_classes mismatch: {model_outputs[0]['model_id']} has {num_classes}, "
                f"{m['model_id']} has {m['num_classes']}"
            )
        if not torch.equal(ref_valid_targets, m["valid_targets"]):
            raise ValueError("Validation targets mismatch across models.")

    val_logits_by_model = [m["valid_logits"] for m in model_outputs]
    test_logits_by_model = [m["test_logits"] for m in model_outputs]
    model_names = [m["model_name"] for m in model_outputs]

    logging.info("")
    logging.info("============================================================")
    logging.info("Single-model validation macro-F1 summary")
    for m in model_outputs:
        logging.info("  %s -> %.6f", m["model_id"], m["single_val_macro_f1"])

    weight_grid = _generate_simplex_weights(num_models, weights_step)
    logging.info("")
    logging.info("============================================================")
    logging.info(
        "Starting grid search on committee weights: step=%.4f, combinations=%d",
        weights_step,
        len(weight_grid),
    )

    os.makedirs(out_dir, exist_ok=True)
    grid_csv_path = os.path.join(out_dir, "committee_grid_search.csv")
    best_weights = None
    best_f1 = -1.0
    best_idx = -1
    all_results = []

    with open(grid_csv_path, "w", newline="") as f_grid:
        writer = csv.writer(f_grid)
        header = ["attempt", "val_macro_f1"] + [f"w_{name}" for name in model_names]
        writer.writerow(header)

        for attempt_idx, weights in enumerate(weight_grid, start=1):
            combined_logits = torch.zeros_like(val_logits_by_model[0])
            for w, logits in zip(weights, val_logits_by_model):
                if w != 0.0:
                    combined_logits.add_(logits, alpha=float(w))

            f1 = _macro_f1_from_logits(combined_logits, ref_valid_targets, num_classes)
            row = [attempt_idx, float(f1)] + [float(w) for w in weights]
            writer.writerow(row)
            all_results.append((float(f1), weights, attempt_idx))

            logging.info(
                "[Grid %d/%d] F1=%.6f | %s",
                attempt_idx,
                len(weight_grid),
                float(f1),
                ", ".join(f"{name}={w:.4f}" for name, w in zip(model_names, weights)),
            )

            if f1 > best_f1:
                best_f1 = float(f1)
                best_weights = weights
                best_idx = attempt_idx
                logging.info(
                    "  -> NEW BEST at attempt %d: F1=%.6f | %s",
                    best_idx,
                    best_f1,
                    ", ".join(f"{name}={w:.4f}" for name, w in zip(model_names, best_weights)),
                )

    if best_weights is None:
        raise RuntimeError("Grid search produced no candidate.")

    all_results.sort(key=lambda x: x[0], reverse=True)
    top_k = max(int(top_k), 1)
    top_results = all_results[:top_k]

    logging.info("")
    logging.info("============================================================")
    logging.info("Top-%d committee settings", top_k)
    for rank, (f1, weights, attempt_idx) in enumerate(top_results, start=1):
        logging.info(
            "#%d | attempt=%d | F1=%.6f | %s",
            rank,
            attempt_idx,
            f1,
            ", ".join(f"{name}={w:.4f}" for name, w in zip(model_names, weights)),
        )

    combined_test_logits = torch.zeros_like(test_logits_by_model[0])
    for w, logits in zip(best_weights, test_logits_by_model):
        if w != 0.0:
            combined_test_logits.add_(logits, alpha=float(w))
    test_preds = torch.argmax(combined_test_logits, dim=1).tolist()

    submission_path = os.path.join(out_dir, submission_name)
    _write_submission(submission_path, ref_test_names, test_preds)

    summary_path = os.path.join(out_dir, "committee_summary.yaml")
    summary = {
        "device": str(device),
        "config_paths": [str(Path(p)) for p in config_paths],
        "model_ids": [m["model_id"] for m in model_outputs],
        "single_model_val_macro_f1": {
            m["model_id"]: float(m["single_val_macro_f1"]) for m in model_outputs
        },
        "selected_tau_by_model": {m["model_id"]: float(m["best_tau"]) for m in model_outputs},
        "weights_step": float(weights_step),
        "num_combinations": int(len(weight_grid)),
        "best_attempt": int(best_idx),
        "best_val_macro_f1": float(best_f1),
        "best_weights": {name: float(w) for name, w in zip(model_names, best_weights)},
        "top_k": [
            {
                "rank": int(rank),
                "attempt": int(attempt_idx),
                "val_macro_f1": float(f1),
                "weights": {name: float(w) for name, w in zip(model_names, weights)},
            }
            for rank, (f1, weights, attempt_idx) in enumerate(top_results, start=1)
        ],
        "grid_csv_path": grid_csv_path,
        "submission_path": submission_path,
    }
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    logging.info("")
    logging.info("============================================================")
    logging.info("Committee completed.")
    logging.info("Best validation macro-F1: %.6f", best_f1)
    logging.info(
        "Best weights: %s",
        ", ".join(f"{name}={w:.4f}" for name, w in zip(model_names, best_weights)),
    )
    logging.info("Saved grid results: %s", grid_csv_path)
    logging.info("Saved summary: %s", summary_path)
    logging.info("Saved submission: %s", submission_path)


def _build_arg_parser():
    """Build arg parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an automatic model committee pipeline: per-model val/test inference, "
            "single-model macro-F1, weight grid-search, and best submission export."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config-resnet.yaml", "config-efficientnet.yaml", "config-convnext.yaml"],
        help="List of model config files to ensemble.",
    )
    parser.add_argument(
        "--weights-step",
        type=float,
        default=0.05,
        help="Grid step for simplex weights (must divide 1 exactly).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top committee candidates to report.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs/committee",
        help="Output directory for grid results, summary, and final submission.",
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default="submission_committee_best.csv",
        help="Filename of final committee submission inside --out-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    return parser


def main():
    """Run the entrypoint."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_committee(
        config_paths=args.configs,
        out_dir=args.out_dir,
        submission_name=args.submission_name,
        weights_step=args.weights_step,
        top_k=args.top_k,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
