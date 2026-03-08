# coding: utf-8

# Standard imports
import contextlib
import copy
import csv
import logging
import os
import pathlib
import random
import sys

# External imports
import numpy as np
import torch
import torchinfo.torchinfo as torchinfo
import yaml
from tqdm import tqdm
import wandb

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def set_global_seed(seed):
    """Set global seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_loss_name(name):
    """Execute canonical loss name."""
    lower = str(name).lower()
    if lower in {"cross_entropy", "crossentropyloss", "ce"}:
        return "CrossEntropyLoss"
    return name


def _canonical_optimizer_name(name):
    """Execute canonical optimizer name."""
    lower = str(name).lower()
    if lower == "adamw":
        return "AdamW"
    if lower == "adam":
        return "Adam"
    if lower == "sgd":
        return "SGD"
    if lower == "rmsprop":
        return "RMSprop"
    return name


def _prepare_loss_config(
    config,
    class_weights,
    device,
    *,
    loss_override=None,
    label_smoothing_override=None,
    use_class_weights_override=None,
):
    """Execute prepare loss config."""
    base_loss_cfg = config["loss"]
    train_cfg = config.get("train", {})
    use_class_weights = bool(train_cfg.get("use_class_weights", False))
    if use_class_weights_override is not None:
        use_class_weights = bool(use_class_weights_override)

    if isinstance(base_loss_cfg, str):
        built = {"name": _canonical_loss_name(base_loss_cfg), "params": {}}
    else:
        built = {
            "name": _canonical_loss_name(base_loss_cfg["name"]),
            "params": dict(base_loss_cfg.get("params", {})),
        }

    if loss_override is not None:
        if isinstance(loss_override, str):
            built["name"] = _canonical_loss_name(loss_override)
            built["params"] = {}
        elif isinstance(loss_override, dict):
            if "name" not in loss_override:
                raise ValueError("Phase loss override dict must include 'name'.")
            built["name"] = _canonical_loss_name(loss_override["name"])
            built["params"] = dict(loss_override.get("params", {}))
        else:
            raise TypeError("loss override must be a string or dictionary")

    if label_smoothing_override is not None:
        built["params"]["label_smoothing"] = float(label_smoothing_override)

    if use_class_weights and class_weights is not None:
        built["params"]["weight"] = class_weights.to(device)

    return built


def _load_model_weights(model, ckpt_path, device, strict=True):
    """Load model weights."""
    def _is_state_dict_like(obj):
        """Execute is state dict like."""
        if not isinstance(obj, dict) or len(obj) == 0:
            return False
        return any(torch.is_tensor(v) for v in obj.values())

    def _strip_prefix_if_all_keys_match(state_dict, prefix):
        """Execute strip prefix if all keys match."""
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return state_dict
        if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict

    def _extract_state_dict_candidates(raw_state):
        """Execute extract state dict candidates."""
        queue = [raw_state]
        visited = set()
        candidates = []
        unwrap_keys = ("model_state_dict", "state_dict", "model", "net", "weights")

        while queue:
            item = queue.pop(0)
            marker = id(item)
            if marker in visited:
                continue
            visited.add(marker)

            if hasattr(item, "state_dict") and callable(item.state_dict):
                item = item.state_dict()

            if _is_state_dict_like(item):
                candidates.append(item)
                continue

            if isinstance(item, dict):
                for key in unwrap_keys:
                    if key in item:
                        queue.append(item[key])

        return candidates

    def _load_with_key_normalization(state_dict):
        """Load with key normalization."""
        attempts = []
        for candidate in (
            state_dict,
            _strip_prefix_if_all_keys_match(state_dict, "module."),
            _strip_prefix_if_all_keys_match(state_dict, "model."),
            _strip_prefix_if_all_keys_match(
                _strip_prefix_if_all_keys_match(state_dict, "model."),
                "module.",
            ),
        ):
            if isinstance(candidate, dict):
                key_tuple = tuple(candidate.keys())
            else:
                key_tuple = None
            if not any(existing == key_tuple for existing, _ in attempts):
                attempts.append((key_tuple, candidate))

        last_error = None
        for _, candidate in attempts:
            try:
                model.load_state_dict(candidate, strict=strict)
                return
            except RuntimeError as err:
                last_error = err

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not load checkpoint: no valid state_dict candidate found.")

    state = torch.load(ckpt_path, map_location=device)
    state_dict_candidates = _extract_state_dict_candidates(state)

    if len(state_dict_candidates) == 0:
        top_keys = list(state.keys()) if isinstance(state, dict) else []
        raise RuntimeError(
            f"Checkpoint format not recognized for '{ckpt_path}'. Top-level keys: {top_keys}"
        )

    last_error = None
    for candidate in state_dict_candidates:
        try:
            _load_with_key_normalization(candidate)
            return
        except RuntimeError as err:
            last_error = err

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load checkpoint from '{ckpt_path}'.")


def _resume_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    resume_checkpoint,
    device,
    strict=True,
    ema=None,
):
    """
    Resume from either:
    - full training checkpoint dict (model/optimizer/scheduler/scaler/epoch/best_score)
    - plain model state_dict (best_model.pt / last_model.pt)
    """
    state = torch.load(resume_checkpoint, map_location=device)

    start_epoch = 0
    best_score = None

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)

        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict", None) is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict", None) is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        if ema is not None and state.get("ema_state_dict", None) is not None:
            ema.load_state_dict(state["ema_state_dict"])

        if state.get("epoch", None) is not None:
            start_epoch = int(state["epoch"]) + 1
        best_score = state.get("best_score", None)
        logging.info(
            f"Resumed full training state from '{resume_checkpoint}' "
            f"(next epoch index: {start_epoch})."
        )
        return start_epoch, best_score

    model.load_state_dict(state, strict=strict)
    logging.info(
        f"Resumed model weights from '{resume_checkpoint}' "
        "(optimizer/scheduler state not available in this checkpoint)."
    )
    logging.info(
        "If you want to skip already-completed epochs with a plain model checkpoint, "
        "set train.resume_epoch in the config."
    )
    return start_epoch, best_score


def _get_phase_for_epoch(train_cfg, epoch_1based):
    """Return phase for epoch."""
    phase_items = []
    for key, value in train_cfg.items():
        if not str(key).startswith("phase") or not isinstance(value, dict):
            continue
        if "epochs" not in value:
            continue
        window = value["epochs"]
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"{key}.epochs must be [start, end]")
        start, end = int(window[0]), int(window[1])
        phase_items.append((start, end, key, value))

    phase_items.sort(key=lambda x: x[0])
    for start, end, key, value in phase_items:
        if start <= epoch_1based <= end:
            return key, value

    return None, {}


def _resolve_img_size_for_epoch(data_cfg, epoch_1based):
    """Resolve img size for epoch."""
    default_size = int(data_cfg.get("img_size", 128))
    schedule = data_cfg.get("progressive_resize", None)
    if not schedule:
        return default_size

    for entry in schedule:
        epochs = entry.get("epochs", None)
        if not isinstance(epochs, (list, tuple)) or len(epochs) != 2:
            continue
        start, end = int(epochs[0]), int(epochs[1])
        if start <= epoch_1based <= end:
            return int(entry.get("image_size", default_size))

    return default_size


def _build_epoch_data_config(base_data_cfg, train_cfg, epoch_1based):
    """Build epoch data config."""
    phase_name, phase_cfg = _get_phase_for_epoch(train_cfg, epoch_1based)
    img_size = _resolve_img_size_for_epoch(base_data_cfg, epoch_1based)

    epoch_data_cfg = copy.deepcopy(base_data_cfg)
    epoch_data_cfg["img_size"] = img_size

    if "sampler" in phase_cfg:
        epoch_data_cfg["sampler_mode"] = phase_cfg["sampler"]
    if "class_weight_formula" in phase_cfg:
        epoch_data_cfg["class_weight_formula"] = phase_cfg["class_weight_formula"]

    return epoch_data_cfg, phase_name, phase_cfg


def _infer_sampler_mode(data_cfg):
    """Execute infer sampler mode."""
    sampler_mode = str(data_cfg.get("sampler_mode", "")).lower().strip()
    if sampler_mode:
        return sampler_mode

    imbalance_cfg = data_cfg.get("imbalance", {})
    if imbalance_cfg.get("use_weighted_sampler", False):
        return "weighted_random"
    if bool(data_cfg.get("sampler", {}).get("enabled", False)):
        return "weighted_random"
    return "natural"


def _infer_class_weight_formula(data_cfg):
    """Execute infer class weight formula."""
    if "class_weight_formula" in data_cfg:
        return str(data_cfg["class_weight_formula"])

    imbalance_cfg = data_cfg.get("imbalance", {})
    if "class_weight_formula" in imbalance_cfg:
        return str(imbalance_cfg["class_weight_formula"])

    if bool(data_cfg.get("sampler", {}).get("enabled", False)):
        return f"legacy_pow_{data_cfg.get('sampler', {}).get('alpha', 0.5)}"
    return "balanced"


def _loader_cache_key(data_cfg):
    """Execute loader cache key."""
    return (
        int(data_cfg.get("img_size", 128)),
        _infer_sampler_mode(data_cfg),
        _infer_class_weight_formula(data_cfg),
    )


def _build_optimizer(model, optim_cfg, train_cfg):
    """Build optimizer."""
    params = dict(optim_cfg.get("params", {}))

    optimizer_name = train_cfg.get("optimizer", optim_cfg.get("algo", "AdamW"))
    optimizer_name = _canonical_optimizer_name(optimizer_name)

    if "weight_decay" in train_cfg:
        params["weight_decay"] = float(train_cfg["weight_decay"])
    if "betas" in train_cfg:
        params["betas"] = tuple(train_cfg["betas"])

    lr_backbone = train_cfg.get("lr_backbone", None)
    lr_head = train_cfg.get("lr_head", None)

    if lr_head is not None and hasattr(model, "get_param_groups"):
        if lr_backbone is None:
            lr_backbone = float(params.get("lr", lr_head))

        param_groups = model.get_param_groups(
            lr_backbone=float(lr_backbone),
            lr_head=float(lr_head),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        params.pop("lr", None)
        return optim.get_optimizer({"algo": optimizer_name, "params": params}, param_groups)

    if "lr" in train_cfg:
        params["lr"] = float(train_cfg["lr"])

    return optim.get_optimizer({"algo": optimizer_name, "params": params}, model.parameters())


def _build_scheduler(optimizer, optim_cfg, train_cfg, nepochs, steps_per_epoch):
    """Build scheduler."""
    scheduler_name = str(train_cfg.get("scheduler", "")).lower().strip()
    if scheduler_name in {"cosine", "cosineannealing", "cosineannealinglr"}:
        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.1))

        if warmup_epochs > 0 and nepochs > warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(nepochs - warmup_epochs, 1),
                eta_min=min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            return scheduler, "epoch", None

        if warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            )
            return scheduler, "epoch", None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(nepochs, 1),
            eta_min=min_lr,
        )
        return scheduler, "epoch", None

    scheduler_cfg = optim_cfg.get("scheduler", None)
    return optim.get_scheduler(
        scheduler_cfg,
        optimizer,
        steps_per_epoch=steps_per_epoch,
    )


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


def _select_best_tau(
    model,
    loader,
    device,
    num_classes,
    tau_grid,
    class_priors,
    amp_enabled,
):
    """Select best tau."""
    if not tau_grid:
        return 0.0, {}

    use_amp = bool(amp_enabled and device.type == "cuda")
    model.eval()

    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Collect validation logits"):
            inputs, metadata, targets = utils.unpack_supervised_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = utils.model_forward(model, inputs, metadata)

            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if len(all_logits) == 0:
        return 0.0, {}

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    class_priors = class_priors.detach().cpu().clamp_min(1e-12)
    log_prior = torch.log(class_priors)

    best_tau = float(tau_grid[0])
    best_f1 = -1.0
    tau_scores = {}

    for tau in tau_grid:
        tau = float(tau)
        preds = torch.argmax(logits - tau * log_prior.unsqueeze(0), dim=1)
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        utils._update_confusion_matrix(confusion, preds, targets, num_classes)
        f1 = utils.macro_f1_from_confusion(confusion)
        tau_scores[tau] = f1

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau, tau_scores


def train(config):
    """Execute train."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    wandb_log = None
    if "wandb" in config.get("logging", {}):
        wandb_config = config["logging"]["wandb"]
        wandb.init(
            project=wandb_config.get("project", "ZooCamChallenge"),
            entity=wandb_config.get("entity", None),
            config=config,
        )
        wandb_log = wandb.log
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    data_config = config["data"]
    model_config = config["model"]
    optim_config = config["optim"]
    logging_config = config["logging"]
    train_cfg = config.get("train", {})

    seed = data_config.get("seed", 0)
    set_global_seed(seed)

    nepochs = int(config.get("nepochs", train_cfg.get("epochs", train_cfg.get("nepochs", 1))))

    # Backward compatibility: older configs define scheduler at the root level.
    if config.get("scheduler", None) is not None and "scheduler" not in optim_config:
        optim_config = copy.deepcopy(optim_config)
        optim_config["scheduler"] = config["scheduler"]

    logging.info("= Building the dataloaders")
    initial_data_cfg, _, _ = _build_epoch_data_config(
        base_data_cfg=data_config,
        train_cfg=train_cfg,
        epoch_1based=1,
    )
    train_loader, valid_loader, _, input_size, num_classes = data.get_dataloaders(
        initial_data_cfg,
        use_cuda,
        build_test=False,
    )

    loader_cache = {}
    initial_loader_key = _loader_cache_key(initial_data_cfg)
    loader_cache[initial_loader_key] = (train_loader, valid_loader, input_size, num_classes)

    logging.info("= Model")
    model = models.build_model(model_config, input_size, num_classes).to(device)

    logging.info("= Optimizer")
    optimizer = _build_optimizer(model, optim_config, train_cfg)

    logging.info("= Scheduler")
    scheduler, scheduler_step, scheduler_monitor = _build_scheduler(
        optimizer,
        optim_config,
        train_cfg,
        nepochs,
        len(train_loader),
    )

    amp_enabled = bool(train_cfg.get("amp", True))
    grad_clip_norm = train_cfg.get("grad_clip_norm", train_cfg.get("max_grad_norm", None))
    unfreeze_backbone_epoch = train_cfg.get("unfreeze_backbone_epoch", None)
    resume_checkpoint = train_cfg.get("resume_checkpoint", None)
    if (resume_checkpoint is None or resume_checkpoint == "") and bool(train_cfg.get("resume", False)):
        resume_checkpoint = train_cfg.get("checkpoint", None)
    if resume_checkpoint == "":
        resume_checkpoint = None

    resume_in_place = bool(train_cfg.get("resume_in_place", bool(train_cfg.get("resume", False))))
    resume_strict = bool(train_cfg.get("resume_strict", True))
    resume_epoch_override = train_cfg.get("resume_epoch", None)
    ema_decay = train_cfg.get("ema_decay", None)

    if resume_checkpoint is not None and not pathlib.Path(resume_checkpoint).is_file():
        raise FileNotFoundError(f"train.resume_checkpoint not found: {resume_checkpoint}")

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))
    ema = utils.ModelEMA(model, decay=float(ema_decay)) if ema_decay is not None else None

    logname = model_config["class"]
    if resume_checkpoint is not None and resume_in_place:
        logdir = str(pathlib.Path(resume_checkpoint).parent)
        os.makedirs(logdir, exist_ok=True)
    else:
        logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
        os.makedirs(logdir, exist_ok=True)
    logging.info(f"Will be logging into {logdir}")

    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    if bool(getattr(model, "expects_metadata", False)):
        summary = torchinfo.summary(
            model,
            input_data=[
                torch.zeros((1,) + tuple(input_size), device=device),
                torch.zeros((1, 4), device=device),
            ],
            device=str(device),
        )
    else:
        summary = torchinfo.summary(model, input_size=(1,) + tuple(input_size), device=str(device))

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{summary}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}\n"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    selection_metric = train_cfg.get("selection_metric", "val_macro_f1")
    selection_mode = train_cfg.get("selection_mode", "max").lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError("train.selection_mode must be 'min' or 'max'")

    start_epoch = 0
    resumed_best_score = None
    if resume_checkpoint:
        start_epoch, resumed_best_score = _resume_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_checkpoint=resume_checkpoint,
            device=device,
            strict=resume_strict,
            ema=ema,
        )
        if resume_epoch_override is not None:
            start_epoch = int(resume_epoch_override)
            logging.info(f"Using train.resume_epoch override: start_epoch={start_epoch}")

    checkpoint_model = ema.ema if ema is not None else model
    model_checkpoint = utils.ModelCheckpoint(
        checkpoint_model,
        str(logdir / "best_model.pt"),
        min_is_best=(selection_mode == "min"),
    )
    if resumed_best_score is not None:
        model_checkpoint.best_score = resumed_best_score

    current_loader_key = initial_loader_key

    for e in range(start_epoch, nepochs):
        epoch_1based = e + 1

        epoch_data_cfg, phase_name, phase_cfg = _build_epoch_data_config(
            base_data_cfg=data_config,
            train_cfg=train_cfg,
            epoch_1based=epoch_1based,
        )
        loader_key = _loader_cache_key(epoch_data_cfg)

        if loader_key != current_loader_key:
            if loader_key not in loader_cache:
                logging.info(
                    "Switching data pipeline for epoch %s: img_size=%s sampler=%s formula=%s",
                    epoch_1based,
                    loader_key[0],
                    loader_key[1],
                    loader_key[2],
                )
                rebuilt_train, rebuilt_valid, _, rebuilt_input_size, rebuilt_num_classes = data.get_dataloaders(
                    epoch_data_cfg,
                    use_cuda,
                    build_test=False,
                )
                loader_cache[loader_key] = (
                    rebuilt_train,
                    rebuilt_valid,
                    rebuilt_input_size,
                    rebuilt_num_classes,
                )
            train_loader, valid_loader, _, _ = loader_cache[loader_key]
            current_loader_key = loader_key
        else:
            train_loader, valid_loader, _, _ = loader_cache[current_loader_key]

        phase_loss = phase_cfg.get("loss", None)
        phase_label_smoothing = phase_cfg.get("label_smoothing", None)
        phase_use_class_weights = phase_cfg.get("use_class_weights", None)

        loss_cfg = _prepare_loss_config(
            config,
            getattr(train_loader, "class_weights", None),
            device,
            loss_override=phase_loss,
            label_smoothing_override=phase_label_smoothing,
            use_class_weights_override=phase_use_class_weights,
        )
        loss = optim.get_loss(loss_cfg, device=device)

        if (
            unfreeze_backbone_epoch is not None
            and e == int(unfreeze_backbone_epoch)
            and hasattr(model, "unfreeze_backbone")
        ):
            model.unfreeze_backbone()

        train_loss = utils.train(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            grad_clip_norm=grad_clip_norm,
            ema=ema,
            wandb_log = wandb_log
        )

        eval_model = ema.ema if ema is not None else model
        val_loss, val_macro_f1 = utils.evaluate(
            eval_model,
            valid_loader,
            loss,
            device,
            amp_enabled=amp_enabled,
            num_classes=num_classes,
        )

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
        }

        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)
    
    if wandb_log is not None:
        wandb.finish()

        if scheduler is not None:
            if scheduler_step == "epoch":
                scheduler.step()
            elif scheduler_step == "metric":
                monitor_key = scheduler_monitor if scheduler_monitor is not None else "val_loss"
                if monitor_key not in metrics:
                    raise ValueError(
                        f"Scheduler monitor '{monitor_key}' not found in metrics "
                        f"{list(metrics.keys())}"
                    )
                scheduler.step(metrics[monitor_key])

        if selection_metric not in metrics:
            raise ValueError(
                f"Selection metric '{selection_metric}' not found in metrics {list(metrics.keys())}"
            )

        best_updated = model_checkpoint.update(metrics[selection_metric])
        torch.save(model.state_dict(), str(logdir / "last_model.pt"))
        if ema is not None:
            torch.save(ema.ema.state_dict(), str(logdir / "last_model_ema.pt"))

        torch.save(
            {
                "epoch": e,
                "best_score": model_checkpoint.best_score,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "selection_metric": selection_metric,
                "current_phase": phase_name,
                "img_size": loader_key[0],
                "sampler_mode": loader_key[1],
            },
            str(logdir / "training_state.pt"),
        )

        current_lr = optimizer.param_groups[0].get("lr", float("nan"))
        phase_tag = phase_name if phase_name is not None else "default"
        logging.info(
            f"[{epoch_1based}/{nepochs}] "
            f"phase={phase_tag} "
            f"img={loader_key[0]} "
            f"sampler={loader_key[1]} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"lr={current_lr:.3e} "
            f"{'[>> BETTER <<]' if best_updated else ''}"
        )


def test(config):
    """Execute test."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, test_loader, input_size, num_classes = data.get_dataloaders(
        data_config,
        use_cuda,
        build_test=True,
    )

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes).to(device)
    model.eval()

    if "test" in config and "checkpoint" in config["test"]:
        ckpt_path = config["test"]["checkpoint"]
    else:
        ckpt_path = os.path.join(config["logging"]["logdir"], "best_model.pt")

    logging.info(f"Loading checkpoint: {ckpt_path}")
    _load_model_weights(model, ckpt_path, device=device, strict=True)

    inference_cfg = dict(config.get("inference", {}))
    legacy_test_cfg = config.get("test", {})
    if "tta" not in inference_cfg and bool(legacy_test_cfg.get("use_tta", False)):
        inference_cfg["tta"] = legacy_test_cfg.get("tta_names", ["orig"])

    tta_modes = inference_cfg.get("tta", ["orig"])
    if isinstance(tta_modes, str):
        tta_modes = [tta_modes]

    tta_norm_mean = inference_cfg.get("tta_norm_mean", data_config.get("mean", None))
    tta_norm_std = inference_cfg.get("tta_norm_std", data_config.get("std", None))

    tau_grid = inference_cfg.get("logit_adjustment_tau_grid", [0.0])
    if isinstance(tau_grid, (int, float)):
        tau_grid = [float(tau_grid)]

    use_tau_sweep = bool(inference_cfg.get("sweep_logit_adjustment", True))
    fixed_tau = inference_cfg.get("selected_tau", None)
    amp_enabled = bool(config.get("train", {}).get("amp", True))

    class_priors = getattr(train_loader, "class_priors", None)
    if class_priors is None:
        class_priors = torch.ones(num_classes, dtype=torch.float32) / float(num_classes)

    if fixed_tau is not None:
        best_tau = float(fixed_tau)
        tau_scores = {}
        logging.info(f"Using fixed logit-adjustment tau from config: {best_tau:.3f}")
    elif use_tau_sweep and tau_grid:
        best_tau, tau_scores = _select_best_tau(
            model=model,
            loader=valid_loader,
            device=device,
            num_classes=num_classes,
            tau_grid=tau_grid,
            class_priors=class_priors,
            amp_enabled=amp_enabled,
        )
        logging.info(
            "Best validation tau=%.3f (macro-F1=%.4f)",
            best_tau,
            tau_scores.get(best_tau, float("nan")),
        )
    else:
        best_tau = 0.0
        tau_scores = {}

    log_prior = torch.log(class_priors.to(device).clamp_min(1e-12))

    all_imgnames = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference on test"):
            inputs, metadata, imgnames = utils.unpack_inference_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)

            logits = _predict_logits_with_tta(
                model=model,
                inputs=inputs,
                metadata=metadata,
                tta_modes=tta_modes,
                device=device,
                amp_enabled=amp_enabled,
                tta_norm_mean=tta_norm_mean,
                tta_norm_std=tta_norm_std,
            )
            logits = logits - best_tau * log_prior.unsqueeze(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()

            if torch.is_tensor(imgnames):
                imgnames = imgnames.detach().cpu().tolist()

            all_imgnames.extend(list(imgnames))
            all_labels.extend(preds)

    out_path = config.get("output", {}).get("submission_path", "submission.csv")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if tau_scores:
        tau_report = os.path.join(out_dir if out_dir else ".", "tau_sweep.yaml")
        with open(tau_report, "w") as f:
            yaml.safe_dump(
                {
                    "selected_tau": float(best_tau),
                    "tau_scores": {float(k): float(v) for k, v in tau_scores.items()},
                },
                f,
                sort_keys=True,
            )
        logging.info(f"Saved tau sweep report: {tau_report}")

    logging.info(f"Writing submission to: {out_path}")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imgname", "label"])
        for name, lab in zip(all_imgnames, all_labels):
            writer.writerow([os.path.basename(str(name)), int(lab)])

    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    command = sys.argv[2]
    if command == "train":
        train(config)
    elif command == "test":
        test(config)
    else:
        raise ValueError("Unknown command. Use 'train' or 'test'.")
