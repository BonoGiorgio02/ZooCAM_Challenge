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
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import yaml
from tqdm import tqdm

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_loss_name(name):
    lower = str(name).lower()
    if lower in {"cross_entropy", "crossentropyloss", "ce"}:
        return "CrossEntropyLoss"
    return name


def _canonical_optimizer_name(name):
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
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)


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
    phase_name, phase_cfg = _get_phase_for_epoch(train_cfg, epoch_1based)
    img_size = _resolve_img_size_for_epoch(base_data_cfg, epoch_1based)

    epoch_data_cfg = copy.deepcopy(base_data_cfg)
    epoch_data_cfg["img_size"] = img_size

    sampler_mode = phase_cfg.get("sampler", epoch_data_cfg.get("sampler_mode", "natural"))
    class_weight_formula = phase_cfg.get(
        "class_weight_formula",
        epoch_data_cfg.get("class_weight_formula", "balanced"),
    )

    epoch_data_cfg["sampler_mode"] = sampler_mode
    epoch_data_cfg["class_weight_formula"] = class_weight_formula

    return epoch_data_cfg, phase_name, phase_cfg


def _build_optimizer(model, optim_cfg, train_cfg):
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


def _predict_logits_with_tta(model, inputs, metadata, tta_modes, device, amp_enabled):
    tta_modes = list(tta_modes) if tta_modes is not None else ["orig"]
    if len(tta_modes) == 0:
        tta_modes = ["orig"]

    use_amp = bool(amp_enabled and device.type == "cuda")
    logits_sum = None
    for mode in tta_modes:
        aug_inputs = utils.apply_tta(inputs, mode)
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

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    model_config = config["model"]
    optim_config = config["optim"]
    logging_config = config["logging"]
    train_cfg = config.get("train", {})

    train_loader, valid_loader, test_loader, input_size, num_classes, tta_transforms = data.get_dataloaders(
        data_config, use_cuda
    )

    loader_cache = {}
    initial_loader_key = (
        int(initial_data_cfg.get("img_size")),
        str(initial_data_cfg.get("sampler_mode", "natural")).lower(),
        str(initial_data_cfg.get("class_weight_formula", "balanced")),
    )
    loader_cache[initial_loader_key] = (train_loader, valid_loader, input_size, num_classes)

    logging.info("= Model")
    model = models.build_model(model_config, input_size, num_classes).to(device)

    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())
    
    # Build the scheduler (optional)
    logging.info("= Scheduler")
    sched_cfg = config.get("scheduler", None)
    scheduler = optim.get_scheduler(sched_cfg, optimizer)
    
    train_cfg = config.get("train", {})
    resume = bool(train_cfg.get("resume", False))
    ckpt_path = train_cfg.get("checkpoint", "")

    start_epoch = 0
    resumed_best = None

    if resume:
        if ckpt_path is None or ckpt_path == "":
            raise ValueError("train.resume is True but train.checkpoint is empty.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logging.info(f"  - Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # New style checkpoint: dict with model/optimizer/scheduler
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)

            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch = int(ckpt.get("epoch", -1)) + 1
            resumed_best = ckpt.get("best_score", None)
        else:
            # Old style: ckpt is directly model weights
            model.load_state_dict(ckpt, strict=True)
            start_epoch = 0
            resumed_best = None

        logging.info(f"Resumed start_epoch={start_epoch}")
    else:
        logging.info("  - Training from scratch (no checkpoint loaded).")

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
        model,
        str(logdir / "best_model.pt"),
        min_is_best=False,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    if resume and resumed_best is not None:
        model_checkpoint.best_score = resumed_best
        logging.info(f"Restored best_score={resumed_best}")

    for e in range(start_epoch, config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device, wandb_log=wandb_log)

        # Test
        test_loss, val_macro_f1 = utils.test(model, valid_loader, loss, device)
        
        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_macro_f1)  # plateau needs a metric
            else:
                scheduler.step()
                
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"LR: {current_lr:.2e}")

        updated = model_checkpoint.update(val_macro_f1, epoch=e)
        logging.info(
            "[%d/%d] Val loss: %.3f | Val macro F1: %.4f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                val_macro_f1,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {
            "epoch": e,
            "train_loss": train_loss,
            "val_loss": test_loss,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
        }
        
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)
    
    if wandb_log is not None:
        wandb.finish()


def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, test_loader, input_size, num_classes, tta_transforms = data.get_dataloaders(
        data_config, use_cuda
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
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    
    test_cfg = config.get("test", {})
    use_tta = test_cfg.get("use_tta", False)
    tta_names = test_cfg.get("tta_names", ["none"])
    tta_weights = test_cfg.get("tta_weights", None)

    all_imgnames = []

    if use_tta:
        logging.info(f"Using TTA with views: {tta_names}")
        
        tta_loaders = []
        reference_dataset = None
        
        for name in tta_names:
            if name not in tta_transforms:
                raise ValueError(f"Unknown TTA transform '{name}'. Available: {list(tta_transforms.keys())}")

            ds = data.InferenceImageDataset(
                root=data_config["testpath"],
                transform=tta_transforms[name],
            )
            
            if reference_dataset is None:
                reference_dataset = ds
            else:
                if [p.name for p in ds.samples] != [p.name for p in reference_dataset.samples]:
                    raise RuntimeError("TTA datasets do not have the same sample ordering.")

            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=data_config.get("batch_size", 64),
                shuffle=False,
                num_workers=data_config.get("num_workers", 4),
                pin_memory=use_cuda,
            )
            tta_loaders.append(dl)
            
        probs = utils.predict_proba_tta(model, tta_loaders, device, weights=tta_weights)
        preds = probs.argmax(dim=1).cpu().tolist()
        all_imgnames = [p.name for p in reference_dataset.samples]
    
    else:
        logging.info("Using standard test inference without TTA")

        probs = utils.predict_proba(model, test_loader, device)
        preds = probs.argmax(dim=1).cpu().tolist()

        for p in test_loader.dataset.samples:
            all_imgnames.append(p.name)


    # with torch.no_grad():
    #     for batch in tqdm(test_loader, desc="Inference on test"):
    #         if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
    #             raise ValueError(
    #                 "test_loader must return (images, imgname/filename). "
    #                 "Update the test dataset to include filename."
    #             )

    #         x, imgnames = batch
    #         x = x.to(device)

    #         logits = model(x)
    #         preds = torch.argmax(logits, dim=1)

    #         preds = preds.detach().cpu().tolist()

    #         if torch.is_tensor(imgnames):
    #             imgnames = imgnames.detach().cpu().tolist()

    #         all_imgnames.extend(list(imgnames))
    #         all_labels.extend(preds)

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
        for name, lab in zip(all_imgnames, preds):
            name = os.path.basename(str(name))
            writer.writerow([name, int(lab)])

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
