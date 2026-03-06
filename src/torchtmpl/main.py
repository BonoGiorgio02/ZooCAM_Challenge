# coding: utf-8

# Standard imports
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


def _prepare_loss_config(config, class_weights, device):
    loss_cfg = config["loss"]
    train_cfg = config.get("train", {})
    use_class_weights = bool(train_cfg.get("use_class_weights", False))

    if isinstance(loss_cfg, str):
        built = {"name": loss_cfg, "params": {}}
    else:
        built = {
            "name": loss_cfg["name"],
            "params": dict(loss_cfg.get("params", {})),
        }

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


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    data_config = config["data"]
    model_config = config["model"]
    optim_config = config["optim"]
    logging_config = config["logging"]
    train_cfg = config.get("train", {})

    seed = data_config.get("seed", 0)
    set_global_seed(seed)

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    train_loader, valid_loader, _, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model = models.build_model(model_config, input_size, num_classes).to(device)

    # Build loss
    logging.info("= Loss")
    class_weights = getattr(train_loader, "class_weights", None)
    loss = optim.get_loss(_prepare_loss_config(config, class_weights, device), device=device)

    # Build optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build scheduler
    scheduler_cfg = optim_config.get("scheduler", None)
    scheduler, scheduler_step, scheduler_monitor = optim.get_scheduler(
        scheduler_cfg,
        optimizer,
        steps_per_epoch=len(train_loader),
    )

    nepochs = int(config.get("nepochs", train_cfg.get("nepochs", 1)))
    amp_enabled = bool(train_cfg.get("amp", True))
    grad_clip_norm = train_cfg.get("max_grad_norm", None)
    unfreeze_backbone_epoch = train_cfg.get("unfreeze_backbone_epoch", None)
    resume_checkpoint = train_cfg.get("resume_checkpoint", None)
    resume_in_place = bool(train_cfg.get("resume_in_place", True))
    resume_strict = bool(train_cfg.get("resume_strict", True))
    resume_epoch_override = train_cfg.get("resume_epoch", None)

    if resume_checkpoint is not None and not pathlib.Path(resume_checkpoint).is_file():
        raise FileNotFoundError(f"train.resume_checkpoint not found: {resume_checkpoint}")

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    # Build the callbacks and output directory
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

    # Keep summary lightweight (batch size 1 to avoid startup OOM).
    summary_input_size = (1,) + tuple(input_size)
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=summary_input_size, device=str(device))}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}\n"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

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
        )
        if resume_epoch_override is not None:
            start_epoch = int(resume_epoch_override)
            logging.info(f"Using train.resume_epoch override: start_epoch={start_epoch}")

    model_checkpoint = utils.ModelCheckpoint(
        model,
        str(logdir / "best_model.pt"),
        min_is_best=(selection_mode == "min"),
    )
    if resumed_best_score is not None:
        model_checkpoint.best_score = resumed_best_score

    for e in range(start_epoch, nepochs):
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
        )

        val_loss, val_macro_f1 = utils.evaluate(
            model,
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

        if scheduler is not None:
            if scheduler_step == "epoch":
                scheduler.step()
            elif scheduler_step == "metric":
                if scheduler_monitor not in metrics:
                    raise ValueError(
                        f"Scheduler monitor '{scheduler_monitor}' not found in metrics "
                        f"{list(metrics.keys())}"
                    )
                scheduler.step(metrics[scheduler_monitor])

        if selection_metric not in metrics:
            raise ValueError(
                f"Selection metric '{selection_metric}' not found in metrics {list(metrics.keys())}"
            )

        best_updated = model_checkpoint.update(metrics[selection_metric])
        torch.save(model.state_dict(), str(logdir / "last_model.pt"))
        torch.save(
            {
                "epoch": e,
                "best_score": model_checkpoint.best_score,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "selection_metric": selection_metric,
            },
            str(logdir / "training_state.pt"),
        )

        current_lr = optimizer.param_groups[0].get("lr", float("nan"))
        logging.info(
            f"[{e + 1}/{nepochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"lr={current_lr:.3e} "
            f"{'[>> BETTER <<]' if best_updated else ''}"
        )


def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("= Building the dataloaders")
    data_config = config["data"]
    _, _, test_loader, input_size, num_classes = data.get_dataloaders(
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
    _load_model_weights(model, ckpt_path, device=device, strict=True)

    all_imgnames = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference on test"):
            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                raise ValueError(
                    "test_loader must return (images, imgname/filename). "
                    "Update the test dataset to include filename."
                )

            x, imgnames = batch
            x = x.to(device, non_blocking=True)

            logits = model(x)
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()

            if torch.is_tensor(imgnames):
                imgnames = imgnames.detach().cpu().tolist()

            all_imgnames.extend(list(imgnames))
            all_labels.extend(preds)

    out_path = config.get("output", {}).get("submission_path", "submission.csv")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
