# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

import csv
from tqdm import tqdm

# Local imports
from . import data
from . import models
from . import optim
from . import utils


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

    train_loader, valid_loader, test_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])

    # Build the optimizer
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

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model,
        str(logdir / "best_model.pt"),
        min_is_best=True,
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
        test_loss = utils.test(model, valid_loader, loss, device)
        
        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)  # plateau needs a metric
            else:
                scheduler.step()
                
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"LR: {current_lr:.2e}")

        updated = model_checkpoint.update(test_loss, epoch=e)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {
        "epoch": e,
        "train_loss": train_loss,
        "val_loss": test_loss,
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
    train_loader, valid_loader, test_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes).to(device)
    model.eval()

    ckpt_path = None
    if "test" in config and "checkpoint" in config["test"]:
        ckpt_path = config["test"]["checkpoint"]
    else:
        ckpt_path = os.path.join(config["logging"]["logdir"], "best_model.pt")

    logging.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    all_imgnames = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference on test"):
            if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
                raise ValueError(
                    "test_loader must return (images, imgname/filename). "
                    "Update the test dataset to include filename."
                )

            x, imgnames = batch
            x = x.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            preds = preds.detach().cpu().tolist()

            if torch.is_tensor(imgnames):
                imgnames = imgnames.detach().cpu().tolist()

            all_imgnames.extend(list(imgnames))
            all_labels.extend(preds)

    out_path = config.get("output", {}).get("submission_path", "submission.csv")
    logging.info(f"Writing submission to: {out_path}")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imgname", "label"])
        for name, lab in zip(all_imgnames, all_labels):
            name = os.path.basename(str(name))
            writer.writerow([name, int(lab)])

    logging.info("Done.")



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
