# coding: utf-8

# External imports
import torch
import torch.nn as nn


def get_loss(loss_cfg, device=None):
    """Return loss."""
    if isinstance(loss_cfg, str):
        loss_name = loss_cfg
        params = {}
    elif isinstance(loss_cfg, dict):
        loss_name = loss_cfg.get("name", None)
        if loss_name is None:
            raise ValueError("Loss config dictionary must define 'name'")
        params = dict(loss_cfg.get("params", {}))
    else:
        raise TypeError("loss must be either a string or a dictionary")

    if "weight" in params and isinstance(params["weight"], (list, tuple)):
        params["weight"] = torch.tensor(params["weight"], dtype=torch.float32, device=device)

    if not hasattr(nn, loss_name):
        raise ValueError(f"Unknown loss '{loss_name}'")
    return getattr(nn, loss_name)(**params)


def get_optimizer(cfg, params):
    """Return optimizer."""
    algo = cfg["algo"]
    params_dict = dict(cfg.get("params", {}))

    if not hasattr(torch.optim, algo):
        raise ValueError(f"Unknown optimizer '{algo}'")
    return getattr(torch.optim, algo)(params, **params_dict)


def get_scheduler(cfg, optimizer, steps_per_epoch=None):
    """Return scheduler."""
    if cfg is None:
        return None, None, None

    if isinstance(cfg, str):
        algo = cfg
        if algo.lower() == "none":
            return None, None, None
        cfg = {"algo": algo}

    algo = cfg.get("algo", None)
    if algo is None or str(algo).lower() == "none":
        return None, None, None

    params = dict(cfg.get("params", {}))
    step_mode = cfg.get("step", None)
    if step_mode is None:
        step_mode = "metric" if algo == "ReduceLROnPlateau" else "epoch"
    step_mode = step_mode.lower()

    monitor = cfg.get("monitor", "val_loss")

    if algo == "OneCycleLR" and steps_per_epoch is not None:
        params.setdefault("steps_per_epoch", steps_per_epoch)

    if not hasattr(torch.optim.lr_scheduler, algo):
        raise ValueError(f"Unknown scheduler '{algo}'")
    scheduler = getattr(torch.optim.lr_scheduler, algo)(optimizer, **params)

    if step_mode not in {"batch", "epoch", "metric"}:
        raise ValueError("scheduler.step must be one of: batch, epoch, metric")

    return scheduler, step_mode, monitor
