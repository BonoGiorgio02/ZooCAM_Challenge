# coding: utf-8

# External imports
import torch
import torch.nn as nn


def get_loss(lossname):
    return eval(f"nn.{lossname}(label_smoothing=0.1)")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

def get_scheduler(cfg, optimizer):
    """
    Build a LR scheduler from config.
    If cfg is None or missing 'algo', returns None.
    """
    if cfg is None:
        return None
    if "algo" not in cfg:
        return None

    params = cfg.get("params", {})
    exec(f"global sched; sched = torch.optim.lr_scheduler.{cfg['algo']}(optimizer, **params)")
    return sched
