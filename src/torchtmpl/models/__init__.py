# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
from .resnet_model import *
from .torchvision_resnet import *


def build_model(cfg, input_size, num_classes):
    model_name = cfg["class"]
    if model_name not in globals():
        raise ValueError(f"Unknown model class '{model_name}'")
    return globals()[model_name](cfg, input_size, num_classes)
