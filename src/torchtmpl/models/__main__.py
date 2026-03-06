# coding: utf-8

# External imports
import torch

# Local imports
from . import build_model


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_resnet():
    cfg = {
        "class": "ResNetModel",
        "stem_channels": 64,
        "stage_channels": [64, 128, 256, 512],
        "blocks_per_stage": [2, 2, 2, 2],
    }
    input_size = (1, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_torchvision_resnet():
    cfg = {
        "class": "TorchvisionResNet",
        "backbone": "resnet18",
        "pretrained": False,
        "dropout": 0.0,
    }
    input_size = (3, 224, 224)
    batch_size = 4
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


if __name__ == "__main__":
    test_linear()
    test_cnn()
    test_resnet()
    test_torchvision_resnet()
