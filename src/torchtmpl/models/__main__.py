# coding: utf-8

# External imports
import torch

# Local imports
from . import build_model


def test_linear():
    """Run linear smoke tests."""
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
    """Run cnn smoke tests."""
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_resnet():
    """Run resnet smoke tests."""
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
    """Run torchvision resnet smoke tests."""
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


def test_convnext_tiny_meta():
    """Run convnext tiny meta smoke tests."""
    cfg = {
        "class": "ConvNeXtTinyMeta",
        "backbone": "convnext_tiny.fb_in22k_ft_in1k",
        "pretrained": False,
        "meta_mlp_dims": [64, 128],
        "fusion_hidden_dim": 256,
        "head_dropout": 0.3,
    }
    input_size = (3, 224, 224)
    batch_size = 2
    num_classes = 86
    model = build_model(cfg, input_size, num_classes)

    images = torch.randn(batch_size, *input_size)
    metadata = torch.randn(batch_size, 4)
    output = model(images, metadata)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_efficientnet_classifier():
    """Run efficientnet classifier smoke tests."""
    cfg = {
        "class": "EfficientNetClassifier",
        "name": "efficientnet_b0",
        "pretrained": False,
        "freeze_backbone": False,
    }
    input_size = (3, 224, 224)
    batch_size = 2
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_convnext_classifier():
    """Run convnext classifier smoke tests."""
    cfg = {
        "class": "ConvNeXtClassifier",
        "name": "convnext_tiny",
        "pretrained": False,
        "freeze_backbone": False,
    }
    input_size = (3, 224, 224)
    batch_size = 2
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_resnet_classifier():
    """Run resnet classifier smoke tests."""
    cfg = {
        "class": "ResNetClassifier",
        "name": "resnet18",
        "pretrained": False,
        "freeze_backbone": False,
    }
    input_size = (3, 224, 224)
    batch_size = 2
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
    test_convnext_tiny_meta()
    test_efficientnet_classifier()
    test_convnext_classifier()
    test_resnet_classifier()
