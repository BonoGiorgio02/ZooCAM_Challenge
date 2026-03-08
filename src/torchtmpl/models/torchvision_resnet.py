# coding: utf-8

# Standard imports
import logging
import math

# External imports
import torch
import torch.nn as nn
from torchvision import models


_RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def _resolve_weights(backbone_name, weights_name):
    enum_name = f"{backbone_name.replace('resnet', 'ResNet')}_Weights"
    weights_enum = getattr(models, enum_name, None)
    if weights_enum is None:
        return None

    if weights_name is None:
        return None

    weights_name = str(weights_name)
    if weights_name.upper() == "DEFAULT":
        return weights_enum.DEFAULT

    if not hasattr(weights_enum, weights_name):
        raise ValueError(
            f"Unknown weights '{weights_name}' for backbone '{backbone_name}'. "
            f"Available: {[w.name for w in list(weights_enum)]}"
        )
    return getattr(weights_enum, weights_name)


def _build_backbone(backbone_name, pretrained=True, weights_name="DEFAULT"):
    if backbone_name not in _RESNET_BUILDERS:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose one of {list(_RESNET_BUILDERS.keys())}"
        )

    builder = _RESNET_BUILDERS[backbone_name]

    if not pretrained:
        return builder(weights=None)

    resolved_weights = _resolve_weights(backbone_name, weights_name)

    # torchvision>=0.14 path
    if resolved_weights is not None:
        return builder(weights=resolved_weights)

    # Fallback for environments with older API.
    return builder(pretrained=True)


def _adapt_first_conv(model, in_channels, pretrained):
    if model.conv1.in_channels == in_channels:
        return

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if pretrained:
            old_w = old_conv.weight.data
            old_in = old_w.shape[1]

            if in_channels == 1 and old_in == 3:
                # Standard adaptation RGB -> grayscale.
                new_conv.weight.copy_(old_w.mean(dim=1, keepdim=True))
            else:
                repeat = math.ceil(in_channels / old_in)
                expanded = old_w.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                expanded = expanded * (old_in / float(in_channels))
                new_conv.weight.copy_(expanded)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.conv1 = new_conv


class TorchvisionResNet(nn.Module):
    """
    Wrapper around torchvision ResNet models with project-compatible signature.
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.num_classes = num_classes

        backbone_name = cfg.get("backbone", "resnet18")
        pretrained = bool(cfg.get("pretrained", True))
        weights_name = cfg.get("weights", "DEFAULT")
        dropout = float(cfg.get("dropout", 0.0))
        self.freeze_backbone_flag = bool(cfg.get("freeze_backbone", False))

        self.model = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_name=weights_name,
        )

        _adapt_first_conv(self.model, in_channels=input_size[0], pretrained=pretrained)

        in_features = self.model.fc.in_features
        if dropout > 0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model.fc = nn.Linear(in_features, num_classes)

        if self.freeze_backbone_flag:
            self._set_backbone_requires_grad(False)
            logging.info("TorchvisionResNet: backbone frozen (fc remains trainable).")

    def _set_backbone_requires_grad(self, trainable):
        for name, param in self.model.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def unfreeze_backbone(self):
        self._set_backbone_requires_grad(True)
        self.freeze_backbone_flag = False
        logging.info("TorchvisionResNet: backbone unfrozen.")

    def forward(self, x):
        return self.model(x)
