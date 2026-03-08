# coding: utf-8

# Standard imports
import logging
import math

# External imports
import torch
import torch.nn as nn
from torchvision import models as tv_models

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None


def _adapt_conv2d_in_channels(conv, in_channels, pretrained):
    """Execute adapt conv2d in channels."""
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if pretrained:
            old_w = conv.weight.data
            old_in = old_w.shape[1]

            if in_channels == 1 and old_in == 3:
                new_conv.weight.copy_(old_w.mean(dim=1, keepdim=True))
            else:
                repeat = math.ceil(in_channels / old_in)
                expanded = old_w.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                expanded = expanded * (old_in / float(in_channels))
                new_conv.weight.copy_(expanded)

            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias.data)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

    return new_conv


class _TorchvisionConvNeXtTinyFeatures(nn.Module):
    """Internal class for torchvision conv ne xt tiny features."""
    def __init__(self, pretrained=True, in_channels=3):
        """Initialize the instance."""
        super().__init__()

        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = tv_models.convnext_tiny(weights=weights)
        if in_channels != 3:
            backbone.features[0][0] = _adapt_conv2d_in_channels(
                backbone.features[0][0],
                in_channels=in_channels,
                pretrained=pretrained,
            )

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.num_features = backbone.classifier[-1].in_features

    def forward(self, x):
        """Run a forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _build_backbone(backbone_name, pretrained, in_channels):
    """Build backbone."""
    if timm is not None:
        try:
            model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                in_chans=in_channels,
            )
            feat_dim = int(getattr(model, "num_features", 768))
            logging.info(
                f"ConvNeXtTinyMeta: using timm backbone '{backbone_name}' "
                f"(num_features={feat_dim})"
            )
            return model, feat_dim
        except Exception as err:
            logging.warning(
                "ConvNeXtTinyMeta: failed to build timm backbone "
                f"'{backbone_name}' ({err}). Falling back to torchvision convnext_tiny."
            )

    fallback = _TorchvisionConvNeXtTinyFeatures(
        pretrained=pretrained,
        in_channels=in_channels,
    )
    logging.info(
        "ConvNeXtTinyMeta: using torchvision convnext_tiny fallback "
        "(22K pretraining unavailable in this mode)."
    )
    return fallback, fallback.num_features


class ConvNeXtTinyMeta(nn.Module):
    """
    ConvNeXt-Tiny image branch + metadata branch (4 scalar features) + fusion head.
    """

    expects_metadata = True

    def __init__(self, cfg, input_size, num_classes):
        """Initialize the instance."""
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.num_classes = num_classes

        backbone_name = cfg.get("backbone", "convnext_tiny.fb_in22k_ft_in1k")
        pretrained = bool(cfg.get("pretrained", True))
        meta_dims = list(cfg.get("meta_mlp_dims", [64, 128]))
        fusion_hidden_dim = int(cfg.get("fusion_hidden_dim", 256))
        head_dropout = float(cfg.get("head_dropout", 0.30))
        meta_dropout = float(cfg.get("meta_dropout", 0.10))
        freeze_backbone = bool(cfg.get("freeze_backbone", False))

        if len(meta_dims) != 2:
            raise ValueError("cfg.meta_mlp_dims must be a list with exactly 2 entries.")

        self.backbone, image_feat_dim = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_channels=int(input_size[0]),
        )
        self.image_feat_dim = int(image_feat_dim)

        meta_dim_1, meta_dim_2 = int(meta_dims[0]), int(meta_dims[1])
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, meta_dim_1),
            nn.GELU(),
            nn.LayerNorm(meta_dim_1),
            nn.Dropout(p=meta_dropout),
            nn.Linear(meta_dim_1, meta_dim_2),
            nn.GELU(),
        )
        self.meta_feat_dim = meta_dim_2

        self.fusion_head = nn.Sequential(
            nn.Linear(self.image_feat_dim + self.meta_feat_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        if freeze_backbone:
            self._set_backbone_requires_grad(False)
            logging.info("ConvNeXtTinyMeta: backbone frozen.")

    def _set_backbone_requires_grad(self, trainable):
        """Set backbone requires grad."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        for param in self.meta_mlp.parameters():
            param.requires_grad = True
        for param in self.fusion_head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Execute unfreeze backbone."""
        self._set_backbone_requires_grad(True)
        logging.info("ConvNeXtTinyMeta: backbone unfrozen.")

    def get_param_groups(self, lr_backbone, lr_head, weight_decay):
        """Return param groups."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for module in (self.meta_mlp, self.fusion_head)
            for p in module.parameters()
            if p.requires_grad
        ]

        groups = []
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": float(lr_backbone),
                    "weight_decay": float(weight_decay),
                }
            )
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": float(lr_head),
                    "weight_decay": float(weight_decay),
                }
            )
        return groups

    def forward(self, x, meta):
        """Run a forward pass."""
        if meta is None:
            raise ValueError("ConvNeXtTinyMeta expects metadata tensor with shape [B, 4].")

        img_feat = self.backbone(x)
        if img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion_head(fused)
