# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torchvision.models as tvm


def _freeze_module(m: nn.Module) -> None:
    """Execute freeze module."""
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze_module(m: nn.Module) -> None:
    """Execute unfreeze module."""
    for p in m.parameters():
        p.requires_grad = True


class MLPHead(nn.Module):
    """
    Classic classification head:
      Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear
    """
    def __init__(self, in_features: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
        """Initialize the instance."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.net(x)


class EfficientNetClassifier(nn.Module):
    """
    Wrap a torchvision EfficientNet backbone + custom MLP head.
    Supports:
      - pretrained weights
      - freezing backbone
      - optional unfreezing of the last feature blocks
    """
    def __init__(self, cfg: dict, input_size, num_classes: int):
        """Initialize the instance."""
        super().__init__()

        name = cfg.get("name", "efficientnet_b0")   # efficientnet_b0, ..., efficientnet_b7, efficientnet_v2_s, ...
        pretrained = bool(cfg.get("pretrained", True))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))
        unfreeze_last_n = int(cfg.get("unfreeze_last_n", 0))  # how many final feature blocks to unfreeze

        head_hidden = int(cfg.get("head_hidden", 256))
        head_dropout = float(cfg.get("head_dropout", 0.2))

        # --- Build backbone
        backbone, feat_dim = self._build_efficientnet_backbone(name=name, pretrained=pretrained)

        # Replace original classifier with identity
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Freeze/unfreeze policy
        if freeze_backbone:
            _freeze_module(self.backbone)

            if unfreeze_last_n > 0:
                # EfficientNet stores convolutional stages in backbone.features
                n_blocks = len(self.backbone.features)
                start_idx = max(0, n_blocks - unfreeze_last_n)
                for i in range(start_idx, n_blocks):
                    _unfreeze_module(self.backbone.features[i])

        # Head
        self.head = MLPHead(
            in_features=feat_dim,
            num_classes=num_classes,
            hidden=head_hidden,
            dropout=head_dropout,
        )

    @staticmethod
    def _build_efficientnet_backbone(name: str, pretrained: bool):
        """
        Returns: (model, feature_dim)
        """
        name = name.lower()

        if name == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b0(weights=weights)
            feat_dim = 1280

        elif name == "efficientnet_b1":
            weights = tvm.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b1(weights=weights)
            feat_dim = 1280

        elif name == "efficientnet_b2":
            weights = tvm.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b2(weights=weights)
            feat_dim = 1408

        elif name == "efficientnet_b3":
            weights = tvm.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b3(weights=weights)
            feat_dim = 1536

        elif name == "efficientnet_b4":
            weights = tvm.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b4(weights=weights)
            feat_dim = 1792

        elif name == "efficientnet_b5":
            weights = tvm.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b5(weights=weights)
            feat_dim = 2048

        elif name == "efficientnet_b6":
            weights = tvm.EfficientNet_B6_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b6(weights=weights)
            feat_dim = 2304

        elif name == "efficientnet_b7":
            weights = tvm.EfficientNet_B7_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_b7(weights=weights)
            feat_dim = 2560

        elif name == "efficientnet_v2_s":
            weights = tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_v2_s(weights=weights)
            feat_dim = 1280

        elif name == "efficientnet_v2_m":
            weights = tvm.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_v2_m(weights=weights)
            feat_dim = 1280

        elif name == "efficientnet_v2_l":
            weights = tvm.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            model = tvm.efficientnet_v2_l(weights=weights)
            feat_dim = 1280

        else:
            raise ValueError(f"Unknown EfficientNet name: {name}")

        return model, feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        feats = self.backbone(x)   # [B, feat_dim]
        return self.head(feats)