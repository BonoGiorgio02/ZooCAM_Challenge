# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torchvision.models as tvm


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


class MLPHead(nn.Module):
    """
    Classic classification head:
      Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear
    """
    def __init__(self, in_features: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
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
        return self.net(x)


class ConvNeXtClassifier(nn.Module):
    """
    Wrap a torchvision ConvNeXt backbone + custom MLP head.
    Supports:
      - pretrained weights
      - freezing backbone
      - optional unfreezing of the last feature blocks
    """
    def __init__(self, cfg: dict, input_size, num_classes: int):
        super().__init__()

        name = cfg.get("name", "convnext_tiny")   # convnext_tiny/small/base/large
        pretrained = bool(cfg.get("pretrained", True))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))
        unfreeze_last_n = int(cfg.get("unfreeze_last_n", 0))

        head_hidden = int(cfg.get("head_hidden", 256))
        head_dropout = float(cfg.get("head_dropout", 0.2))

        # --- Build backbone
        backbone, feat_dim = self._build_convnext_backbone(name=name, pretrained=pretrained)

        # Replace original classifier with identity
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Freeze/unfreeze policy
        if freeze_backbone:
            _freeze_module(self.backbone)

            if unfreeze_last_n > 0:
                # ConvNeXt stores stages in backbone.features
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
    def _build_convnext_backbone(name: str, pretrained: bool):
        """
        Returns: (model, feature_dim)
        """
        name = name.lower()

        if name == "convnext_tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_tiny(weights=weights)
            feat_dim = 768

        elif name == "convnext_small":
            weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_small(weights=weights)
            feat_dim = 768

        elif name == "convnext_base":
            weights = tvm.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_base(weights=weights)
            feat_dim = 1024

        elif name == "convnext_large":
            weights = tvm.ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_large(weights=weights)
            feat_dim = 1536

        else:
            raise ValueError(f"Unknown ConvNeXt name: {name}")

        return model, feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)   # [B, feat_dim]
        return self.head(feats)