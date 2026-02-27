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
      Linear -> BN -> ReLU -> Dropout -> Linear
    """
    def __init__(self, in_features: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden//2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetClassifier(nn.Module):
    """
    Wrap a torchvision ResNet backbone + custom MLP head.
    Supports:
      - pretrained weights
      - freezing backbone
      - (optional) unfreezing last residual stage: layer4
    """
    def __init__(self, cfg: dict, input_size, num_classes: int):
        super().__init__()

        name = cfg.get("name", "resnet18")  # resnet18/resnet34/resnet50/...
        pretrained = bool(cfg.get("pretrained", True))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))
        unfreeze_layer4 = bool(cfg.get("unfreeze_layer4", False))

        head_hidden = int(cfg.get("head_hidden", 256))
        head_dropout = float(cfg.get("head_dropout", 0.2))

        # --- Build backbone
        backbone, feat_dim = self._build_resnet_backbone(name=name, pretrained=pretrained)

        # Replace original classifier with identity, we will attach our own head
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Freeze/unfreeze policy
        if freeze_backbone:
            _freeze_module(self.backbone)
            # Common fine-tuning trick: unfreeze only last stage
            if unfreeze_layer4:
                _unfreeze_module(self.backbone.layer4)

        # Head
        self.head = MLPHead(in_features=feat_dim, num_classes=num_classes, hidden=head_hidden, dropout=head_dropout)

    @staticmethod
    def _build_resnet_backbone(name: str, pretrained: bool):
        """
        Returns: (model, feature_dim)
        """
        name = name.lower()

        # We use the new torchvision API with weights objects.
        # For pretrained=False we pass weights=None.
        if name == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            model = tvm.resnet18(weights=weights)
            feat_dim = 512
        elif name == "resnet34":
            weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
            model = tvm.resnet34(weights=weights)
            feat_dim = 512
        elif name == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
            model = tvm.resnet50(weights=weights)
            feat_dim = 2048
        elif name == "resnet101":
            weights = tvm.ResNet101_Weights.DEFAULT if pretrained else None
            model = tvm.resnet101(weights=weights)
            feat_dim = 2048
        elif name == "resnet152":
            weights = tvm.ResNet152_Weights.DEFAULT if pretrained else None
            model = tvm.resnet152(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet name: {name}")

        return model, feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # [B, feat_dim]
        return self.head(feats)