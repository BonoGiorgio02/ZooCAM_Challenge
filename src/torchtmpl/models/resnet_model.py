# coding: utf-8

# External imports
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Residual basic block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + shortcut -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetModel(nn.Module):
    """
    ResNet model skeleton.

    Step 2 goal: define the project-compatible interface and forward entry point.
    The actual architecture (stem, residual stages, classifier head) is added later.
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.num_classes = num_classes

        in_channels = input_size[0]
        stem_channels = cfg.get("stem_channels", 64)

        # Step 3: lightweight stem for 128x128 grayscale inputs.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stage_channels = cfg.get("stage_channels", [64, 128, 256, 512])
        blocks_per_stage = cfg.get("blocks_per_stage", [2, 2, 2, 2])
        if len(stage_channels) != len(blocks_per_stage):
            raise ValueError(
                "cfg['stage_channels'] and cfg['blocks_per_stage'] must have same length."
            )

        stages = []
        current_channels = stem_channels
        for stage_idx, (out_channels, num_blocks) in enumerate(
            zip(stage_channels, blocks_per_stage)
        ):
            stage_stride = 1 if stage_idx == 0 else 2
            stage = self._make_stage(
                in_channels=current_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                first_stride=stage_stride,
            )
            stages.append(stage)
            current_channels = out_channels

        self.stages = nn.Sequential(*stages)
        self.features_channels = current_channels

        # Step 5: classification head producing raw logits.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.features_channels, num_classes),
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, first_stride):
        if num_blocks < 1:
            raise ValueError("Each stage must contain at least one BasicBlock.")

        blocks = [BasicBlock(in_channels, out_channels, stride=first_stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
