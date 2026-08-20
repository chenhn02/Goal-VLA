"""ResNet building blocks for AggregationNetwork.

Adapted from detectron2's ResNet implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels,
                 stride=1, num_groups=1, norm="GN", num_norm_groups=32,
                 kernel_size=(1, 3, 1)):
        super().__init__()
        k1, k2, k3 = kernel_size
        p2 = k2 // 2

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, k1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, k2,
                               stride=stride, padding=p2, groups=num_groups, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, k3, stride=1, bias=False)

        if norm == "GN":
            self.norm1 = nn.GroupNorm(num_norm_groups, bottleneck_channels)
            self.norm2 = nn.GroupNorm(num_norm_groups, bottleneck_channels)
            self.norm3 = nn.GroupNorm(num_norm_groups, out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(bottleneck_channels)
            self.norm2 = nn.BatchNorm2d(bottleneck_channels)
            self.norm3 = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_norm_groups, out_channels) if norm == "GN"
                else nn.BatchNorm2d(out_channels),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = F.relu(self.norm2(self.conv2(out)), inplace=True)
        out = self.norm3(self.conv3(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return F.relu(out + identity, inplace=True)


class ResNet:
    @staticmethod
    def make_stage(block_class, num_blocks, in_channels, bottleneck_channels,
                   out_channels, norm="GN", num_norm_groups=32, kernel_size=(1, 3, 1)):
        blocks = []
        for i in range(num_blocks):
            ch_in = in_channels if i == 0 else out_channels
            blocks.append(block_class(
                in_channels=ch_in,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels,
                norm=norm,
                num_norm_groups=num_norm_groups,
                kernel_size=kernel_size,
            ))
        return blocks
