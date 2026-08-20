"""AggregationNetwork: fuses SD + DINOv2 features into dense descriptors.

Design inspired by the Feature Extractor from ODISE (Xu et al., CVPR 2023).
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from goalvla.feature_matching.resnet import ResNet, BottleneckBlock


class AggregationNetwork(nn.Module):

    def __init__(
        self,
        feature_dims=(640, 1280, 1280, 768),
        projection_dim=768,
        num_norm_groups=32,
        kernel_size=(1, 3, 1),
        feat_map_dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self.feature_dims = list(feature_dims)
        self.feat_map_dropout = feat_map_dropout
        self.device = device

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.bottleneck_layers = nn.ModuleList()
        for dim in self.feature_dims:
            layer = nn.Sequential(*ResNet.make_stage(
                BottleneckBlock,
                num_blocks=1,
                in_channels=dim,
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                norm="GN",
                num_norm_groups=num_norm_groups,
                kernel_size=kernel_size,
            ))
            self.bottleneck_layers.append(layer)
        self.bottleneck_layers = self.bottleneck_layers.to(device)

        mixing_weights = torch.ones(len(self.feature_dims))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    @staticmethod
    def _remap_detectron2_keys(state_dict):
        """Translate detectron2/ODISE BottleneckBlock key names to this module's.

        The pretrained checkpoint stores each conv's norm as a `.norm` submodule
        of the conv (detectron2 CNNBlockBase style: conv1.norm, conv2.norm,
        conv3.norm, shortcut.norm) and the shortcut conv as `shortcut.weight`.
        This ResNet reimplementation instead uses separate norm1/norm2/norm3
        layers and a Sequential shortcut (shortcut.0 conv, shortcut.1 norm).
        Without this remap, all norm and shortcut weights (33 of 47 tensors)
        silently fail to load and stay randomly initialised, which collapses the
        feature discriminability.
        """
        remapped = {}
        for k, v in state_dict.items():
            if k == "self_logit_scale":  # extra tensor absent from this module
                continue
            nk = k
            for n in ("1", "2", "3"):
                nk = nk.replace(f".conv{n}.norm.", f".norm{n}.")
            nk = nk.replace(".shortcut.norm.", ".shortcut.1.")
            if nk.endswith(".shortcut.weight"):
                nk = nk[: -len(".shortcut.weight")] + ".shortcut.0.weight"
            remapped[nk] = v
        return remapped

    def load_pretrained_weights(self, state_dict):
        state_dict = self._remap_detectron2_keys(state_dict)
        own = self.state_dict()
        if "mixing_weights" in own and "mixing_weights" in state_dict:
            n = min(own["mixing_weights"].shape[0], state_dict["mixing_weights"].shape[0])
            own["mixing_weights"][:n] = state_dict["mixing_weights"][:n]
        matching = {
            k: v for k, v in state_dict.items()
            if k in own and k != "mixing_weights" and v.shape == own[k].shape
        }
        own.update(matching)
        self.load_state_dict(own, strict=False)

        loaded = len(matching) + ("mixing_weights" in own and "mixing_weights" in state_dict)
        missing = [k for k in own if k not in matching and k != "mixing_weights"]
        if missing:
            print(f"[AggregationNetwork] loaded {loaded}/{len(own)} params; "
                  f"{len(missing)} still uninitialised from checkpoint: {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        else:
            print(f"[AggregationNetwork] loaded all {loaded}/{len(own)} params from checkpoint.")

    def forward(self, batch):
        """Forward pass. batch: (B, C_total, H, W) with C_total = sum(feature_dims)."""
        if self.feat_map_dropout > 0 and self.training:
            batch = F.dropout(batch, p=self.feat_map_dropout)

        weights = torch.softmax(self.mixing_weights, dim=0)
        output = None
        start = 0
        for i, dim in enumerate(self.feature_dims):
            feats = batch[:, start:start + dim, :, :]
            start += dim
            bottlenecked = weights[i] * self.bottleneck_layers[i](feats)
            output = bottlenecked if output is None else output + bottlenecked
        return output
