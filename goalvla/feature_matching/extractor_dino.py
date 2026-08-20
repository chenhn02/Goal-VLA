"""DINOv2 feature extractor.

Extracts dense token features from DINOv2 ViT-B/14.
"""

import math
import types
from typing import List, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
from PIL import Image


class ViTExtractor:

    def __init__(self, model_type: str = "dinov2_vitb14", stride: int = 14, device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.model = self._create_model(model_type)
        self.model = self._patch_resolution(self.model, stride)
        self.model.eval().to(device)

        self.p = self.model.patch_embed.patch_size
        if isinstance(self.p, tuple):
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self._hook_handlers = []
        self.num_patches = None

    @staticmethod
    def _create_model(model_type: str) -> nn.Module:
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        if "v2" in model_type:
            return torch.hub.load("facebookresearch/dinov2", model_type)
        elif "dino" in model_type:
            return torch.hub.load("facebookresearch/dino:main", model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _patch_resolution(model: nn.Module, stride: int) -> nn.Module:
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if stride == patch_size:
            return model

        stride_hw = nn_utils._pair(stride)
        model.patch_embed.proj.stride = stride_hw

        def interpolate_pos_encoding(self, x, w, h):
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos = self.pos_embed[:, 0]
            patch_pos = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos = nn.functional.interpolate(
                patch_pos.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic", align_corners=False, recompute_scale_factor=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1)

        model.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, model)
        return model

    def preprocess_pil(self, pil_image: Image.Image) -> torch.Tensor:
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return prep(pil_image)[None, ...]

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11,
                            facet: str = "token") -> torch.Tensor:
        """Extract descriptors. Returns (B, 1, num_patches, dim)."""
        B, C, H, W = batch.shape
        self._feats = []

        def hook(model, input, output):
            self._feats.append(output)

        handle = self.model.blocks[layer].register_forward_hook(hook)
        with torch.no_grad():
            self.model(batch)
        handle.remove()

        x = self._feats[0]
        x = x[:, 1:, :]  # remove CLS token
        return x.unsqueeze(1)  # (B, 1, num_patches, dim)
