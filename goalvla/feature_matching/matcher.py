"""Dense feature matching between init and goal images."""

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from goalvla.config import Config
from goalvla.feature_matching.aggregation_network import AggregationNetwork
from goalvla.feature_matching.extractor_sd import load_model as load_sd_model, process_features_and_mask
from goalvla.feature_matching.extractor_dino import ViTExtractor


def _resize(img: Image.Image, target_res: int, resize: bool = True) -> Image.Image:
    if not resize:
        return img
    return img.resize((target_res, target_res), Image.LANCZOS)


class FeatureMatcher:

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        self.device = device
        self.num_patches = Config.NUM_PATCHES

        self.aggre_net = AggregationNetwork(
            feature_dims=Config.FEATURE_DIMS,
            projection_dim=Config.PROJECTION_DIM,
            device=device,
        )
        self.aggre_net.load_pretrained_weights(
            torch.load(checkpoint_path, map_location=device)
        )
        self.aggre_net.eval()

        self.sd_model = None
        self.sd_aug = None
        self.extractor_vit = None

    def _ensure_models_loaded(self):
        if self.sd_model is None:
            self.sd_model, self.sd_aug = load_sd_model(
                diffusion_ver="v1-5",
                image_size=self.num_patches * 16,
                num_timesteps=50,
                block_indices=[2, 5, 8, 11],
            )
        if self.extractor_vit is None:
            self.extractor_vit = ViTExtractor(
                "dinov2_vitb14", stride=14, device=self.device,
            )

    def extract_features(self, img: Image.Image) -> torch.Tensor:
        """Extract aggregated 768-dim dense features from an image.

        Returns: (1, projection_dim, num_patches, num_patches)
        """
        self._ensure_models_loaded()
        np_ = self.num_patches

        img_sd = _resize(img, np_ * 16)
        features_sd = process_features_and_mask(self.sd_model, self.sd_aug, img_sd)
        del features_sd["s2"]

        img_dino = _resize(img, np_ * 14)
        batch_dino = self.extractor_vit.preprocess_pil(img_dino)
        features_dino = self.extractor_vit.extract_descriptors(
            batch_dino.to(self.device), layer=11, facet="token",
        ).permute(0, 1, 3, 2).reshape(1, -1, np_, np_)

        desc = torch.cat([
            F.interpolate(features_sd["s3"], size=(np_, np_), mode="bilinear", align_corners=False),
            F.interpolate(features_sd["s4"], size=(np_, np_), mode="bilinear", align_corners=False),
            F.interpolate(features_sd["s5"], size=(np_, np_), mode="bilinear", align_corners=False),
            features_dino,
        ], dim=1)

        desc = self.aggre_net(desc)
        desc = desc / (torch.linalg.norm(desc, dim=1, keepdim=True) + 1e-8)
        return desc

    def match(
        self,
        img_init: Image.Image,
        img_goal: Image.Image,
        mask_init: np.ndarray,
        mask_goal: np.ndarray,
        threshold: float = None,
        num_matchings: int = None,
    ) -> dict[tuple[int, int], tuple[int, int]]:
        """Find dense pixel correspondences between init and goal images.

        Args:
            img_init: RGB init image, resized to Config.IMG_SIZE.
            img_goal: RGB goal image, resized to Config.IMG_SIZE.
            mask_init: Binary mask (H, W) for init image.
            mask_goal: Binary mask (H, W) for goal image.
            threshold: Cosine similarity threshold.
            num_matchings: Max number of source pixels to sample.

        Returns:
            Dict mapping (row, col) in init to (row, col) in goal.
        """
        if threshold is None:
            threshold = Config.MATCHING_THRESHOLD
        if num_matchings is None:
            num_matchings = Config.NUM_MATCHINGS

        feat_init = self.extract_features(img_init)
        feat_goal = self.extract_features(img_goal)
        feats = torch.cat([feat_init, feat_goal], dim=0)

        H, W = mask_init.shape
        min_disp = Config.MIN_DISPLACEMENT_RATIO * H

        src_ft = nn.Upsample(size=(H, W), mode="bilinear")(feats[0:1])[0]  # (D, H, W)
        trg_ft = nn.Upsample(size=(H, W), mode="bilinear")(feats[1:2])[0]

        src_idx = np.argwhere(mask_init > 0.5)  # (Ns, 2), rows (x, y)
        trg_idx = np.argwhere(mask_goal > 0.5)  # (Nt, 2)
        if len(src_idx) == 0 or len(trg_idx) == 0:
            return {}

        # Reproducible subsampling of source pixels (was np.random.permutation
        # with no seed, so which pixels got checked drifted across processes).
        rng = np.random.default_rng(Config.MATCHING_SEED)
        if len(src_idx) > num_matchings:
            src_idx = src_idx[rng.choice(len(src_idx), num_matchings, replace=False)]

        # Gather per-pixel feature vectors and L2-normalise (bilinear upsampling
        # breaks the normalisation done in extract_features), so a dot product is
        # the cosine similarity.
        src_feat = src_ft[:, src_idx[:, 0], src_idx[:, 1]].t()  # (Ns, D)
        trg_feat = trg_ft[:, trg_idx[:, 0], trg_idx[:, 1]].t()  # (Nt, D)
        src_feat = F.normalize(src_feat, dim=1)
        trg_feat = F.normalize(trg_feat, dim=1)

        # Remove the per-image DC component shared across all masked pixels: the
        # descriptors are dominated by an object-common vector that flattens the
        # similarity map, so the top-1/top-2 gap collapses to ~0 and mutual-NN
        # finds almost nothing. Subtracting the mask-mean and renormalising
        # restores contrast (verified: mean sim -> 0, gap -> 0.01+).
        if Config.FEATURE_MEAN_SUBTRACT:
            src_feat = F.normalize(src_feat - src_feat.mean(dim=0, keepdim=True), dim=1)
            trg_feat = F.normalize(trg_feat - trg_feat.mean(dim=0, keepdim=True), dim=1)

        S = (src_feat @ trg_feat.t()).detach().cpu().numpy()  # (Ns, Nt) cosine

        fwd = S.argmax(axis=1)          # best target for each source
        rev = S.argmax(axis=0)          # best source for each target

        # Ratio test: gap between the best and second-best target per source.
        if Config.RATIO_MARGIN > 0.0 and S.shape[1] >= 2:
            part = np.partition(S, -2, axis=1)
            gap = part[:, -1] - part[:, -2]
        else:
            gap = np.full(S.shape[0], np.inf)

        matchings = {}
        for i in range(len(src_idx)):
            j = int(fwd[i])
            # Mutual nearest neighbour: i must also be the best source for j.
            if Config.MUTUAL_CONSISTENCY and rev[j] != i:
                continue
            if gap[i] < Config.RATIO_MARGIN:
                continue
            sim = float(S[i, j])
            x, y = int(src_idx[i][0]), int(src_idx[i][1])
            tgt_x, tgt_y = int(trg_idx[j][0]), int(trg_idx[j][1])
            if sim > threshold and np.sqrt((tgt_x - x) ** 2 + (tgt_y - y) ** 2) >= min_disp:
                matchings[(x, y)] = (tgt_x, tgt_y)

        del src_ft, trg_ft
        gc.collect()
        torch.cuda.empty_cache()

        return matchings
