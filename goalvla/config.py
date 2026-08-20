import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
OUTPUTS_DIR = Path(os.environ.get("GOALVLA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


class CameraIntrinsics:
    """Camera intrinsic parameters."""

    # RealSense D435 (example real-world camera)
    REAL = {
        "fx": 1164.5043680682502,
        "fy": 1164.5043680682502,
        "cx": 923.019597861359,
        "cy": 533.7860687031805,
        "raw_resolution": (1080, 1920),
        "crop_width": 1440,
    }

    # RLBench simulation camera
    SIM = {
        "fx": 703.3542416,
        "fy": 703.3542416,
        "cx": 256.0,
        "cy": 256.0,
    }

    @staticmethod
    def get_matrix(mode: str, img_size: tuple[int, int] = (200, 200)) -> np.ndarray:
        if mode == "sim":
            p = CameraIntrinsics.SIM
            return -np.array([
                [-p["fx"], 0.0, p["cx"]],
                [0.0, -p["fy"], p["cy"]],
                [0.0, 0.0, 1.0],
            ])
        elif mode == "real":
            p = CameraIntrinsics.REAL
            fx_scaled = p["fx"] / p["cx"] * img_size[0] / 2
            fy_scaled = p["fy"] / p["cy"] * img_size[1] / 2
            return -np.array([
                [-fx_scaled, 0.0, img_size[0] / 2],
                [0.0, -fy_scaled, img_size[1] / 2],
                [0.0, 0.0, 1.0],
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sim' or 'real'.")


class Config:
    """Global configuration."""

    IMG_SIZE = (200, 200)
    NUM_PATCHES = 60
    FEATURE_DIMS = [640, 1280, 1280, 768]
    PROJECTION_DIM = 768
    MATCHING_THRESHOLD = 0.0
    MIN_DISPLACEMENT_RATIO = 0.0
    NUM_MATCHINGS = 1000

    # Matching robustness. One-way argmax collapses many source pixels onto a
    # single "most-similar-to-everything" target pixel, leaving the target cloud
    # degenerate (rank-deficient) and the Kabsch rotation unrecoverable. Mutual
    # nearest-neighbour (cycle consistency) enforces one-to-one correspondences;
    # the ratio margin rejects ambiguous matches where the top-1 and top-2 target
    # similarities are too close (flat similarity map).
    MUTUAL_CONSISTENCY = False  # RANSAC (below) provides robustness; feed it the
                                # full one-way match set rather than the sparse
                                # mutual subset (too few points for a stable fit).
    RATIO_MARGIN = 0.0  # min (top1 - top2) cosine gap; 0.0 disables the ratio test
    MATCHING_SEED = 0   # seed for source-pixel subsampling (reproducibility)

    # Feature discriminability. The aggregated SD+DINO descriptors share a large
    # common component across all pixels of an object (cosine sims cluster around
    # 0.55-0.72, top1-vs-top2 gap ~0.002), so the argmax is nearly a coin-flip and
    # mutual-NN finds almost nothing on low-texture objects. Subtracting the
    # per-image mask-mean feature and renormalising removes that DC component and
    # restores contrast (mean sim -> 0, gap -> 0.01+), which rescues the sparse
    # scenes. (Per-channel z-scoring over-whitens and was worse.)
    FEATURE_MEAN_SUBTRACT = True

    # Rigid fit robustness. A handful of correspondences with even one bad match
    # swings the Kabsch rotation wildly. RANSAC samples minimal triplets, scores
    # by 3D inlier count, and refits on the consensus set — far more stable than a
    # single least-squares fit over all (possibly contaminated) matches.
    RANSAC_ENABLE = True
    RANSAC_ITERS = 2000
    RANSAC_THRESH = 0.03      # meters; inlier if |R@p+t - q| < this
    RANSAC_MIN_INLIERS = 6    # need at least this many inliers to trust the fit

    # Gemini models (image editing)
    GEMINI_EDIT_MODEL = "gemini-3.1-flash-image-preview"
    GEMINI_EDIT_FALLBACK = "gemini-2.5-flash-image"
    GEMINI_LLM_MODEL = "gemini-2.5-pro"

    # Depth estimation
    DEPTH_ENCODER = "vitl"
    DEPTH_DATASET = "hypersim"
    DEPTH_MAX = 20

    camera = CameraIntrinsics
