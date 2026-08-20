"""Metric depth estimation using Depth Anything V2."""

import sys
from pathlib import Path

import cv2
import torch
import matplotlib.pyplot as plt


def load_depth_model(encoder: str = "vitl", checkpoint: str = None, device: str = "cuda"):
    """Load DepthAnythingV2 model.

    The Depth-Anything-V2 repo must be cloned and its metric_depth directory
    must be accessible. Set checkpoint to the .pth file path.
    """
    from goalvla.config import THIRD_PARTY_DIR
    sys.path.insert(0, str(THIRD_PARTY_DIR / "Depth-Anything-V2" / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**configs[encoder])
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()
    return model


def estimate_depth(model, image_path: str | Path, output_path: str | Path = None):
    """Run metric depth estimation on a single image.

    Args:
        model: Loaded DepthAnythingV2 model.
        image_path: Path to input RGB image.
        output_path: Optional path to save depth visualization.

    Returns:
        depth: (H, W) numpy array of metric depth in meters.
    """
    raw_img = cv2.imread(str(image_path))
    depth = model.infer_image(raw_img)

    if output_path:
        plt.imshow(depth, cmap="gray")
        plt.axis("off")
        plt.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
        plt.close()

    return depth


def generate_depth_maps(model, img_dir: str | Path):
    """Generate depth maps for all rgb_*.png files in a directory.

    Saves dpt_*.png alongside the original images.
    """
    img_dir = Path(img_dir)
    for img_path in sorted(img_dir.glob("rgb_*.png")):
        out_path = img_path.parent / img_path.name.replace("rgb_", "dpt_")
        estimate_depth(model, img_path, out_path)
        print(f"Depth estimated: {img_path.name} -> {out_path.name}")
