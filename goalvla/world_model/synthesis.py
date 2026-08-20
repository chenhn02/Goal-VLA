"""Synthesis pipeline: edit image -> segment objects -> overlay on original."""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pycocotools.mask as mask_util

from goalvla.config import OUTPUTS_DIR


def _run_subprocess(cmd: list[str], description: str):
    print(f"[synthesis] Running {description}...")
    subprocess.run(cmd, check=True)


def run_edit_image(input_image: Path, instruction: str, prefix: str,
                   target_dir: Path = None) -> Path:
    """Generate an edited image using Gemini."""
    from goalvla.world_model.edit_image import edit_image
    output_dir = target_dir or (OUTPUTS_DIR / "synthesis")
    return edit_image(input_image, instruction, output_dir=output_dir, prefix=prefix)


def run_extract_objects(instruction: str) -> list[str]:
    """Extract target object names from instruction."""
    from goalvla.world_model.extract_objects import extract_objects
    objs = extract_objects(instruction)
    print(f"[synthesis] Extracted objects: {objs}")
    return objs


def run_grounded_sam(input_image: Path, objects: list[str], output_dir: Path,
                     box_threshold: float = 0.35, text_threshold: float = 0.25) -> Path:
    """Run Grounded SAM to segment objects. Returns path to results JSON."""
    from goalvla.segmentation.grounded_sam import segment_objects
    return segment_objects(
        image_path=input_image,
        objects=objects,
        output_dir=output_dir,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )


def _rle_to_mask(rle_obj: dict, height: int, width: int) -> np.ndarray:
    counts = rle_obj["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle_c = {"counts": counts, "size": [height, width]}
    return mask_util.decode(rle_c).astype(bool)


def _overlay_objects(base_img: np.ndarray, source_img: np.ndarray,
                     masks: list[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend segmented objects from source onto base image."""
    out = base_img.copy()
    for m in masks:
        if m.shape[:2] != base_img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8),
                           (base_img.shape[1], base_img.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        out[m] = (alpha * source_img[m] + (1 - alpha) * out[m]).astype(out.dtype)
    return out


def synthesize(
    input_image: Path,
    instruction: str,
    prefix: str = "synthesis",
    target_dir: Path = None,
    alpha: float = 0.5,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> tuple[Path, Path, Path]:
    """Full synthesis pipeline: edit -> segment -> overlay.

    Returns (overlay_path, edited_path, run_dir).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    synthesis_dir = OUTPUTS_DIR / "synthesis"
    run_dir = synthesis_dir / f"{prefix}_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Edit image
    edit_prefix = f"{prefix}_edit_{timestamp}"
    edited_path = run_edit_image(input_image, instruction, edit_prefix, target_dir)

    # 2) Extract objects
    objects = run_extract_objects(instruction)
    if not objects:
        print("[synthesis] No objects extracted, using edited image as overlay")
        overlay_path = synthesis_dir / f"{prefix}_overlay_{timestamp}.png"
        edited_img = cv2.imread(str(edited_path))
        cv2.imwrite(str(overlay_path), edited_img)
        return overlay_path, edited_path, run_dir

    # 3) Segment objects in edited image. segment_objects() returns the mask PNG
    # path, but also writes segmentation_results.json (RLE masks + image size)
    # into run_dir; that JSON is what we need here. When nothing is detected no
    # JSON is written, so fall back to an empty result.
    run_grounded_sam(edited_path, objects, run_dir,
                     box_threshold, text_threshold)

    results_file = run_dir / "segmentation_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            result = json.load(f)
    else:
        result = {"annotations": [], "img_height": 0, "img_width": 0}

    ann = result.get("annotations", [])
    H = int(result.get("img_height", 0))
    W = int(result.get("img_width", 0))

    masks = []
    for a in ann:
        rle = a.get("segmentation")
        if rle:
            try:
                masks.append(_rle_to_mask(rle, H, W))
            except Exception:
                continue

    if not masks:
        print("[synthesis] No masks decoded, using edited image as overlay")
        overlay_path = synthesis_dir / f"{prefix}_overlay_{timestamp}.png"
        cv2.imwrite(str(overlay_path), cv2.imread(str(edited_path)))
        return overlay_path, edited_path, run_dir

    # 4) Overlay edited objects onto original
    print("[synthesis] Overlaying objects from edited image onto original...")
    original_img = cv2.imread(str(input_image))
    edited_img = cv2.imread(str(edited_path))

    if edited_img.shape[:2] != original_img.shape[:2]:
        edited_img = cv2.resize(edited_img,
                                (original_img.shape[1], original_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    blended = _overlay_objects(original_img, edited_img, masks, alpha=alpha)

    overlay_path = synthesis_dir / f"{prefix}_overlay_{timestamp}.png"
    cv2.imwrite(str(overlay_path), blended)
    print(f"[synthesis] Overlay saved: {overlay_path}")

    return overlay_path, edited_path, run_dir
