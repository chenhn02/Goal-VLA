"""Grounded SAM 2: zero-shot object segmentation using GroundingDINO + SAM2."""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import pycocotools.mask as mask_util
from torchvision.ops import box_convert

from goalvla.config import THIRD_PARTY_DIR

GSAM2_ROOT = THIRD_PARTY_DIR / "Grounded-SAM-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam2_predictor = None
_grounding_model = None


def _ensure_imports():
    if str(GSAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(GSAM2_ROOT))


def _load_models():
    global _sam2_predictor, _grounding_model
    if _sam2_predictor is not None:
        return

    _ensure_imports()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model

    sam2_ckpt = GSAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    gdino_cfg = GSAM2_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = GSAM2_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

    sam2_model = build_sam2(sam2_cfg, str(sam2_ckpt), device=DEVICE)
    _sam2_predictor = SAM2ImagePredictor(sam2_model)

    _grounding_model = load_model(
        model_config_path=str(gdino_cfg),
        model_checkpoint_path=str(gdino_ckpt),
        device=DEVICE,
    )


def _normalize_prompt(items: str) -> str:
    raw = [t.strip().lower() for t in items.replace("\n", " ").replace(",", ".").split(".")]
    toks = [t for t in raw if t]
    return ". ".join(toks) + "." if toks else ""


def _single_mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def segment_objects(
    image_path: str | Path,
    objects: list[str],
    output_dir: str | Path,
    mask_filename: str = "mask.png",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    target_labels: list[str] | None = None,
) -> Path:
    """Segment specified objects in an image using GroundingDINO + SAM2.

    Args:
        image_path: Path to input image.
        objects: List of object names to detect and segment. When
            ``target_labels`` is given this is the full set of contrastive
            phrases (target + distractors) fed to the detector.
        output_dir: Directory to save results.
        mask_filename: Name for the output binary mask file.
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text similarity threshold.
        target_labels: If set, only detections whose predicted label
            unambiguously matches one of these phrases are unioned into the mask
            — a box is kept when its label contains a target word AND no other
            (distractor) word from ``objects``. This separates the grasped
            target from look-alike distractors that GroundingDINO would
            otherwise ground onto the same phrase. If None, all detections are
            unioned (legacy behaviour).

    Returns:
        Path to the binary mask file.
    """
    _load_models()
    _ensure_imports()
    from grounding_dino.groundingdino.util.inference import load_image, predict

    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_prompt = _normalize_prompt(", ".join(objects))
    image_source, image = load_image(str(image_path))
    _sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=_grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE,
    )

    h, w, _ = image_source.shape
    results_file = output_dir / "segmentation_results.json"

    mask_file = output_dir / mask_filename

    if boxes.numel() == 0:
        union_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(mask_file), union_mask)
        print("[grounded_sam] No detections found, saved empty mask")
        return mask_file

    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = _sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()

    # Decide which detections to union. When target_labels is given, keep only
    # boxes whose predicted phrase unambiguously names a target: it must contain
    # a target word and none of the distractor words. This drops both distractor
    # objects (e.g. a dustpan grounded as "brush") and ambiguous multi-phrase
    # hits (e.g. label "brush dustpan").
    n_det = len(labels)
    if target_labels:
        def _toks(s: str) -> set:
            return set(re.findall(r"[a-z0-9]+", s.lower()))

        target_words = set().union(*[_toks(t) for t in target_labels]) if target_labels else set()
        distractor_words = set()
        for o in objects:
            ow = _toks(o)
            if not (ow & target_words):
                distractor_words |= ow

        keep_idx = []
        for i, lab in enumerate(labels):
            lw = _toks(lab)
            if (lw & target_words) and not (lw & distractor_words):
                keep_idx.append(i)
        print(f"[grounded_sam] target={target_labels} kept {len(keep_idx)}/{n_det} "
              f"detections (labels: {list(labels)})")
        if not keep_idx:
            # Target object not present/visible in this image. Keep an empty mask
            # rather than falling back to distractors (e.g. the robot arm), so the
            # run fails cleanly instead of estimating a bogus transform onto a
            # look-alike. (Common on WM goal images where the tool is occluded.)
            print("[grounded_sam] no target detection; leaving mask empty")
    else:
        keep_idx = list(range(n_det))

    keep_set = set(keep_idx)

    # Save visualization: kept boxes/masks in red, dropped ones in gray
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_bgr = cv2.imread(str(image_path))
    annotated = img_bgr.copy()

    for i, (x1, y1, x2, y2) in enumerate(input_boxes.astype(int)):
        color = (0, 255, 0) if i in keep_set else (128, 128, 128)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{labels[i]} {confidences[i]:.2f}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for i, m in enumerate(masks.astype(bool)):
        if i not in keep_set:
            continue
        overlay = annotated.copy()
        overlay[m] = (overlay[m] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
        annotated = overlay

    seg_file = output_dir / f"segmentation_{timestamp}.jpg"
    cv2.imwrite(str(seg_file), annotated)

    # Save binary mask (union of kept instances)
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(masks.astype(bool)):
        if i in keep_set:
            union_mask[m] = 255
    cv2.imwrite(str(mask_file), union_mask)

    # Save JSON results
    mask_rles = [_single_mask_to_rle(m) for m in masks]
    raw_scores = scores.tolist() if hasattr(scores, "tolist") else scores
    norm_scores = []
    for s in raw_scores:
        if isinstance(s, (list, tuple, np.ndarray)):
            s = s[0] if len(s) > 0 else 0.0
        norm_scores.append(float(s))

    results = {
        "image_path": str(image_path),
        "annotations": [
            {
                "class_name": name,
                "bbox": box.tolist(),
                "segmentation": rle,
                "score": score,
            }
            for name, box, rle, score in zip(labels, input_boxes, mask_rles, norm_scores)
        ],
        "box_format": "xyxy",
        "img_width": int(w),
        "img_height": int(h),
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[grounded_sam] Segmented {len(labels)} object(s): {', '.join(labels)}")
    return mask_file
