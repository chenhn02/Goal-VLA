"""GoalVLA World Model pipeline.

Given an input image and a text instruction, generates a goal image
through iterative Gemini-based editing with validation.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from goalvla.config import OUTPUTS_DIR


def WorldModel(
    image: str | Path,
    text: str,
    generate_masks: bool = True,
    max_iterations: int = 3,
    use_enhancer: bool = True,
    use_reflector: bool = True,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    target_dir: Path = None,
) -> Path:
    """Generate a goal image from an input image and instruction.

    Pipeline:
      1. Enhance instruction (optional) -> precise editing prompt
      2. Edit image with Gemini -> synthesize overlay
      3. Validate overlay with Gemini -> accept or revise (iterative)
      4. Generate object masks (optional)

    Args:
        image: Path to the input RGB image.
        text: Natural language instruction (e.g., "move the cup to the plate").
        generate_masks: Whether to generate segmentation masks.
        max_iterations: Max validation-revision iterations.
        use_enhancer: Whether to enhance the instruction before editing.
        use_reflector: Whether to validate and iteratively improve.
        box_threshold: GroundingDINO box detection threshold.
        text_threshold: GroundingDINO text matching threshold.
        target_dir: Optional directory for final outputs.

    Returns:
        Path to the final edited/goal image.
    """
    from goalvla.world_model.enhancer import enhance_instruction
    from goalvla.world_model.synthesis import synthesize
    from goalvla.world_model.reflector import validate_overlay, generate_revised_instruction
    from goalvla.world_model.extract_objects import extract_objects

    input_image = Path(image).expanduser().resolve()
    if not input_image.exists():
        raise FileNotFoundError(f"Image not found: {input_image}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if use_reflector and not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (required for reflector)")

    print(f"[world_model] Input: {input_image.name}")
    print(f"[world_model] Instruction: {text}")
    print(f"[world_model] Enhancer: {use_enhancer}, Reflector: {use_reflector}")

    # Step 1: Instruction enhancement
    if use_enhancer:
        print("[world_model] Step 1: Enhancing instruction...")
        current_instruction = enhance_instruction(text)
    else:
        print("[world_model] Step 1: Using original instruction")
        current_instruction = text

    final_edited_image = None
    overlay_path = None
    run_dir = None

    if use_reflector:
        # Iterative improvement loop
        for iteration in range(1, max_iterations + 1):
            print(f"[world_model] Iteration {iteration}/{max_iterations}")
            print(f"[world_model] Instruction: {current_instruction}")

            overlay_path, edited_path, run_dir = synthesize(
                input_image=input_image,
                instruction=current_instruction,
                prefix=f"world_model_iter_{iteration}",
                target_dir=target_dir,
            )

            print("[world_model] Validating with Gemini...")
            is_valid, feedback = validate_overlay(
                text, current_instruction, overlay_path, api_key,
            )

            if edited_path:
                final_edited_image = edited_path

            if is_valid:
                print("[world_model] Validation passed.")
                break
            else:
                print(f"[world_model] Validation failed: {feedback}")
                if iteration < max_iterations:
                    print("[world_model] Generating revised instruction...")
                    current_instruction = generate_revised_instruction(
                        text, current_instruction, feedback, api_key,
                    )
                else:
                    print("[world_model] Max iterations reached, using best result.")

        print(f"[world_model] Completed after {iteration} iteration(s).")
    else:
        # Single-pass synthesis
        print("[world_model] Running single-pass synthesis...")
        overlay_path, edited_path, run_dir = synthesize(
            input_image=input_image,
            instruction=current_instruction,
            prefix="world_model",
            target_dir=target_dir,
        )
        if edited_path:
            final_edited_image = edited_path
        print("[world_model] Synthesis completed.")

    result_path = final_edited_image or overlay_path
    if result_path is None:
        raise RuntimeError("No output image produced")

    # Optional mask generation
    if generate_masks:
        _generate_masks(input_image, result_path, text, target_dir,
                        box_threshold, text_threshold)

    print(f"[world_model] Final result: {result_path}")
    return result_path


def _generate_masks(
    original_image: Path,
    edited_image: Path,
    instruction: str,
    target_dir: Optional[Path],
    box_threshold: float,
    text_threshold: float,
):
    """Generate segmentation masks for both original and edited images."""
    import re
    from goalvla.world_model.extract_objects import extract_scene_objects

    scene = extract_scene_objects(instruction)
    target = scene["target"]
    if not target:
        print("[world_model] No target objects detected, skipping mask generation")
        return

    distractors = scene["distractors"]
    prompt_objects = [target] + distractors
    target_labels = [target]
    print(f"[world_model] Grounding {prompt_objects}, keeping only: {target_labels}")

    try:
        from goalvla.segmentation.grounded_sam import segment_objects

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_dir = OUTPUTS_DIR / "mask" / f"run_{timestamp}"
        mask_dir.mkdir(parents=True, exist_ok=True)

        orig_mask = segment_objects(
            original_image, prompt_objects,
            output_dir=mask_dir / "original",
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_labels=target_labels,
        )
        edit_mask = segment_objects(
            edited_image, prompt_objects,
            output_dir=mask_dir / "edited",
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_labels=target_labels,
        )

        shutil.copy2(original_image, mask_dir / "original.png")
        shutil.copy2(edited_image, mask_dir / "edited.png")

        # segment_objects() returns the binary mask PNG path for each image.
        shutil.copy2(orig_mask, mask_dir / "original_mask.png")
        shutil.copy2(edit_mask, mask_dir / "edited_mask.png")

        if target_dir:
            for fname in ["original.png", "edited.png", "original_mask.png", "edited_mask.png"]:
                src = mask_dir / fname
                if src.exists():
                    shutil.copy2(src, Path(target_dir) / fname)

        print(f"[world_model] Masks saved to: {mask_dir}")

    except Exception as e:
        print(f"[world_model] Mask generation failed: {e}")
