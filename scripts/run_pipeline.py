"""End-to-end GoalVLA pipeline.

Usage:
    python scripts/run_pipeline.py \
        --image path/to/rgb_init.png \
        --instruction "move the cup to the plate" \
        --mode real \
        --visualize
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="GoalVLA: Goal Image Generation + Transformation Estimation")
    parser.add_argument("--image", type=str, required=True, help="Path to input RGB image (rgb_init.png)")
    parser.add_argument("--instruction", type=str, required=True, help="Natural language instruction")
    parser.add_argument("--mode", type=str, default="real", choices=["sim", "real"], help="Camera mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_856.PTH",
                        help="AggregationNetwork checkpoint path")
    parser.add_argument("--no-enhancer", action="store_true", help="Skip instruction enhancement")
    parser.add_argument("--no-reflector", action="store_true", help="Skip iterative validation")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max reflector iterations")
    parser.add_argument("--visualize", action="store_true", help="Launch viser 3D visualization")
    parser.add_argument("--viser-port", type=int, default=8081, help="Viser server port")
    args = parser.parse_args()

    input_image = Path(args.image).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else input_image.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    from goalvla.config import Config

    # Fail fast: the transform step needs real metric depth (sensor .npy) for
    # scale. Check before spending a Gemini call on the World Model.
    if not list(output_dir.glob("depth_*.npy")):
        raise FileNotFoundError(
            f"No real metric depth found in {output_dir}. Place the sensor depth as "
            f"'depth_init.npy' (a HxW float array in meters) next to the input image "
            f"before running the pipeline."
        )

    # Step 1: World Model -> goal image
    print("=" * 60)
    print("Step 1: Generating goal image with World Model")
    print("=" * 60)

    from goalvla.world_model.pipeline import WorldModel

    goal_image_path = WorldModel(
        image=input_image,
        text=args.instruction,
        use_enhancer=not args.no_enhancer,
        use_reflector=not args.no_reflector,
        max_iterations=args.max_iterations,
        target_dir=output_dir,
    )

    goal_image_path = Path(goal_image_path)
    print(f"Goal image: {goal_image_path}")

    # The World Model writes original/edited(+_mask).png into output_dir, but the
    # remaining steps read the rgb_init/rgb_goal/mask_init/mask_goal names. Bridge
    # them so depth/matching/transform can find their inputs.
    if not (output_dir / "rgb_init.png").exists():
        shutil.copy2(input_image, output_dir / "rgb_init.png")
    shutil.copy2(goal_image_path, output_dir / "rgb_goal.png")
    for src, dst in [("original_mask.png", "mask_init.png"),
                     ("edited_mask.png", "mask_goal.png")]:
        if (output_dir / src).exists():
            shutil.copy2(output_dir / src, output_dir / dst)

    missing = [n for n in ("mask_init.png", "mask_goal.png")
               if not (output_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"World Model did not produce {missing}. Segmentation found no objects "
            f"for the instruction — use a clear object noun in --instruction "
            f"(e.g. 'move the bottle ...')."
        )

    # Step 2: Depth estimation
    print("\n" + "=" * 60)
    print("Step 2: Estimating metric depth")
    print("=" * 60)

    from goalvla.transformation.depth_estimation import load_depth_model, generate_depth_maps

    depth_model = load_depth_model(
        encoder=Config.DEPTH_ENCODER,
        checkpoint=f"checkpoints/depth_anything_v2_metric_{Config.DEPTH_DATASET}_{Config.DEPTH_ENCODER}.pth",
    )
    generate_depth_maps(depth_model, output_dir)

    # Step 3: Feature matching
    print("\n" + "=" * 60)
    print("Step 3: Dense feature matching")
    print("=" * 60)

    from goalvla.feature_matching.matcher import FeatureMatcher

    matcher = FeatureMatcher(checkpoint_path=args.checkpoint)
    img_size = Config.IMG_SIZE

    img_init = Image.open(output_dir / "rgb_init.png").convert("RGB").resize(img_size)
    img_goal = Image.open(goal_image_path).convert("RGB").resize(img_size)
    mask_init = np.array(Image.open(output_dir / "mask_init.png").convert("1").resize(img_size))
    mask_goal = np.array(Image.open(output_dir / "mask_goal.png").convert("1").resize(img_size))

    matchings = matcher.match(img_init, img_goal, mask_init, mask_goal)
    print(f"Found {len(matchings)} matches")

    matching_path = output_dir / "point_wise_matching.pkl"
    with open(matching_path, "wb") as f:
        pickle.dump(matchings, f)

    # Step 4: Transformation estimation
    print("\n" + "=" * 60)
    print("Step 4: Estimating rigid transformation")
    print("=" * 60)

    from goalvla.transformation.kabsch import load_and_estimate

    R, t, info = load_and_estimate(output_dir, mode=args.mode, img_size=img_size)
    print(f"Rotation:\n{R}")
    print(f"Translation: {t}")

    transform_path = output_dir / "transformation.pkl"
    with open(transform_path, "wb") as f:
        pickle.dump((R, t), f)
    print(f"Saved: {transform_path}")

    # Step 5: Visualization (optional)
    if args.visualize:
        print("\n" + "=" * 60)
        print("Step 5: Launching 3D visualization")
        print("=" * 60)

        import glob

        depth_files = glob.glob(str(output_dir / "depth_*.npy"))
        depth_real = np.load(depth_files[0])
        if depth_real.shape == (1080, 1920):
            depth_real = depth_real[:, :1440]
        depth_real = np.array(Image.fromarray(depth_real).resize(img_size))

        from goalvla.transformation.visualizer import visualize_transformation

        visualize_transformation(
            img_init=np.array(img_init),
            depth_real=depth_real,
            dpt_init_scaled=info["dpt_init_scaled"],
            dpt_goal_scaled=info["dpt_goal_scaled"],
            seg_init=mask_init,
            seg_goal=mask_goal,
            R=R, t=t, K=info["K"],
            img_goal=np.array(img_goal),
            P=info["P"], Q=info["Q"],
            port=args.viser_port,
        )


if __name__ == "__main__":
    main()
