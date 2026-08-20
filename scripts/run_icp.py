"""Run ICP pipeline only (given pre-generated init/goal images and masks).

Usage:
    python scripts/run_icp.py \
        --img-path path/to/data_dir \
        --mode real \
        --visualize

Expects in img-path:
    rgb_init.png, rgb_goal.png, mask_init.png, mask_goal.png, depth_*.npy
"""

import argparse
import pickle
import glob
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="GoalVLA ICP: Feature Matching + Transformation")
    parser.add_argument("--img-path", type=str, required=True, help="Directory with init/goal images")
    parser.add_argument("--mode", type=str, default="real", choices=["sim", "real"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_856.PTH")
    parser.add_argument("--visualize", action="store_true", help="Launch viser visualization")
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation (use existing dpt_*.png)")
    parser.add_argument("--skip-matching", action="store_true", help="Skip matching (use existing .pkl)")
    args = parser.parse_args()

    img_dir = Path(args.img_path)
    from goalvla.config import Config
    img_size = Config.IMG_SIZE

    # Step 1: Depth estimation
    if not args.skip_depth:
        print("Step 1: Depth estimation...")
        from goalvla.transformation.depth_estimation import load_depth_model, generate_depth_maps
        model = load_depth_model(
            encoder=Config.DEPTH_ENCODER,
            checkpoint=f"checkpoints/depth_anything_v2_metric_{Config.DEPTH_DATASET}_{Config.DEPTH_ENCODER}.pth",
        )
        generate_depth_maps(model, img_dir)
        del model
    else:
        print("Step 1: Skipping depth estimation")

    # Step 2: Feature matching
    if not args.skip_matching:
        print("Step 2: Feature matching...")
        from goalvla.feature_matching.matcher import FeatureMatcher

        matcher = FeatureMatcher(checkpoint_path=args.checkpoint)
        img_init = Image.open(img_dir / "rgb_init.png").convert("RGB").resize(img_size)
        img_goal = Image.open(img_dir / "rgb_goal.png").convert("RGB").resize(img_size)
        mask_init = np.array(Image.open(img_dir / "mask_init.png").convert("1").resize(img_size))
        mask_goal = np.array(Image.open(img_dir / "mask_goal.png").convert("1").resize(img_size))

        matchings = matcher.match(img_init, img_goal, mask_init, mask_goal)
        print(f"Found {len(matchings)} matches")

        with open(img_dir / "point_wise_matching.pkl", "wb") as f:
            pickle.dump(matchings, f)
    else:
        print("Step 2: Skipping feature matching")

    # Step 3: Transformation estimation
    print("Step 3: Transformation estimation...")
    from goalvla.transformation.kabsch import load_and_estimate

    R, t, info = load_and_estimate(img_dir, mode=args.mode, img_size=img_size)
    print(f"Rotation:\n{R}")
    print(f"Translation: {t}")
    print(f"Error: {info['error']:.6f}")

    with open(img_dir / "transformation.pkl", "wb") as f:
        pickle.dump((R, t), f)
    print(f"Saved: {img_dir / 'transformation.pkl'}")

    # Step 4: Visualization
    if args.visualize:
        print("Step 4: Launching 3D visualization...")

        depth_files = glob.glob(str(img_dir / "depth_*.npy"))
        depth_real = np.load(depth_files[0])
        if depth_real.shape == (1080, 1920):
            depth_real = depth_real[:, :1440]
        depth_real = np.array(Image.fromarray(depth_real).resize(img_size))

        img_init = np.array(Image.open(img_dir / "rgb_init.png").convert("RGB").resize(img_size))
        img_goal = np.array(Image.open(img_dir / "rgb_goal.png").convert("RGB").resize(img_size))
        mask_init = np.array(Image.open(img_dir / "mask_init.png").convert("1").resize(img_size))
        mask_goal = np.array(Image.open(img_dir / "mask_goal.png").convert("1").resize(img_size))

        from goalvla.transformation.visualizer import visualize_transformation
        visualize_transformation(
            img_init=img_init, depth_real=depth_real,
            dpt_init_scaled=info["dpt_init_scaled"],
            dpt_goal_scaled=info["dpt_goal_scaled"],
            seg_init=mask_init, seg_goal=mask_goal,
            R=R, t=t, K=info["K"],
            img_goal=img_goal,
            P=info["P"], Q=info["Q"],
            port=args.viser_port,
        )


if __name__ == "__main__":
    main()
