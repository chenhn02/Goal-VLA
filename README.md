# Goal-VLA: Image-Generative VLMs as Object-Centric World Models Enable Zero-shot Robot Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2506.23919-b31b1b.svg)](https://arxiv.org/abs/2506.23919)

Goal-conditioned manipulation via World Model and visual correspondence.

Given an RGB image and a natural language instruction, Goal-VLA generates a goal image using a World Model (Gemini), estimates dense feature correspondences between the current and goal scenes, and computes the 6-DoF rigid transformation for the target object.

Paper: [Goal-VLA: Image-Generative VLMs as Object-Centric World Models Enable Zero-shot Robot Manipulation](https://arxiv.org/abs/2506.23919)

## Pipeline

```
Input: rgb_init.png + "move the cup to the plate"
  |
  v
[World Model] Gemini image editing + iterative validation --> rgb_goal.png
  |
  v
[Segmentation] Grounded SAM 2 --> mask_init.png, mask_goal.png
  |
  v
[Depth Estimation] Depth Anything V2 --> metric depth
  |
  v
[Feature Matching] Stable Diffusion + DINOv2 + AggregationNetwork --> correspondences
  |
  v
[Transformation] Kabsch algorithm --> (R, t)
  |
  v
[Visualization] Viser 3D point cloud with predicted object pose
```

## Project Structure

```
goalvla/
├── config.py                     # Camera intrinsics, thresholds, model paths
├── world_model/                  # Goal image generation (Gemini-based)
│   ├── pipeline.py               # WorldModel() main entry point
│   ├── enhancer.py               # Instruction enhancement
│   ├── reflector.py              # Iterative validation and revision
│   ├── synthesis.py              # Edit -> segment -> overlay
│   ├── edit_image.py             # Gemini image editing API
│   └── extract_objects.py        # Object name extraction from text
├── segmentation/
│   └── grounded_sam.py           # Grounded SAM 2 (GroundingDINO + SAM2)
├── feature_matching/
│   ├── aggregation_network.py    # Trainable SD+DINOv2 feature fusion
│   ├── extractor_sd.py           # Stable Diffusion feature extraction
│   ├── extractor_dino.py         # DINOv2 feature extraction
│   ├── matcher.py                # Dense pixel correspondence
│   └── resnet.py                 # ResNet building blocks
├── transformation/
│   ├── kabsch.py                 # Kabsch algorithm + depth scale estimation
│   ├── depth_estimation.py       # Depth Anything V2 wrapper
│   └── visualizer.py             # Viser 3D visualization
└── utils/
    └── correspondence.py         # Image resize utilities

scripts/
├── run_pipeline.py               # Full pipeline: instruction --> 3D visualization
└── run_icp.py                    # ICP only: given init/goal images --> transformation
```

## Installation

### Prerequisites

- Python >= 3.10 (tested with 3.10.12)
- CUDA 11.8 + GCC 11 (for compiling CUDA extensions)
- [Gemini API key](https://ai.google.dev/) (for World Model)
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/chenhn02/Goal-VLA.git
cd Goal-VLA

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

# Create environment with uv (Python 3.10)
uv venv .venv --python 3.10
source .venv/bin/activate

# Install PyTorch (CUDA 11.8)
uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install Goal-VLA and dependencies
uv pip install -e .
uv pip install "numpy<2" "setuptools==69.5.1" "hydra-core>=1.3" ftfy
```

### Third-party dependencies

Third-party code lives under `third_party/`. Grounded SAM 2 and Depth Anything V2 are
pulled in as git submodules from their official upstream repositories (see
`.gitmodules`); ODISE (with a bundled Mask2Former) is vendored directly in the tree.
Build them with GCC 11 (required for CUDA 11.8 compatibility):

```bash
# detectron2
CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1 pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# ODISE + Mask2Former (Mask2Former is bundled inside ODISE)
cd third_party/ODISE/third_party/Mask2Former
CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    FORCE_CUDA=1 python setup.py develop
cd ../..
python setup.py develop
cd ../..

# Grounded SAM 2 + GroundingDINO
cd third_party/Grounded-SAM-2
python setup.py develop
pip install --no-build-isolation -e grounding_dino
cd ../..
```

Upstream sources for the submodules:

- Grounded SAM 2 — https://github.com/IDEA-Research/Grounded-SAM-2
- Depth Anything V2 — https://github.com/DepthAnything/Depth-Anything-V2
- ODISE — https://github.com/NVlabs/ODISE

### Checkpoints

```bash
# SAM 2 checkpoints
cd third_party/Grounded-SAM-2/checkpoints && bash download_ckpts.sh && cd ../../..

# GroundingDINO checkpoint
cd third_party/Grounded-SAM-2/gdino_checkpoints && bash download_ckpts.sh && cd ../../..

# Depth Anything V2 (download to checkpoints/)
# wget -P checkpoints/ <depth_anything_v2_metric_hypersim_vitl.pth URL>
```

Place the AggregationNetwork checkpoint at `checkpoints/best_856.PTH`.

## Usage

### Full pipeline (instruction to 3D visualization)

```bash
export GEMINI_API_KEY="your-api-key"

python scripts/run_pipeline.py \
    --image path/to/rgb_init.png \
    --instruction "move the cup to the plate" \
    --mode real \
    --visualize
```

### ICP only (given pre-generated images)

Prepare a directory with:
```
data_dir/
├── rgb_init.png
├── rgb_goal.png
├── mask_init.png
├── mask_goal.png
└── depth_*.npy        # real metric depth from sensor
```

Run:
```bash
python scripts/run_icp.py \
    --img-path path/to/data_dir \
    --mode real \
    --visualize
```

### Python API

```python
from goalvla.world_model import WorldModel
from goalvla.feature_matching import FeatureMatcher
from goalvla.transformation import estimate_transformation

# Generate goal image
goal_path = WorldModel(
    image="rgb_init.png",
    text="move the cup to the plate",
)

# Feature matching
matcher = FeatureMatcher(checkpoint_path="checkpoints/best_856.PTH")
matchings = matcher.match(img_init, img_goal, mask_init, mask_goal)

# Transformation estimation
R, t, info = estimate_transformation(
    matchings, dpt_init, dpt_goal, depth_real,
    seg_init, seg_goal, mode="real",
)
```

## Configuration

Camera intrinsics and other parameters are centralized in `goalvla/config.py`. Modify `CameraIntrinsics.REAL` to match your camera setup.

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2025goal,
  title={Goal-vla: Image-generative vlms as object-centric world models empowering zero-shot robot manipulation},
  author={Chen, Haonan and Guo, Jingxiang and Wang, Bangjun and Zhang, Tianrui and Huang, Xuchuan and Zheng, Boren and Hou, Yiwen and Tie, Chenrui and Deng, Jiajun and Shao, Lin},
  journal={arXiv preprint arXiv:2506.23919},
  year={2025}
}
```

> Chen, Haonan, et al. "Goal-VLA: Image-generative VLMs as object-centric world models empowering zero-shot robot manipulation." arXiv preprint arXiv:2506.23919 (2025).

## Acknowledgements

- Feature correspondence adapted from [Telling Left from Right](https://github.com/Junyi42/GeoAware-SC) (Junyi Zhang et al.)
- [ODISE](https://github.com/NVlabs/ODISE) for Stable Diffusion feature extraction
- [DINOv2](https://github.com/facebookresearch/dinov2) for vision transformer features
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) for object segmentation
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for metric depth estimation
