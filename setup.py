from setuptools import setup, find_packages

setup(
    name="goalvla",
    version="0.1.0",
    description="GoalVLA: Goal-Conditioned Visual Language Action via World Model",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0,<2.4",
        "torchvision>=0.16.0,<0.19",
        "numpy<2",
        "pillow",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "opencv-python>=4.8.0",
        "viser",
        "tqdm",
        "pycocotools>=2.0.6",
        "google-genai",
        "timm",
        "ftfy",
        "hydra-core>=1.3",
        "transformers>=4.40,<5",
        "supervision>=0.16.0",
        "addict",
    ],
)
