"""Image resize utilities for feature extraction."""

from PIL import Image


def resize(img: Image.Image, target_res: int, resize: bool = True,
           to_pil: bool = False) -> Image.Image:
    if not resize:
        return img
    img = img.resize((target_res, target_res), Image.LANCZOS)
    return img
