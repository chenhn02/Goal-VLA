"""Gemini-based image editing: given an image + instruction, generate an edited image."""

import os
import re
import time
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from goalvla.config import Config

HIGH_PERF_MODEL = Config.GEMINI_EDIT_MODEL
FALLBACK_MODEL = Config.GEMINI_EDIT_FALLBACK


def _decode_to_bytes(raw):
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("data:"):
            comma_idx = s.find(",")
            if comma_idx != -1:
                s = s[comma_idx + 1:]
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            pass
        try:
            pad = "=" * (-len(s) % 4)
            return base64.urlsafe_b64decode(s + pad)
        except Exception:
            pass
    return None


def _extract_retry_delay(error_message: str) -> int:
    """Extract retry delay in seconds from error message. Returns -1 if not found."""
    patterns = [
        r'"retryDelay":\s*"(\d+)([smhd])"',
        r'retryDelay["\s]*:\s*["\s]*(\d+)([smhd])',
        r'(\d+)\s*seconds?',
    ]
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}

    for pattern in patterns:
        m = re.search(pattern, error_message, re.IGNORECASE)
        if m:
            try:
                value = int(m.group(1))
                unit = m.group(2) if m.lastindex >= 2 else "s"
                return value * multiplier.get(unit, 1)
            except (ValueError, IndexError):
                continue
    return -1


def get_available_model(client: genai.Client) -> str:
    """Select best available model, preferring high-performance."""
    try:
        models = [m.name for m in client.models.list()]
        for name in [HIGH_PERF_MODEL, f"models/{HIGH_PERF_MODEL}"]:
            if name in models:
                print(f"[edit_image] Using model: {HIGH_PERF_MODEL}")
                return HIGH_PERF_MODEL
    except Exception:
        pass
    print(f"[edit_image] Using fallback model: {FALLBACK_MODEL}")
    return FALLBACK_MODEL


def generate_edited_image(
    client: genai.Client,
    model: str,
    image_bytes: bytes,
    instruction: str,
    max_retries: int = 3,
) -> list[Image.Image]:
    """Generate an edited image with automatic retry on quota limits.

    Returns a list of PIL Images extracted from the response.
    """
    for attempt in range(max_retries):
        try:
            contents = [types.Content(
                role="user",
                parts=[
                    types.Part(text=instruction),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
                ],
            )]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )

            images = []
            for cand in getattr(response, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        img_bytes = _decode_to_bytes(inline.data)
                        if img_bytes:
                            try:
                                images.append(Image.open(BytesIO(img_bytes)))
                            except Exception:
                                pass
            if images:
                return images

            # A successful call can still carry no image (text-only reply or a
            # safety block). generate_content did not raise, so the exception
            # retry path above never triggers — treat an empty image list as a
            # transient failure and retry here instead of giving up.
            fb = getattr(response, "prompt_feedback", None)
            print(f"[edit_image] No image in response "
                  f"(attempt {attempt + 1}/{max_retries})"
                  f"{f'; prompt_feedback={fb}' if fb else ''}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return []

        except ClientError as e:
            if e.code == 429:
                delay = _extract_retry_delay(str(e))
                if 0 < delay <= 60:
                    print(f"[edit_image] Quota limit, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[edit_image] Quota limit, suggested wait: {delay}s")
                    break
            raise
        except Exception as e:
            print(f"[edit_image] Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise

    raise RuntimeError(f"Failed to generate image after {max_retries} attempts")


def edit_image(
    image_path: str | Path,
    instruction: str,
    output_dir: str | Path = None,
    prefix: str = "edited",
    model: str = None,
) -> Path:
    """Edit an image using Gemini and save the result.

    Args:
        image_path: Path to input image.
        instruction: Text instruction for editing.
        output_dir: Directory to save output. Defaults to same dir as input.
        prefix: Filename prefix for saved images.
        model: Gemini model name. Auto-selected if None.

    Returns:
        Path to the saved edited image.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    if model is None:
        model = get_available_model(client)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    images = generate_edited_image(client, model, img_bytes, instruction)
    if not images and model != FALLBACK_MODEL:
        print(f"[edit_image] '{model}' returned no image; "
              f"retrying with fallback '{FALLBACK_MODEL}'")
        images = generate_edited_image(client, FALLBACK_MODEL, img_bytes, instruction)
    if not images:
        raise RuntimeError("No images generated")

    if output_dir is None:
        output_dir = Path(image_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{prefix}_{timestamp}.png"
    images[0].save(out_path)
    print(f"[edit_image] Saved: {out_path}")
    return out_path
