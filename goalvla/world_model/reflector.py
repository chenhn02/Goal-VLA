"""Iterative validation and reflection loop using Gemini.

Validates generated overlay images against the original instruction,
and generates revised prompts when validation fails.
"""

import os
import time

from pathlib import Path

from google import genai
from google.genai import types

GEMINI_MODEL_ORDER = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def _parse_quota_retry_seconds(err_msg: str) -> int:
    import re
    m = re.search(r"retry[- ]?after[: ]\s*(\d+)", err_msg, flags=re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)))
    m2 = re.search(r"(\d+)\s*seconds", err_msg, flags=re.IGNORECASE)
    if m2:
        return max(1, int(m2.group(1)))
    return 15


def _call_gemini_with_fallback(
    client: genai.Client,
    contents: list,
    config: types.GenerateContentConfig,
) -> tuple:
    """Try primary model with quota backoff, then fall through model chain.

    Returns (response, model_used, used_fallback).
    """
    primary = GEMINI_MODEL_ORDER[0]
    try:
        print(f"[reflector] Trying model: {primary}")
        return client.models.generate_content(
            model=primary, contents=contents, config=config,
        ), primary, False
    except Exception as e:
        msg = str(e)
        print(f"[reflector] Primary model failed: {msg}")
        if "quota" in msg.lower():
            wait_s = _parse_quota_retry_seconds(msg)
            print(f"[reflector] Quota limit, waiting {wait_s}s...")
            time.sleep(wait_s)
            try:
                return client.models.generate_content(
                    model=primary, contents=contents, config=config,
                ), primary, False
            except Exception as e2:
                print(f"[reflector] Retry failed: {e2}")

    for m in GEMINI_MODEL_ORDER[1:]:
        try:
            print(f"[reflector] Trying fallback: {m}")
            resp = client.models.generate_content(model=m, contents=contents, config=config)
            return resp, m, True
        except Exception as ee:
            print(f"[reflector] Fallback {m} failed: {ee}")

    raise RuntimeError("All Gemini models failed")


def validate_overlay(
    original_text: str,
    enhanced_text: str,
    overlay_image_path: Path,
    api_key: str,
) -> tuple[bool, str]:
    """Validate whether an overlay image meets the original instruction.

    Returns (is_valid, feedback_text).
    """
    client = genai.Client(api_key=api_key)

    with open(overlay_image_path, "rb") as f:
        image_bytes = f.read()

    prompt = f"""\
You are an expert image analysis assistant. Evaluate whether an overlay image \
successfully demonstrates the requested image editing task.

**Original User Request:** {original_text}
**Enhanced Instruction:** {enhanced_text}

**Evaluation Criteria:**
1. Does the overlay image show the main object(s) from the original instruction?
2. Are the objects positioned/arranged as requested?
3. Is the visual result clear and matches the intent?
4. Are there any obvious errors or missing elements?

**Response Format:**
- First line: "VALID: Yes" or "VALID: No"
- If valid, provide a brief confirmation
- If not valid, provide specific feedback on what needs improvement

Be strict but fair. Only mark as valid if the image clearly demonstrates the requested changes.
"""

    try:
        mime = "image/png" if overlay_image_path.suffix.lower() == ".png" else "image/jpeg"
        content = types.Content(
            role="user",
            parts=[
                types.Part(text=prompt),
                types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes)),
            ],
        )

        response, model_used, _ = _call_gemini_with_fallback(
            client,
            contents=[content],
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1024),
        )

        if hasattr(response, "text") and response.text:
            result = response.text.strip()
            print(f"[reflector] Validation result ({model_used}): {result}")

            if result.startswith("VALID: Yes"):
                return True, "Image meets requirements"
            elif result.startswith("VALID: No"):
                return False, result.replace("VALID: No", "").strip()
            elif "valid" in result.lower() and "yes" in result.lower():
                return True, "Image meets requirements"
            else:
                return False, "Image does not meet requirements"

        raise RuntimeError("No text in Gemini response")

    except Exception as e:
        print(f"[reflector] Validation failed: {e}")
        return False, f"Validation error: {e}"


def generate_revised_instruction(
    original_text: str,
    enhanced_text: str,
    feedback: str,
    api_key: str,
) -> str:
    """Generate a revised instruction based on validation feedback."""
    client = genai.Client(api_key=api_key)

    prompt = f"""\
You are an expert prompt engineer for AI image editing. Improve an enhanced prompt based on feedback.

**Important context:**
- The feedback is based on an OVERLAY image (edited objects overlaid on the original)
- Your task is to create a better prompt for generating the EDITED IMAGE from the original
- The overlay is only for validation purposes

**Original User Request:** {original_text}
**Previous Enhanced Prompt:** {enhanced_text}
**Validation Feedback:** {feedback}

**Guidelines:**
1. Focus on improving the edited image generation, not the overlay
2. Address specific issues from the feedback
3. Make instructions more precise and clear
4. Maintain the same format as the enhanced prompt
5. Keep the SAME object as the "moved object". In particular, if the request uses a handheld tool ("with/using the <tool>"), the tool (e.g. brush, sponge) must stay the moved object and must be shown displaced to where it finished its action — never revert to moving only the material it acts on.

Return only the revised enhanced prompt without additional explanation.
"""

    try:
        response, model_used, used_fallback = _call_gemini_with_fallback(
            client,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048),
        )

        if hasattr(response, "text") and response.text:
            revised = response.text.strip()
            print(f"[reflector] Revised instruction ({model_used}): {revised}")
            if used_fallback:
                from goalvla.world_model.enhancer import enhance_instruction
                revised = enhance_instruction(revised)
            return revised

        print("[reflector] No revised text generated, using original")
        return enhanced_text

    except Exception as e:
        print(f"[reflector] Revision failed: {e}")
        return enhanced_text
