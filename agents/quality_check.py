"""Image Quality Check Agent -- uses Pillow to assess brightness, contrast, and sharpness."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from graph.state import ImagePipelineState

# ── Quality thresholds ──────────────────────────────────────────────
BRIGHTNESS_LOW = 100
BRIGHTNESS_HIGH = 180
CONTRAST_MIN = 50
SHARPNESS_MIN = 30
OVERALL_THRESHOLD = 0.6  # 0-1 scale; ≥ this means "pass"


def _brightness_score(img: Image.Image) -> tuple[float, float]:
    """Return (raw_brightness, normalised 0-1 score).

    Raw brightness = mean pixel intensity of the grayscale conversion (0-255).
    Score = 1.0 when brightness is in the ideal range [100, 180], tapering
    linearly to 0.0 at the extremes (0 and 255).
    """
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    raw = stat.mean[0]  # mean intensity 0-255

    if BRIGHTNESS_LOW <= raw <= BRIGHTNESS_HIGH:
        score = 1.0
    elif raw < BRIGHTNESS_LOW:
        score = max(raw / BRIGHTNESS_LOW, 0.0)
    else:
        score = max((255 - raw) / (255 - BRIGHTNESS_HIGH), 0.0)
    return raw, round(score, 3)


def _contrast_score(img: Image.Image) -> tuple[float, float]:
    """Return (raw_contrast, normalised 0-1 score).

    Raw contrast = standard deviation of grayscale pixel intensities.
    Score ramps linearly from 0 at stddev=0 to 1 at stddev=CONTRAST_MIN,
    staying at 1 above.
    """
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    raw = stat.stddev[0]

    score = min(raw / CONTRAST_MIN, 1.0) if CONTRAST_MIN > 0 else 1.0
    return round(raw, 2), round(score, 3)


def _sharpness_score(img: Image.Image) -> tuple[float, float]:
    """Return (raw_sharpness, normalised 0-1 score).

    Raw sharpness = variance of edge-detected image (Pillow FIND_EDGES).
    Score ramps linearly from 0 at variance=0 to 1 at variance=SHARPNESS_MIN,
    staying at 1 above.
    """
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges, dtype=np.float64)
    raw = float(edge_array.var())

    score = min(raw / SHARPNESS_MIN, 1.0) if SHARPNESS_MIN > 0 else 1.0
    return round(raw, 2), round(score, 3)


def quality_check_agent(state: ImagePipelineState) -> dict:
    """LangGraph node: evaluate image quality and update state."""
    image_path = state["image_path"]
    img = Image.open(image_path)

    brightness_raw, brightness_norm = _brightness_score(img)
    contrast_raw, contrast_norm = _contrast_score(img)
    sharpness_raw, sharpness_norm = _sharpness_score(img)

    # Weighted overall score (equal weights)
    overall = round((brightness_norm + contrast_norm + sharpness_norm) / 3, 3)
    quality_passed = overall >= OVERALL_THRESHOLD

    quality_scores = {
        "brightness": {"raw": brightness_raw, "score": brightness_norm},
        "contrast": {"raw": contrast_raw, "score": contrast_norm},
        "sharpness": {"raw": sharpness_raw, "score": sharpness_norm},
        "overall": overall,
    }

    iteration = state.get("enhancement_iteration", 0)
    tag = "Initial" if iteration == 0 else f"Iteration {iteration}"
    print(f"\n📊 [{tag}] Quality Check Results:")
    print(f"   Brightness : {brightness_raw:.1f} (score {brightness_norm})")
    print(f"   Contrast   : {contrast_raw:.1f} (score {contrast_norm})")
    print(f"   Sharpness  : {sharpness_raw:.1f} (score {sharpness_norm})")
    print(f"   Overall    : {overall}  →  {'PASS ✅' if quality_passed else 'FAIL ❌'}")

    return {
        "quality_scores": quality_scores,
        "quality_passed": quality_passed,
        "status": f"Quality check {'passed' if quality_passed else 'failed'} (overall={overall})",
    }

