"""Image Enhancement Agent -- uses Pillow ImageEnhance to improve the image."""

from __future__ import annotations

import os
from PIL import Image, ImageEnhance

from graph.state import ImagePipelineState

# ── Enhancement factors ─────────────────────────────────────────────
# When a metric score is below 1.0 we boost the corresponding enhancer.
# The factor is: 1.0 + BOOST_STEP * (1.0 - score), so a score of 0.5
# yields a factor of 1.0 + 0.4 * 0.5 = 1.20  (moderate boost).
BOOST_STEP = 0.4


def _output_path(original_path: str, iteration: int) -> str:
    """Generate an output path for the enhanced image."""
    base, ext = os.path.splitext(original_path)
    directory = os.path.dirname(original_path) or "."
    basename = os.path.basename(base)
    out_dir = os.path.join(directory, "enhanced")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{basename}_iter{iteration}{ext}")


def enhancement_agent(state: ImagePipelineState) -> dict:
    """LangGraph node: enhance the image based on quality scores."""
    image_path = state["image_path"]
    quality_scores = state["quality_scores"]
    iteration = state.get("enhancement_iteration", 0) + 1

    img = Image.open(image_path)
    applied: list[str] = []

    # ── Brightness ──────────────────────────────────────────────────
    b_score = quality_scores["brightness"]["score"]
    if b_score < 1.0:
        factor = 1.0 + BOOST_STEP * (1.0 - b_score)
        # If the raw brightness is too high, we need to *reduce* it
        if quality_scores["brightness"]["raw"] > 180:
            factor = 1.0 / factor  # invert
        img = ImageEnhance.Brightness(img).enhance(factor)
        applied.append(f"Brightness x{factor:.2f}")

    # ── Contrast ────────────────────────────────────────────────────
    c_score = quality_scores["contrast"]["score"]
    if c_score < 1.0:
        factor = 1.0 + BOOST_STEP * (1.0 - c_score)
        img = ImageEnhance.Contrast(img).enhance(factor)
        applied.append(f"Contrast x{factor:.2f}")

    # ── Sharpness ───────────────────────────────────────────────────
    s_score = quality_scores["sharpness"]["score"]
    if s_score < 1.0:
        factor = 1.0 + BOOST_STEP * (1.0 - s_score)
        img = ImageEnhance.Sharpness(img).enhance(factor)
        applied.append(f"Sharpness x{factor:.2f}")

    # ── Color saturation (small general boost) ──────────────────────
    img = ImageEnhance.Color(img).enhance(1.05)
    applied.append("Color x1.05")

    # Save enhanced image
    out_path = _output_path(state["original_image_path"], iteration)
    img.save(out_path)

    enhancement_log = list(state.get("enhancement_log", []))
    enhancement_log.append({"iteration": iteration, "applied": applied, "output": out_path})

    print(f"\n🔧 [Iteration {iteration}] Enhancement Applied:")
    for a in applied:
        print(f"   • {a}")
    print(f"   Saved to: {out_path}")

    return {
        "image_path": out_path,
        "enhancement_iteration": iteration,
        "enhancement_log": enhancement_log,
        "status": f"Enhanced image (iteration {iteration}): {', '.join(applied)}",
    }

