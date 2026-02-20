#!/usr/bin/env python3
"""
Multi-Agent Image Enhancement Pipeline
=======================================
Entry point that:
  1. Loads environment variables (.env) for Galileo credentials.
  2. Accepts an input image path via CLI.
  3. Runs the LangGraph workflow: quality check → enhance (loop) → video generation.
  4. Prints a summary of the pipeline execution.

Usage:
    python main.py <path/to/image.jpg> [--max-iterations 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import json

from dotenv import load_dotenv


def _create_sample_image(path: str) -> str:
    """Generate a simple low-quality test image if none is provided."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 480), color=(60, 60, 60))  # dark, low contrast
    draw = ImageDraw.Draw(img)

    # Add some shapes so there is *something* to sharpen
    draw.rectangle([50, 50, 200, 200], fill=(90, 40, 40))
    draw.ellipse([300, 100, 500, 350], fill=(40, 90, 40))
    draw.polygon([(320, 50), (400, 200), (240, 200)], fill=(40, 40, 90))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((150, 400), "Sample Image", fill=(200, 200, 200), font=font)

    img.save(path)
    print(f"📷 Created sample test image: {path}")
    return path


def main() -> None:
    # ── Load environment ────────────────────────────────────────────
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the multi-agent image enhancement pipeline."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to the input image. If omitted a sample image is generated.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum enhancement iterations (default: 5).",
    )
    args = parser.parse_args()

    # Resolve image path
    if args.image is None:
        image_path = os.path.join(os.path.dirname(__file__) or ".", "sample_input.jpg")
        _create_sample_image(image_path)
    else:
        image_path = os.path.abspath(args.image)
        if not os.path.isfile(image_path):
            print(f"❌ Image not found: {image_path}")
            sys.exit(1)

    # ── Build and run the workflow ──────────────────────────────────
    from graph.workflow import run_pipeline  # late import after dotenv

    initial_state = {
        "image_path": image_path,
        "original_image_path": image_path,
        "quality_scores": {},
        "quality_passed": False,
        "enhancement_iteration": 0,
        "max_iterations": args.max_iterations,
        "enhancement_log": [],
        "video_output_path": "",
        "status": "Starting pipeline",
    }

    print("\n" + "=" * 60)
    print("  Multi-Agent Image Enhancement Pipeline")
    print("=" * 60)
    print(f"  Input image    : {image_path}")
    print(f"  Max iterations : {args.max_iterations}")
    print("=" * 60)

    final_state = run_pipeline(initial_state)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Quality passed : {final_state.get('quality_passed')}")
    print(f"  Iterations     : {final_state.get('enhancement_iteration')}")
    print(f"  Final image    : {final_state.get('image_path')}")
    print(f"  Video output   : {final_state.get('video_output_path')}")
    print(f"  Status         : {final_state.get('status')}")

    if final_state.get("enhancement_log"):
        print("\n  Enhancement History:")
        for entry in final_state["enhancement_log"]:
            print(f"    Iter {entry['iteration']}: {', '.join(entry['applied'])}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

