#!/usr/bin/env python3
"""
Bulk Pipeline Runner – Galileo Scaling Test
============================================
Generates a batch of varied synthetic images and runs the multi-agent
image-enhancement pipeline on each one, measuring wall-clock time and
Galileo upload behaviour.

Three modes:
  1. sequential-per-run  – upload to Galileo after every single workflow
  2. sequential-batch    – run all workflows, upload in one batch at the end
  3. concurrent-batch    – run workflows in parallel threads, upload once

Usage:
    python bulk_run.py [--count 20] [--mode sequential-per-run]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Image generators – each produces a different quality profile so the
# pipeline exercises different enhancement paths.
# ---------------------------------------------------------------------------

PALETTES = [
    # (bg, shape1, shape2, shape3, text)  – intentionally varied quality
    ((30, 30, 30), (80, 30, 30), (30, 80, 30), (30, 30, 80), (160, 160, 160)),   # very dark
    ((240, 240, 240), (200, 180, 180), (180, 200, 180), (180, 180, 200), (80, 80, 80)),  # very bright
    ((60, 60, 60), (90, 40, 40), (40, 90, 40), (40, 40, 90), (200, 200, 200)),   # low contrast
    ((120, 80, 80), (200, 100, 60), (60, 160, 100), (80, 80, 200), (255, 255, 200)),  # decent
    ((50, 70, 50), (110, 70, 50), (50, 110, 70), (70, 50, 110), (190, 190, 170)),  # greenish dark
    ((100, 100, 100), (180, 60, 60), (60, 180, 60), (60, 60, 180), (230, 230, 230)),  # mid-grey
]

SIZES = [(320, 240), (640, 480), (800, 600), (1024, 768), (480, 480)]


def generate_test_image(path: str, index: int) -> str:
    """Create a synthetic image with varied quality characteristics."""
    palette = PALETTES[index % len(PALETTES)]
    w, h = random.choice(SIZES)
    bg, s1, s2, s3, txt = palette

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)

    # Shapes
    draw.rectangle([w * 0.05, h * 0.1, w * 0.35, h * 0.55], fill=s1)
    draw.ellipse([w * 0.4, h * 0.15, w * 0.75, h * 0.65], fill=s2)
    draw.polygon(
        [(w * 0.5, h * 0.05), (w * 0.7, h * 0.45), (w * 0.3, h * 0.45)],
        fill=s3,
    )
    draw.text((w * 0.2, h * 0.8), f"Test Image #{index + 1}", fill=txt)

    # Optionally blur to lower sharpness
    if index % 3 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=2))

    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _make_initial_state(image_path: str, max_iter: int = 3) -> dict:
    return {
        "image_path": image_path,
        "original_image_path": image_path,
        "quality_scores": {},
        "quality_passed": False,
        "enhancement_iteration": 0,
        "max_iterations": max_iter,
        "enhancement_log": [],
        "video_output_path": "",
        "status": "Starting pipeline",
    }


# ---------------------------------------------------------------------------
# Bulk execution modes
# ---------------------------------------------------------------------------

def run_sequential_per_run(image_paths: list[str]) -> list[dict]:
    """Run each pipeline sequentially; upload to Galileo after every run."""
    from graph.workflow import run_pipeline, reset_galileo

    results = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(image_paths)}] Processing {os.path.basename(path)}")
        print(f"{'─' * 60}")

        reset_galileo()  # fresh ObserveWorkflows per run
        t0 = time.perf_counter()
        final = run_pipeline(_make_initial_state(path), defer_upload=False)
        elapsed = time.perf_counter() - t0
        results.append({"index": i, "elapsed": elapsed, "state": final})
        print(f"  ⏱  Finished in {elapsed:.2f}s")

    return results


def run_sequential_batch(image_paths: list[str]) -> list[dict]:
    """Run all pipelines sequentially; upload everything in one Galileo batch."""
    from graph.workflow import run_pipeline, flush_galileo

    results = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(image_paths)}] Processing {os.path.basename(path)}")
        print(f"{'─' * 60}")

        t0 = time.perf_counter()
        final = run_pipeline(_make_initial_state(path), defer_upload=True)
        elapsed = time.perf_counter() - t0
        results.append({"index": i, "elapsed": elapsed, "state": final})
        print(f"  ⏱  Finished in {elapsed:.2f}s")

    # Single bulk upload
    print(f"\n{'═' * 60}")
    print(f"  Uploading {len(results)} workflows to Galileo in one batch…")
    print(f"{'═' * 60}")
    t0 = time.perf_counter()
    uploaded = flush_galileo()
    upload_time = time.perf_counter() - t0
    print(f"  ⏱  Upload took {upload_time:.2f}s for {uploaded} workflow(s)")
    results.append({"_upload_time": upload_time, "_uploaded_count": uploaded})
    return results


def run_concurrent_batch(image_paths: list[str], workers: int = 4) -> list[dict]:
    """Run pipelines in parallel threads; upload everything once at the end."""
    from graph.workflow import run_pipeline, flush_galileo

    results: list[dict] = []

    def _worker(idx: int, path: str) -> dict:
        t0 = time.perf_counter()
        final = run_pipeline(_make_initial_state(path), defer_upload=True)
        elapsed = time.perf_counter() - t0
        return {"index": idx, "elapsed": elapsed, "state": final}

    print(f"\n  Launching {len(image_paths)} workflows across {workers} threads…\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, i + 1, p): i
            for i, p in enumerate(image_paths)
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"  ✔ Workflow #{res['index']} done in {res['elapsed']:.2f}s")

    # Single bulk upload
    print(f"\n{'═' * 60}")
    print(f"  Uploading {len(results)} workflows to Galileo in one batch…")
    print(f"{'═' * 60}")
    t0 = time.perf_counter()
    uploaded = flush_galileo()
    upload_time = time.perf_counter() - t0
    print(f"  ⏱  Upload took {upload_time:.2f}s for {uploaded} workflow(s)")
    results.append({"_upload_time": upload_time, "_uploaded_count": uploaded})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES = {
    "sequential-per-run": run_sequential_per_run,
    "sequential-batch": run_sequential_batch,
    "concurrent-batch": run_concurrent_batch,
}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Bulk pipeline runner for Galileo scaling test.")
    parser.add_argument("--count", type=int, default=20, help="Number of images to process (default 20).")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="sequential-batch",
        help="Execution mode (default: sequential-batch).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Thread count for concurrent-batch mode.")
    args = parser.parse_args()

    # Prepare temp directory for test images
    tmp_dir = os.path.join(os.path.dirname(__file__) or ".", "bulk_test_images")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print("=" * 60)
    print("  Bulk Pipeline Runner – Galileo Scaling Test")
    print("=" * 60)
    print(f"  Mode           : {args.mode}")
    print(f"  Image count    : {args.count}")
    if args.mode == "concurrent-batch":
        print(f"  Worker threads : {args.workers}")
    print(f"  Temp dir       : {tmp_dir}")
    print("=" * 60)

    # Generate varied test images
    print("\n📷 Generating test images…")
    image_paths: list[str] = []
    for i in range(args.count):
        p = os.path.join(tmp_dir, f"test_{i + 1:03d}.jpg")
        generate_test_image(p, i)
        image_paths.append(p)
    print(f"   Created {len(image_paths)} images.\n")

    # Run
    overall_start = time.perf_counter()

    if args.mode == "concurrent-batch":
        results = run_concurrent_batch(image_paths, workers=args.workers)
    else:
        runner = MODES[args.mode]
        results = runner(image_paths)

    overall_elapsed = time.perf_counter() - overall_start

    # Summary
    pipeline_results = [r for r in results if "index" in r]
    times = [r["elapsed"] for r in pipeline_results]

    print("\n" + "=" * 60)
    print("  Bulk Run Summary")
    print("=" * 60)
    print(f"  Mode               : {args.mode}")
    print(f"  Total workflows    : {len(pipeline_results)}")
    print(f"  Total wall-clock   : {overall_elapsed:.2f}s")
    if times:
        print(f"  Avg per workflow   : {sum(times) / len(times):.2f}s")
        print(f"  Min / Max          : {min(times):.2f}s / {max(times):.2f}s")
        print(f"  Throughput         : {len(times) / overall_elapsed:.1f} workflows/s")

    upload_rows = [r for r in results if "_upload_time" in r]
    if upload_rows:
        u = upload_rows[0]
        print(f"  Galileo upload     : {u['_uploaded_count']} workflows in {u['_upload_time']:.2f}s")
    print("=" * 60 + "\n")

    # Cleanup enhanced + output dirs that the pipeline creates per image
    for d in ["enhanced", "output"]:
        full = os.path.join(os.path.dirname(__file__) or ".", d)
        if os.path.isdir(full):
            shutil.rmtree(full)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    print("🧹 Cleaned up temporary files.\n")


if __name__ == "__main__":
    main()

