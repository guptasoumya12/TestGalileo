#!/usr/bin/env python3
"""
Anomaly Test Suite – Inject Bad Inputs into the Pipeline
=========================================================
Generates a suite of deliberately malformed / edge-case images and runs
each through the multi-agent pipeline.  Every run (success or crash) is
reported to Galileo with the correct status code so that the Tool Error
Rate and Agentic Workflow Success scorers can detect them.

Test cases:
  1. missing_file        – path that doesn't exist
  2. zero_byte_file      – empty file with .jpg extension
  3. corrupted_random    – random bytes masquerading as .jpg
  4. truncated_jpeg      – valid JPEG header, then cut off
  5. wrong_format_txt    – a plain-text file renamed to .jpg
  6. tiny_1x1            – a valid but 1×1 pixel image
  7. huge_dimension       – extremely large canvas (10 000 × 10 000)
  8. grayscale_only      – single-channel grayscale image (mode "L")
  9. rgba_with_alpha     – 4-channel RGBA PNG (alpha mismatch)
  10. normal_control     – a well-formed image (should succeed)

Usage:
    python anomaly_test.py [--defer-upload]
"""

from __future__ import annotations

import os
import random
import shutil
import struct
import sys
import time

from dotenv import load_dotenv
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Test-image generators
# ---------------------------------------------------------------------------

TMP_DIR: str = ""  # set in main()


def _path(name: str) -> str:
    return os.path.join(TMP_DIR, name)


def gen_missing_file() -> tuple[str, str]:
    """Return a path to a file that does not exist."""
    p = _path("this_file_does_not_exist.jpg")
    # Make sure it really doesn't exist
    if os.path.exists(p):
        os.remove(p)
    return p, "missing_file"


def gen_zero_byte_file() -> tuple[str, str]:
    """Create an empty 0-byte file with a .jpg extension."""
    p = _path("zero_byte.jpg")
    open(p, "wb").close()
    return p, "zero_byte_file"


def gen_corrupted_random() -> tuple[str, str]:
    """Create a file filled with random bytes."""
    p = _path("corrupted_random.jpg")
    with open(p, "wb") as f:
        f.write(os.urandom(4096))
    return p, "corrupted_random_bytes"


def gen_truncated_jpeg() -> tuple[str, str]:
    """Write a valid JPEG SOI header + a few bytes, then stop."""
    p = _path("truncated.jpg")
    with open(p, "wb") as f:
        # JPEG SOI marker + APP0 marker start
        f.write(b"\xff\xd8\xff\xe0")
        f.write(struct.pack(">H", 16))  # APP0 length
        f.write(b"JFIF\x00")
        f.write(b"\x01\x01\x00\x00\x01\x00\x01\x00\x00")
        # Abruptly end — no image data, no EOI
    return p, "truncated_jpeg"


def gen_wrong_format_txt() -> tuple[str, str]:
    """Create a plain-text file with a .jpg extension."""
    p = _path("actually_text.jpg")
    with open(p, "w") as f:
        f.write("This is not an image. It is plain text.\n" * 50)
    return p, "wrong_format_txt"


def gen_tiny_1x1() -> tuple[str, str]:
    """Create a valid 1×1 pixel JPEG."""
    p = _path("tiny_1x1.jpg")
    img = Image.new("RGB", (1, 1), color=(128, 128, 128))
    img.save(p)
    return p, "tiny_1x1_pixel"


def gen_huge_dimension() -> tuple[str, str]:
    """Create a very large image (10 000 × 10 000)."""
    p = _path("huge_10k.jpg")
    # Use a solid color to keep file size manageable
    img = Image.new("RGB", (10_000, 10_000), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([1000, 1000, 9000, 9000], fill=(50, 80, 120))
    img.save(p, quality=30)  # low quality to keep file small-ish
    return p, "huge_10000x10000"


def gen_grayscale() -> tuple[str, str]:
    """Create a single-channel grayscale image (mode 'L')."""
    p = _path("grayscale.jpg")
    img = Image.new("L", (640, 480), color=90)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 300, 300], fill=200)
    img.save(p)
    return p, "grayscale_single_channel"


def gen_rgba_png() -> tuple[str, str]:
    """Create a 4-channel RGBA PNG (not JPEG-compatible mode)."""
    p = _path("rgba_image.png")
    img = Image.new("RGBA", (640, 480), color=(100, 50, 50, 128))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 400, 400], fill=(200, 200, 50, 200))
    img.save(p)
    return p, "rgba_with_alpha"


def gen_normal_control() -> tuple[str, str]:
    """Create a normal, well-formed test image (control case)."""
    p = _path("normal_control.jpg")
    img = Image.new("RGB", (640, 480), color=(60, 60, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill=(90, 40, 40))
    draw.ellipse([300, 100, 500, 350], fill=(40, 90, 40))
    draw.text((150, 400), "Normal Control Image", fill=(200, 200, 200))
    img.save(p)
    return p, "normal_control"


# All generators in order
GENERATORS = [
    gen_missing_file,
    gen_zero_byte_file,
    gen_corrupted_random,
    gen_truncated_jpeg,
    gen_wrong_format_txt,
    gen_tiny_1x1,
    gen_huge_dimension,
    gen_grayscale,
    gen_rgba_png,
    gen_normal_control,
]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _make_state(image_path: str, max_iter: int = 3) -> dict:
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


def run_anomaly_suite(*, defer_upload: bool = False) -> list[dict]:
    """Run every anomaly test case and collect results."""
    from graph.workflow import run_pipeline, flush_galileo

    results: list[dict] = []

    for i, gen_fn in enumerate(GENERATORS, 1):
        image_path, label = gen_fn()

        print(f"\n{'━' * 60}")
        print(f"  [{i}/{len(GENERATORS)}] TEST: {label}")
        print(f"  Image: {image_path}")
        print(f"  Exists: {os.path.exists(image_path)}"
              f"  Size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} bytes")
        print(f"{'━' * 60}")

        state = _make_state(image_path)
        t0 = time.perf_counter()

        try:
            final = run_pipeline(
                state,
                defer_upload=True,   # batch upload at the end
                raise_on_error=False,  # don't crash the test suite
            )
            elapsed = time.perf_counter() - t0
            errored = "PIPELINE_ERROR" in final.get("status", "")
            results.append({
                "index": i,
                "label": label,
                "elapsed": elapsed,
                "errored": errored,
                "status": final.get("status", ""),
                "quality_passed": final.get("quality_passed", False),
            })
            tag = "❌ ERRORED" if errored else "✅ OK"
            print(f"\n  Result: {tag}  ({elapsed:.2f}s)")
            print(f"  Status: {final.get('status', '')[:120]}")

        except Exception as exc:
            # Shouldn't normally reach here since raise_on_error=False
            elapsed = time.perf_counter() - t0
            results.append({
                "index": i,
                "label": label,
                "elapsed": elapsed,
                "errored": True,
                "status": f"UNCAUGHT: {type(exc).__name__}: {exc}",
                "quality_passed": False,
            })
            print(f"\n  Result: 💥 UNCAUGHT EXCEPTION ({elapsed:.2f}s)")
            print(f"  {type(exc).__name__}: {exc}")

    # Upload all workflows (successes and failures) in one batch
    print(f"\n{'═' * 60}")
    print(f"  Uploading {len(results)} test workflows to Galileo…")
    print(f"{'═' * 60}")
    t0 = time.perf_counter()
    uploaded = flush_galileo()
    upload_time = time.perf_counter() - t0
    print(f"  ⏱  Upload: {uploaded} workflow(s) in {upload_time:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    """Print a formatted summary table."""

    print(f"\n{'═' * 70}")
    print(f"  Anomaly Test Report")
    print(f"{'═' * 70}")
    print(f"  {'#':<3} {'Test Case':<28} {'Time':>6} {'Result':<10} {'Quality'}")
    print(f"  {'─' * 3} {'─' * 28} {'─' * 6} {'─' * 10} {'─' * 7}")

    passed = 0
    failed = 0
    for r in results:
        tag = "❌ ERROR" if r["errored"] else "✅ OK"
        qp = "PASS" if r["quality_passed"] else "FAIL"
        if r["errored"]:
            failed += 1
        else:
            passed += 1
        print(f"  {r['index']:<3} {r['label']:<28} {r['elapsed']:>5.2f}s {tag:<10} {qp}")

    print(f"  {'─' * 3} {'─' * 28} {'─' * 6} {'─' * 10} {'─' * 7}")
    print(f"  Total: {len(results)}  |  Succeeded: {passed}  |  Errored: {failed}")
    print(f"{'═' * 70}")

    # Expected-result validation
    print(f"\n  Expected Behaviour:")
    expected_errors = {
        "missing_file", "zero_byte_file", "corrupted_random_bytes",
        "truncated_jpeg", "wrong_format_txt",
    }
    expected_ok = {
        "tiny_1x1_pixel", "huge_10000x10000", "grayscale_single_channel",
        "rgba_with_alpha", "normal_control",
    }
    for r in results:
        label = r["label"]
        if label in expected_errors:
            match = "✅" if r["errored"] else "⚠️  UNEXPECTED SUCCESS"
            print(f"    {label:<28} expected=ERROR   got={'ERROR' if r['errored'] else 'OK'}  {match}")
        elif label in expected_ok:
            match = "✅" if not r["errored"] else "⚠️  UNEXPECTED ERROR"
            print(f"    {label:<28} expected=OK      got={'ERROR' if r['errored'] else 'OK'}  {match}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    global TMP_DIR
    load_dotenv()

    TMP_DIR = os.path.join(os.path.dirname(__file__) or ".", "anomaly_test_images")
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    print("=" * 60)
    print("  Anomaly Test Suite – Bad Input Injection")
    print("=" * 60)
    print(f"  Test cases : {len(GENERATORS)}")
    print(f"  Temp dir   : {TMP_DIR}")
    print("=" * 60)

    overall_start = time.perf_counter()
    results = run_anomaly_suite()
    overall_elapsed = time.perf_counter() - overall_start

    print_report(results)
    print(f"  Total wall-clock: {overall_elapsed:.1f}s\n")

    # Cleanup
    for d in ["enhanced", "output", TMP_DIR]:
        if os.path.isdir(d):
            shutil.rmtree(d)
    print("🧹 Cleaned up temporary files.\n")


if __name__ == "__main__":
    main()

