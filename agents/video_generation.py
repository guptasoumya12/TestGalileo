"""Video Generation Agent -- uses MoviePy to create a short video from the enhanced image."""

from __future__ import annotations

import os

import numpy as np
from moviepy import ImageClip
from PIL import Image

from graph.state import ImagePipelineState

# ── Video settings ──────────────────────────────────────────────────
VIDEO_DURATION = 5  # seconds
VIDEO_FPS = 24
ZOOM_START = 1.0
ZOOM_END = 1.2  # subtle Ken-Burns zoom


def _make_zoom_clip(image_path: str, duration: float, fps: int) -> ImageClip:
    """Create a clip with a slow Ken-Burns zoom effect.

    Uses MoviePy 2.x API: ImageClip(..., duration=...) and
    .transform() for per-frame manipulation.
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    base_clip = ImageClip(img_array, duration=duration).with_fps(fps)

    def zoom_effect(get_frame, t):
        """Apply a smooth zoom that scales from ZOOM_START to ZOOM_END."""
        progress = t / duration
        scale = ZOOM_START + (ZOOM_END - ZOOM_START) * progress

        # Calculate crop box centred on the image
        new_w = int(w / scale)
        new_h = int(h / scale)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2

        frame = get_frame(t)
        cropped = frame[y1 : y1 + new_h, x1 : x1 + new_w]

        # Resize back to original dimensions
        resized = np.array(
            Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        )
        return resized

    zoomed = base_clip.transform(zoom_effect)
    return zoomed


def video_generation_agent(state: ImagePipelineState) -> dict:
    """LangGraph node: generate a short video from the final image."""
    image_path = state["image_path"]

    # Determine output path next to the project root
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "final_video.mp4")

    print(f"\n🎬 Generating video from: {image_path}")
    print(f"   Duration : {VIDEO_DURATION}s @ {VIDEO_FPS}fps")
    print(f"   Zoom     : {ZOOM_START}x → {ZOOM_END}x")

    clip = _make_zoom_clip(image_path, VIDEO_DURATION, VIDEO_FPS)
    clip.write_videofile(
        video_path,
        fps=VIDEO_FPS,
        codec="libx264",
        audio=False,
        logger="bar",
    )

    print(f"   ✅ Video saved to: {video_path}")

    return {
        "video_output_path": video_path,
        "status": f"Video generated at {video_path}",
    }

