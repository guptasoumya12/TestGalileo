"""State definition for the image processing pipeline."""

from typing import TypedDict


class ImagePipelineState(TypedDict):
    """Shared state that flows through the LangGraph pipeline."""

    # Paths
    image_path: str  # Path to current (possibly enhanced) image
    original_image_path: str  # Path to the original input image

    # Quality assessment
    quality_scores: dict  # {"brightness": float, "contrast": float, "sharpness": float, "overall": float}
    quality_passed: bool  # Whether quality thresholds are met

    # Enhancement tracking
    enhancement_iteration: int  # Current iteration count
    max_iterations: int  # Max allowed enhancement loops (default 5)
    enhancement_log: list  # History of enhancements applied per iteration

    # Output
    video_output_path: str  # Final video output path
    status: str  # Current pipeline status message

