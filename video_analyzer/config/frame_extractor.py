"""
Configuration for the FrameExtractor.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, validator


class ExtractionStrategy(str, Enum):
    """
    Strategies for frame extraction.
    """

    UNIFORM = "uniform"
    SCENE_CHANGE = "scene_change"
    KEYFRAME = "keyframe"


class FrameExtractorConfig(BaseModel):
    """
    Configuration for the FrameExtractor.
    """

    strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.UNIFORM,
        description="Strategy for frame extraction",
    )

    # Uniform strategy parameters
    uniform_interval_seconds: float = Field(
        default=1.0,
        description="Interval between frames in seconds for uniform extraction",
    )

    # Scene change strategy parameters
    scene_change_threshold: float = Field(
        default=30.0,
        description="Threshold for scene change detection (higher values mean fewer scene changes)",
    )

    # Keyframe strategy parameters
    keyframe_method: str = Field(
        default="content_based",
        description="Method for keyframe extraction (content_based, motion_based)",
    )

    # General parameters
    max_frames: Optional[int] = Field(
        default=None,
        description="Maximum number of frames to extract (None for no limit)",
    )

    batch_size: int = Field(
        default=100,
        description="Number of frames to process in a batch for large videos",
    )

    @validator("uniform_interval_seconds")
    def validate_uniform_interval(cls, v):
        if v <= 0:
            raise ValueError("Uniform interval must be positive")
        return v

    @validator("scene_change_threshold")
    def validate_scene_change_threshold(cls, v):
        if v <= 0:
            raise ValueError("Scene change threshold must be positive")
        return v

    @validator("batch_size")
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v
