"""
Data models for video information.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field, validator


class Frame(BaseModel):
    """
    Data model for a video frame.
    """

    image: Any  # np.ndarray, but Pydantic doesn't directly support numpy types
    timestamp: float = Field(..., description="Timestamp of the frame in seconds")
    index: int = Field(..., description="Frame index in the video")

    class Config:
        arbitrary_types_allowed = True


class VideoData(BaseModel):
    """
    Data model for video information.
    """

    path: Path = Field(..., description="Path to the video file")
    frames: List[Frame] = Field(default_factory=list, description="Extracted frames")
    duration: float = Field(..., description="Duration of the video in seconds")
    fps: float = Field(..., description="Frames per second")
    resolution: Tuple[int, int] = Field(
        ..., description="Video resolution (width, height)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional video metadata"
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("path")
    def validate_path(cls, v):
        if not v.exists():
            raise ValueError(f"Video file does not exist: {v}")
        return v
