"""
Data models for video information.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator


class Frame(BaseModel):
    """
    Data model for a video frame.

    Attributes:
        image: The actual frame image data as numpy array
        timestamp: Timestamp of the frame in seconds
        index: Frame index in the video
    """

    image: Any  # np.ndarray, but Pydantic doesn't directly support numpy types
    timestamp: float = Field(..., description="Timestamp of the frame in seconds")
    index: int = Field(..., description="Frame index in the video")

    class Config:
        arbitrary_types_allowed = True

    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate that timestamp is positive."""
        if v < 0:
            raise ValueError(f"Frame timestamp must be positive, got {v}")
        return v

    @validator("index")
    def validate_index(cls, v):
        """Validate that index is non-negative."""
        if v < 0:
            raise ValueError(f"Frame index must be non-negative, got {v}")
        return v

    @validator("image")
    def validate_image(cls, v):
        """Validate that image is a numpy array."""
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Frame image must be a numpy array, got {type(v)}")
        if v.ndim < 2:
            raise ValueError(
                f"Frame image must be at least 2-dimensional, got {v.ndim}"
            )
        return v


class VideoData(BaseModel):
    """
    Data model for video information.

    Attributes:
        path: Path to the video file
        frames: List of extracted frames
        duration: Duration of the video in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        metadata: Additional video metadata
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
        """Validate that the video file exists."""
        if not v.exists():
            raise ValueError(f"Video file does not exist: {v}")
        return v

    @validator("duration")
    def validate_duration(cls, v):
        """Validate that duration is positive."""
        if v <= 0:
            raise ValueError(f"Video duration must be positive, got {v}")
        return v

    @validator("fps")
    def validate_fps(cls, v):
        """Validate that fps is positive."""
        if v <= 0:
            raise ValueError(f"Video fps must be positive, got {v}")
        return v

    @validator("resolution")
    def validate_resolution(cls, v):
        """Validate that resolution dimensions are positive."""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"Video resolution dimensions must be positive, got {v}")
        return v

    @root_validator(skip_on_failure=True)
    def validate_frames_timestamps(cls, values):
        """Validate that frame timestamps are within video duration."""
        frames = values.get("frames", [])
        duration = values.get("duration")

        if frames and duration:
            for frame in frames:
                if frame.timestamp > duration:
                    raise ValueError(
                        f"Frame timestamp {frame.timestamp} exceeds video duration {duration}"
                    )

        return values

    def get_frame_at_time(self, timestamp: float) -> Optional[Frame]:
        """
        Get the frame closest to the specified timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            The closest frame or None if no frames are available
        """
        if not self.frames:
            return None

        # Find the frame with the closest timestamp
        return min(self.frames, key=lambda f: abs(f.timestamp - timestamp))

    def get_frames_in_range(self, start_time: float, end_time: float) -> List[Frame]:
        """
        Get all frames within the specified time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of frames within the range
        """
        return [
            frame for frame in self.frames if start_time <= frame.timestamp <= end_time
        ]
