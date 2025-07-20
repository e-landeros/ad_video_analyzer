"""
Configuration for the VideoProcessor.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class VideoProcessorConfig(BaseModel):
    """
    Configuration for the VideoProcessor.
    """

    supported_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "wmv"],
        description="List of supported video formats",
    )
    max_file_size_mb: int = Field(
        default=1000,
        description="Maximum file size in MB",
    )
    min_file_size_kb: int = Field(
        default=10,
        description="Minimum file size in KB",
    )

    @validator("supported_formats")
    def validate_supported_formats(cls, v):
        """Ensure all formats are lowercase."""
        return [fmt.lower() for fmt in v]
