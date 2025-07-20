"""
VideoProcessor class for handling video input, validation, and preprocessing.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.models.video import VideoData
from video_analyzer.utils.errors import (
    VideoFormatError,
    VideoSizeError,
    VideoProcessingError,
)


class ValidationResult:
    """
    Result of video validation.
    """

    def __init__(self, is_valid: bool, errors: Dict[str, str] = None):
        self.is_valid = is_valid
        self.errors = errors or {}


class ProcessedVideo:
    """
    Result of video preprocessing.
    """

    def __init__(self, path: Path, metadata: Dict[str, Any] = None):
        self.path = path
        self.metadata = metadata or {}


class VideoProcessor:
    """
    Handles video input, validation, and preprocessing.
    """

    def __init__(self, config: Optional[VideoProcessorConfig] = None):
        """
        Initialize the VideoProcessor with the given configuration.

        Args:
            config: Configuration for the VideoProcessor. If None, default configuration is used.
        """
        self.config = config or VideoProcessorConfig()

    def validate_video(self, video_path: Path) -> ValidationResult:
        """
        Validate the video file format and size.

        Args:
            video_path: Path to the video file.

        Returns:
            ValidationResult: Result of the validation.

        Raises:
            VideoFormatError: If the video format is not supported.
            VideoSizeError: If the video size exceeds the maximum allowed.
        """
        errors = {}

        # Check if file exists
        if not video_path.exists():
            errors["file"] = f"Video file does not exist: {video_path}"
            return ValidationResult(is_valid=False, errors=errors)

        # Check file format
        file_extension = video_path.suffix.lower().lstrip(".")
        if file_extension not in self.config.supported_formats:
            error_msg = f"Unsupported video format: {file_extension}. Supported formats: {', '.join(self.config.supported_formats)}"
            errors["format"] = error_msg
            raise VideoFormatError(error_msg, format=file_extension)

        # Check file size
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_size_kb = file_size_bytes / 1024

        if file_size_mb > self.config.max_file_size_mb:
            error_msg = f"Video file size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.config.max_file_size_mb} MB)"
            errors["size"] = error_msg
            raise VideoSizeError(
                error_msg,
                size=file_size_bytes,
                max_size=self.config.max_file_size_mb * 1024 * 1024,
            )

        if file_size_kb < self.config.min_file_size_kb:
            error_msg = f"Video file size ({file_size_kb:.2f} KB) is below minimum required size ({self.config.min_file_size_kb} KB)"
            errors["size"] = error_msg
            raise VideoSizeError(
                error_msg,
                size=file_size_bytes,
                max_size=self.config.min_file_size_kb * 1024,
            )

        # Try to open the video to ensure it's a valid video file
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                errors["format"] = f"Could not open video file: {video_path}"
                cap.release()
                return ValidationResult(is_valid=False, errors=errors)
            cap.release()
        except Exception as e:
            errors["processing"] = f"Error processing video file: {str(e)}"
            raise VideoProcessingError(
                f"Error processing video file: {str(e)}", details={"exception": str(e)}
            )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def preprocess_video(self, video_path: Path) -> ProcessedVideo:
        """
        Preprocess the video for analysis.

        Args:
            video_path: Path to the video file.

        Returns:
            ProcessedVideo: Processed video information.

        Raises:
            VideoProcessingError: If there's an error processing the video.
        """
        # First validate the video
        validation_result = self.validate_video(video_path)
        if not validation_result.is_valid:
            raise VideoProcessingError(
                "Invalid video file", details=validation_result.errors
            )

        # Extract metadata
        try:
            cap = cv2.VideoCapture(str(video_path))

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            metadata = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "resolution": (width, height),
            }

            return ProcessedVideo(path=video_path, metadata=metadata)

        except Exception as e:
            raise VideoProcessingError(
                f"Error preprocessing video: {str(e)}", details={"exception": str(e)}
            )

    def create_video_data(self, video_path: Path) -> VideoData:
        """
        Create a VideoData object from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoData: Video data object.

        Raises:
            VideoProcessingError: If there's an error processing the video.
        """
        processed_video = self.preprocess_video(video_path)

        return VideoData(
            path=video_path,
            frames=[],  # Frames will be populated by the FrameExtractor
            duration=processed_video.metadata["duration"],
            fps=processed_video.metadata["fps"],
            resolution=processed_video.metadata["resolution"],
            metadata=processed_video.metadata,
        )
