"""
Tests for the VideoProcessor class.
"""

import os
import tempfile
from pathlib import Path
import pytest

from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.services.video_processor import VideoProcessor
from video_analyzer.utils.errors import (
    VideoFormatError,
    VideoSizeError,
    VideoProcessingError,
)


class TestVideoProcessor:
    """
    Tests for the VideoProcessor class.
    """

    def test_validate_video_valid_format(self, tmp_path):
        """Test validation with a valid video format."""
        # Create a dummy MP4 file
        video_path = tmp_path / "test_video.mp4"
        with open(video_path, "wb") as f:
            # Write some dummy content to make the file non-empty
            f.write(b"x" * 20 * 1024)  # 20KB

        processor = VideoProcessor()

        # This will fail because it's not a real video file, but we can check the format validation
        with pytest.raises(VideoProcessingError):
            processor.validate_video(video_path)

    def test_validate_video_invalid_format(self, tmp_path):
        """Test validation with an invalid video format."""
        # Create a dummy file with unsupported extension
        video_path = tmp_path / "test_video.xyz"
        with open(video_path, "wb") as f:
            f.write(b"x" * 20 * 1024)  # 20KB

        processor = VideoProcessor()

        with pytest.raises(VideoFormatError) as excinfo:
            processor.validate_video(video_path)

        assert "Unsupported video format" in str(excinfo.value)
        assert excinfo.value.format == "xyz"

    def test_validate_video_file_too_large(self, tmp_path):
        """Test validation with a file that's too large."""
        # Create a dummy MP4 file
        video_path = tmp_path / "test_video.mp4"

        # Configure a very small max file size for testing
        config = VideoProcessorConfig(max_file_size_mb=0.01)  # 10KB
        processor = VideoProcessor(config)

        with open(video_path, "wb") as f:
            f.write(b"x" * 20 * 1024)  # 20KB

        with pytest.raises(VideoSizeError) as excinfo:
            processor.validate_video(video_path)

        assert "exceeds maximum allowed size" in str(excinfo.value)

    def test_validate_video_file_too_small(self, tmp_path):
        """Test validation with a file that's too small."""
        # Create a dummy MP4 file
        video_path = tmp_path / "test_video.mp4"

        # Configure a large min file size for testing
        config = VideoProcessorConfig(min_file_size_kb=50)  # 50KB
        processor = VideoProcessor(config)

        with open(video_path, "wb") as f:
            f.write(b"x" * 20 * 1024)  # 20KB

        with pytest.raises(VideoSizeError) as excinfo:
            processor.validate_video(video_path)

        assert "below minimum required size" in str(excinfo.value)

    def test_validate_video_nonexistent_file(self):
        """Test validation with a nonexistent file."""
        processor = VideoProcessor()

        result = processor.validate_video(Path("/nonexistent/path/to/video.mp4"))

        assert not result.is_valid
        assert "does not exist" in result.errors["file"]
