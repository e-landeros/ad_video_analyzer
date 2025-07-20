"""
Tests for the FrameExtractor class.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import cv2

from video_analyzer.config.frame_extractor import (
    FrameExtractorConfig,
    ExtractionStrategy,
)
from video_analyzer.services.frame_extractor import FrameExtractor
from video_analyzer.models.video import VideoData
from video_analyzer.utils.errors import VideoProcessingError


class TestFrameExtractor:
    """
    Tests for the FrameExtractor class.
    """

    @pytest.fixture
    def sample_video_path(self, tmp_path):
        """Create a sample video file for testing."""
        video_path = tmp_path / "test_video.mp4"

        # Create a simple test video with 30 frames
        width, height = 320, 240
        fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # Create frames with different colors to simulate scene changes
        for i in range(30):
            # Create a colored frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Change color every 10 frames to simulate scene changes
            if i < 10:
                frame[:, :, 0] = 255  # Blue
            elif i < 20:
                frame[:, :, 1] = 255  # Green
            else:
                frame[:, :, 2] = 255  # Red

            out.write(frame)

        out.release()

        return video_path

    def test_extract_frames_uniform(self, sample_video_path):
        """Test extracting frames using uniform strategy."""
        config = FrameExtractorConfig(
            strategy=ExtractionStrategy.UNIFORM,
            uniform_interval_seconds=0.1,  # Extract every 3 frames at 30fps
        )
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames(sample_video_path)

        # We should have about 10 frames (30 frames / 3)
        # Allow for some rounding errors
        assert 9 <= len(frames) <= 11

        # Check that frames have the expected properties
        for frame in frames:
            assert isinstance(frame.image, np.ndarray)
            assert frame.timestamp >= 0
            assert frame.index >= 0

    def test_extract_frames_scene_change(self, sample_video_path):
        """Test extracting frames at scene changes."""
        config = FrameExtractorConfig(
            strategy=ExtractionStrategy.SCENE_CHANGE,
            scene_change_threshold=10,  # Lower threshold to detect more scene changes
        )
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames(sample_video_path)

        # We should have at least 3 frames (one for each color section)
        assert len(frames) >= 3

        # Check that frames have the expected properties
        for frame in frames:
            assert isinstance(frame.image, np.ndarray)
            assert frame.timestamp >= 0
            assert frame.index >= 0

    def test_extract_frames_keyframe(self, sample_video_path):
        """Test extracting keyframes."""
        config = FrameExtractorConfig(strategy=ExtractionStrategy.KEYFRAME)
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames(sample_video_path)

        # We should have at least 2 frames (first frame and at least one keyframe)
        assert len(frames) >= 2

        # Check that frames have the expected properties
        for frame in frames:
            assert isinstance(frame.image, np.ndarray)
            assert frame.timestamp >= 0
            assert frame.index >= 0

    def test_extract_frames_with_max_limit(self, sample_video_path):
        """Test extracting frames with a maximum limit."""
        config = FrameExtractorConfig(
            strategy=ExtractionStrategy.UNIFORM,
            uniform_interval_seconds=0.01,  # This would give many frames
            max_frames=5,  # But we limit to 5
        )
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames(sample_video_path)

        # We should have exactly 5 frames
        assert len(frames) == 5

    def test_extract_frames_to_video_data(self, sample_video_path):
        """Test extracting frames and adding them to VideoData."""
        config = FrameExtractorConfig(
            strategy=ExtractionStrategy.UNIFORM, uniform_interval_seconds=0.1
        )
        extractor = FrameExtractor(config)

        # Create a VideoData object
        video_data = VideoData(
            path=sample_video_path,
            frames=[],
            duration=1.0,
            fps=30.0,
            resolution=(320, 240),
            metadata={},
        )

        # Extract frames and add to VideoData
        updated_video_data = extractor.extract_frames_to_video_data(video_data)

        # Check that frames were added to the VideoData object
        assert len(updated_video_data.frames) > 0
        assert updated_video_data.frames[0].image is not None

    def test_invalid_strategy(self, sample_video_path):
        """Test extracting frames with an invalid strategy."""
        extractor = FrameExtractor()

        with pytest.raises(ValueError) as excinfo:
            extractor.extract_frames(sample_video_path, strategy="invalid_strategy")

        assert "Unsupported extraction strategy" in str(excinfo.value)

    def test_batch_processing(self, sample_video_path):
        """Test batch processing of frames."""
        config = FrameExtractorConfig(
            strategy=ExtractionStrategy.UNIFORM,
            uniform_interval_seconds=0.03,  # Extract many frames
            batch_size=2,  # Process in small batches
        )
        extractor = FrameExtractor(config)

        frames = extractor.extract_frames(sample_video_path)

        # Check that we got frames
        assert len(frames) > 0
