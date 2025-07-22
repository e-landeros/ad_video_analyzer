"""
Tests for the AnalysisManager class.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from video_analyzer.services.analysis_manager import AnalysisManager
from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import AnalysisResult


# Mock analyzer for testing
class MockAnalyzer(BaseAnalyzer):
    """Mock analyzer for testing."""

    def __init__(self, analyzer_id="mock_analyzer", config=None):
        super().__init__(config)
        self._analyzer_id = analyzer_id

    @property
    def analyzer_id(self):
        return self._analyzer_id

    async def analyze(self, video_data):
        # Simple mock implementation
        return AnalysisResult(
            analyzer_id=self.analyzer_id,
            confidence=0.9,
            video_id=str(video_data.path),
        )


# Register the mock analyzer
AnalyzerRegistry._registry["mock"] = MockAnalyzer


@pytest.fixture
def mock_video_data():
    """Create mock video data for testing."""
    # Create a simple frame with a small numpy array
    frame = Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.0,
        index=0,
    )

    # Create video data with the frame
    return VideoData(
        path=Path("test_video.mp4"),
        frames=[frame],
        duration=10.0,
        fps=30.0,
        resolution=(1920, 1080),
    )


@pytest.fixture
def analysis_manager():
    """Create an AnalysisManager instance for testing."""
    return AnalysisManager()


@patch("video_analyzer.services.video_processor.VideoProcessor.process_video")
@patch("video_analyzer.services.frame_extractor.FrameExtractor.extract_frames_async")
@patch("video_analyzer.services.frame_extractor.FrameExtractor.get_video_info")
async def test_analyze_video(
    mock_get_video_info,
    mock_extract_frames,
    mock_process_video,
    analysis_manager,
    mock_video_data,
):
    """Test the analyze_video method."""
    # Set up mocks
    mock_process_video.return_value = Path("processed_video.mp4")

    mock_frame = Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=0.0,
        index=0,
    )
    mock_extract_frames.return_value = [mock_frame]

    mock_get_video_info.return_value = {
        "duration": 10.0,
        "fps": 30.0,
        "resolution": (1920, 1080),
    }

    # Register a mock analyzer
    analysis_manager.register_analyzer("mock")

    # Run the analysis
    results = await analysis_manager.analyze_video("test_video.mp4")

    # Verify the results
    assert len(results) == 1
    assert "mock_analyzer" in results
    assert results["mock_analyzer"].analyzer_id == "mock_analyzer"
    assert results["mock_analyzer"].confidence == 0.9


def test_register_analyzer(analysis_manager):
    """Test registering analyzers."""
    # Register by instance
    analyzer = MockAnalyzer(analyzer_id="custom_analyzer")
    analysis_manager.register_analyzer(analyzer)

    # Register by type
    analysis_manager.register_analyzer("mock")

    # Verify registration
    assert "custom_analyzer" in analysis_manager._registered_analyzers
    assert "mock_analyzer" in analysis_manager._registered_analyzers


def test_register_analyzers(analysis_manager):
    """Test registering multiple analyzers."""
    # Register multiple analyzers
    analysis_manager.register_analyzers(["mock", "mock"])

    # Verify registration
    assert len(analysis_manager._registered_analyzers) == 2


def test_get_combined_frame_requirements(analysis_manager):
    """Test combining frame requirements from analyzers."""
    # Create analyzers with different requirements
    analyzer1 = MockAnalyzer(analyzer_id="analyzer1")
    analyzer1.required_frames = {
        "min_frames": 5,
        "frame_interval": 1.0,
        "specific_timestamps": [1.0, 2.0],
    }

    analyzer2 = MockAnalyzer(analyzer_id="analyzer2")
    analyzer2.required_frames = {
        "min_frames": 10,
        "frame_interval": 0.5,
        "specific_timestamps": [2.0, 3.0],
    }

    # Register the analyzers
    analysis_manager.register_analyzer(analyzer1)
    analysis_manager.register_analyzer(analyzer2)

    # Get combined requirements
    requirements = analysis_manager._get_combined_frame_requirements()

    # Verify requirements
    assert requirements["min_frames"] == 10  # Max of 5 and 10
    assert requirements["frame_interval"] == 0.5  # Min of 1.0 and 0.5
    assert set(requirements["specific_timestamps"]) == {
        1.0,
        2.0,
        3.0,
    }  # Combined timestamps


def test_generate_report(analysis_manager):
    """Test generating a report from analysis results."""
    # Create mock results
    results = {
        "hook_analyzer": AnalysisResult(
            analyzer_id="hook_analyzer",
            confidence=0.9,
            video_id="test_video",
        ),
        "progression_analyzer": AnalysisResult(
            analyzer_id="progression_analyzer",
            confidence=0.8,
            video_id="test_video",
        ),
    }

    # Generate report
    report = analysis_manager.generate_report(results, "test_video")

    # Verify report
    assert report.video_id == "test_video"
    assert "hook" in report.sections
    assert "progression" in report.sections
    assert len(report.sections) == 2
