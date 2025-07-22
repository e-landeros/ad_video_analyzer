"""
Tests for the progression analyzer.
"""

import pytest
import numpy as np
from pathlib import Path
import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)


# Create mock classes for testing
class Frame:
    def __init__(self, image, timestamp, index):
        self.image = image
        self.timestamp = timestamp
        self.index = index


class VideoData:
    def __init__(self, path, frames, duration, fps, resolution, metadata=None):
        self.path = path
        self.frames = frames
        self.duration = duration
        self.fps = fps
        self.resolution = resolution
        self.metadata = metadata or {}

    def get_frames_in_range(self, start_time, end_time):
        return [f for f in self.frames if start_time <= f.timestamp <= end_time]


class AnalysisResult:
    def __init__(
        self, analyzer_id, confidence=1.0, data=None, video_id=None, timestamps=None
    ):
        self.analyzer_id = analyzer_id
        self.confidence = confidence
        self.data = data or {}
        self.video_id = video_id or "test_video"
        self.timestamps = timestamps or []


class ProgressionAnalysisResult(AnalysisResult):
    def __init__(
        self,
        analyzer_id,
        sections=None,
        pacing_changes=None,
        transitions=None,
        narrative_flow_score=0.7,
        retention_strategies=None,
        **kwargs,
    ):
        super().__init__(analyzer_id, **kwargs)
        self.sections = sections or []
        self.pacing_changes = pacing_changes or []
        self.transitions = transitions or []
        self.narrative_flow_score = narrative_flow_score
        self.retention_strategies = retention_strategies or []


class BaseAnalyzer:
    def __init__(self, config=None):
        self._config = config or {}
        self._start_time = None
        self._end_time = None
        self._dependencies = []
        self._result_cache = {}


class AnalyzerRegistry:
    @staticmethod
    def register(analyzer_type):
        def decorator(cls):
            return cls

        return decorator


# Mock the imports
sys.modules["video_analyzer.models.video"] = MagicMock()
sys.modules["video_analyzer.models.video"].VideoData = VideoData
sys.modules["video_analyzer.models.video"].Frame = Frame

sys.modules["video_analyzer.models.analysis"] = MagicMock()
sys.modules["video_analyzer.models.analysis"].AnalysisResult = AnalysisResult
sys.modules[
    "video_analyzer.models.analysis"
].ProgressionAnalysisResult = ProgressionAnalysisResult

sys.modules["video_analyzer.analyzers.base"] = MagicMock()
sys.modules["video_analyzer.analyzers.base"].BaseAnalyzer = BaseAnalyzer
sys.modules["video_analyzer.analyzers.base"].AnalyzerRegistry = AnalyzerRegistry
sys.modules["video_analyzer.analyzers.base"].ANALYZER_TYPES = {
    "progression": "Video structure and pacing analysis"
}

# Import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from video_analyzer.analyzers.progression_analyzer import ProgressionAnalyzer


@pytest.fixture
def mock_video_data():
    """Create mock video data for testing."""
    # Create a series of frames with timestamps
    frames = []
    for i in range(30):
        # Create a simple gradient image that changes over time
        img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 5)

        # Add some variation to simulate scene changes
        if i in [5, 15, 25]:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        frames.append(
            Frame(
                image=img,
                timestamp=i * 2.0,  # 2 seconds per frame
                index=i,
            )
        )

    # Create the video data
    return VideoData(
        path=Path("test_video.mp4"),
        frames=frames,
        duration=60.0,  # 60 seconds
        fps=0.5,  # 1 frame every 2 seconds
        resolution=(100, 100),
    )


@pytest.mark.asyncio
async def test_progression_analyzer_initialization():
    """Test that the progression analyzer initializes correctly."""
    analyzer = ProgressionAnalyzer()
    assert analyzer.analyzer_id == "progression_analyzer"
    assert analyzer.supports_progress is True


@pytest.mark.asyncio
async def test_progression_analyzer_analyze(mock_video_data):
    """Test that the progression analyzer produces valid results."""
    analyzer = ProgressionAnalyzer()
    result = await analyzer.analyze(mock_video_data)

    # Check that the result has the expected fields
    assert result.analyzer_id == "progression_analyzer"
    assert result.video_id == str(mock_video_data.path)
    assert 0 <= result.narrative_flow_score <= 1.0

    # Check that sections were identified
    assert len(result.sections) > 0
    for section in result.sections:
        assert "start_time" in section
        assert "end_time" in section
        assert "title" in section
        assert "description" in section
        assert "type" in section
        assert section["start_time"] < section["end_time"]

    # Check that transitions were identified
    assert len(result.transitions) > 0
    for transition in result.transitions:
        assert "timestamp" in transition
        assert "type" in transition
        assert 0 <= transition["confidence"] <= 1.0

    # Check that retention strategies were identified
    assert len(result.retention_strategies) > 0


@pytest.mark.asyncio
async def test_progression_analyzer_with_short_video():
    """Test the progression analyzer with a very short video."""
    # Create a short video with just 2 frames
    frames = []
    for i in range(2):
        img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
        frames.append(Frame(image=img, timestamp=i * 1.0, index=i))

    short_video = VideoData(
        path=Path("short_video.mp4"),
        frames=frames,
        duration=2.0,
        fps=1.0,
        resolution=(100, 100),
    )

    analyzer = ProgressionAnalyzer()
    result = await analyzer.analyze(short_video)

    # Check that the result is still valid
    assert result.analyzer_id == "progression_analyzer"

    # Short videos should have lower confidence
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_identify_sections(mock_video_data):
    """Test the section identification functionality."""
    analyzer = ProgressionAnalyzer()
    sections = await analyzer._identify_sections(mock_video_data)

    # Check that sections were identified
    assert len(sections) > 0

    # Check that sections cover the entire video
    assert sections[0]["start_time"] == 0.0
    assert sections[-1]["end_time"] == mock_video_data.duration

    # Check that sections don't overlap
    for i in range(len(sections) - 1):
        assert sections[i]["end_time"] <= sections[i + 1]["start_time"]


@pytest.mark.asyncio
async def test_detect_pacing_changes(mock_video_data):
    """Test the pacing change detection functionality."""
    analyzer = ProgressionAnalyzer()
    pacing_changes = await analyzer._detect_pacing_changes(mock_video_data)

    # Check that pacing changes were detected
    assert isinstance(pacing_changes, list)

    # If pacing changes were detected, check their structure
    for change in pacing_changes:
        assert "timestamp" in change
        assert "type" in change
        assert change["type"] in ["increase", "decrease"]
        assert "magnitude" in change
        assert 0 <= change["magnitude"]


@pytest.mark.asyncio
async def test_identify_transitions(mock_video_data):
    """Test the transition identification functionality."""
    analyzer = ProgressionAnalyzer()
    transitions = await analyzer._identify_transitions(mock_video_data)

    # Check that transitions were identified
    assert len(transitions) > 0

    # Check transition structure
    for transition in transitions:
        assert "timestamp" in transition
        assert "type" in transition
        assert "confidence" in transition
        assert 0 <= transition["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_evaluate_narrative_flow(mock_video_data):
    """Test the narrative flow evaluation functionality."""
    analyzer = ProgressionAnalyzer()
    sections = await analyzer._identify_sections(mock_video_data)
    transitions = await analyzer._identify_transitions(mock_video_data)

    flow_score = await analyzer._evaluate_narrative_flow(
        mock_video_data, sections, transitions
    )

    # Check that the flow score is valid
    assert 0.0 <= flow_score <= 1.0


@pytest.mark.asyncio
async def test_identify_retention_strategies(mock_video_data):
    """Test the retention strategy identification functionality."""
    analyzer = ProgressionAnalyzer()
    sections = await analyzer._identify_sections(mock_video_data)
    pacing_changes = await analyzer._detect_pacing_changes(mock_video_data)

    strategies = await analyzer._identify_retention_strategies(
        mock_video_data, sections, pacing_changes
    )

    # Check that strategies were identified
    assert isinstance(strategies, list)

    # If strategies were identified, check they are valid
    for strategy in strategies:
        assert strategy in ProgressionAnalyzer.RETENTION_STRATEGIES
