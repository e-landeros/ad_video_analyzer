"""
Tests for the hook analyzer.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from video_analyzer.analyzers.hook_analyzer import HookAnalyzer, HOOK_TECHNIQUES
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import HookAnalysisResult


@pytest.fixture
def mock_video_data():
    """Create mock video data for testing."""
    # Create a 3x3 red frame
    red_frame = np.zeros((3, 3, 3), dtype=np.uint8)
    red_frame[:, :, 0] = 255  # Red channel

    # Create a 3x3 blue frame
    blue_frame = np.zeros((3, 3, 3), dtype=np.uint8)
    blue_frame[:, :, 2] = 255  # Blue channel

    # Create frames at different timestamps
    frames = [
        Frame(image=red_frame, timestamp=0.0, index=0),
        Frame(image=blue_frame, timestamp=5.0, index=1),
        Frame(image=red_frame, timestamp=10.0, index=2),
        Frame(image=blue_frame, timestamp=15.0, index=3),
    ]

    # Create video data
    return VideoData(
        path=Path("test_video.mp4"),
        frames=frames,
        duration=30.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={"title": "Test Video"},
    )


def test_hook_analyzer_init():
    """Test hook analyzer initialization."""
    # Test with default config
    analyzer = HookAnalyzer()
    assert analyzer.analyzer_id == "hook_analyzer"
    assert analyzer.supports_progress is True

    # Test with custom config
    custom_config = {
        "default_hook_duration": 20,
        "min_hook_duration": 5,
        "max_hook_duration": 40,
    }
    analyzer = HookAnalyzer(custom_config)
    assert analyzer._config["default_hook_duration"] == 20
    assert analyzer._config["min_hook_duration"] == 5
    assert analyzer._config["max_hook_duration"] == 40


@pytest.mark.asyncio
async def test_analyze_with_no_frames():
    """Test analyzing a video with no frames."""
    # Create video data with no frames
    video_data = VideoData(
        path=Path("empty_video.mp4"),
        frames=[],
        duration=30.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={},
    )

    # Create analyzer
    analyzer = HookAnalyzer()

    # Analyze
    result = await analyzer.analyze(video_data)

    # Check result
    assert isinstance(result, HookAnalysisResult)
    assert result.analyzer_id == "hook_analyzer"
    assert result.hook_start_time == 0
    assert result.hook_end_time == 15  # Default hook duration
    assert result.hook_effectiveness == 0.5  # Default middle value
    assert "unknown" in result.hook_techniques
    assert result.confidence == 0.5  # Low confidence due to lack of frames


@pytest.mark.asyncio
async def test_analyze_basic(mock_video_data):
    """Test basic hook analysis."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Analyze
    result = await analyzer.analyze(mock_video_data)

    # Check result
    assert isinstance(result, HookAnalysisResult)
    assert result.analyzer_id == "hook_analyzer"
    assert result.hook_start_time == 0
    assert result.hook_end_time == 15  # Default hook duration
    assert len(result.hook_techniques) > 0
    assert 0 <= result.hook_effectiveness <= 1
    assert len(result.key_moments) > 0
    assert len(result.recommendations) > 0
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_identify_hook_techniques_basic(mock_video_data):
    """Test basic hook technique identification."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Get hook frames
    hook_frames = mock_video_data.get_frames_in_range(0, 15)

    # Identify techniques
    techniques = analyzer._identify_hook_techniques_basic(mock_video_data, hook_frames)

    # Check result
    assert isinstance(techniques, list)
    assert len(techniques) > 0
    assert all(technique in HOOK_TECHNIQUES for technique in techniques)


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_identify_hook_techniques_with_llm(mock_openai, mock_video_data):
    """Test hook technique identification with LLM."""
    # Create mock OpenAI client
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Mock the chat completions create method
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = "question, direct_address, visual_pattern_interrupt"
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create analyzer with OpenAI API key
    analyzer = HookAnalyzer({"openai_api_key": "test_key"})

    # Get hook frames
    hook_frames = mock_video_data.get_frames_in_range(0, 15)

    # Identify techniques
    techniques = await analyzer._identify_hook_techniques_with_llm(
        mock_video_data, hook_frames
    )

    # Check result
    assert isinstance(techniques, list)
    assert len(techniques) == 3
    assert "question" in techniques
    assert "direct_address" in techniques
    assert "visual_pattern_interrupt" in techniques

    # Verify OpenAI was called
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_hook_effectiveness(mock_video_data):
    """Test hook effectiveness evaluation."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Get hook frames
    hook_frames = mock_video_data.get_frames_in_range(0, 15)

    # Evaluate effectiveness
    effectiveness = await analyzer._evaluate_hook_effectiveness(
        mock_video_data, hook_frames, ["question", "direct_address"]
    )

    # Check result
    assert isinstance(effectiveness, float)
    assert 0 <= effectiveness <= 1


@pytest.mark.asyncio
async def test_identify_key_moments(mock_video_data):
    """Test key moment identification."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Get hook frames
    hook_frames = mock_video_data.get_frames_in_range(0, 15)

    # Identify key moments
    key_moments = await analyzer._identify_key_moments(mock_video_data, hook_frames)

    # Check result
    assert isinstance(key_moments, list)
    assert len(key_moments) > 0

    # Check that each key moment has required fields
    for moment in key_moments:
        assert "timestamp" in moment
        assert "description" in moment
        assert "importance" in moment
        assert 0 <= moment["importance"] <= 1


@pytest.mark.asyncio
async def test_generate_recommendations(mock_video_data):
    """Test recommendation generation."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Generate recommendations
    recommendations = await analyzer._generate_recommendations(
        mock_video_data, ["question"], 0.5
    )

    # Check result
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    # Test with high effectiveness
    recommendations_high = await analyzer._generate_recommendations(
        mock_video_data, ["question", "direct_address", "visual_pattern_interrupt"], 0.9
    )

    # Check result
    assert isinstance(recommendations_high, list)
    assert len(recommendations_high) > 0


@pytest.mark.asyncio
async def test_hook_analyzer_integration(mock_video_data):
    """Test the complete hook analyzer workflow."""
    # Create analyzer
    analyzer = HookAnalyzer()

    # Run analysis
    result = await analyzer.analyze(mock_video_data)

    # Verify result structure
    assert isinstance(result, HookAnalysisResult)
    assert result.analyzer_id == "hook_analyzer"
    assert result.hook_start_time >= 0
    assert result.hook_end_time > result.hook_start_time
    assert 0 <= result.hook_effectiveness <= 1
    assert len(result.hook_techniques) > 0
    assert len(result.key_moments) > 0
    assert len(result.recommendations) > 0

    # Verify key moments have correct structure
    for moment in result.key_moments:
        assert "timestamp" in moment
        assert "description" in moment
        assert "importance" in moment

    # Verify timestamps are within video duration
    assert all(
        moment["timestamp"] <= mock_video_data.duration for moment in result.key_moments
    )
