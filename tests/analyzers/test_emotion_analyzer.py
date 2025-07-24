"""
Tests for the emotion analyzer.
"""

import pytest
import numpy as np
from pathlib import Path
import os
from unittest.mock import AsyncMock, MagicMock, patch

from video_analyzer.analyzers.emotion_analyzer import EmotionAnalyzer
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import EmotionAnalysisResult


@pytest.fixture
def mock_video_data():
    """Create mock video data for testing."""
    # Create a few test frames with different properties
    frames = []

    # Bright, high saturation frame (joyful)
    bright_frame = MagicMock()
    bright_frame.timestamp = 0.0
    bright_frame.image = np.ones((100, 100, 3), dtype=np.uint8) * 220  # Bright
    # Add some color variation for saturation
    bright_frame.image[:50, :50, 0] = 180
    bright_frame.image[:50, :50, 1] = 240
    bright_frame.image[:50, :50, 2] = 200
    frames.append(bright_frame)

    # Dark, high saturation frame (tense)
    dark_frame = MagicMock()
    dark_frame.timestamp = 10.0
    dark_frame.image = np.ones((100, 100, 3), dtype=np.uint8) * 80  # Dark
    # Add some color variation for saturation
    dark_frame.image[:50, :50, 0] = 40
    dark_frame.image[:50, :50, 1] = 100
    dark_frame.image[:50, :50, 2] = 60
    frames.append(dark_frame)

    # Medium brightness, low saturation frame (calm)
    neutral_frame = MagicMock()
    neutral_frame.timestamp = 20.0
    neutral_frame.image = np.ones((100, 100, 3), dtype=np.uint8) * 150  # Medium
    frames.append(neutral_frame)

    # Create mock video data
    video_data = MagicMock()
    video_data.path = Path("test_video.mp4")
    video_data.frames = frames
    video_data.duration = 30.0
    video_data.fps = 30.0
    video_data.resolution = (1920, 1080)

    # Mock the get_frames_at_intervals method
    video_data.get_frames_at_intervals = MagicMock(return_value=frames)

    return video_data


@pytest.mark.asyncio
async def test_emotion_analyzer_initialization():
    """Test that the emotion analyzer initializes correctly."""
    analyzer = EmotionAnalyzer()
    assert analyzer.analyzer_id == "emotion_analyzer"
    assert analyzer.supports_progress is True
    assert analyzer.required_frames["min_frames"] == 10


@pytest.mark.asyncio
async def test_emotion_analyzer_basic_analysis(mock_video_data):
    """Test the basic emotion analysis functionality."""
    analyzer = EmotionAnalyzer()

    # Mock the OpenAI client to ensure we use basic analysis
    analyzer.openai_client = None

    result = await analyzer.analyze(mock_video_data)

    # Verify the result is of the correct type
    assert isinstance(result, EmotionAnalysisResult)

    # Check that required fields are populated
    assert result.analyzer_id == "emotion_analyzer"
    assert result.overall_mood is not None
    assert isinstance(result.emotional_shifts, list)
    assert isinstance(result.emotional_elements, dict)
    assert "visual" in result.emotional_elements
    assert "audio" in result.emotional_elements
    assert isinstance(result.emotion_techniques, list)
    assert isinstance(result.emotional_journey, list)

    # Check that the emotional journey has entries for each frame
    assert len(result.emotional_journey) == len(mock_video_data.frames)

    # Verify each emotional journey entry has the required fields
    for entry in result.emotional_journey:
        assert "timestamp" in entry
        assert "emotion" in entry
        assert "intensity" in entry
        assert 0 <= entry["intensity"] <= 1


@pytest.mark.asyncio
async def test_emotion_analyzer_empty_video():
    """Test the emotion analyzer with an empty video."""
    analyzer = EmotionAnalyzer()

    # Create an empty video data mock
    empty_video_data = MagicMock()
    empty_video_data.path = Path("empty_video.mp4")
    empty_video_data.frames = []
    empty_video_data.duration = 0.0
    empty_video_data.fps = 30.0
    empty_video_data.resolution = (1920, 1080)
    empty_video_data.get_frames_at_intervals = MagicMock(return_value=[])

    result = await analyzer.analyze(empty_video_data)

    # Verify the result has default values
    assert result.overall_mood == "neutral"
    assert result.emotional_shifts == []
    assert result.emotional_journey == []
    assert result.confidence == 0.5  # Lower confidence due to lack of frames


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_emotion_analyzer_with_openai(mock_openai, mock_video_data):
    """Test the emotion analyzer with OpenAI integration."""
    # Set up the OpenAI mock
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Mock the chat completions create method
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "joy"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create analyzer with mock OpenAI API key
    analyzer = EmotionAnalyzer({"openai_api_key": "test_key"})

    # Ensure the OpenAI client is set
    assert analyzer.openai_client is not None

    # Mock the emotional journey method to avoid additional OpenAI calls
    analyzer._create_emotional_journey = AsyncMock(
        return_value=[
            {"timestamp": 0.0, "emotion": "joy", "intensity": 0.8},
            {"timestamp": 10.0, "emotion": "tension", "intensity": 0.7},
            {"timestamp": 20.0, "emotion": "calm", "intensity": 0.5},
        ]
    )

    result = await analyzer.analyze(mock_video_data)

    # Verify OpenAI was called for mood identification
    assert mock_client.chat.completions.create.call_count > 0

    # Check that the result uses the OpenAI response
    assert result.overall_mood == "joy"


@pytest.mark.asyncio
async def test_detect_emotional_shifts():
    """Test the detection of emotional shifts."""
    analyzer = EmotionAnalyzer()

    # Create a test emotional journey with clear shifts
    emotional_journey = [
        {"timestamp": 0.0, "emotion": "joy", "intensity": 0.8},
        {"timestamp": 10.0, "emotion": "joy", "intensity": 0.7},  # No shift
        {"timestamp": 20.0, "emotion": "sadness", "intensity": 0.3},  # Shift
        {"timestamp": 30.0, "emotion": "sadness", "intensity": 0.4},  # No shift
        {"timestamp": 40.0, "emotion": "fear", "intensity": 0.9},  # Shift
    ]

    shifts = await analyzer._detect_emotional_shifts(emotional_journey)

    # Should detect two shifts
    assert len(shifts) == 2

    # Check first shift
    assert shifts[0]["timestamp"] == 20.0
    assert shifts[0]["from_emotion"] == "joy"
    assert shifts[0]["to_emotion"] == "sadness"

    # Check second shift
    assert shifts[1]["timestamp"] == 40.0
    assert shifts[1]["from_emotion"] == "sadness"
    assert shifts[1]["to_emotion"] == "fear"


@pytest.mark.asyncio
async def test_identify_emotion_techniques(mock_video_data):
    """Test the identification of emotion techniques."""
    analyzer = EmotionAnalyzer()

    # Mock the overall mood identification
    analyzer._identify_overall_mood = AsyncMock(return_value="joy")

    # Create a test emotional journey with a pattern
    emotional_journey = [
        {"timestamp": 0.0, "emotion": "calm", "intensity": 0.3},
        {"timestamp": 10.0, "emotion": "joy", "intensity": 0.5},
        {"timestamp": 20.0, "emotion": "excitement", "intensity": 0.7},
        {"timestamp": 30.0, "emotion": "excitement", "intensity": 0.9},  # Buildup
    ]

    techniques = await analyzer._identify_emotion_techniques(
        mock_video_data, mock_video_data.frames, emotional_journey
    )

    # Should identify at least one technique
    assert len(techniques) > 0

    # Should identify emotional buildup
    assert "emotional_buildup" in techniques

    # Should identify positive reinforcement based on the "joy" mood
    assert "positive_reinforcement" in techniques
