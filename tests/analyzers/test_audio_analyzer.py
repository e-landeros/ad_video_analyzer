"""
Tests for the audio analyzer.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from video_analyzer.analyzers.audio_analyzer import (
    AudioAnalyzer,
    SOUND_EFFECT_TYPES,
    MUSIC_MOODS,
)
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import AudioAnalysisResult


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
        Frame(image=blue_frame, timestamp=10.0, index=1),
        Frame(image=red_frame, timestamp=20.0, index=2),
        Frame(image=blue_frame, timestamp=30.0, index=3),
        Frame(image=red_frame, timestamp=40.0, index=4),
        Frame(image=blue_frame, timestamp=50.0, index=5),
    ]

    # Create video data
    return VideoData(
        path=Path("test_video.mp4"),
        frames=frames,
        duration=60.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={"title": "Test Video", "audio_bitrate": 192000},
    )


def test_audio_analyzer_init():
    """Test audio analyzer initialization."""
    # Test with default config
    analyzer = AudioAnalyzer()
    assert analyzer.analyzer_id == "audio_analyzer"
    assert analyzer.supports_progress is True

    # Test with custom config
    custom_config = {
        "min_speech_segment_duration": 2.0,
        "min_music_segment_duration": 5.0,
        "sound_effect_detection_threshold": 0.7,
    }
    analyzer = AudioAnalyzer(custom_config)
    assert analyzer._config["min_speech_segment_duration"] == 2.0
    assert analyzer._config["min_music_segment_duration"] == 5.0
    assert analyzer._config["sound_effect_detection_threshold"] == 0.7


@pytest.mark.asyncio
async def test_analyze_with_no_frames():
    """Test analyzing a video with no frames."""
    # Create video data with no frames
    video_data = VideoData(
        path=Path("empty_video.mp4"),
        frames=[],
        duration=60.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={},
    )

    # Create analyzer
    analyzer = AudioAnalyzer()

    # Analyze
    result = await analyzer.analyze(video_data)

    # Check result
    assert isinstance(result, AudioAnalysisResult)
    assert result.analyzer_id == "audio_analyzer"
    assert result.sound_quality == 0.5  # Default middle value
    assert "pacing" in result.speech_analysis
    assert "tone" in result.speech_analysis
    assert "clarity" in result.speech_analysis
    assert result.speech_analysis["pacing"] == "unknown"
    assert result.speech_analysis["tone"] == "unknown"
    assert result.speech_analysis["clarity"] == "unknown"
    assert result.confidence == 0.5  # Low confidence due to lack of frames


@pytest.mark.asyncio
async def test_analyze_basic(mock_video_data):
    """Test basic audio analysis."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Analyze
    result = await analyzer.analyze(mock_video_data)

    # Check result
    assert isinstance(result, AudioAnalysisResult)
    assert result.analyzer_id == "audio_analyzer"
    assert 0 <= result.sound_quality <= 1
    assert "pacing" in result.speech_analysis
    assert "tone" in result.speech_analysis
    assert "clarity" in result.speech_analysis
    assert len(result.background_music) > 0
    assert len(result.sound_effects) > 0
    assert isinstance(result.transcription, str)
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_evaluate_sound_quality(mock_video_data):
    """Test sound quality evaluation."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Evaluate sound quality
    sound_quality = await analyzer._evaluate_sound_quality(mock_video_data)

    # Check result
    assert isinstance(sound_quality, float)
    assert 0 <= sound_quality <= 1

    # Test with different audio bitrates
    high_bitrate_video = VideoData(
        path=Path("high_quality.mp4"),
        frames=mock_video_data.frames,
        duration=60.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={"audio_bitrate": 320000},
    )
    low_bitrate_video = VideoData(
        path=Path("low_quality.mp4"),
        frames=mock_video_data.frames,
        duration=60.0,
        fps=30.0,
        resolution=(1920, 1080),
        metadata={"audio_bitrate": 96000},
    )

    high_quality = await analyzer._evaluate_sound_quality(high_bitrate_video)
    low_quality = await analyzer._evaluate_sound_quality(low_bitrate_video)

    # Higher bitrate should result in higher quality score
    assert high_quality > low_quality


@pytest.mark.asyncio
async def test_analyze_speech(mock_video_data):
    """Test speech analysis."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Analyze speech
    speech_analysis = await analyzer._analyze_speech(mock_video_data)

    # Check result
    assert isinstance(speech_analysis, dict)
    assert "pacing" in speech_analysis
    assert "tone" in speech_analysis
    assert "clarity" in speech_analysis
    assert "delivery_style" in speech_analysis
    assert "vocal_variety" in speech_analysis
    assert "filler_words_frequency" in speech_analysis
    assert 0 <= speech_analysis["vocal_variety"] <= 1
    assert 0 <= speech_analysis["filler_words_frequency"] <= 1


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_analyze_speech_with_llm(mock_openai, mock_video_data):
    """Test speech analysis with LLM."""
    # Create mock OpenAI client
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Mock the chat completions create method
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    {
        "pacing": "moderate",
        "tone": "enthusiastic",
        "clarity": "clear",
        "delivery_style": "conversational",
        "vocal_variety": 0.8,
        "filler_words_frequency": 0.2
    }
    """
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create analyzer with OpenAI API key
    analyzer = AudioAnalyzer({"openai_api_key": "test_key"})

    # Analyze speech
    speech_analysis = await analyzer._analyze_speech_with_llm(mock_video_data)

    # Check result
    assert isinstance(speech_analysis, dict)
    assert speech_analysis["pacing"] == "moderate"
    assert speech_analysis["tone"] == "enthusiastic"
    assert speech_analysis["clarity"] == "clear"
    assert speech_analysis["delivery_style"] == "conversational"
    assert speech_analysis["vocal_variety"] == 0.8
    assert speech_analysis["filler_words_frequency"] == 0.2

    # Verify OpenAI was called
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_identify_background_music(mock_video_data):
    """Test background music identification."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Identify background music
    background_music = await analyzer._identify_background_music(mock_video_data)

    # Check result
    assert isinstance(background_music, list)
    assert len(background_music) > 0

    # Check that each music segment has required fields
    for music in background_music:
        assert "start_time" in music
        assert "end_time" in music
        assert "mood" in music
        assert "volume" in music
        assert "description" in music
        assert music["start_time"] < music["end_time"]
        assert 0 <= music["volume"] <= 1
        assert music["mood"] in MUSIC_MOODS


@pytest.mark.asyncio
async def test_detect_sound_effects(mock_video_data):
    """Test sound effect detection."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Detect sound effects
    sound_effects = await analyzer._detect_sound_effects(mock_video_data)

    # Check result
    assert isinstance(sound_effects, list)
    assert len(sound_effects) > 0

    # Check that each sound effect has required fields
    for effect in sound_effects:
        assert "timestamp" in effect
        assert "type" in effect
        assert "purpose" in effect
        assert "intensity" in effect
        assert "duration" in effect
        assert 0 <= effect["timestamp"] <= mock_video_data.duration
        assert 0 <= effect["intensity"] <= 1
        assert effect["type"] in SOUND_EFFECT_TYPES


@pytest.mark.asyncio
async def test_transcribe_speech(mock_video_data):
    """Test speech transcription."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Transcribe speech
    transcription = await analyzer._transcribe_speech(mock_video_data)

    # Check result
    assert isinstance(transcription, str)
    assert len(transcription) > 0


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_transcribe_with_llm(mock_openai, mock_video_data):
    """Test speech transcription with LLM."""
    # Create mock OpenAI client
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Mock the chat completions create method
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = "This is a test transcription for the video."
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Create analyzer with OpenAI API key
    analyzer = AudioAnalyzer({"openai_api_key": "test_key"})

    # Transcribe speech
    transcription = await analyzer._transcribe_with_llm(mock_video_data)

    # Check result
    assert isinstance(transcription, str)
    assert transcription == "This is a test transcription for the video."

    # Verify OpenAI was called
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_audio_analyzer_integration(mock_video_data):
    """Test the complete audio analyzer workflow."""
    # Create analyzer
    analyzer = AudioAnalyzer()

    # Run analysis
    result = await analyzer.analyze(mock_video_data)

    # Verify result structure
    assert isinstance(result, AudioAnalysisResult)
    assert result.analyzer_id == "audio_analyzer"
    assert 0 <= result.sound_quality <= 1
    assert "pacing" in result.speech_analysis
    assert "tone" in result.speech_analysis
    assert "clarity" in result.speech_analysis
    assert len(result.background_music) > 0
    assert len(result.sound_effects) > 0
    assert isinstance(result.transcription, str)

    # Verify background music has correct structure
    for music in result.background_music:
        assert "start_time" in music
        assert "end_time" in music
        assert "mood" in music
        assert music["start_time"] < music["end_time"]
        assert music["start_time"] >= 0
        assert music["end_time"] <= mock_video_data.duration

    # Verify sound effects have correct structure
    for effect in result.sound_effects:
        assert "timestamp" in effect
        assert "type" in effect
        assert "purpose" in effect
        assert 0 <= effect["timestamp"] <= mock_video_data.duration
