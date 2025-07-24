"""
Tests for the storytelling analyzer.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from pathlib import Path

from video_analyzer.analyzers.storytelling_analyzer import StorytellingAnalyzer
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import StorytellingAnalysisResult


@pytest.fixture
def mock_video_data():
    """Create a mock VideoData object for testing."""
    # Create a mock video data object
    video_data = MagicMock(spec=VideoData)
    video_data.path = Path("/path/to/test_video.mp4")
    video_data.duration = 300  # 5 minutes
    video_data.fps = 30
    video_data.resolution = (1920, 1080)

    # Create mock frames
    frames = []
    for i in range(10):
        # Create a mock frame with a random image
        frame = MagicMock(spec=Frame)
        frame.timestamp = i * 30  # Every 30 seconds
        frame.index = i
        frame.image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(frame)

    # Set up the get_frames_at_intervals method to return our mock frames
    video_data.get_frames_at_intervals.return_value = frames

    return video_data


@pytest.fixture
def storytelling_analyzer():
    """Create a StorytellingAnalyzer instance for testing."""
    return StorytellingAnalyzer()


@pytest.mark.asyncio
async def test_analyze_with_frames(storytelling_analyzer, mock_video_data):
    """Test the analyze method with frames."""
    # Run the analysis
    result = await storytelling_analyzer.analyze(mock_video_data)

    # Check that the result is of the correct type
    assert isinstance(result, StorytellingAnalysisResult)

    # Check that the analyzer ID is correct
    assert result.analyzer_id == "storytelling_analyzer"

    # Check that the video ID is correct
    assert result.video_id == str(mock_video_data.path)

    # Check that the confidence is reasonable
    assert 0 <= result.confidence <= 1

    # Check that the narrative structure is identified
    assert result.narrative_structure is not None
    assert result.narrative_structure != ""

    # Check that character development entries are present
    assert isinstance(result.character_development, list)
    if result.character_development:
        entry = result.character_development[0]
        assert "timestamp" in entry
        assert "character" in entry
        assert "development_type" in entry
        assert "description" in entry

    # Check that conflict patterns are present
    assert isinstance(result.conflict_patterns, list)
    if result.conflict_patterns:
        pattern = result.conflict_patterns[0]
        assert "conflict_timestamp" in pattern
        assert "resolution_timestamp" in pattern
        assert "conflict_type" in pattern
        assert "description" in pattern
        # Resolution should come after conflict
        assert pattern["resolution_timestamp"] > pattern["conflict_timestamp"]

    # Check that persuasion techniques are identified
    assert isinstance(result.persuasion_techniques, list)
    assert len(result.persuasion_techniques) > 0

    # Check that engagement strategies are identified
    assert isinstance(result.engagement_strategies, list)
    assert len(result.engagement_strategies) > 0


@pytest.mark.asyncio
async def test_analyze_without_frames(storytelling_analyzer, mock_video_data):
    """Test the analyze method without frames."""
    # Set up the get_frames_at_intervals method to return an empty list
    mock_video_data.get_frames_at_intervals.return_value = []

    # Run the analysis
    result = await storytelling_analyzer.analyze(mock_video_data)

    # Check that the result is of the correct type
    assert isinstance(result, StorytellingAnalysisResult)

    # Check that the analyzer ID is correct
    assert result.analyzer_id == "storytelling_analyzer"

    # Check that the video ID is correct
    assert result.video_id == str(mock_video_data.path)

    # Check that the confidence is lower due to lack of frames
    assert result.confidence == 0.5

    # Check that the narrative structure is "unknown"
    assert result.narrative_structure == "unknown"

    # Check that other fields are empty
    assert len(result.character_development) == 0
    assert len(result.conflict_patterns) == 0
    assert len(result.persuasion_techniques) == 0
    assert len(result.engagement_strategies) == 0


@pytest.mark.asyncio
async def test_identify_narrative_structure_basic(
    storytelling_analyzer, mock_video_data
):
    """Test the _identify_narrative_structure_basic method."""
    # Test with different video durations

    # Very short video (< 60s)
    mock_video_data.duration = 30
    frames = mock_video_data.get_frames_at_intervals()
    structure = storytelling_analyzer._identify_narrative_structure_basic(
        mock_video_data, frames
    )
    assert structure == "explainer"

    # Short video (60-180s)
    mock_video_data.duration = 120
    structure = storytelling_analyzer._identify_narrative_structure_basic(
        mock_video_data, frames
    )
    assert structure == "vlog_style"

    # Medium video (180-600s)
    mock_video_data.duration = 400
    structure = storytelling_analyzer._identify_narrative_structure_basic(
        mock_video_data, frames
    )
    assert structure == "problem_solution"

    # Long video (> 600s)
    mock_video_data.duration = 900
    structure = storytelling_analyzer._identify_narrative_structure_basic(
        mock_video_data, frames
    )
    assert structure == "three_act"


@pytest.mark.asyncio
async def test_identify_character_development(storytelling_analyzer, mock_video_data):
    """Test the _identify_character_development method."""
    frames = mock_video_data.get_frames_at_intervals()

    # Run the method
    character_development = await storytelling_analyzer._identify_character_development(
        mock_video_data, frames
    )

    # Check that we have character development entries
    assert isinstance(character_development, list)
    assert len(character_development) > 0

    # Check that each entry has the required fields
    for entry in character_development:
        assert "timestamp" in entry
        assert "character" in entry
        assert "development_type" in entry
        assert "description" in entry

        # Check that the timestamp is valid
        assert entry["timestamp"] >= 0

        # Check that the character is a string
        assert isinstance(entry["character"], str)

        # Check that the development type is a string
        assert isinstance(entry["development_type"], str)

        # Check that the description is a string
        assert isinstance(entry["description"], str)


@pytest.mark.asyncio
async def test_identify_conflict_patterns(storytelling_analyzer, mock_video_data):
    """Test the _identify_conflict_patterns method."""
    frames = mock_video_data.get_frames_at_intervals()

    # Test with a short video
    mock_video_data.duration = 200  # < 300s
    conflict_patterns = await storytelling_analyzer._identify_conflict_patterns(
        mock_video_data, frames
    )

    # Check that we have conflict patterns
    assert isinstance(conflict_patterns, list)
    if conflict_patterns:
        # Check that each pattern has the required fields
        for pattern in conflict_patterns:
            assert "conflict_timestamp" in pattern
            assert "resolution_timestamp" in pattern
            assert "conflict_type" in pattern
            assert "description" in pattern

            # Check that the timestamps are valid
            assert pattern["conflict_timestamp"] >= 0
            assert pattern["resolution_timestamp"] > pattern["conflict_timestamp"]

    # Test with a longer video
    mock_video_data.duration = 600  # > 300s
    conflict_patterns = await storytelling_analyzer._identify_conflict_patterns(
        mock_video_data, frames
    )

    # Check that we have conflict patterns
    assert isinstance(conflict_patterns, list)
    if conflict_patterns:
        # Check that each pattern has the required fields
        for pattern in conflict_patterns:
            assert "conflict_timestamp" in pattern
            assert "resolution_timestamp" in pattern
            assert "conflict_type" in pattern
            assert "description" in pattern

            # Check that the timestamps are valid
            assert pattern["conflict_timestamp"] >= 0
            assert pattern["resolution_timestamp"] > pattern["conflict_timestamp"]


@pytest.mark.asyncio
async def test_identify_persuasion_techniques(storytelling_analyzer, mock_video_data):
    """Test the _identify_persuasion_techniques method."""
    frames = mock_video_data.get_frames_at_intervals()

    # Test with different video durations

    # Short video (< 60s)
    mock_video_data.duration = 30
    techniques = await storytelling_analyzer._identify_persuasion_techniques(
        mock_video_data, frames
    )

    # Check that we have techniques
    assert isinstance(techniques, list)
    assert len(techniques) > 0

    # Check that "storytelling" and "urgency" are included for short videos
    assert "storytelling" in techniques
    assert "urgency" in techniques

    # Medium video (60-300s)
    mock_video_data.duration = 120
    techniques = await storytelling_analyzer._identify_persuasion_techniques(
        mock_video_data, frames
    )

    # Check that we have techniques
    assert isinstance(techniques, list)
    assert len(techniques) > 0

    # Check that "storytelling" and "logos_appeal" are included for medium videos
    assert "storytelling" in techniques
    assert "logos_appeal" in techniques

    # Long video (> 300s)
    mock_video_data.duration = 400
    techniques = await storytelling_analyzer._identify_persuasion_techniques(
        mock_video_data, frames
    )

    # Check that we have techniques
    assert isinstance(techniques, list)
    assert len(techniques) > 0

    # Check that "storytelling" and "ethos_appeal" are included for long videos
    assert "storytelling" in techniques
    assert "ethos_appeal" in techniques


@pytest.mark.asyncio
async def test_identify_engagement_strategies(storytelling_analyzer, mock_video_data):
    """Test the _identify_engagement_strategies method."""
    frames = mock_video_data.get_frames_at_intervals()

    # Test with different video durations

    # Short video (< 60s)
    mock_video_data.duration = 30
    strategies = await storytelling_analyzer._identify_engagement_strategies(
        mock_video_data, frames
    )

    # Check that we have strategies
    assert isinstance(strategies, list)
    assert len(strategies) > 0

    # Check that "direct_address" and "surprise_reveal" are included for short videos
    assert "direct_address" in strategies
    assert "surprise_reveal" in strategies

    # Medium video (60-300s)
    mock_video_data.duration = 120
    strategies = await storytelling_analyzer._identify_engagement_strategies(
        mock_video_data, frames
    )

    # Check that we have strategies
    assert isinstance(strategies, list)
    assert len(strategies) > 0

    # Check that "direct_address" and "question_hooks" are included for medium videos
    assert "direct_address" in strategies
    assert "question_hooks" in strategies

    # Long video (> 300s)
    mock_video_data.duration = 400
    strategies = await storytelling_analyzer._identify_engagement_strategies(
        mock_video_data, frames
    )

    # Check that we have strategies
    assert isinstance(strategies, list)
    assert len(strategies) > 0

    # Check that "direct_address" and "pacing_variation" are included for long videos
    assert "direct_address" in strategies
    assert "pacing_variation" in strategies


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_openai_integration(mock_openai, storytelling_analyzer, mock_video_data):
    """Test the OpenAI integration."""
    # Set up the OpenAI client mock
    mock_client = AsyncMock()
    mock_openai.return_value = mock_client

    # Set up the chat completions mock
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "three_act"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Set the OpenAI API key
    storytelling_analyzer.openai_api_key = "test_key"
    storytelling_analyzer.openai_client = mock_client

    # Test the narrative structure identification with LLM
    frames = mock_video_data.get_frames_at_intervals()
    structure = await storytelling_analyzer._identify_narrative_structure(
        mock_video_data, frames
    )

    # Check that the OpenAI API was called
    assert mock_client.chat.completions.create.called

    # Check that the structure is what we mocked
    assert structure == "three_act"
