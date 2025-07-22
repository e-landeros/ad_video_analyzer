"""
Tests for the object detector analyzer.
"""

import os
import pytest
import numpy as np
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock

# Mock cv2 if it's not available
try:
    import cv2
except ImportError:
    cv2 = MagicMock()
    cv2.rectangle = MagicMock()
    cv2.data = MagicMock()
    cv2.data.haarcascades = ""

from video_analyzer.analyzers.object_detector import ObjectDetector
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import ObjectDetectionResult


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a simple 100x100 test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a simple shape to the image
    cv2.rectangle(image, (30, 30), (70, 70), (255, 255, 255), -1)

    return Frame(image=image, timestamp=1.0, index=0)


@pytest.fixture
def sample_video_data(sample_frame):
    """Create sample video data for testing."""
    # Create multiple frames at different timestamps
    frames = [
        sample_frame,
        Frame(image=sample_frame.image, timestamp=2.0, index=1),
        Frame(image=sample_frame.image, timestamp=3.0, index=2),
        Frame(image=sample_frame.image, timestamp=4.0, index=3),
        Frame(image=sample_frame.image, timestamp=5.0, index=4),
    ]

    return VideoData(
        path=Path("test_video.mp4"),
        frames=frames,
        duration=5.0,
        fps=1.0,
        resolution=(100, 100),
    )


def test_object_detector_initialization():
    """Test that the object detector initializes correctly."""
    detector = ObjectDetector()

    assert detector.analyzer_id == "object_detector"
    assert detector.supports_progress is True
    assert detector.required_frames["min_frames"] == 10
    assert detector.required_frames["frame_interval"] == 0.5


@pytest.mark.asyncio
async def test_analyze_with_insufficient_frames():
    """Test analysis with insufficient frames."""
    detector = ObjectDetector()

    # Create video data with only one frame
    frame = Frame(image=np.zeros((100, 100, 3), dtype=np.uint8), timestamp=0.0, index=0)

    video_data = VideoData(
        path=Path("test_video.mp4"),
        frames=[frame],
        duration=1.0,
        fps=1.0,
        resolution=(100, 100),
    )

    # Analyze the video
    result = await detector.analyze(video_data)

    # Check that a minimal result was returned
    assert isinstance(result, ObjectDetectionResult)
    assert result.analyzer_id == "object_detector"
    assert result.confidence == 0.5  # Low confidence due to lack of frames
    assert len(result.objects) == 0
    assert len(result.faces) == 0
    assert len(result.brands) == 0


@pytest.mark.asyncio
async def test_analyze_with_sample_video(sample_video_data):
    """Test analysis with sample video data."""
    detector = ObjectDetector()

    # Mock the face detector to avoid OpenCV dependencies in tests
    detector.face_detector = MagicMock()
    detector.face_detector.detectMultiScale.return_value = [(30, 30, 40, 40)]

    # Analyze the video
    result = await detector.analyze(sample_video_data)

    # Check the result
    assert isinstance(result, ObjectDetectionResult)
    assert result.analyzer_id == "object_detector"
    assert result.confidence == 0.8

    # Check that objects were detected
    assert len(result.objects) > 0
    for obj in result.objects:
        assert "timestamp" in obj
        assert "label" in obj
        assert "confidence" in obj
        assert "bounding_box" in obj
        assert "tracking_id" in obj

    # Check that faces were detected
    assert len(result.faces) > 0
    for face in result.faces:
        assert "timestamp" in face
        assert "bounding_box" in face
        assert "expression" in face
        assert "face_id" in face

    # Check screen time analysis
    assert "by_label" in result.screen_time_analysis
    assert "by_brand" in result.screen_time_analysis
    assert "prominent_objects" in result.screen_time_analysis

    # Check brand integration score
    assert 0 <= result.brand_integration_score <= 1


@pytest.mark.asyncio
async def test_object_tracking():
    """Test object tracking functionality."""
    detector = ObjectDetector()

    # Create sample objects at different timestamps but similar positions
    objects = [
        {
            "timestamp": 1.0,
            "label": "person",
            "confidence": 0.9,
            "bounding_box": [10, 10, 50, 50],
            "frame_index": 0,
            "position": {"center_x": 0.35, "center_y": 0.35, "size": 0.25},
        },
        {
            "timestamp": 2.0,
            "label": "person",
            "confidence": 0.85,
            "bounding_box": [12, 12, 50, 50],
            "frame_index": 1,
            "position": {"center_x": 0.37, "center_y": 0.37, "size": 0.25},
        },
        {
            "timestamp": 3.0,
            "label": "person",
            "confidence": 0.8,
            "bounding_box": [15, 15, 50, 50],
            "frame_index": 2,
            "position": {"center_x": 0.4, "center_y": 0.4, "size": 0.25},
        },
        # Different object (different position)
        {
            "timestamp": 2.5,
            "label": "person",
            "confidence": 0.75,
            "bounding_box": [70, 70, 50, 50],
            "frame_index": 3,
            "position": {"center_x": 0.95, "center_y": 0.95, "size": 0.25},
        },
    ]

    # Create a mock video data
    video_data = MagicMock()

    # Track the objects
    tracked_objects = await detector._track_objects(objects, video_data)

    # Check that tracking IDs were assigned correctly
    assert len(tracked_objects) == 4

    # The first three objects should have the same tracking ID
    assert tracked_objects[0]["tracking_id"] == "person_1"
    assert tracked_objects[1]["tracking_id"] == "person_1"
    assert tracked_objects[2]["tracking_id"] == "person_1"

    # The fourth object should have a different tracking ID
    assert tracked_objects[3]["tracking_id"] == "person_2"

    # Check track start and end times
    assert tracked_objects[0]["track_start"] == 1.0
    assert tracked_objects[0]["track_end"] == 3.0  # Updated by later objects
    assert tracked_objects[3]["track_start"] == 2.5
    assert tracked_objects[3]["track_end"] == 2.5


@pytest.mark.asyncio
async def test_screen_time_analysis():
    """Test screen time analysis functionality."""
    detector = ObjectDetector()

    # Create sample objects, faces, and brands
    objects = [
        {
            "label": "person",
            "tracking_id": "person_1",
            "track_start": 1.0,
            "track_end": 3.0,
        },
        {"label": "car", "tracking_id": "car_1", "track_start": 2.0, "track_end": 4.0},
    ]

    faces = [{"face_id": "face_1", "track_start": 0.5, "track_end": 2.5}]

    brands = [
        {"name": "TechCorp", "track_start": 1.0, "track_end": 2.0},
        {"name": "TechCorp", "track_start": 3.0, "track_end": 4.0},
    ]

    # Analyze screen time
    screen_time = await detector._analyze_screen_time(objects, faces, brands, 5.0)

    # Check screen time calculations
    assert screen_time["by_label"]["person"] == 2.0  # 3.0 - 1.0
    assert screen_time["by_label"]["car"] == 2.0  # 4.0 - 2.0
    assert screen_time["by_brand"]["TechCorp"] == 2.0  # (2.0 - 1.0) + (4.0 - 3.0)

    # Check percentages
    assert screen_time["percentage_by_label"]["person"] == 40.0  # 2.0 / 5.0 * 100
    assert screen_time["percentage_by_label"]["car"] == 40.0  # 2.0 / 5.0 * 100
    assert screen_time["percentage_by_brand"]["TechCorp"] == 40.0  # 2.0 / 5.0 * 100

    # Check prominent objects and brands
    assert len(screen_time["prominent_objects"]) > 0
    assert len(screen_time["prominent_brands"]) > 0

    # Check face time
    assert screen_time["face_time"] == 2.0  # 2.5 - 0.5
    assert screen_time["face_percentage"] == 40.0  # 2.0 / 5.0 * 100


@pytest.mark.asyncio
async def test_brand_integration_score():
    """Test brand integration score calculation."""
    detector = ObjectDetector()

    # Create sample brands with different characteristics
    brands = [
        # Well-integrated brand (centered, moderate size)
        {
            "name": "TechCorp",
            "timestamp": 1.0,
            "track_start": 1.0,
            "track_end": 2.0,
            "position": {"center_x": 0.5, "center_y": 0.5, "size": 0.05},
        },
        {
            "name": "TechCorp",
            "timestamp": 3.0,
            "track_start": 3.0,
            "track_end": 4.0,
            "position": {"center_x": 0.52, "center_y": 0.48, "size": 0.05},
        },
        # Poorly integrated brand (corner, too large)
        {
            "name": "SportsBrand",
            "timestamp": 2.0,
            "track_start": 2.0,
            "track_end": 2.5,
            "position": {"center_x": 0.1, "center_y": 0.1, "size": 0.3},
        },
    ]

    # Create mock video data
    video_data = MagicMock()
    video_data.duration = 5.0

    # Calculate brand integration score
    score = await detector._calculate_brand_integration_score(brands, video_data)

    # Check that score is between 0 and 1
    assert 0 <= score <= 1

    # Test with no brands
    empty_score = await detector._calculate_brand_integration_score([], video_data)
    assert empty_score == 0.0
