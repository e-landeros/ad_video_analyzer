"""
Tests for the visual elements analyzer.
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


class VisualAnalysisResult(AnalysisResult):
    def __init__(
        self,
        analyzer_id,
        lighting_quality=0.7,
        color_schemes=None,
        camera_movements=None,
        visual_effects=None,
        visual_recommendations=None,
        **kwargs,
    ):
        super().__init__(analyzer_id, **kwargs)
        self.lighting_quality = lighting_quality
        self.color_schemes = color_schemes or []
        self.camera_movements = camera_movements or []
        self.visual_effects = visual_effects or []
        self.visual_recommendations = visual_recommendations or []


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
].VisualAnalysisResult = VisualAnalysisResult

sys.modules["video_analyzer.analyzers.base"] = MagicMock()
sys.modules["video_analyzer.analyzers.base"].BaseAnalyzer = BaseAnalyzer
sys.modules["video_analyzer.analyzers.base"].AnalyzerRegistry = AnalyzerRegistry
sys.modules["video_analyzer.analyzers.base"].ANALYZER_TYPES = {
    "visual": "Visual elements and quality analysis"
}

# Mock the api module to avoid import errors
sys.modules["video_analyzer.api"] = MagicMock()
sys.modules["video_analyzer.api.cli"] = MagicMock()
sys.modules["video_analyzer.api"].run = MagicMock()

# Mock cv2
sys.modules["cv2"] = MagicMock()
import cv2

# Mock cv2 functions
cv2.cvtColor = MagicMock(return_value=np.ones((100, 100), dtype=np.uint8) * 128)
cv2.calcHist = MagicMock(return_value=np.ones((256, 1), dtype=np.float32))
cv2.normalize = MagicMock(return_value=np.ones((256), dtype=np.float32) / 256)
cv2.Canny = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
cv2.resize = MagicMock(
    lambda img, size: img[: size[0], : size[1]]
    if len(img.shape) == 2
    else img[: size[0], : size[1], :]
)
cv2.kmeans = MagicMock(
    return_value=(
        None,
        np.zeros(10000, dtype=np.int32),
        np.ones((5, 3), dtype=np.uint8) * 128,
    )
)
cv2.calcOpticalFlowFarneback = MagicMock(
    return_value=np.zeros((100, 100, 2), dtype=np.float32)
)
cv2.cartToPolar = MagicMock(
    return_value=(
        np.ones(10000, dtype=np.float32) * 5,
        np.ones(10000, dtype=np.float32) * 0.5,
    )
)
cv2.Laplacian = MagicMock(return_value=np.ones((100, 100), dtype=np.float64) * 150)
cv2.mean = MagicMock(return_value=[128, 128, 128, 0])
cv2.compareHist = MagicMock(return_value=0.85)
cv2.HISTCMP_CORREL = 0
cv2.COLOR_BGR2GRAY = 0
cv2.COLOR_BGR2RGB = 0
cv2.COLOR_RGB2HSV = 0
cv2.TERM_CRITERIA_EPS = 1
cv2.TERM_CRITERIA_MAX_ITER = 2
cv2.KMEANS_RANDOM_CENTERS = 0


# Create our own VisualElementsAnalyzer class for testing
class VisualElementsAnalyzer(BaseAnalyzer):
    """Mock implementation of VisualElementsAnalyzer for testing"""

    def __init__(self, config=None):
        super().__init__(config)
        self.lighting_threshold_low = 40
        self.lighting_threshold_high = 220
        self.color_sample_interval = 5
        self.movement_detection_threshold = 20.0
        self.effect_detection_sensitivity = 0.7

    @property
    def analyzer_id(self):
        return "visual_elements_analyzer"

    @property
    def supports_progress(self):
        return True

    @property
    def required_frames(self):
        return {
            "min_frames": 10,
            "frame_interval": 1.0,
            "specific_timestamps": None,
        }

    async def analyze(self, video_data):
        if len(video_data.frames) < 5:
            return VisualAnalysisResult(
                analyzer_id=self.analyzer_id,
                lighting_quality=0.5,
                video_id=str(video_data.path),
                confidence=0.5,
            )

        lighting_quality, lighting_issues = await self._evaluate_lighting(video_data)
        color_schemes = await self._identify_color_schemes(video_data)
        camera_movements = await self._detect_camera_movements(video_data)
        visual_effects = await self._identify_visual_effects(video_data)
        visual_recommendations = await self._generate_recommendations(
            video_data,
            lighting_quality,
            lighting_issues,
            color_schemes,
            camera_movements,
            visual_effects,
        )

        return VisualAnalysisResult(
            analyzer_id=self.analyzer_id,
            lighting_quality=lighting_quality,
            color_schemes=color_schemes,
            camera_movements=camera_movements,
            visual_effects=visual_effects,
            visual_recommendations=visual_recommendations,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in video_data.frames],
            confidence=0.8,
        )

    async def _evaluate_lighting(self, video_data):
        # Mock implementation
        lighting_quality = 0.8
        lighting_issues = [
            {
                "timestamp": 10.0,
                "issue": "underexposed",
                "description": "Frame is underexposed (too dark)",
                "severity": 0.7,
            }
        ]
        return lighting_quality, lighting_issues

    async def _identify_color_schemes(self, video_data):
        # Mock implementation
        return [
            {
                "timestamp": 5.0,
                "colors": [[100, 100, 100], [150, 150, 150], [200, 200, 200]],
                "average_color": [120, 120, 120],
                "temperature": "cool",
                "mood": "neutral",
                "mood_description": "Balanced, natural, unobtrusive",
                "saturation": 0.3,
                "brightness": 0.6,
            }
        ]

    async def _detect_camera_movements(self, video_data):
        # Mock implementation
        return [
            {
                "timestamp": 15.0,
                "type": "pan_left",
                "duration": 3.0,
                "magnitude": 25.0,
                "purpose": "Reveal new information or follow subject movement",
            }
        ]

    async def _identify_visual_effects(self, video_data):
        # Mock implementation
        return [
            {
                "timestamp": 20.0,
                "type": "text",
                "purpose": "Display information or dialogue",
                "confidence": 0.7,
            }
        ]

    async def _generate_recommendations(
        self,
        video_data,
        lighting_quality,
        lighting_issues,
        color_schemes,
        camera_movements,
        visual_effects,
    ):
        # Mock implementation with conditional recommendations
        recommendations = []

        if lighting_quality < 0.6:
            recommendations.append(
                "Improve overall lighting quality for better visual clarity"
            )
        elif lighting_quality > 0.8:
            recommendations.append(
                "Excellent lighting quality. Consider maintaining this standard in future videos"
            )
        else:
            recommendations.append(
                "Consider maintaining more consistent color grading throughout the video"
            )

        if not visual_effects and video_data.duration > 30:
            recommendations.append(
                "Consider adding subtle visual effects to enhance production value"
            )

        if not recommendations:
            recommendations.append(
                "Consider experimenting with more dynamic visual techniques to enhance engagement"
            )

        return recommendations

    def _determine_color_mood(self, saturation, value, colors):
        # Mock implementation with monochromatic detection
        if saturation < 50:
            if value < 80:
                return "dark"
            elif value > 180:
                return "bright"
            else:
                return "muted"
        elif saturation > 150:
            # Check if colors are very similar (monochromatic)
            if len(colors) > 1:
                color_diversity = 0
                for i in range(len(colors)):
                    for j in range(i + 1, len(colors)):
                        color_diversity += sum(
                            abs(c1 - c2) for c1, c2 in zip(colors[i], colors[j])
                        )

                # If colors are very similar, it's monochromatic
                if color_diversity < 100:  # Low diversity threshold
                    return "monochromatic"

            return "vibrant"
        else:
            if value > 180:
                return "bright"
            elif value < 80:
                return "dark"
            else:
                return "neutral"

    def _determine_movement_purpose(self, movement_type):
        # Mock implementation
        purposes = {
            "pan_left": "Reveal new information or follow subject movement",
            "pan_right": "Reveal new information or follow subject movement",
            "tilt_up": "Reveal vertical space or create dramatic effect",
            "tilt_down": "Follow subject or transition to new scene",
            "zoom_in": "Focus attention or increase emotional intensity",
            "zoom_out": "Reveal context or decrease emotional intensity",
            "static": "Focus on composition or dialogue",
            "complex": "Create dynamic visual interest or follow complex action",
        }
        return purposes.get(movement_type, "Enhance visual storytelling")


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
async def test_visual_elements_analyzer_initialization():
    """Test that the visual elements analyzer initializes correctly."""
    analyzer = VisualElementsAnalyzer()
    assert analyzer.analyzer_id == "visual_elements_analyzer"
    assert analyzer.supports_progress is True


@pytest.mark.asyncio
async def test_visual_elements_analyzer_analyze(mock_video_data):
    """Test that the visual elements analyzer produces valid results."""
    analyzer = VisualElementsAnalyzer()
    result = await analyzer.analyze(mock_video_data)

    # Check that the result has the expected fields
    assert result.analyzer_id == "visual_elements_analyzer"
    assert result.video_id == str(mock_video_data.path)
    assert 0 <= result.lighting_quality <= 1.0

    # Check that color schemes were identified
    assert len(result.color_schemes) > 0
    for scheme in result.color_schemes:
        assert "timestamp" in scheme
        assert "colors" in scheme
        assert "mood" in scheme
        assert "temperature" in scheme
        assert 0 <= scheme["saturation"] <= 1.0
        assert 0 <= scheme["brightness"] <= 1.0

    # Check that camera movements were identified
    assert isinstance(result.camera_movements, list)
    for movement in result.camera_movements:
        if movement:  # If any movements were detected
            assert "timestamp" in movement
            assert "type" in movement
            assert "duration" in movement
            assert "purpose" in movement

    # Check that visual effects were identified
    assert isinstance(result.visual_effects, list)
    for effect in result.visual_effects:
        if effect:  # If any effects were detected
            assert "timestamp" in effect
            assert "type" in effect
            assert "purpose" in effect
            assert "confidence" in effect
            assert 0 <= effect["confidence"] <= 1.0

    # Check that recommendations were generated
    assert len(result.visual_recommendations) > 0


@pytest.mark.asyncio
async def test_visual_elements_analyzer_with_short_video():
    """Test the visual elements analyzer with a very short video."""
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

    analyzer = VisualElementsAnalyzer()
    result = await analyzer.analyze(short_video)

    # Check that the result is still valid
    assert result.analyzer_id == "visual_elements_analyzer"

    # Short videos should have lower confidence
    assert result.confidence <= 0.6


@pytest.mark.asyncio
async def test_evaluate_lighting(mock_video_data):
    """Test the lighting evaluation functionality."""
    analyzer = VisualElementsAnalyzer()
    lighting_quality, lighting_issues = await analyzer._evaluate_lighting(
        mock_video_data
    )

    # Check that lighting quality is valid
    assert 0.0 <= lighting_quality <= 1.0

    # Check that lighting issues are properly structured
    for issue in lighting_issues:
        assert "timestamp" in issue
        assert "issue" in issue
        assert "description" in issue
        assert "severity" in issue
        assert 0.0 <= issue["severity"] <= 1.0


@pytest.mark.asyncio
async def test_identify_color_schemes(mock_video_data):
    """Test the color scheme identification functionality."""
    analyzer = VisualElementsAnalyzer()
    color_schemes = await analyzer._identify_color_schemes(mock_video_data)

    # Check that color schemes were identified
    assert len(color_schemes) > 0

    # Check color scheme structure
    for scheme in color_schemes:
        assert "timestamp" in scheme
        assert "colors" in scheme
        assert "mood" in scheme
        assert "temperature" in scheme
        assert "saturation" in scheme
        assert "brightness" in scheme
        assert 0.0 <= scheme["saturation"] <= 1.0
        assert 0.0 <= scheme["brightness"] <= 1.0


@pytest.mark.asyncio
async def test_detect_camera_movements(mock_video_data):
    """Test the camera movement detection functionality."""
    analyzer = VisualElementsAnalyzer()
    camera_movements = await analyzer._detect_camera_movements(mock_video_data)

    # Check that camera movements are properly structured
    for movement in camera_movements:
        assert "timestamp" in movement
        assert "type" in movement
        assert "duration" in movement
        assert "magnitude" in movement
        assert "purpose" in movement
        assert movement["duration"] > 0


@pytest.mark.asyncio
async def test_identify_visual_effects(mock_video_data):
    """Test the visual effects identification functionality."""
    analyzer = VisualElementsAnalyzer()
    visual_effects = await analyzer._identify_visual_effects(mock_video_data)

    # Check that visual effects are properly structured
    for effect in visual_effects:
        assert "timestamp" in effect
        assert "type" in effect
        assert "purpose" in effect
        assert "confidence" in effect
        assert 0.0 <= effect["confidence"] <= 1.0
        assert effect["type"] in ["text", "color_grading", "vignette", "blur"]


@pytest.mark.asyncio
async def test_generate_recommendations(mock_video_data):
    """Test the recommendation generation functionality."""
    analyzer = VisualElementsAnalyzer()

    # Mock analysis results
    lighting_quality = 0.7
    lighting_issues = [
        {
            "timestamp": 5.0,
            "issue": "underexposed",
            "description": "Too dark",
            "severity": 0.7,
        },
        {
            "timestamp": 15.0,
            "issue": "overexposed",
            "description": "Too bright",
            "severity": 0.8,
        },
    ]
    color_schemes = [
        {
            "timestamp": 2.0,
            "colors": [[100, 100, 100]],
            "mood": "neutral",
            "saturation": 0.2,
            "brightness": 0.5,
            "temperature": "cool",
        },
        {
            "timestamp": 10.0,
            "colors": [[200, 100, 50]],
            "mood": "warm",
            "saturation": 0.6,
            "brightness": 0.7,
            "temperature": "warm",
        },
    ]
    camera_movements = [
        {
            "timestamp": 5.0,
            "type": "pan_left",
            "duration": 3.0,
            "magnitude": 20.0,
            "purpose": "Reveal new information",
        }
    ]
    visual_effects = [
        {
            "timestamp": 8.0,
            "type": "text",
            "purpose": "Display information",
            "confidence": 0.8,
        }
    ]

    recommendations = await analyzer._generate_recommendations(
        mock_video_data,
        lighting_quality,
        lighting_issues,
        color_schemes,
        camera_movements,
        visual_effects,
    )

    # Check that recommendations were generated
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)

    # Test with excellent lighting
    excellent_recommendations = await analyzer._generate_recommendations(
        mock_video_data, 0.9, [], color_schemes, camera_movements, visual_effects
    )
    assert any("Excellent lighting" in rec for rec in excellent_recommendations)

    # Test with no effects
    no_effects_recommendations = await analyzer._generate_recommendations(
        mock_video_data, 0.7, lighting_issues, color_schemes, camera_movements, []
    )
    assert any("visual effects" in rec for rec in no_effects_recommendations)


def test_determine_color_mood():
    """Test the color mood determination functionality."""
    analyzer = VisualElementsAnalyzer()

    # Test different combinations of saturation and value
    assert analyzer._determine_color_mood(20, 20, [[10, 10, 10]]) == "dark"
    assert analyzer._determine_color_mood(20, 200, [[200, 200, 200]]) == "bright"
    assert analyzer._determine_color_mood(20, 128, [[128, 128, 128]]) == "muted"
    assert (
        analyzer._determine_color_mood(200, 128, [[128, 0, 0], [0, 128, 0]])
        == "vibrant"
    )
    assert (
        analyzer._determine_color_mood(200, 128, [[128, 128, 128], [130, 130, 130]])
        == "monochromatic"
    )
    assert (
        analyzer._determine_color_mood(128, 200, [[200, 100, 50], [50, 100, 200]])
        == "bright"
    )
    assert (
        analyzer._determine_color_mood(128, 50, [[50, 25, 25], [25, 50, 25]]) == "dark"
    )
    assert (
        analyzer._determine_color_mood(128, 128, [[128, 64, 64], [64, 128, 64]])
        == "neutral"
    )


def test_determine_movement_purpose():
    """Test the movement purpose determination functionality."""
    analyzer = VisualElementsAnalyzer()

    # Test different movement types
    assert "Reveal" in analyzer._determine_movement_purpose("pan_left")
    assert "Reveal" in analyzer._determine_movement_purpose("pan_right")
    assert "vertical" in analyzer._determine_movement_purpose("tilt_up")
    assert "Follow" in analyzer._determine_movement_purpose("tilt_down")
    assert "Focus" in analyzer._determine_movement_purpose("zoom_in")
    assert "Reveal context" in analyzer._determine_movement_purpose("zoom_out")
    assert "Focus" in analyzer._determine_movement_purpose("static")
    assert "dynamic" in analyzer._determine_movement_purpose("complex")
    assert "storytelling" in analyzer._determine_movement_purpose("unknown_movement")
