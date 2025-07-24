"""
Tests for the report generator.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import (
    AnalysisResult,
    Report,
    HookAnalysisResult,
    ProgressionAnalysisResult,
    VisualAnalysisResult,
    AudioAnalysisResult,
    ObjectDetectionResult,
    EmotionAnalysisResult,
    StorytellingAnalysisResult,
)
from video_analyzer.services.report_generator import (
    ReportGenerator,
    ReportGeneratorConfig,
)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    # Create a simple 10x10 RGB image
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    # Add some color to make it more interesting
    image[2:8, 2:8] = [255, 0, 0]  # Red square

    return Frame(image=image, timestamp=1.0, index=0)


@pytest.fixture
def sample_video_data(sample_frame, tmp_path):
    """Create sample video data for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Create an empty file
    video_path.touch()

    return VideoData(
        path=video_path,
        frames=[sample_frame],
        duration=10.0,
        fps=30.0,
        resolution=(1280, 720),
        metadata={"codec": "h264", "bitrate": "5000k"},
    )


@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results for testing."""
    # Hook analysis
    hook_result = HookAnalysisResult(
        analyzer_id="hook_analyzer",
        hook_start_time=0.0,
        hook_end_time=5.0,
        hook_techniques=["question", "surprise", "curiosity"],
        hook_effectiveness=0.85,
        key_moments=[
            {"timestamp": 1.5, "description": "Surprising statement"},
            {"timestamp": 3.0, "description": "Question posed to audience"},
        ],
        recommendations=[
            "Start with a stronger question",
            "Add visual element in first 2 seconds",
        ],
    )

    # Progression analysis
    progression_result = ProgressionAnalysisResult(
        analyzer_id="progression_analyzer",
        sections=[
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "title": "Introduction",
                "description": "Video introduction and hook",
            },
            {
                "start_time": 5.0,
                "end_time": 10.0,
                "title": "Main content",
                "description": "Core message delivery",
            },
        ],
        pacing_changes=[
            {
                "timestamp": 5.0,
                "from_pace": "slow",
                "to_pace": "medium",
                "description": "Picks up pace after intro",
            }
        ],
        transitions=[
            {
                "timestamp": 5.0,
                "type": "fade",
                "description": "Fade transition to main content",
            }
        ],
        narrative_flow_score=0.78,
        retention_strategies=["Frequent callbacks", "Open loops", "Visual consistency"],
    )

    # Visual analysis
    visual_result = VisualAnalysisResult(
        analyzer_id="visual_analyzer",
        lighting_quality=0.92,
        color_schemes=[
            {
                "timestamp": 1.0,
                "colors": ["#FF5733", "#33FF57", "#3357FF"],
                "mood": "energetic",
            },
            {
                "timestamp": 6.0,
                "colors": ["#5733FF", "#FF33F5", "#33FFF5"],
                "mood": "calm",
            },
        ],
        camera_movements=[
            {
                "timestamp": 2.0,
                "type": "pan",
                "duration": 1.5,
                "description": "Pan from left to right",
            }
        ],
        visual_effects=[
            {
                "timestamp": 4.0,
                "type": "zoom",
                "purpose": "emphasis",
                "description": "Zoom in on product",
            }
        ],
        visual_recommendations=[
            "Improve lighting consistency",
            "Use more complementary colors",
        ],
    )

    # Audio analysis
    audio_result = AudioAnalysisResult(
        analyzer_id="audio_analyzer",
        sound_quality=0.88,
        speech_analysis={
            "pacing": "good",
            "tone": "enthusiastic",
            "clarity": "excellent",
        },
        background_music=[
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "mood": "upbeat",
                "description": "Energetic background track",
            }
        ],
        sound_effects=[
            {
                "timestamp": 3.5,
                "type": "whoosh",
                "purpose": "transition",
                "description": "Transition sound",
            }
        ],
        transcription="Welcome to this video about video analysis. Today we'll explore...",
    )

    # Object detection
    object_result = ObjectDetectionResult(
        analyzer_id="object_detector",
        objects=[
            {
                "timestamp": 1.0,
                "label": "person",
                "confidence": 0.95,
                "bounding_box": [10, 10, 100, 200],
            },
            {
                "timestamp": 2.0,
                "label": "laptop",
                "confidence": 0.87,
                "bounding_box": [150, 100, 250, 180],
            },
        ],
        faces=[
            {
                "timestamp": 1.0,
                "expression": "happy",
                "confidence": 0.92,
                "bounding_box": [20, 20, 80, 80],
            }
        ],
        brands=[
            {
                "timestamp": 3.0,
                "name": "Example Brand",
                "confidence": 0.85,
                "bounding_box": [200, 150, 300, 200],
            }
        ],
        screen_time_analysis={
            "by_label": {"person": 8.5, "laptop": 5.2},
            "total_tracked_time": 10.0,
        },
        brand_integration_score=0.75,
    )

    # Emotion analysis
    emotion_result = EmotionAnalysisResult(
        analyzer_id="emotion_analyzer",
        overall_mood="positive",
        emotional_shifts=[
            {
                "timestamp": 5.0,
                "from_emotion": "curiosity",
                "to_emotion": "excitement",
                "trigger": "product reveal",
            }
        ],
        emotional_elements={
            "visual": ["bright colors", "smiling faces"],
            "audio": ["upbeat music", "enthusiastic tone"],
        },
        emotion_techniques=["contrast", "anticipation", "resolution"],
        emotional_journey=[
            {"timestamp": 1.0, "emotion": "curiosity", "intensity": 0.6},
            {"timestamp": 5.0, "emotion": "excitement", "intensity": 0.8},
            {"timestamp": 9.0, "emotion": "satisfaction", "intensity": 0.7},
        ],
    )

    # Storytelling analysis
    storytelling_result = StorytellingAnalysisResult(
        analyzer_id="storytelling_analyzer",
        narrative_structure="problem-solution-benefit",
        character_development=[
            {
                "timestamp": 2.0,
                "character": "presenter",
                "development_type": "establishes credibility",
                "description": "Presenter shares relevant experience",
            }
        ],
        conflict_patterns=[
            {
                "conflict_timestamp": 3.0,
                "resolution_timestamp": 7.0,
                "conflict_type": "problem identification",
                "description": "Identifies pain point and later resolves it",
            }
        ],
        persuasion_techniques=["social proof", "scarcity", "authority"],
        engagement_strategies=["storytelling", "asking questions", "demonstration"],
    )

    return {
        "hook": hook_result,
        "progression": progression_result,
        "visual": visual_result,
        "audio": audio_result,
        "object": object_result,
        "emotion": emotion_result,
        "storytelling": storytelling_result,
    }


def test_report_generator_initialization():
    """Test that the report generator initializes correctly."""
    # Default initialization
    generator = ReportGenerator()
    assert generator.config is not None

    # Custom configuration
    custom_config = ReportGeneratorConfig(
        output_dir="custom_output", include_visual_examples=False
    )
    generator = ReportGenerator(config=custom_config)
    assert generator.config.output_dir == Path("custom_output")
    assert generator.config.include_visual_examples is False


def test_generate_report(sample_video_data, sample_analysis_results):
    """Test generating a report from analysis results."""
    generator = ReportGenerator()

    # Generate the report
    report = generator.generate_report(
        results=sample_analysis_results,
        video_data=sample_video_data,
        analysis_duration=2.5,
    )

    # Verify report structure
    assert isinstance(report, Report)
    assert report.video_id is not None
    assert isinstance(report.analysis_timestamp, datetime)
    assert report.analysis_duration == 2.5
    assert report.summary is not None

    # Check that all sections are present
    assert "hook" in report.sections
    assert "progression" in report.sections
    assert "visual" in report.sections
    assert "audio" in report.sections
    assert "object" in report.sections
    assert "emotion" in report.sections
    assert "storytelling" in report.sections

    # Check recommendations
    assert len(report.recommendations) > 0

    # Check visual examples if enabled
    if generator.config.include_visual_examples:
        assert len(report.visual_examples) > 0


def test_save_json_report(sample_video_data, sample_analysis_results, tmp_path):
    """Test saving a report in JSON format."""
    # Create a report generator with output to temp directory
    config = ReportGeneratorConfig(output_dir=tmp_path)
    generator = ReportGenerator(config=config)

    # Generate a report
    report = generator.generate_report(
        results=sample_analysis_results,
        video_data=sample_video_data,
        analysis_duration=2.5,
    )

    # Save the report as JSON
    output_path = generator.save_report(report, format="json")

    # Verify the file exists
    assert output_path.exists()

    # Verify the content is valid JSON
    with open(output_path, "r") as f:
        report_data = json.load(f)

    assert report_data["video_id"] == report.video_id
    assert "analysis_timestamp" in report_data
    assert report_data["analysis_duration"] == 2.5
    assert "summary" in report_data
    assert "sections" in report_data
    assert "recommendations" in report_data


def test_save_html_report(sample_video_data, sample_analysis_results, tmp_path):
    """Test saving a report in HTML format."""
    # Create a report generator with output to temp directory
    config = ReportGeneratorConfig(output_dir=tmp_path)
    generator = ReportGenerator(config=config)

    # Generate a report
    report = generator.generate_report(
        results=sample_analysis_results,
        video_data=sample_video_data,
        analysis_duration=2.5,
    )

    # Save the report as HTML
    output_path = generator.save_report(report, format="html")

    # Verify the file exists
    assert output_path.exists()

    # Verify the content is HTML
    with open(output_path, "r") as f:
        html_content = f.read()

    assert "<!DOCTYPE html>" in html_content
    assert "<html>" in html_content
    assert report.video_id in html_content
    assert "Summary" in html_content
    assert "Recommendations" in html_content


def test_save_pdf_report(
    sample_video_data, sample_analysis_results, tmp_path, monkeypatch
):
    """Test saving a report in PDF format."""

    # Mock the weasyprint import and HTML class to avoid actual PDF generation
    class MockHTML:
        def __init__(self, string=None):
            self.string = string

        def write_pdf(self, output_path):
            # Just create an empty file
            with open(output_path, "wb") as f:
                f.write(b"%PDF-1.5")  # Minimal PDF header

    # Apply the mock
    monkeypatch.setattr("video_analyzer.services.report_generator.HTML", MockHTML)

    # Create a report generator with output to temp directory
    config = ReportGeneratorConfig(output_dir=tmp_path)
    generator = ReportGenerator(config=config)

    # Generate a report
    report = generator.generate_report(
        results=sample_analysis_results,
        video_data=sample_video_data,
        analysis_duration=2.5,
    )

    # Save the report as PDF
    output_path = generator.save_report(report, format="pdf")

    # Verify the file exists
    assert output_path.exists()

    # Verify it has PDF header
    with open(output_path, "rb") as f:
        content = f.read()
    assert content.startswith(b"%PDF")


def test_extract_visual_examples(sample_video_data, sample_analysis_results):
    """Test extracting visual examples from video frames."""
    # Create a report generator with visual examples enabled
    config = ReportGeneratorConfig(include_visual_examples=True, max_visual_examples=3)
    generator = ReportGenerator(config=config)

    # Access the private method directly for testing
    visual_examples = generator._extract_visual_examples(
        results=sample_analysis_results, video_data=sample_video_data
    )

    # Verify visual examples were extracted
    assert len(visual_examples) > 0
    assert len(visual_examples) <= config.max_visual_examples

    # Check structure of visual examples
    for example in visual_examples:
        assert "timestamp" in example
        assert "category" in example
        assert "description" in example
        assert "image_data" in example  # Base64 encoded image


def test_collect_recommendations(sample_analysis_results):
    """Test collecting recommendations from all analyzers."""
    generator = ReportGenerator()

    # Access the private method directly for testing
    recommendations = generator._collect_recommendations(
        results=sample_analysis_results
    )

    # Verify recommendations were collected
    assert len(recommendations) > 0

    # Check that hook recommendations are included
    hook_recommendations = sample_analysis_results["hook"].recommendations
    for rec in hook_recommendations:
        assert rec in recommendations


def test_unsupported_format(sample_video_data, sample_analysis_results, tmp_path):
    """Test that an error is raised for unsupported formats."""
    # Create a report generator
    config = ReportGeneratorConfig(output_dir=tmp_path)
    generator = ReportGenerator(config=config)

    # Generate a report
    report = generator.generate_report(
        results=sample_analysis_results,
        video_data=sample_video_data,
        analysis_duration=2.5,
    )

    # Try to save in an unsupported format
    with pytest.raises(ValueError, match="Unsupported report format"):
        generator.save_report(report, format="invalid")
