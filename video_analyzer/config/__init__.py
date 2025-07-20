"""
Configuration management for the video analyzer.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """
    Base configuration class.
    """

    pass


class VideoProcessorConfig(BaseConfig):
    """
    Configuration for the video processor.
    """

    allowed_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "wmv"],
        description="List of allowed video formats",
    )
    max_file_size_mb: int = Field(
        default=500, description="Maximum allowed file size in MB"
    )


class FrameExtractorConfig(BaseConfig):
    """
    Configuration for the frame extractor.
    """

    uniform_interval: float = Field(
        default=1.0, description="Interval in seconds for uniform frame extraction"
    )
    scene_change_threshold: float = Field(
        default=0.35, description="Threshold for scene change detection"
    )
    max_frames: int = Field(
        default=1000, description="Maximum number of frames to extract"
    )


class AnalysisPipelineConfig(BaseConfig):
    """
    Configuration for the analysis pipeline.
    """

    parallel_analyzers: bool = Field(
        default=True, description="Whether to run analyzers in parallel"
    )
    timeout_seconds: int = Field(
        default=300, description="Timeout for analysis in seconds"
    )


class OpenAIConfig(BaseConfig):
    """
    Configuration for OpenAI integration.
    """

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4", description="OpenAI model to use")
    temperature: float = Field(
        default=0.7, description="Temperature for OpenAI API calls"
    )


class ComputerVisionConfig(BaseConfig):
    """
    Configuration for computer vision services.
    """

    object_detection_confidence: float = Field(
        default=0.5, description="Confidence threshold for object detection"
    )
    face_detection_confidence: float = Field(
        default=0.7, description="Confidence threshold for face detection"
    )


class ReportGeneratorConfig(BaseConfig):
    """
    Configuration for the report generator.
    """

    include_frames: bool = Field(
        default=True, description="Whether to include frames in the report"
    )
    max_recommendations: int = Field(
        default=10, description="Maximum number of recommendations to include"
    )


class AppConfig(BaseConfig):
    """
    Main application configuration.
    """

    video_processor: VideoProcessorConfig = Field(default_factory=VideoProcessorConfig)
    frame_extractor: FrameExtractorConfig = Field(default_factory=FrameExtractorConfig)
    analysis_pipeline: AnalysisPipelineConfig = Field(
        default_factory=AnalysisPipelineConfig
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    computer_vision: ComputerVisionConfig = Field(default_factory=ComputerVisionConfig)
    report_generator: ReportGeneratorConfig = Field(
        default_factory=ReportGeneratorConfig
    )


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a file or environment variables.

    Args:
        config_path: Path to the configuration file. If None, will look for
                    config.json in the current directory and then fall back to
                    environment variables.

    Returns:
        AppConfig: The loaded configuration.
    """
    config_data = {}

    # Try to load from file if provided
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)

    # Override with environment variables
    # Format: VIDEO_ANALYZER_SECTION_KEY=value
    # Example: VIDEO_ANALYZER_OPENAI_API_KEY=sk-123456
    for env_name, env_value in os.environ.items():
        if env_name.startswith("VIDEO_ANALYZER_"):
            parts = env_name.split("_")[2:]
            if len(parts) >= 2:
                section = parts[0].lower()
                key = "_".join(parts[1:]).lower()

                if section not in config_data:
                    config_data[section] = {}

                # Try to parse as JSON, fall back to string if it fails
                try:
                    config_data[section][key] = json.loads(env_value)
                except json.JSONDecodeError:
                    config_data[section][key] = env_value

    return AppConfig(**config_data)


# Create a global config instance
config = load_config()
