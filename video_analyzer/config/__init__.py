"""
Configuration module for the Video Analyzer.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import os

from video_analyzer.config.frame_extractor import (
    FrameExtractorConfig,
    ExtractionStrategy,
)
from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig


class AppConfig:
    """
    Application configuration class.
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize the application configuration.

        Args:
            config_dict: Optional configuration dictionary
        """
        self.config_dict = config_dict or {}

        # Initialize sub-configs
        self.frame_extractor = FrameExtractorConfig(
            **self.config_dict.get("frame_extractor", {})
        )
        self.video_processor = VideoProcessorConfig(
            **self.config_dict.get("video_processor", {})
        )
        self.analysis_pipeline = AnalysisPipelineConfig(
            **self.config_dict.get("analysis_pipeline", {})
        )


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a file.

    Args:
        config_path: Path to the configuration file

    Returns:
        AppConfig: The loaded configuration
    """
    config_dict = {}

    # Load from default locations if no path provided
    if config_path is None:
        # Try user home directory
        home_config = Path.home() / ".video_analyzer" / "config.json"
        if home_config.exists():
            with open(home_config, "r") as f:
                config_dict = json.load(f)
    else:
        # Load from specified path
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)

    return AppConfig(config_dict)


__all__ = [
    "FrameExtractorConfig",
    "ExtractionStrategy",
    "VideoProcessorConfig",
    "AnalysisPipelineConfig",
    "AppConfig",
    "load_config",
]
