"""
Configuration module for the Video Analyzer.
"""

from video_analyzer.config.frame_extractor import (
    FrameExtractorConfig,
    ExtractionStrategy,
)
from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig

__all__ = [
    "FrameExtractorConfig",
    "ExtractionStrategy",
    "VideoProcessorConfig",
    "AnalysisPipelineConfig",
]
