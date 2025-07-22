"""
Analysis manager for coordinating video analysis.

This module provides a high-level interface for analyzing videos using the
analysis pipeline and various analyzers.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import time

from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import AnalysisResult, Report
from video_analyzer.services.pipeline import AnalysisPipeline
from video_analyzer.services.video_processor import VideoProcessor
from video_analyzer.services.frame_extractor import FrameExtractor
from video_analyzer.analyzers.base import (
    BaseAnalyzer,
    AnalyzerRegistry,
    AnalysisProgressCallback,
    AnalysisCancellationToken,
)
from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig
from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.config.frame_extractor import FrameExtractorConfig
from video_analyzer.utils.errors import VideoAnalyzerError, VideoProcessingError

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisManager:
    """
    High-level manager for video analysis.

    This class coordinates the video processing, frame extraction, and analysis pipeline
    to provide a simple interface for analyzing videos.
    """

    def __init__(
        self,
        pipeline_config: Optional[AnalysisPipelineConfig] = None,
        video_processor_config: Optional[VideoProcessorConfig] = None,
        frame_extractor_config: Optional[FrameExtractorConfig] = None,
    ):
        """
        Initialize the analysis manager.

        Args:
            pipeline_config: Configuration for the analysis pipeline
            video_processor_config: Configuration for the video processor
            frame_extractor_config: Configuration for the frame extractor
        """
        # Create default configs if not provided
        self.pipeline_config = pipeline_config or AnalysisPipelineConfig()
        self.video_processor_config = video_processor_config or VideoProcessorConfig()
        self.frame_extractor_config = frame_extractor_config or FrameExtractorConfig()

        # Create components
        self.pipeline = AnalysisPipeline(self.pipeline_config)
        self.video_processor = VideoProcessor(self.video_processor_config)
        self.frame_extractor = FrameExtractor(self.frame_extractor_config)

        # Initialize state
        self._registered_analyzers: Dict[str, BaseAnalyzer] = {}
        self._progress_callback: Optional[Callable] = None
        self._cancellation_token: Optional[AnalysisCancellationToken] = None
        self._start_time = None
        self._end_time = None

        logger.debug("Initialized AnalysisManager")

    def register_analyzer(
        self, analyzer: Union[BaseAnalyzer, str], config: Dict[str, Any] = None
    ) -> None:
        """
        Register an analyzer with the analysis manager.

        Args:
            analyzer: The analyzer to register, or the name of an analyzer type
            config: Optional configuration for the analyzer (used if analyzer is a string)
        """
        if isinstance(analyzer, str):
            # Create analyzer from registry
            analyzer_instance = AnalyzerRegistry.create(analyzer, config)
        else:
            analyzer_instance = analyzer

        # Register with pipeline
        self.pipeline.register_analyzer(analyzer_instance)

        # Store for reference
        self._registered_analyzers[analyzer_instance.analyzer_id] = analyzer_instance

        logger.debug(f"Registered analyzer: {analyzer_instance.analyzer_id}")

    def register_analyzers(self, analyzer_types: List[str]) -> None:
        """
        Register multiple analyzers by type.

        Args:
            analyzer_types: List of analyzer types to register
        """
        for analyzer_type in analyzer_types:
            self.register_analyzer(analyzer_type)

    def set_progress_callback(
        self, callback: Callable[[str, float, Dict[str, Any]], None]
    ) -> None:
        """
        Set a callback function to track progress.

        Args:
            callback: Function to call with progress updates.
                     Arguments: analyzer_id, progress (0.0-1.0), metadata
        """
        self._progress_callback = callback
        self.pipeline.set_progress_callback(callback)
        logger.debug("Progress callback set")

    def create_cancellation_token(self) -> AnalysisCancellationToken:
        """
        Create a cancellation token for analysis.

        Returns:
            AnalysisCancellationToken: A token that can be used to cancel the analysis
        """
        self._cancellation_token = self.pipeline.create_cancellation_token()
        logger.debug("Cancellation token created")
        return self._cancellation_token

    async def analyze_video(
        self, video_path: Union[str, Path]
    ) -> Dict[str, AnalysisResult]:
        """
        Analyze a video file.

        This method handles the entire analysis process:
        1. Process the video file
        2. Extract frames
        3. Run the analysis pipeline

        Args:
            video_path: Path to the video file

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results

        Raises:
            VideoProcessingError: If there's an error processing the video
        """
        self._start_time = time.time()
        logger.info(f"Starting analysis for video: {video_path}")

        # Convert string path to Path if needed
        if isinstance(video_path, str):
            video_path = Path(video_path)

        try:
            # Process the video
            if self._progress_callback:
                self._progress_callback("manager", 0.1, {"status": "processing_video"})

            processed_video = self.video_processor.process_video(video_path)

            # Extract frames based on analyzer requirements
            if self._progress_callback:
                self._progress_callback("manager", 0.2, {"status": "extracting_frames"})

            frame_requirements = self._get_combined_frame_requirements()
            video_data = await self._extract_frames(processed_video, frame_requirements)

            # Run the analysis pipeline
            if self._progress_callback:
                self._progress_callback("manager", 0.3, {"status": "running_analysis"})

            results = await self.pipeline.run_analysis(video_data)

            self._end_time = time.time()
            elapsed_time = self._end_time - self._start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

            if self._progress_callback:
                self._progress_callback(
                    "manager",
                    1.0,
                    {"status": "completed", "elapsed_time": elapsed_time},
                )

            return results

        except Exception as e:
            self._end_time = time.time()
            elapsed_time = self._end_time - self._start_time
            logger.error(
                f"Analysis failed after {elapsed_time:.2f} seconds: {str(e)}",
                exc_info=True,
            )

            if self._progress_callback:
                self._progress_callback(
                    "manager",
                    0.0,
                    {"status": "error", "error": str(e), "elapsed_time": elapsed_time},
                )

            if isinstance(e, VideoAnalyzerError):
                raise
            else:
                raise VideoProcessingError(f"Analysis failed: {str(e)}")

    async def _extract_frames(
        self, video_path: Path, requirements: Dict[str, Any]
    ) -> VideoData:
        """
        Extract frames from a video based on requirements.

        Args:
            video_path: Path to the video file
            requirements: Frame extraction requirements

        Returns:
            VideoData: Video data with extracted frames
        """
        # Extract frames based on requirements
        frames = await self.frame_extractor.extract_frames_async(
            video_path,
            min_frames=requirements.get("min_frames", 1),
            frame_interval=requirements.get("frame_interval"),
            specific_timestamps=requirements.get("specific_timestamps"),
        )

        # Create VideoData object
        video_info = self.frame_extractor.get_video_info(video_path)

        video_data = VideoData(
            path=video_path,
            frames=frames,
            duration=video_info["duration"],
            fps=video_info["fps"],
            resolution=video_info["resolution"],
            metadata=video_info,
        )

        return video_data

    def _get_combined_frame_requirements(self) -> Dict[str, Any]:
        """
        Combine frame requirements from all registered analyzers.

        Returns:
            Dict[str, Any]: Combined frame requirements
        """
        combined_requirements = {
            "min_frames": 1,
            "frame_interval": None,
            "specific_timestamps": set(),
        }

        for analyzer in self._registered_analyzers.values():
            requirements = analyzer.required_frames

            # Take the maximum min_frames
            if "min_frames" in requirements:
                combined_requirements["min_frames"] = max(
                    combined_requirements["min_frames"], requirements["min_frames"]
                )

            # Take the smallest frame_interval (most dense sampling)
            if requirements.get("frame_interval") is not None:
                if combined_requirements["frame_interval"] is None:
                    combined_requirements["frame_interval"] = requirements[
                        "frame_interval"
                    ]
                else:
                    combined_requirements["frame_interval"] = min(
                        combined_requirements["frame_interval"],
                        requirements["frame_interval"],
                    )

            # Combine specific timestamps
            if requirements.get("specific_timestamps"):
                combined_requirements["specific_timestamps"].update(
                    requirements["specific_timestamps"]
                )

        # Convert set to list for specific_timestamps
        if combined_requirements["specific_timestamps"]:
            combined_requirements["specific_timestamps"] = sorted(
                combined_requirements["specific_timestamps"]
            )
        else:
            combined_requirements["specific_timestamps"] = None

        return combined_requirements

    def generate_report(
        self, results: Dict[str, AnalysisResult], video_id: str
    ) -> Report:
        """
        Generate a comprehensive report from analysis results.

        Args:
            results: Analysis results from different analyzers
            video_id: Identifier for the analyzed video

        Returns:
            Report: Comprehensive analysis report
        """
        # This is a placeholder implementation that would be enhanced in a real system
        from video_analyzer.models.analysis import Report

        # Create sections for each analyzer type
        sections = {}
        for analyzer_id, result in results.items():
            analyzer_type = analyzer_id.split("_")[0]  # Simple heuristic
            sections[analyzer_type] = result.dict()

        # Create a summary
        summary = f"Analysis of video {video_id} with {len(results)} analyzers"

        # Collect recommendations from all analyzers
        recommendations = []
        for result in results.values():
            if hasattr(result, "recommendations") and result.recommendations:
                recommendations.extend(result.recommendations)

        # Create the report
        report = Report(
            video_id=video_id,
            analysis_duration=self.execution_time or 0,
            summary=summary,
            sections=sections,
            recommendations=recommendations,
            visual_examples=[],  # Would be populated in a real implementation
        )

        return report

    @property
    def execution_time(self) -> Optional[float]:
        """
        Get the execution time of the last analysis in seconds.

        Returns:
            Optional[float]: The execution time, or None if no analysis has been run
        """
        if self._start_time is None or self._end_time is None:
            return None
        return self._end_time - self._start_time

    @property
    def registered_analyzers(self) -> List[str]:
        """
        Get a list of registered analyzer IDs.

        Returns:
            List[str]: The registered analyzer IDs
        """
        return list(self._registered_analyzers.keys())


# Example usage of the AnalysisManager
async def example_analysis_manager_usage():
    """
    Example of how to use the AnalysisManager.
    """
    # Create an analysis manager
    manager = AnalysisManager()

    # Register analyzers
    manager.register_analyzer("hook")
    manager.register_analyzer("progression")

    # Set up progress tracking
    def on_progress(
        analyzer_id: str, progress: float, metadata: Dict[str, Any]
    ) -> None:
        status = metadata.get("status", "in_progress")
        print(f"{analyzer_id}: {progress:.1%} - {status}")

    manager.set_progress_callback(on_progress)

    # Create a cancellation token
    cancellation_token = manager.create_cancellation_token()

    try:
        # Analyze a video
        results = await manager.analyze_video("example.mp4")

        # Generate a report
        report = manager.generate_report(results, "example_video")

        print(f"Analysis completed with {len(results)} results")
        print(f"Report summary: {report.summary}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
