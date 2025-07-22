"""
Analysis pipeline for orchestrating video analysis.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
import time
import logging

from video_analyzer.analyzers.base import (
    BaseAnalyzer,
    AnalysisProgressCallback,
    AnalysisCancellationToken,
    CancellationError,
)
from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import AnalysisResult, Report
from video_analyzer.config import AnalysisPipelineConfig
from video_analyzer.utils.errors import VideoAnalyzerError, VideoProcessingError

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Orchestrates the various analyzers.

    This class manages the execution of multiple analyzers on a video, handling
    dependencies, progress tracking, and cancellation.
    """

    def __init__(self, config: AnalysisPipelineConfig):
        self.config = config
        self.analyzers: List[BaseAnalyzer] = []
        self._progress_callback: Optional[AnalysisProgressCallback] = None
        self._cancellation_token: Optional[AnalysisCancellationToken] = None
        self._overall_progress: float = 0.0
        self._start_time = None
        self._end_time = None
        self._running = False
        self._total_analyzers = 0
        self._completed_analyzers = 0

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """
        Register an analyzer with the pipeline.

        Args:
            analyzer: The analyzer to register.
        """
        self.analyzers.append(analyzer)
        logger.debug(f"Registered analyzer: {analyzer.analyzer_id}")

    def set_progress_callback(
        self, callback: Callable[[str, float, Dict[str, Any]], None]
    ) -> None:
        """
        Set a callback function to track progress.

        Args:
            callback: Function to call with progress updates.
                     Arguments: analyzer_id, progress (0.0-1.0), metadata
        """
        self._progress_callback = AnalysisProgressCallback(callback)
        logger.debug("Progress callback set")

    def create_cancellation_token(self) -> AnalysisCancellationToken:
        """
        Create a cancellation token for this analysis pipeline.

        Returns:
            AnalysisCancellationToken: A token that can be used to cancel the analysis.
        """
        self._cancellation_token = AnalysisCancellationToken()
        logger.debug("Cancellation token created")
        return self._cancellation_token

    @property
    def is_running(self) -> bool:
        """Check if the analysis pipeline is currently running."""
        return self._running

    @property
    def overall_progress(self) -> float:
        """Get the overall progress of the analysis pipeline (0.0-1.0)."""
        return self._overall_progress

    @property
    def execution_time(self) -> Optional[float]:
        """
        Get the execution time of the last analysis in seconds.

        Returns:
            Optional[float]: The execution time, or None if no analysis has been run.
        """
        if self._start_time is None or self._end_time is None:
            return None
        return self._end_time - self._start_time

    async def run_analysis(self, video_data: VideoData) -> Dict[str, AnalysisResult]:
        """
        Run all registered analyzers on the video data.

        Args:
            video_data: The video data to analyze.

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results.

        Raises:
            CancellationError: If the analysis is cancelled.
            VideoProcessingError: If there's an error processing the video.
        """
        if self._running:
            raise VideoProcessingError("Analysis pipeline is already running")

        self._running = True
        self._start_time = time.time()
        self._overall_progress = 0.0
        self._total_analyzers = len(self.analyzers)
        self._completed_analyzers = 0

        results: Dict[str, AnalysisResult] = {}

        # Create a cancellation token if none exists
        if not self._cancellation_token:
            self._cancellation_token = AnalysisCancellationToken()

        # Report initial progress
        self._update_overall_progress(0.0, {"status": "started"})

        try:
            logger.info(
                f"Starting analysis pipeline with {self._total_analyzers} analyzers"
            )

            if self.config.parallel_analyzers:
                # Run analyzers in parallel
                results = await self._run_analyzers_parallel(video_data)
            else:
                # Run analyzers sequentially
                results = await self._run_analyzers_sequential(video_data)

            self._end_time = time.time()
            elapsed_time = self._end_time - self._start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

            # Report completion
            self._update_overall_progress(
                1.0, {"status": "completed", "elapsed_time": elapsed_time}
            )

            return results

        except CancellationError:
            self._end_time = time.time()
            elapsed_time = self._end_time - self._start_time
            logger.warning(f"Analysis cancelled after {elapsed_time:.2f} seconds")

            # Report cancellation
            self._update_overall_progress(
                self._overall_progress,
                {"status": "cancelled", "elapsed_time": elapsed_time},
            )

            raise
        except Exception as e:
            self._end_time = time.time()
            elapsed_time = self._end_time - self._start_time
            logger.error(
                f"Analysis failed after {elapsed_time:.2f} seconds: {str(e)}",
                exc_info=True,
            )

            # Report error
            self._update_overall_progress(
                self._overall_progress,
                {"status": "error", "error": str(e), "elapsed_time": elapsed_time},
            )

            raise VideoProcessingError(f"Analysis pipeline failed: {str(e)}")
        finally:
            self._running = False

    async def _run_analyzers_parallel(
        self, video_data: VideoData
    ) -> Dict[str, AnalysisResult]:
        """
        Run analyzers in parallel.

        Args:
            video_data: The video data to analyze.

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results.
        """
        results: Dict[str, AnalysisResult] = {}

        # Create tasks for all analyzers
        tasks = []
        for analyzer in self.analyzers:
            # Check if the analyzer is enabled
            if (
                self.config.enabled_analyzers is not None
                and analyzer.analyzer_category not in self.config.enabled_analyzers
            ):
                logger.debug(f"Skipping disabled analyzer: {analyzer.analyzer_id}")
                continue

            tasks.append(self._run_analyzer(analyzer, video_data))

        # Run all tasks in parallel
        analyzer_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for analyzer, result in zip(self.analyzers, analyzer_results):
            if isinstance(result, Exception):
                logger.error(f"Error in analyzer {analyzer.analyzer_id}: {str(result)}")
                # Continue with other analyzers unless configured to fail fast
                if self.config.get("fail_fast", False):
                    if isinstance(result, CancellationError):
                        raise result
                    raise VideoProcessingError(
                        f"Analyzer {analyzer.analyzer_id} failed: {str(result)}"
                    )
            else:
                results[analyzer.analyzer_id] = result

        return results

    async def _run_analyzers_sequential(
        self, video_data: VideoData
    ) -> Dict[str, AnalysisResult]:
        """
        Run analyzers sequentially.

        Args:
            video_data: The video data to analyze.

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results.
        """
        results: Dict[str, AnalysisResult] = {}

        for analyzer in self.analyzers:
            # Check if the analyzer is enabled
            if (
                self.config.enabled_analyzers is not None
                and analyzer.analyzer_category not in self.config.enabled_analyzers
            ):
                logger.debug(f"Skipping disabled analyzer: {analyzer.analyzer_id}")
                continue

            try:
                result = await self._run_analyzer(analyzer, video_data)
                results[analyzer.analyzer_id] = result
            except CancellationError:
                # Propagate cancellation
                raise
            except Exception as e:
                logger.error(
                    f"Error in analyzer {analyzer.analyzer_id}: {str(e)}", exc_info=True
                )
                # Continue with other analyzers unless configured to fail fast
                if self.config.get("fail_fast", False):
                    raise VideoProcessingError(
                        f"Analyzer {analyzer.analyzer_id} failed: {str(e)}"
                    )

        return results

    async def _run_analyzer(
        self, analyzer: BaseAnalyzer, video_data: VideoData
    ) -> AnalysisResult:
        """
        Run a single analyzer with progress tracking and cancellation support.

        Args:
            analyzer: The analyzer to run.
            video_data: The video data to analyze.

        Returns:
            AnalysisResult: The analysis result.

        Raises:
            CancellationError: If the analysis is cancelled.
            Exception: If the analyzer fails.
        """
        # Create a progress tracking function for this analyzer
        analyzer_weight = self.config.analyzer_weights.get(
            analyzer.analyzer_category, 1.0
        )

        def on_analyzer_progress(
            analyzer_id: str, progress: float, metadata: Dict[str, Any]
        ) -> None:
            # Update the overall progress based on this analyzer's progress
            self._update_analyzer_progress(
                analyzer_id, progress, metadata, analyzer_weight
            )

        # Create a progress callback for this analyzer if we have a main callback
        analyzer_progress_callback = None
        if self._progress_callback:
            analyzer_progress_callback = AnalysisProgressCallback(on_analyzer_progress)

        try:
            # Run the analyzer with timeout, progress tracking, and cancellation support
            result = await self._run_analyzer_with_timeout(
                analyzer, video_data, analyzer_progress_callback
            )

            # Update completion status
            self._completed_analyzers += 1
            self._update_overall_progress_from_completed()

            return result

        except Exception as e:
            # Update completion status even on failure
            self._completed_analyzers += 1
            self._update_overall_progress_from_completed()

            # Re-raise the exception
            raise

    async def _run_analyzer_with_timeout(
        self,
        analyzer: BaseAnalyzer,
        video_data: VideoData,
        progress_callback: Optional[AnalysisProgressCallback] = None,
    ) -> AnalysisResult:
        """
        Run an analyzer with a timeout, progress tracking, and cancellation support.

        Args:
            analyzer: The analyzer to run.
            video_data: The video data to analyze.
            progress_callback: Optional callback for progress updates.

        Returns:
            AnalysisResult: The analysis result.

        Raises:
            CancellationError: If the analysis is cancelled.
            TimeoutError: If the analyzer times out.
            Exception: If the analyzer fails.
        """
        try:
            # Check for cancellation before starting
            if self._cancellation_token and self._cancellation_token.is_cancelled:
                raise CancellationError()

            # Run the analyzer with progress tracking and cancellation support
            return await asyncio.wait_for(
                analyzer.analyze_with_progress(
                    video_data, progress_callback, self._cancellation_token
                ),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Analyzer {analyzer.analyzer_id} timed out after {self.config.timeout_seconds} seconds"
            )
            raise TimeoutError(
                f"Analyzer {analyzer.analyzer_id} timed out after {self.config.timeout_seconds} seconds"
            )
        except CancellationError:
            logger.warning(f"Analyzer {analyzer.analyzer_id} was cancelled")
            raise
        except Exception as e:
            logger.error(
                f"Analyzer {analyzer.analyzer_id} failed: {str(e)}", exc_info=True
            )
            raise

    def _update_analyzer_progress(
        self,
        analyzer_id: str,
        progress: float,
        metadata: Dict[str, Any],
        weight: float = 1.0,
    ) -> None:
        """
        Update the progress of an analyzer and the overall progress.

        Args:
            analyzer_id: ID of the analyzer.
            progress: Progress value between 0.0 and 1.0.
            metadata: Additional metadata about the progress.
            weight: Weight of this analyzer in the overall progress calculation.
        """
        # Forward the progress update to the main callback if available
        if self._progress_callback:
            self._progress_callback.update(analyzer_id, progress, metadata)

        # Update the overall progress
        self._update_overall_progress_from_completed()

    def _update_overall_progress_from_completed(self) -> None:
        """Update the overall progress based on the number of completed analyzers."""
        if self._total_analyzers > 0:
            progress = self._completed_analyzers / self._total_analyzers
            self._update_overall_progress(progress)

    def _update_overall_progress(
        self, progress: float, metadata: Dict[str, Any] = None
    ) -> None:
        """
        Update the overall progress of the analysis pipeline.

        Args:
            progress: Progress value between 0.0 and 1.0.
            metadata: Additional metadata about the progress.
        """
        self._overall_progress = max(0.0, min(progress, 1.0))  # Clamp to valid range

        # Forward the progress update to the main callback if available
        if self._progress_callback:
            self._progress_callback.update(
                "pipeline", self._overall_progress, metadata or {}
            )


# Example usage of the AnalysisPipeline with progress tracking and cancellation
async def example_pipeline_usage():
    """
    Example of how to use the AnalysisPipeline with progress tracking and cancellation.
    """
    from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig
    from video_analyzer.analyzers.base import AnalyzerRegistry
    import asyncio

    # Create a pipeline configuration
    config = AnalysisPipelineConfig(
        parallel_analyzers=True, timeout_seconds=60, max_retries=1
    )

    # Create a pipeline
    pipeline = AnalysisPipeline(config)

    # Register some analyzers
    for analyzer_type in AnalyzerRegistry.get_available_types():
        analyzer = AnalyzerRegistry.create(analyzer_type)
        pipeline.register_analyzer(analyzer)

    # Set up progress tracking
    def on_progress(
        analyzer_id: str, progress: float, metadata: Dict[str, Any]
    ) -> None:
        status = metadata.get("status", "in_progress")
        if analyzer_id == "pipeline":
            print(f"Overall progress: {progress:.1%} - Status: {status}")
        else:
            print(f"Analyzer {analyzer_id}: {progress:.1%} - Status: {status}")

    pipeline.set_progress_callback(on_progress)

    # Create a cancellation token
    cancellation_token = pipeline.create_cancellation_token()

    # Set up a task to cancel the analysis after 10 seconds
    async def cancel_after_delay():
        await asyncio.sleep(10)
        print("Cancelling analysis...")
        cancellation_token.cancel()

    # Run the analysis and cancellation task concurrently
    try:
        # Create a dummy video data object for the example
        from pathlib import Path
        from video_analyzer.models.video import VideoData
        import numpy as np

        video_data = VideoData(
            path=Path("example.mp4"), duration=60.0, fps=30.0, resolution=(1920, 1080)
        )

        # Run the analysis and cancellation task concurrently
        analysis_task = asyncio.create_task(pipeline.run_analysis(video_data))
        cancel_task = asyncio.create_task(cancel_after_delay())

        # Wait for the analysis to complete or be cancelled
        results = await analysis_task

        # Cancel the cancellation task if the analysis completes first
        cancel_task.cancel()

        # Process the results
        print(f"Analysis completed with {len(results)} results:")
        for analyzer_id, result in results.items():
            print(f"- {analyzer_id}: {result.confidence:.2f} confidence")

    except CancellationError:
        print("Analysis was cancelled")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
