"""
Base analyzer interface for all video analyzers.

This module defines the common interface that all video analyzers must implement,
as well as utility classes for tracking progress and handling cancellation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable, Type, Union
import asyncio
import time
from datetime import datetime
import logging

from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import AnalysisResult
from video_analyzer.utils.errors import VideoAnalyzerError

# Set up logging
logger = logging.getLogger(__name__)

# Define analyzer types
ANALYZER_TYPES = {
    "hook": "Hook analysis for video introductions",
    "progression": "Video structure and pacing analysis",
    "visual": "Visual elements and quality analysis",
    "audio": "Audio quality and speech analysis",
    "object": "Object and brand detection",
    "emotion": "Mood and emotional impact analysis",
    "storytelling": "Narrative structure analysis",
}


class AnalysisProgressCallback:
    """
    Callback for tracking analysis progress.
    """

    def __init__(self, on_progress: Callable[[str, float, Dict[str, Any]], None]):
        """
        Initialize the progress callback.

        Args:
            on_progress: Function to call with progress updates.
                         Arguments: analyzer_id, progress (0.0-1.0), metadata
        """
        self.on_progress = on_progress

    def update(
        self, analyzer_id: str, progress: float, metadata: Dict[str, Any] = None
    ):
        """
        Update the progress.

        Args:
            analyzer_id: ID of the analyzer reporting progress
            progress: Progress value between 0.0 and 1.0
            metadata: Additional metadata about the progress
        """
        if not 0.0 <= progress <= 1.0:
            progress = max(0.0, min(progress, 1.0))  # Clamp to valid range

        self.on_progress(analyzer_id, progress, metadata or {})


class AnalysisCancellationToken:
    """
    Token for cancelling analysis.
    """

    def __init__(self):
        """Initialize the cancellation token."""
        self._cancelled = False
        self._callbacks = []

    def cancel(self):
        """Cancel the analysis."""
        self._cancelled = True
        for callback in self._callbacks:
            callback()

    @property
    def is_cancelled(self) -> bool:
        """Check if the analysis has been cancelled."""
        return self._cancelled

    def register_callback(self, callback: Callable[[], None]):
        """
        Register a callback to be called when the analysis is cancelled.

        Args:
            callback: Function to call when cancelled
        """
        self._callbacks.append(callback)

    def check_cancelled(self):
        """
        Check if the analysis has been cancelled and raise an exception if it has.

        Raises:
            CancellationError: If the analysis has been cancelled
        """
        if self._cancelled:
            raise CancellationError("Analysis was cancelled")


class CancellationError(VideoAnalyzerError):
    """
    Exception raised when an analysis is cancelled.
    """

    def __init__(self, message: str = "Analysis was cancelled"):
        super().__init__(message, code="CANCELLED")


class AnalysisError(VideoAnalyzerError):
    """
    Exception raised when an analysis fails.
    """

    def __init__(
        self, message: str, analyzer_id: str = None, details: Dict[str, Any] = None
    ):
        super().__init__(message, code="ANALYSIS_ERROR")
        self.analyzer_id = analyzer_id
        self.details = details or {}


class BaseAnalyzer(ABC):
    """
    Base class for all analyzers.

    This abstract class defines the common interface that all video analyzers must implement.
    It provides functionality for analyzing video data, tracking progress, handling cancellation,
    and reporting metadata about the analyzer.

    Attributes:
        _start_time: Time when the analysis started
        _end_time: Time when the analysis ended
        _config: Configuration for the analyzer
        _dependencies: Other analyzers this analyzer depends on
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the analyzer.

        Args:
            config: Optional configuration for the analyzer
        """
        self._start_time = None
        self._end_time = None
        self._config = config or {}
        self._dependencies = []
        self._result_cache = {}
        logger.debug(f"Initializing {self.__class__.__name__} analyzer")

    @abstractmethod
    async def analyze(self, video_data: VideoData) -> AnalysisResult:
        """
        Analyze the video data.

        Args:
            video_data: The video data to analyze.

        Returns:
            AnalysisResult: The analysis result.
        """
        pass

    async def analyze_with_progress(
        self,
        video_data: VideoData,
        progress_callback: Optional[AnalysisProgressCallback] = None,
        cancellation_token: Optional[AnalysisCancellationToken] = None,
    ) -> AnalysisResult:
        """
        Analyze the video data with progress tracking and cancellation support.

        Args:
            video_data: The video data to analyze
            progress_callback: Optional callback for progress updates
            cancellation_token: Optional token for cancellation

        Returns:
            AnalysisResult: The analysis result

        Raises:
            CancellationError: If the analysis is cancelled
            AnalysisError: If the analysis fails
        """
        self._start_time = time.time()
        logger.info(f"Starting analysis with {self.analyzer_id}")

        # Report initial progress
        if progress_callback:
            progress_callback.update(
                self.analyzer_id,
                0.0,
                {"status": "started", "analyzer_type": self.analyzer_type},
            )

        try:
            # Check for cancellation before starting
            if cancellation_token and cancellation_token.is_cancelled:
                raise CancellationError()

            # Check if we have dependencies and run them first
            if self._dependencies:
                await self._run_dependencies(
                    video_data, progress_callback, cancellation_token
                )

            # Create a wrapper for the analyze method that periodically checks for cancellation
            result = await self._analyze_with_cancellation_checks(
                video_data, progress_callback, cancellation_token
            )

            # Cache the result
            self._result_cache[video_data.path] = result

            # Report completion
            if progress_callback:
                progress_callback.update(
                    self.analyzer_id,
                    1.0,
                    {"status": "completed", "analyzer_type": self.analyzer_type},
                )

            logger.info(f"Analysis completed for {self.analyzer_id}")
            return result

        except CancellationError:
            # Report cancellation
            logger.warning(f"Analysis cancelled for {self.analyzer_id}")
            if progress_callback:
                progress_callback.update(
                    self.analyzer_id,
                    0.0,
                    {"status": "cancelled", "analyzer_type": self.analyzer_type},
                )
            raise
        except Exception as e:
            # Report error
            logger.error(
                f"Analysis failed for {self.analyzer_id}: {str(e)}", exc_info=True
            )
            if progress_callback:
                progress_callback.update(
                    self.analyzer_id,
                    0.0,
                    {
                        "status": "error",
                        "error": str(e),
                        "analyzer_type": self.analyzer_type,
                    },
                )
            raise AnalysisError(f"Analysis failed for {self.analyzer_id}: {str(e)}")
        finally:
            self._end_time = time.time()

    async def _analyze_with_cancellation_checks(
        self,
        video_data: VideoData,
        progress_callback: Optional[AnalysisProgressCallback],
        cancellation_token: Optional[AnalysisCancellationToken],
    ) -> AnalysisResult:
        """
        Wrapper around analyze that periodically checks for cancellation.

        Args:
            video_data: The video data to analyze
            progress_callback: Optional callback for progress updates
            cancellation_token: Optional token for cancellation

        Returns:
            AnalysisResult: The analysis result
        """
        # If no cancellation token, just call analyze directly
        if not cancellation_token:
            return await self.analyze(video_data)

        # Create a task for the analysis
        analysis_task = asyncio.create_task(self.analyze(video_data))

        # Check for cancellation every 0.5 seconds
        while not analysis_task.done():
            if cancellation_token.is_cancelled:
                analysis_task.cancel()
                try:
                    await analysis_task
                except asyncio.CancelledError:
                    pass
                raise CancellationError()

            try:
                # Wait for a short time or until the task completes
                await asyncio.wait_for(asyncio.shield(analysis_task), timeout=0.5)
            except asyncio.TimeoutError:
                # This is expected, just continue the loop
                pass

        # Task is done, get the result
        return await analysis_task

    async def _run_dependencies(
        self,
        video_data: VideoData,
        progress_callback: Optional[AnalysisProgressCallback],
        cancellation_token: Optional[AnalysisCancellationToken],
    ) -> Dict[str, AnalysisResult]:
        """
        Run all dependencies and return their results.

        Args:
            video_data: The video data to analyze
            progress_callback: Optional callback for progress updates
            cancellation_token: Optional token for cancellation

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results
        """
        results = {}
        for dependency in self._dependencies:
            # Check if the dependency has already been run for this video
            if video_data.path in dependency._result_cache:
                results[dependency.analyzer_id] = dependency._result_cache[
                    video_data.path
                ]
                continue

            # Run the dependency
            result = await dependency.analyze_with_progress(
                video_data, progress_callback, cancellation_token
            )
            results[dependency.analyzer_id] = result

        return results

    def add_dependency(self, analyzer: "BaseAnalyzer") -> None:
        """
        Add a dependency to this analyzer.

        Args:
            analyzer: The analyzer this analyzer depends on
        """
        if analyzer not in self._dependencies:
            self._dependencies.append(analyzer)
            logger.debug(
                f"Added dependency {analyzer.analyzer_id} to {self.analyzer_id}"
            )

    @property
    @abstractmethod
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID.
        """
        pass

    @property
    def analyzer_type(self) -> str:
        """
        Get the type of this analyzer.

        Returns:
            str: The analyzer type.
        """
        return self.__class__.__name__

    @property
    def analyzer_category(self) -> str:
        """
        Get the category of this analyzer.

        The category should be one of the keys in ANALYZER_TYPES.

        Returns:
            str: The analyzer category.
        """
        # Default implementation tries to infer from class name
        for category in ANALYZER_TYPES:
            if category.lower() in self.__class__.__name__.lower():
                return category
        return "unknown"

    @property
    def description(self) -> str:
        """
        Get a description of this analyzer.

        Returns:
            str: The analyzer description.
        """
        category = self.analyzer_category
        if category in ANALYZER_TYPES:
            return ANALYZER_TYPES[category]
        return f"{self.analyzer_type} analyzer"

    @property
    def supports_cancellation(self) -> bool:
        """
        Check if this analyzer supports cancellation.

        Returns:
            bool: True if cancellation is supported, False otherwise.
        """
        return True

    @property
    def supports_progress(self) -> bool:
        """
        Check if this analyzer supports progress reporting.

        Returns:
            bool: True if progress reporting is supported, False otherwise.
        """
        return False

    @property
    def required_frames(self) -> Dict[str, Any]:
        """
        Get the frame requirements for this analyzer.

        Returns:
            Dict[str, Any]: The frame requirements, which may include:
                - min_frames: Minimum number of frames required
                - frame_interval: Preferred interval between frames in seconds
                - specific_timestamps: List of specific timestamps to extract frames at
        """
        return {
            "min_frames": 1,
            "frame_interval": None,
            "specific_timestamps": None,
        }

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

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this analyzer.

        Returns:
            Dict[str, Any]: The analyzer metadata.
        """
        metadata = {
            "analyzer_id": self.analyzer_id,
            "analyzer_type": self.analyzer_type,
            "analyzer_category": self.analyzer_category,
            "description": self.description,
            "supports_cancellation": self.supports_cancellation,
            "supports_progress": self.supports_progress,
            "dependencies": [dep.analyzer_id for dep in self._dependencies],
        }

        if self._start_time is not None:
            metadata["last_run"] = datetime.fromtimestamp(self._start_time).isoformat()

        if self.execution_time is not None:
            metadata["execution_time"] = self.execution_time

        return metadata

    @classmethod
    def get_result_type(cls) -> Type[AnalysisResult]:
        """
        Get the type of result this analyzer produces.

        Returns:
            Type[AnalysisResult]: The result type.
        """
        return AnalysisResult


class AnalyzerRegistry:
    """
    Registry for analyzer classes.

    This class maintains a registry of analyzer classes and provides methods for
    creating analyzers by type.
    """

    _registry: Dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(
        cls, analyzer_type: str
    ) -> Callable[[Type[BaseAnalyzer]], Type[BaseAnalyzer]]:
        """
        Decorator for registering analyzer classes.

        Args:
            analyzer_type: The type of analyzer to register

        Returns:
            A decorator function
        """

        def decorator(analyzer_class: Type[BaseAnalyzer]) -> Type[BaseAnalyzer]:
            cls._registry[analyzer_type] = analyzer_class
            logger.debug(
                f"Registered analyzer class {analyzer_class.__name__} for type {analyzer_type}"
            )
            return analyzer_class

        return decorator

    @classmethod
    def create(cls, analyzer_type: str, config: Dict[str, Any] = None) -> BaseAnalyzer:
        """
        Create an analyzer of the specified type.

        Args:
            analyzer_type: The type of analyzer to create
            config: Optional configuration for the analyzer

        Returns:
            BaseAnalyzer: The created analyzer

        Raises:
            ValueError: If the analyzer type is not registered
        """
        if analyzer_type not in cls._registry:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

        analyzer_class = cls._registry[analyzer_type]
        return analyzer_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Get a list of available analyzer types.

        Returns:
            List[str]: The available analyzer types
        """
        return list(cls._registry.keys())

    @classmethod
    def get_analyzer_class(cls, analyzer_type: str) -> Type[BaseAnalyzer]:
        """
        Get the analyzer class for the specified type.

        Args:
            analyzer_type: The type of analyzer

        Returns:
            Type[BaseAnalyzer]: The analyzer class

        Raises:
            ValueError: If the analyzer type is not registered
        """
        if analyzer_type not in cls._registry:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

        return cls._registry[analyzer_type]


# Note: The AnalysisPipelineOrchestrator class has been deprecated in favor of
# the more comprehensive AnalysisPipeline class in video_analyzer/services/pipeline.py.
# This class is kept here for backward compatibility but will be removed in a future version.
class AnalysisPipelineOrchestrator:
    """
    Orchestrates the analysis pipeline.

    DEPRECATED: Use AnalysisPipeline from video_analyzer.services.pipeline instead.

    This class manages the execution of multiple analyzers on a video, handling
    dependencies, progress tracking, and cancellation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Optional configuration for the orchestrator
        """
        import warnings
        from video_analyzer.services.pipeline import AnalysisPipeline
        from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig

        warnings.warn(
            "AnalysisPipelineOrchestrator is deprecated. Use AnalysisPipeline from "
            "video_analyzer.services.pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Create a proper config object if a dict was provided
        if config is not None and not isinstance(config, AnalysisPipelineConfig):
            config = AnalysisPipelineConfig(**config)

        # Create an AnalysisPipeline instance that we'll delegate to
        self._pipeline = AnalysisPipeline(config)
        self.analyzers: List[BaseAnalyzer] = []
        logger.debug("Initializing AnalysisPipelineOrchestrator (deprecated)")

    def add_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """
        Add an analyzer to the pipeline.

        Args:
            analyzer: The analyzer to add
        """
        self.analyzers.append(analyzer)
        self._pipeline.register_analyzer(analyzer)
        logger.debug(f"Added analyzer {analyzer.analyzer_id} to pipeline")

    def remove_analyzer(self, analyzer_id: str) -> None:
        """
        Remove an analyzer from the pipeline.

        Args:
            analyzer_id: The ID of the analyzer to remove
        """
        self.analyzers = [a for a in self.analyzers if a.analyzer_id != analyzer_id]
        # Note: AnalysisPipeline doesn't support removing analyzers, so we recreate it
        from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig

        config = self._pipeline.config
        self._pipeline = AnalysisPipeline(config)
        for analyzer in self.analyzers:
            self._pipeline.register_analyzer(analyzer)
        logger.debug(f"Removed analyzer {analyzer_id} from pipeline")

    def set_progress_callback(self, callback: AnalysisProgressCallback) -> None:
        """
        Set the progress callback.

        Args:
            callback: The progress callback
        """
        if isinstance(callback, AnalysisProgressCallback):
            self._pipeline.set_progress_callback(callback.on_progress)
        else:
            self._pipeline.set_progress_callback(callback)

    def set_cancellation_token(self, token: AnalysisCancellationToken) -> None:
        """
        Set the cancellation token.

        Args:
            token: The cancellation token
        """
        self._pipeline.create_cancellation_token = lambda: token

    async def run_analysis(self, video_data: VideoData) -> Dict[str, AnalysisResult]:
        """
        Run all analyzers on the video data.

        Args:
            video_data: The video data to analyze

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results
        """
        return await self._pipeline.run_analysis(video_data)

    @property
    def execution_time(self) -> float:
        """
        Get the execution time of the last analysis in seconds.

        Returns:
            float: The execution time, or 0 if no analysis has been run
        """
        return self._pipeline.execution_time or 0

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the orchestrator.

        Returns:
            Dict[str, Any]: The orchestrator metadata
        """
        return {
            "analyzers": [a.analyzer_id for a in self.analyzers],
            "execution_time": self.execution_time,
            "config": self._pipeline.config.dict()
            if hasattr(self._pipeline.config, "dict")
            else self._pipeline.config,
        }


# Example of how to implement and register an analyzer
@AnalyzerRegistry.register("example")
class ExampleAnalyzer(BaseAnalyzer):
    """
    Example analyzer implementation.

    This is a simple example of how to implement an analyzer.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    async def analyze(self, video_data: VideoData) -> AnalysisResult:
        """
        Analyze the video data.

        Args:
            video_data: The video data to analyze

        Returns:
            AnalysisResult: The analysis result
        """
        # Simulate some analysis work
        await asyncio.sleep(1)

        # Create and return a result
        return AnalysisResult(
            analyzer_id=self.analyzer_id,
            confidence=0.9,
            data={"example": "This is an example analysis result"},
            video_id=str(video_data.path),
        )

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "example_analyzer"

    @property
    def supports_progress(self) -> bool:
        """
        Check if this analyzer supports progress reporting.

        Returns:
            bool: True if progress reporting is supported, False otherwise
        """
        return True
