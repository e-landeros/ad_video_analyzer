"""
Base analyzer interface for all video analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import AnalysisResult


class BaseAnalyzer(ABC):
    """
    Base class for all analyzers.
    """

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
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this analyzer.

        Returns:
            Dict[str, Any]: The analyzer metadata.
        """
        return {
            "analyzer_id": self.analyzer_id,
            "analyzer_type": self.__class__.__name__,
        }
