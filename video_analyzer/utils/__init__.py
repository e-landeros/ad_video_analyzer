"""
Utility functions for the video analyzer.
"""

from typing import Dict, Any, Optional


class VideoAnalyzerError(Exception):
    """
    Base exception class for the Video Analyzer.
    """

    def __init__(self, message: str, code: Optional[str] = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class VideoFormatError(VideoAnalyzerError):
    """
    Exception raised when there is an issue with the video format.
    """

    pass


class ProcessingError(VideoAnalyzerError):
    """
    Exception raised when there is an issue with video processing.
    """

    pass


class AnalysisError(VideoAnalyzerError):
    """
    Exception raised when there is an issue with video analysis.
    """

    pass


class ReportGenerationError(VideoAnalyzerError):
    """
    Exception raised when there is an issue with report generation.
    """

    pass


class ExternalServiceError(VideoAnalyzerError):
    """
    Exception raised when there is an issue with an external service.
    """

    pass
