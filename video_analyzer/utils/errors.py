"""
Custom error classes for the Video Analyzer.
"""


class VideoAnalyzerError(Exception):
    """
    Base exception class for the Video Analyzer.
    """

    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class VideoFormatError(VideoAnalyzerError):
    """
    Exception raised when a video format is not supported.
    """

    def __init__(self, message: str, format: str = None):
        super().__init__(message, code="FORMAT_ERROR")
        self.format = format


class VideoSizeError(VideoAnalyzerError):
    """
    Exception raised when a video size exceeds the maximum allowed.
    """

    def __init__(self, message: str, size: int = None, max_size: int = None):
        super().__init__(message, code="SIZE_ERROR")
        self.size = size
        self.max_size = max_size


class VideoProcessingError(VideoAnalyzerError):
    """
    Exception raised when there's an error processing a video.
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="PROCESSING_ERROR")
        self.details = details or {}
