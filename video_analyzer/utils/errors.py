"""
Custom error classes for the Video Analyzer.

This module defines a hierarchy of exception classes used throughout the Video Analyzer
to provide consistent error handling and reporting. Each exception class includes
relevant context information to help diagnose and resolve issues.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Union

# Get logger for this module
logger = logging.getLogger(__name__)


class VideoAnalyzerError(Exception):
    """
    Base exception class for the Video Analyzer.

    All other error classes in the application should inherit from this class
    to ensure consistent error handling and logging.
    """

    def __init__(
        self,
        message: str,
        code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.cause = cause

        # Log the error when it's created
        self._log_error()

        super().__init__(self.message)

    def _log_error(self) -> None:
        """Log the error with appropriate level and details."""
        log_message = f"{self.__class__.__name__}: {self.message} (code: {self.code})"

        if self.details:
            log_message += f" - Details: {self.details}"

        if self.cause:
            log_message += (
                f" - Caused by: {type(self.cause).__name__}: {str(self.cause)}"
            )
            logger.error(log_message, exc_info=self.cause)
        else:
            logger.error(log_message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the error
        """
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
        }

        if self.details:
            result["details"] = self.details

        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        return result


class VideoFormatError(VideoAnalyzerError):
    """
    Exception raised when a video format is not supported.
    """

    def __init__(
        self,
        message: str,
        format: str = None,
        supported_formats: List[str] = None,
        cause: Exception = None,
    ):
        details = {}
        if format:
            details["format"] = format
        if supported_formats:
            details["supported_formats"] = supported_formats

        super().__init__(message, code="FORMAT_ERROR", details=details, cause=cause)
        self.format = format
        self.supported_formats = supported_formats


class VideoSizeError(VideoAnalyzerError):
    """
    Exception raised when a video size exceeds the maximum allowed.
    """

    def __init__(
        self,
        message: str,
        size: int = None,
        max_size: int = None,
        cause: Exception = None,
    ):
        details = {}
        if size is not None:
            details["size"] = size
        if max_size is not None:
            details["max_size"] = max_size

        super().__init__(message, code="SIZE_ERROR", details=details, cause=cause)
        self.size = size
        self.max_size = max_size


class VideoProcessingError(VideoAnalyzerError):
    """
    Exception raised when there's an error processing a video.

    This can include issues with reading, decoding, or manipulating video files.
    """

    def __init__(
        self,
        message: str,
        video_path: str = None,
        operation: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if video_path:
            error_details["video_path"] = video_path
        if operation:
            error_details["operation"] = operation

        super().__init__(
            message, code="PROCESSING_ERROR", details=error_details, cause=cause
        )
        self.video_path = video_path
        self.operation = operation


class FrameExtractionError(VideoProcessingError):
    """
    Exception raised when there's an error extracting frames from a video.
    """

    def __init__(
        self,
        message: str,
        video_path: str = None,
        frame_index: Union[int, str] = None,
        strategy: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if frame_index is not None:
            error_details["frame_index"] = frame_index
        if strategy:
            error_details["strategy"] = strategy

        super().__init__(
            message,
            video_path=video_path,
            operation="frame_extraction",
            details=error_details,
            cause=cause,
        )
        self.frame_index = frame_index
        self.strategy = strategy


class AnalysisError(VideoAnalyzerError):
    """
    Exception raised when an analysis fails.
    """

    def __init__(
        self,
        message: str,
        analyzer_id: str = None,
        video_path: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if analyzer_id:
            error_details["analyzer_id"] = analyzer_id
        if video_path:
            error_details["video_path"] = video_path

        super().__init__(
            message, code="ANALYSIS_ERROR", details=error_details, cause=cause
        )
        self.analyzer_id = analyzer_id
        self.video_path = video_path


class TimeoutError(AnalysisError):
    """
    Exception raised when an analysis operation times out.
    """

    def __init__(
        self,
        message: str,
        analyzer_id: str = None,
        timeout_seconds: int = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message, analyzer_id=analyzer_id, details=error_details, cause=cause
        )
        self.timeout_seconds = timeout_seconds


class ReportGenerationError(VideoAnalyzerError):
    """
    Exception raised when report generation fails.
    """

    def __init__(
        self,
        message: str,
        report_type: str = None,
        output_path: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if report_type:
            error_details["report_type"] = report_type
        if output_path:
            error_details["output_path"] = output_path

        super().__init__(
            message, code="REPORT_ERROR", details=error_details, cause=cause
        )
        self.report_type = report_type
        self.output_path = output_path


class ConfigurationError(VideoAnalyzerError):
    """
    Exception raised when there's an error in the configuration.
    """

    def __init__(
        self,
        message: str,
        config_section: str = None,
        config_key: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if config_section:
            error_details["config_section"] = config_section
        if config_key:
            error_details["config_key"] = config_key

        super().__init__(
            message, code="CONFIG_ERROR", details=error_details, cause=cause
        )
        self.config_section = config_section
        self.config_key = config_key


class ExternalServiceError(VideoAnalyzerError):
    """
    Exception raised when an external service call fails.

    This can include API calls, model inference, or other external dependencies.
    """

    def __init__(
        self,
        message: str,
        service: str = None,
        operation: str = None,
        status_code: Optional[int] = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if service:
            error_details["service"] = service
        if operation:
            error_details["operation"] = operation
        if status_code is not None:
            error_details["status_code"] = status_code

        super().__init__(
            message, code="EXTERNAL_SERVICE_ERROR", details=error_details, cause=cause
        )
        self.service = service
        self.operation = operation
        self.status_code = status_code


class ResourceError(VideoAnalyzerError):
    """
    Exception raised when there's an issue with system resources.

    This can include memory issues, disk space, or other resource constraints.
    """

    def __init__(
        self,
        message: str,
        resource_type: str = None,
        required: Union[int, str] = None,
        available: Union[int, str] = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
    ):
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if required is not None:
            error_details["required"] = required
        if available is not None:
            error_details["available"] = available

        super().__init__(
            message, code="RESOURCE_ERROR", details=error_details, cause=cause
        )
        self.resource_type = resource_type
        self.required = required
        self.available = available
