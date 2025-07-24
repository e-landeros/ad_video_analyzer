"""
Tests for the error handling utilities.
"""

import sys
import time
import pytest
import logging
from unittest.mock import MagicMock, patch

from video_analyzer.utils.error_handling import (
    handle_errors,
    retry,
    safe_execute,
    convert_exception,
    create_error_context,
    is_recoverable_error,
    get_error_recovery_strategy,
)
from video_analyzer.utils.errors import (
    VideoAnalyzerError,
    VideoProcessingError,
    AnalysisError,
    ExternalServiceError,
    TimeoutError,
    ConfigurationError,
)


# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestErrorHandling:
    """Tests for error handling utilities."""

    def test_handle_errors_decorator(self):
        """Test the handle_errors decorator."""

        # Test function that raises an exception
        @handle_errors(fallback_value="fallback")
        def failing_function():
            raise ValueError("Test error")

        # Test function that succeeds
        @handle_errors(fallback_value="fallback")
        def successful_function():
            return "success"

        # Test with reraise=True
        @handle_errors(fallback_value="fallback", reraise=True)
        def reraise_function():
            raise ValueError("Test error")

        # Test with specific error types
        @handle_errors(fallback_value="fallback", error_types=[ValueError])
        def specific_error_function():
            raise ValueError("Test error")

        @handle_errors(fallback_value="fallback", error_types=[TypeError])
        def wrong_error_function():
            raise ValueError("Test error")

        # Test with error handler
        error_handler = MagicMock()

        @handle_errors(fallback_value="fallback", error_handler=error_handler)
        def handler_function():
            raise ValueError("Test error")

        # Run tests
        assert failing_function() == "fallback"
        assert successful_function() == "success"
        assert specific_error_function() == "fallback"

        with pytest.raises(ValueError):
            reraise_function()

        with pytest.raises(ValueError):
            wrong_error_function()

        assert handler_function() == "fallback"
        error_handler.assert_called_once()

    def test_retry_decorator(self):
        """Test the retry decorator."""
        mock_func = MagicMock()
        mock_func.side_effect = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            "success",
        ]

        on_retry = MagicMock()

        @retry(max_attempts=3, delay=0.01, error_types=[ValueError], on_retry=on_retry)
        def retried_function():
            return mock_func()

        # Function should succeed on the third attempt
        result = retried_function()
        assert result == "success"
        assert mock_func.call_count == 3
        assert on_retry.call_count == 2

        # Test with max_attempts exceeded
        mock_func.reset_mock()
        mock_func.side_effect = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            ValueError("Error 3"),
            "success",
        ]

        @retry(max_attempts=2, delay=0.01)
        def failed_retry_function():
            return mock_func()

        with pytest.raises(ValueError):
            failed_retry_function()

        assert mock_func.call_count == 2

    def test_safe_execute(self):
        """Test the safe_execute function."""

        # Test with a function that succeeds
        def success_func():
            return "success"

        result = safe_execute(success_func)
        assert result == "success"

        # Test with a function that fails
        def fail_func():
            raise ValueError("Test error")

        result = safe_execute(fail_func, fallback_value="fallback")
        assert result == "fallback"

        # Test with arguments
        def arg_func(a, b, c=0):
            return a + b + c

        result = safe_execute(arg_func, 1, 2, c=3)
        assert result == 6

    def test_convert_exception(self):
        """Test the convert_exception decorator."""

        @convert_exception(ValueError, VideoProcessingError, "Converted error")
        def convert_function():
            raise ValueError("Original error")

        with pytest.raises(VideoProcessingError) as excinfo:
            convert_function()

        assert "Converted error" in str(excinfo.value)
        assert isinstance(excinfo.value.cause, ValueError)

        # Test with additional kwargs
        @convert_exception(ValueError, AnalysisError, analyzer_id="test_analyzer")
        def convert_with_kwargs():
            raise ValueError("Original error")

        with pytest.raises(AnalysisError) as excinfo:
            convert_with_kwargs()

        assert excinfo.value.analyzer_id == "test_analyzer"

    def test_create_error_context(self):
        """Test the create_error_context function."""
        error = ValueError("Test error")
        context = create_error_context(error, {"custom": "value"})

        assert context["error_type"] == "ValueError"
        assert context["error_message"] == "Test error"
        assert "traceback" in context
        assert context["custom"] == "value"

    def test_is_recoverable_error(self):
        """Test the is_recoverable_error function."""
        # Test recoverable errors
        assert is_recoverable_error(TimeoutError("Timeout"))
        assert is_recoverable_error(ConnectionError("Connection failed"))
        assert is_recoverable_error(ExternalServiceError("Service error"))

        # Test non-recoverable errors
        assert not is_recoverable_error(ValueError("Value error"))
        assert not is_recoverable_error(ConfigurationError("Config error"))

    def test_get_error_recovery_strategy(self):
        """Test the get_error_recovery_strategy function."""
        # Test ConnectionError strategy
        strategy = get_error_recovery_strategy(ConnectionError("Connection failed"))
        assert strategy["recoverable"] is True
        assert strategy["retry"] is True
        assert strategy["max_retries"] == 3

        # Test TimeoutError strategy
        strategy = get_error_recovery_strategy(TimeoutError("Timeout"))
        assert strategy["recoverable"] is True
        assert strategy["retry"] is True
        assert strategy["retry_delay"] == 5

        # Test ExternalServiceError strategy
        strategy = get_error_recovery_strategy(ExternalServiceError("Service error"))
        assert strategy["recoverable"] is True
        assert strategy["fallback_available"] is True

        # Test non-recoverable error
        strategy = get_error_recovery_strategy(ValueError("Value error"))
        assert strategy["recoverable"] is False
        assert strategy["retry"] is False
