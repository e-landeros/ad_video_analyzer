"""
Error handling utilities for the Video Analyzer.

This module provides functions and decorators for graceful error handling
and recovery throughout the application.
"""

import sys
import time
import logging
import functools
import traceback
from typing import Callable, Any, Dict, Optional, Type, List, Union, TypeVar, cast

from video_analyzer.utils.errors import (
    VideoAnalyzerError,
    VideoProcessingError,
    AnalysisError,
    ExternalServiceError,
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


def handle_errors(
    fallback_value: Any = None,
    reraise: bool = False,
    log_level: int = logging.ERROR,
    error_types: Optional[List[Type[Exception]]] = None,
    error_handler: Optional[Callable[[Exception], Any]] = None,
) -> Callable:
    """
    Decorator for handling exceptions in functions.

    Args:
        fallback_value: Value to return if an exception occurs
        reraise: Whether to re-raise the exception after handling
        log_level: Logging level for the error
        error_types: List of exception types to catch (defaults to Exception)
        error_handler: Optional function to call with the exception

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Only handle specified error types if provided
                if error_types and not any(isinstance(e, t) for t in error_types):
                    raise

                # Log the error
                logger.log(
                    log_level, f"Error in {func.__name__}: {str(e)}", exc_info=True
                )

                # Call error handler if provided
                if error_handler:
                    error_handler(e)

                # Re-raise if specified
                if reraise:
                    raise

                # Return fallback value
                return cast(T, fallback_value)

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    error_types: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator for retrying functions that may fail temporarily.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Factor by which the delay increases with each attempt
        error_types: List of exception types to retry on (defaults to Exception)
        on_retry: Optional callback function called before each retry

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            _error_types = error_types or [Exception]
            _delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(_error_types) as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(attempt, e)
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {_delay:.2f}s..."
                        )
                        time.sleep(_delay)
                        _delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
                        raise

            # This should never happen, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry logic")

        return wrapper

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    fallback_value: Any = None,
    error_message: str = "Function execution failed",
    **kwargs: Any,
) -> T:
    """
    Safely execute a function and handle any exceptions.

    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        fallback_value: Value to return if an exception occurs
        error_message: Message to log if an exception occurs
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}", exc_info=True)
        return cast(T, fallback_value)


def convert_exception(
    from_exception: Type[Exception],
    to_exception: Type[VideoAnalyzerError],
    message: Optional[str] = None,
    **kwargs: Any,
) -> Callable:
    """
    Decorator to convert external exceptions to VideoAnalyzer exceptions.

    Args:
        from_exception: Exception type to convert
        to_exception: VideoAnalyzerError type to convert to
        message: Optional message for the new exception
        **kwargs: Additional keyword arguments for the new exception

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **func_kwargs: Any) -> T:
            try:
                return func(*args, **func_kwargs)
            except from_exception as e:
                error_msg = message or str(e)
                raise to_exception(error_msg, cause=e, **kwargs)

        return wrapper

    return decorator


def create_error_context(
    error: Exception, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a context dictionary for an error with useful debugging information.

    Args:
        error: The exception that occurred
        context: Additional context information

    Returns:
        Dict[str, Any]: Error context dictionary
    """
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }

    # Add custom context if provided
    if context:
        error_context.update(context)

    return error_context


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is potentially recoverable.

    Args:
        error: The exception to check

    Returns:
        bool: True if the error is potentially recoverable
    """
    # Check error type by name to handle both built-in and custom exceptions
    error_type_name = type(error).__name__

    if error_type_name == "TimeoutError":
        return True

    if error_type_name == "ConnectionError":
        return True

    if error_type_name == "ExternalServiceError" or isinstance(
        error, ExternalServiceError
    ):
        return True

    return False


def get_error_recovery_strategy(error: Exception) -> Dict[str, Any]:
    """
    Get a recovery strategy for a given error.

    Args:
        error: The exception to get a recovery strategy for

    Returns:
        Dict[str, Any]: Recovery strategy information
    """
    # Default strategy
    strategy = {
        "recoverable": False,
        "retry": False,
        "retry_delay": 0,
        "max_retries": 0,
        "fallback_available": False,
        "message": "This error cannot be automatically recovered from.",
    }

    # Check error type by name to handle both built-in and custom exceptions
    error_type_name = type(error).__name__

    # Update strategy based on error type
    if error_type_name == "ConnectionError":
        strategy.update(
            {
                "recoverable": True,
                "retry": True,
                "retry_delay": 2,
                "max_retries": 3,
                "message": "Connection error - will retry automatically.",
            }
        )
    elif error_type_name == "TimeoutError":
        strategy.update(
            {
                "recoverable": True,
                "retry": True,
                "retry_delay": 5,
                "max_retries": 2,
                "message": "Timeout error - will retry with increased timeout.",
            }
        )
    elif error_type_name == "ExternalServiceError" or isinstance(
        error, ExternalServiceError
    ):
        strategy.update(
            {
                "recoverable": True,
                "retry": True,
                "retry_delay": 3,
                "max_retries": 3,
                "fallback_available": True,
                "message": "External service error - will retry or use fallback.",
            }
        )

    return strategy
