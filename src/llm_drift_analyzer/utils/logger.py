"""
Logging configuration for LLM Drift Analyzer.

This module provides logging setup with configurable levels,
formatters, and handlers for both file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "llm_drift_analyzer",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure the logger.

    Creates a logger with both console and optional file handlers.
    The logger uses a consistent format showing timestamp, level,
    module name, and message.

    Args:
        name: Name of the logger. Default is 'llm_drift_analyzer'.
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If provided, logs will be
            written to this file in addition to console.
        console_output: Whether to output logs to console. Default is True.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = setup_logger(level="DEBUG", log_file=Path("analysis.log"))
        >>> logger.info("Starting drift analysis")
        2024-01-15 10:30:00 | INFO | Starting drift analysis

        >>> logger = setup_logger(level="WARNING", console_output=False)
    """
    global _logger

    # Create or get logger
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create parent directories if they don't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Store global reference
    _logger = logger

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the configured logger instance.

    If the logger hasn't been set up yet, creates a default logger.
    Can optionally return a child logger with a specific name.

    Args:
        name: Optional name for a child logger. If provided, returns
            a logger with name 'llm_drift_analyzer.{name}'.

    Returns:
        logging.Logger: Logger instance.

    Example:
        >>> logger = get_logger()
        >>> logger.info("General message")

        >>> client_logger = get_logger("clients.openai")
        >>> client_logger.debug("OpenAI client initialized")
    """
    global _logger

    if _logger is None:
        _logger = setup_logger()

    if name:
        return _logger.getChild(name)

    return _logger


class LogContext:
    """
    Context manager for temporary log level changes.

    Useful for temporarily increasing verbosity during specific
    operations without affecting global logging settings.

    Attributes:
        logger: The logger instance to modify.
        new_level: The temporary log level.
        original_level: The original log level (restored on exit).

    Example:
        >>> logger = get_logger()
        >>> with LogContext(logger, "DEBUG"):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This won't be shown if level is INFO")
    """

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize log context.

        Args:
            logger: Logger instance to modify.
            level: Temporary log level to use.
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.original_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter context and set new log level."""
        self.logger.setLevel(self.new_level)
        for handler in self.logger.handlers:
            handler.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original log level."""
        self.logger.setLevel(self.original_level)
        for handler in self.logger.handlers:
            handler.setLevel(self.original_level)


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Logs the start, end, and duration of function execution
    at DEBUG level.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with timing logs.

    Example:
        >>> @log_execution_time
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> result = slow_function()
        # Logs: "Starting slow_function"
        # Logs: "Completed slow_function in 1.00s"
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__qualname__

        logger.debug(f"Starting {func_name}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Completed {func_name} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func_name} after {elapsed:.2f}s: {e}")
            raise

    return wrapper
