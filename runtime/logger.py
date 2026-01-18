"""Logging utilities for Symphony Network.

This module provides centralized logging functionality for the Symphony Network
with configurable log levels and standardized formatting.
"""

import logging
import sys
from typing import Optional


# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Configure basic logging
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# Create logger instance
logger = logging.getLogger("symphony")


def set_level(level: str) -> None:
    """Set the logging level.
    
    Updates the logger level based on the provided string level.
    
    Args:
        level (str): Logging level as string. Valid options:
                    'debug', 'info', 'warning', 'error', 'critical'
                    Case-insensitive.
    """
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    
    log_level = level_map.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)
    
    # Also update the root logger level
    logging.getLogger().setLevel(log_level)


def info(msg: str) -> None:
    """Log an info level message.
    
    Args:
        msg (str): Message to log
    """
    logger.info(msg)


def debug(msg: str) -> None:
    """Log a debug level message.
    
    Args:
        msg (str): Message to log
    """
    logger.debug(msg)


def warning(msg: str) -> None:
    """Log a warning level message.
    
    Args:
        msg (str): Message to log
    """
    logger.warning(msg)


def error(msg: str) -> None:
    """Log an error level message.
    
    Args:
        msg (str): Message to log
    """
    logger.error(msg)


def critical(msg: str) -> None:
    """Log a critical level message.
    
    Args:
        msg (str): Message to log
    """
    logger.critical(msg)


def configure_logger(name: str, level: Optional[str] = None, 
                    format_str: Optional[str] = None) -> logging.Logger:
    """Configure a custom logger with specified parameters.
    
    Args:
        name (str): Name of the logger
        level (Optional[str]): Log level string. Defaults to 'info'
        format_str (Optional[str]): Custom format string. Uses default if None
        
    Returns:
        logging.Logger: Configured logger instance
    """
    custom_logger = logging.getLogger(name)
    
    if level:
        set_level(level)
    
    if format_str:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        custom_logger.handlers.clear()
        custom_logger.addHandler(handler)
    
    return custom_logger
