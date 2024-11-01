import logging
import os
import sys
from typing import Optional

from library import __library__
from version import __version__


LOG_LEVEL = logging.INFO


def setup_logging(level: Optional[str | int] = None) -> None:
    """
    Setup logging configuration for the library.
    
    Args:
        level: Logging level. If None, defaults to INFO.
               Can be string ('DEBUG', 'INFO') or logging constant.
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), LOG_LEVEL)
    
    # Default to INFO if no level specified
    level = level or LOG_LEVEL

    # Define a console handler with the appropriate format
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt=f'[{__library__} v{__version__}] %(asctime)s - %(levelname)s - %(message)s'
        if level == logging.DEBUG else '%(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Configure the root logger
    logging.basicConfig(level=level, handlers=[console_handler])
