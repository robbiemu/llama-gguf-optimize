import logging
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
        level = getattr(logging, level.upper())
    
    # Default to INFO if no level specified
    level = level or LOG_LEVEL
    
    # Setup console handler with format based on level
    console_handler = logging.StreamHandler(sys.stdout)
    
    if level == logging.DEBUG:
        formatter = logging.Formatter(
            fmt=f'[{__library__} v{__version__}]' + '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    
    # Setup root logger for the library
    root_logger = logging.getLogger('src')
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Prevent propagation to Python's root logger
    root_logger.propagate = False
