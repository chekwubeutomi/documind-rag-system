"""
Centralized logging configuration module.

This module sets up structured logging for the entire DocMind RAG application.
It configures both file and console logging with automatic log rotation to prevent
disk space issues from accumulating log files.

Logging Architecture:
- Root Logger: Captures all logs under "docmind" namespace
- File Handler: Writes logs to a rotating file (archives when it reaches 10MB)
- Console Handler: Outputs logs to the terminal/console in real-time
- Formatter: Standardizes log format with timestamp, logger name, level, and message

This centralized setup ensures consistent logging across all modules without
requiring configuration in each file.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime

from app.core.config import settings


def setup_logging() -> logging.Logger:
    """
    Configure and initialize the logging system for the entire application.
    
    This function sets up structured logging with two outputs:
    1. Rotating File Handler - Writes to disk with automatic rotation
    2. Console Handler - Real-time output to terminal
    
    Key Features:
    - Automatic log directory creation if it doesn't exist
    - Log rotation when files reach 10MB (prevents disk space issues)
    - Keeps up to 5 old log files for historical reference
    - Consistent timestamp and format across all logs
    - Configurable log level from settings (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: A configured logger instance named "docmind" that can be
                       imported and used throughout the application
    
    Example Usage:
        from app.core.logging import logger
        logger.info("Application started")  # Goes to both file and console
        logger.error("An error occurred", exc_info=True)  # Includes stack trace
    
    How Log Rotation Works:
        - Log file starts as app.log
        - When it reaches 10MB, it's renamed to app.log.1
        - Previous backups rotate: app.log.1 → app.log.2, etc.
        - Only 5 backup files are kept; older ones are deleted
        - New app.log is created and logging continues
    """
    
    # ============================================================================
    # STEP 1: Create logs directory
    # Extract the directory path from the full log file path and create it
    # ============================================================================
    log_dir = Path(settings.LOG_FILE).parent  # Get parent directory (./logs)
    log_dir.mkdir(parents=True, exist_ok=True)  # Create directory with parents
    # mkdir(parents=True) = create parent dirs if needed
    # exist_ok=True = don't error if directory already exists
    
    # ============================================================================
    # STEP 2: Configure the root logger
    # Get or create a logger with namespace "docmind"
    # ============================================================================
    logger = logging.getLogger("docmind")  # All logs in app use this namespace
    # getattr converts string "INFO" to logging.INFO (20)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # ============================================================================
    # STEP 3: Clear existing handlers
    # If this function is called multiple times, remove old handlers first
    # Prevents duplicate log messages
    # ============================================================================
    logger.handlers = []  # Reset handlers list to empty
    
    # ============================================================================
    # STEP 4: Define log message format
    # The format string controls what information appears in each log line
    # ============================================================================
    formatter = logging.Formatter(
        # Format string with placeholders:
        # %(asctime)s   = timestamp (e.g., "2024-01-15 14:30:45")
        # %(name)s      = logger name (e.g., "docmind.services")
        # %(levelname)s = log level (e.g., "INFO", "ERROR")
        # %(message)s   = actual log message
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # datefmt controls only the timestamp format, not the full asctime format
        datefmt='%Y-%m-%d %H:%M:%S'  # YYYY-MM-DD HH:MM:SS format
    )
    
    # ============================================================================
    # STEP 5: Configure File Handler with Rotation
    # Writes logs to disk with automatic rotation when file gets too large
    # ============================================================================
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE,  # Path where logs are written (e.g., ./logs/app.log)
        maxBytes=10485760,  # 10MB in bytes (1024*1024*10 = 10485760)
        # When log file reaches 10MB, it's automatically rotated
        backupCount=5  # Keep 5 old backup files (app.log.1, app.log.2, ..., app.log.5)
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    file_handler.setFormatter(formatter)  # Apply the format defined above
    logger.addHandler(file_handler)  # Attach this handler to the logger
    
    # ============================================================================
    # STEP 6: Configure Console Handler
    # Writes logs to the terminal/console for real-time monitoring
    # ============================================================================
    console_handler = logging.StreamHandler()  # Defaults to sys.stderr
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    console_handler.setFormatter(formatter)  # Use same format as file handler
    logger.addHandler(console_handler)  # Attach to the logger
    
    # ============================================================================
    # STEP 7: Return the configured logger
    # ============================================================================
    return logger


# ============================================================================
# Initialize the global logger instance
# This module-level variable is imported and used throughout the app
# ============================================================================
# When this module is imported, setup_logging() is called automatically
# This ensures logging is configured before any other code runs
logger = setup_logging()  # Global logger instance used everywhere in the app

# Usage example in other modules:
# 
# from app.core.logging import logger
# 
# logger.debug("Detailed debug information")  # Lowest level
# logger.info("General information")
# logger.warning("Something unexpected happened")
# logger.error("An error occurred", exc_info=True)  # Shows stack trace
# logger.critical("Critical system failure")
