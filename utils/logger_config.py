"""Logging configuration for WhisperDeck."""
import logging
import os
import coloredlogs
from dotenv import load_dotenv

def setup_logger():
    """
    Configure logging with colored output and appropriate log level from .env.
    """
    load_dotenv()
    
    # Get log level from environment or default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Configure colored logging
    coloredlogs.install(
        level=log_level,
        logger=logger,
        fmt='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    return logger