"""Structured logging setup for EU Parliament scraper."""

import logging
import logging.config
import yaml
import structlog
from pathlib import Path
from typing import Dict, Any


def setup_logging(config_path: str = "config/logging.yaml") -> None:
    """Setup structured logging configuration."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Use simplified logging configuration to avoid structlog version issues
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'eu_scrape.log')
        ]
    )
    
    # Configure structlog with compatible processors
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()  # Use ConsoleRenderer for better compatibility
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding structured logging context."""
    
    def __init__(self, **context):
        self.context = context
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        structlog.contextvars.bind_contextvars(**self.context)
        return self.logger.bind(**self.context)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()


def log_api_request(url: str, method: str = "GET", response_time: float = None, 
                   status_code: int = None) -> Dict[str, Any]:
    """Create structured log data for API requests."""
    log_data = {
        "event": "api_request",
        "url": url,
        "method": method,
    }
    
    if response_time is not None:
        log_data["response_time"] = response_time
    if status_code is not None:
        log_data["status_code"] = status_code
    
    return log_data


def log_processing_progress(phase: str, completed: int, total: int, 
                          item_type: str = "items") -> Dict[str, Any]:
    """Create structured log data for processing progress."""
    percentage = (completed / total * 100) if total > 0 else 0
    
    return {
        "event": "processing_progress",
        "phase": phase,
        "completed": completed,
        "total": total,
        "percentage": round(percentage, 2),
        "item_type": item_type
    }