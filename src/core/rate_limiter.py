"""Rate limiting utilities for EU Parliament scraper."""

import time
import random
from typing import Optional
from .exceptions import RateLimitExceededError
import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Rate limiter to respect API service limits."""
    
    def __init__(self, requests_per_second: float, max_burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            max_burst: Maximum burst requests allowed
        """
        self.requests_per_second = requests_per_second
        self.max_burst = max_burst
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Optional[float] = None
        self.request_count = 0
        
        logger.info(
            "Rate limiter initialized",
            requests_per_second=requests_per_second,
            min_interval_seconds=self.min_interval
        )
    
    def wait_if_needed(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        current_time = time.time()
        
        if self.last_request_time is None:
            self.last_request_time = current_time
            self.request_count += 1
            return
        
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            # Add small random jitter to avoid thundering herd
            jitter = random.uniform(0, min(0.1, sleep_time * 0.1))
            total_sleep = sleep_time + jitter
            
            logger.debug(
                "Rate limiting: sleeping",
                sleep_time=total_sleep,
                time_since_last=time_since_last,
                min_interval=self.min_interval
            )
            
            time.sleep(total_sleep)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self.last_request_time = None
        self.request_count = 0
        logger.debug("Rate limiter reset")


class ExponentialBackoff:
    """Exponential backoff for handling failures."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, max_retries: int = 5):
        """
        Initialize exponential backoff.
        
        Args:
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            max_retries: Maximum number of retries
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.attempt = 0
    
    def wait(self) -> bool:
        """
        Wait for exponential backoff delay.
        
        Returns:
            True if should continue (under max_retries), False otherwise
        """
        if self.attempt >= self.max_retries:
            return False
        
        if self.attempt > 0:  # Don't wait on first attempt
            delay = min(self.base_delay * (2 ** (self.attempt - 1)), self.max_delay)
            # Add jitter to avoid thundering herd
            jitter = random.uniform(0.1, 0.3) * delay
            total_delay = delay + jitter
            
            logger.info(
                "Exponential backoff delay",
                attempt=self.attempt,
                delay_seconds=total_delay,
                max_retries=self.max_retries
            )
            
            time.sleep(total_delay)
        
        self.attempt += 1
        return True
    
    def reset(self) -> None:
        """Reset backoff state."""
        self.attempt = 0
        logger.debug("Exponential backoff reset")