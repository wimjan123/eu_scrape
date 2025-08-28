"""Circuit breaker pattern implementation for API resilience."""

import time
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from threading import Lock

from .logging import get_logger
from .exceptions import APIError

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5      # Failures to open circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying recovery
    success_threshold: int = 3      # Successes to close circuit from half-open
    timeout: float = 30.0          # Request timeout for circuit breaker


class CircuitBreakerError(APIError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, service_name: str, recovery_time: float):
        self.service_name = service_name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker open for {service_name}. "
            f"Recovery expected in {recovery_time:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker implementation for API calls."""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker.
        
        Args:
            service_name: Name of the service being protected
            config: Circuit breaker configuration
        """
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        # Thread safety
        self._lock = Lock()
        
        logger.info(
            "Circuit breaker initialized",
            service_name=service_name,
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function call with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
            Other exceptions: From the wrapped function
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we should attempt recovery
            if self.state == CircuitState.OPEN:
                if current_time - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(
                        "Circuit breaker entering half-open state",
                        service_name=self.service_name
                    )
                else:
                    recovery_time = self.config.recovery_timeout - (current_time - self.last_failure_time)
                    logger.debug(
                        "Circuit breaker open, blocking request",
                        service_name=self.service_name,
                        recovery_time=recovery_time
                    )
                    raise CircuitBreakerError(self.service_name, recovery_time)
            
            # Allow limited requests in half-open state
            if self.state == CircuitState.HALF_OPEN:
                logger.debug(
                    "Circuit breaker testing recovery",
                    service_name=self.service_name,
                    success_count=self.success_count
                )
        
        # Execute the function call
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.debug(
                    "Circuit breaker success recorded",
                    service_name=self.service_name,
                    success_count=self.success_count,
                    execution_time=execution_time
                )
                
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(
                        "Circuit breaker closed - service recovered",
                        service_name=self.service_name
                    )
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on successful call
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(
                "Circuit breaker failure recorded",
                service_name=self.service_name,
                failure_count=self.failure_count,
                exception=str(exception)
            )
            
            # Open circuit if threshold exceeded
            if self.failure_count >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    logger.error(
                        "Circuit breaker opened due to failures",
                        service_name=self.service_name,
                        failure_count=self.failure_count,
                        recovery_timeout=self.config.recovery_timeout
                    )
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        with self._lock:
            current_time = time.time()
            return {
                'service_name': self.service_name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_since_failure': current_time - self.last_failure_time if self.last_failure_time > 0 else 0,
                'recovery_timeout': self.config.recovery_timeout,
                'is_healthy': self.state == CircuitState.CLOSED
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            
            logger.info(
                "Circuit breaker manually reset",
                service_name=self.service_name,
                previous_state=old_state.value
            )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize registry."""
        self._breakers = {}
        self._lock = Lock()
    
    def get_breaker(self, service_name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """
        Get or create circuit breaker for service.
        
        Args:
            service_name: Name of the service
            config: Configuration for new breakers
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = CircuitBreaker(service_name, config)
            return self._breakers[service_name]
    
    def get_all_states(self) -> dict:
        """Get states of all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_state()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry instance
circuit_registry = CircuitBreakerRegistry()