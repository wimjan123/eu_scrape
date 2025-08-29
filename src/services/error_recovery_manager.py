#!/usr/bin/env python3
"""
Error Recovery and Retry Management System

Intelligent error recovery with classification, retry strategies, and automatic
recovery routing. Integrates with session management and checkpoint systems
for comprehensive fault tolerance.

Key Features:
- Error classification with severity levels and recovery strategies
- Exponential backoff with jitter and configurable limits
- Circuit breaker pattern for failing services
- Recovery routing with fallback mechanisms
- Integration with checkpoint system for state recovery
- Metrics collection and alerting
- Background recovery tasks for async error handling
"""

import asyncio
import json
import time
import logging
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set, Type
from enum import Enum
import random
import traceback
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps, partial

from ..core.logging import get_logger
from ..core.database import get_connection
from ..core.cache import get_cache_manager
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and routing"""
    CRITICAL = "critical"      # System-threatening, immediate intervention
    HIGH = "high"             # Service degradation, needs attention
    MEDIUM = "medium"         # Recoverable, automatic retry appropriate
    LOW = "low"               # Minor issues, background recovery
    INFO = "info"             # Informational, no recovery needed


class ErrorCategory(Enum):
    """Error categories for specialized handling"""
    NETWORK = "network"             # Connection, timeout, DNS issues
    AUTHENTICATION = "auth"         # Auth failures, token expiry
    RATE_LIMIT = "rate_limit"      # API rate limiting
    DATA_CORRUPTION = "data_corruption"  # Invalid or corrupted data
    DEPENDENCY = "dependency"       # External service failures
    RESOURCE = "resource"          # Memory, disk, CPU constraints
    CONFIGURATION = "config"        # Config errors, missing settings
    VALIDATION = "validation"       # Data validation failures
    PERMISSION = "permission"       # Access control issues
    UNKNOWN = "unknown"            # Unclassified errors


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_SERVICE = "fallback_service"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SKIP_AND_CONTINUE = "skip_and_continue"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorClassification:
    """Error classification result"""
    severity: ErrorSeverity
    category: ErrorCategory
    strategy: RecoveryStrategy
    max_retries: int
    backoff_factor: float
    timeout_seconds: float
    requires_manual: bool = False
    fallback_available: bool = False
    checkpoint_recovery: bool = False


@dataclass
class RetryConfig:
    """Retry configuration for different error types"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 300.0
    backoff_factor: float = 2.0
    jitter_factor: float = 0.1
    timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    enabled: bool = True


@dataclass
class ErrorEvent:
    """Individual error event"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    classification: ErrorClassification
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False
    resolution_method: Optional[str] = None


@dataclass
class RecoveryAttempt:
    """Individual recovery attempt"""
    attempt_id: str
    error_id: str
    strategy: RecoveryStrategy
    timestamp: datetime
    success: bool
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    service_name: str
    state: CircuitState
    failure_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    state_change_time: datetime
    half_open_calls: int = 0


class ErrorClassifier:
    """Classifies errors and determines recovery strategies"""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        
    def _build_classification_rules(self) -> Dict[str, ErrorClassification]:
        """Build error classification rules"""
        return {
            # Network errors
            'ConnectionError': ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=5,
                backoff_factor=2.0,
                timeout_seconds=30.0
            ),
            'TimeoutError': ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                strategy=RecoveryStrategy.LINEAR_BACKOFF,
                max_retries=3,
                backoff_factor=1.5,
                timeout_seconds=45.0
            ),
            'requests.exceptions.ConnectionError': ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=5,
                backoff_factor=2.0,
                timeout_seconds=30.0
            ),
            
            # Authentication errors
            'AuthenticationError': ErrorClassification(
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.AUTHENTICATION,
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                max_retries=1,
                backoff_factor=1.0,
                timeout_seconds=10.0,
                requires_manual=True
            ),
            'TokenExpiredError': ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.AUTHENTICATION,
                strategy=RecoveryStrategy.IMMEDIATE_RETRY,
                max_retries=2,
                backoff_factor=1.0,
                timeout_seconds=15.0
            ),
            
            # Rate limiting
            'RateLimitError': ErrorClassification(
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.RATE_LIMIT,
                strategy=RecoveryStrategy.LINEAR_BACKOFF,
                max_retries=10,
                backoff_factor=1.0,
                timeout_seconds=60.0
            ),
            
            # Data corruption
            'DataCorruptionError': ErrorClassification(
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATA_CORRUPTION,
                strategy=RecoveryStrategy.CHECKPOINT_RESTORE,
                max_retries=1,
                backoff_factor=1.0,
                timeout_seconds=30.0,
                checkpoint_recovery=True
            ),
            
            # Resource constraints
            'MemoryError': ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.RESOURCE,
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_retries=1,
                backoff_factor=1.0,
                timeout_seconds=10.0
            ),
            'DiskSpaceError': ErrorClassification(
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.RESOURCE,
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                max_retries=0,
                backoff_factor=1.0,
                timeout_seconds=0.0,
                requires_manual=True
            ),
            
            # Default classification
            'Exception': ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.UNKNOWN,
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                backoff_factor=2.0,
                timeout_seconds=30.0
            )
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorClassification:
        """Classify an error and determine recovery strategy"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Try exact type match first
        if error_type in self.classification_rules:
            classification = self.classification_rules[error_type]
        else:
            # Try pattern matching on error message
            classification = self._classify_by_message(error_message, context or {})
            
        # Adjust classification based on context
        return self._adjust_classification(classification, error, context or {})
    
    def _classify_by_message(self, error_message: str, context: Dict[str, Any]) -> ErrorClassification:
        """Classify error by message patterns"""
        patterns = {
            'connection': ErrorCategory.NETWORK,
            'timeout': ErrorCategory.NETWORK,
            'rate limit': ErrorCategory.RATE_LIMIT,
            'authentication': ErrorCategory.AUTHENTICATION,
            'permission': ErrorCategory.PERMISSION,
            'not found': ErrorCategory.DEPENDENCY,
            'invalid': ErrorCategory.VALIDATION,
            'corrupt': ErrorCategory.DATA_CORRUPTION,
        }
        
        for pattern, category in patterns.items():
            if pattern in error_message:
                return self._get_default_classification_for_category(category)
                
        return self.classification_rules['Exception']
    
    def _get_default_classification_for_category(self, category: ErrorCategory) -> ErrorClassification:
        """Get default classification for a category"""
        defaults = {
            ErrorCategory.NETWORK: ErrorClassification(
                severity=ErrorSeverity.MEDIUM,
                category=category,
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                backoff_factor=2.0,
                timeout_seconds=30.0
            ),
            ErrorCategory.RATE_LIMIT: ErrorClassification(
                severity=ErrorSeverity.LOW,
                category=category,
                strategy=RecoveryStrategy.LINEAR_BACKOFF,
                max_retries=5,
                backoff_factor=1.0,
                timeout_seconds=60.0
            ),
            ErrorCategory.AUTHENTICATION: ErrorClassification(
                severity=ErrorSeverity.HIGH,
                category=category,
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                max_retries=1,
                backoff_factor=1.0,
                timeout_seconds=10.0,
                requires_manual=True
            ),
        }
        return defaults.get(category, self.classification_rules['Exception'])
    
    def _adjust_classification(self, classification: ErrorClassification, 
                             error: Exception, context: Dict[str, Any]) -> ErrorClassification:
        """Adjust classification based on context"""
        # Increase severity for repeated failures
        if context.get('retry_count', 0) > 2:
            if classification.severity == ErrorSeverity.LOW:
                classification.severity = ErrorSeverity.MEDIUM
            elif classification.severity == ErrorSeverity.MEDIUM:
                classification.severity = ErrorSeverity.HIGH
                
        # Enable fallback for network errors if available
        if classification.category == ErrorCategory.NETWORK:
            classification.fallback_available = context.get('fallback_service_available', False)
            
        return classification


class RetryManager:
    """Manages retry logic with various backoff strategies"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        
    async def retry_async(self, func: Callable, *args, 
                         classification: ErrorClassification = None,
                         context: Dict[str, Any] = None, **kwargs) -> Any:
        """Retry an async function with backoff strategy"""
        if not self.config.enabled:
            return await func(*args, **kwargs)
            
        max_attempts = classification.max_retries if classification else self.config.max_attempts
        backoff_factor = classification.backoff_factor if classification else self.config.backoff_factor
        
        for attempt in range(max_attempts + 1):
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
                return result
                
            except Exception as e:
                if attempt == max_attempts:
                    raise e
                    
                delay = self._calculate_delay(attempt, backoff_factor, classification)
                logger.warning(f"Retry attempt {attempt + 1}/{max_attempts + 1} failed: {e}. "
                              f"Retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
                
        raise RuntimeError(f"All retry attempts exhausted")
    
    def retry_sync(self, func: Callable, *args,
                  classification: ErrorClassification = None,
                  context: Dict[str, Any] = None, **kwargs) -> Any:
        """Retry a sync function with backoff strategy"""
        if not self.config.enabled:
            return func(*args, **kwargs)
            
        max_attempts = classification.max_retries if classification else self.config.max_attempts
        backoff_factor = classification.backoff_factor if classification else self.config.backoff_factor
        
        for attempt in range(max_attempts + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                if attempt == max_attempts:
                    raise e
                    
                delay = self._calculate_delay(attempt, backoff_factor, classification)
                logger.warning(f"Retry attempt {attempt + 1}/{max_attempts + 1} failed: {e}. "
                              f"Retrying in {delay:.2f}s")
                
                time.sleep(delay)
                
        raise RuntimeError(f"All retry attempts exhausted")
    
    def _calculate_delay(self, attempt: int, backoff_factor: float,
                        classification: ErrorClassification = None) -> float:
        """Calculate delay for retry attempt"""
        if classification and classification.strategy == RecoveryStrategy.LINEAR_BACKOFF:
            base_delay = self.config.initial_delay * (attempt + 1)
        else:
            # Exponential backoff
            base_delay = self.config.initial_delay * (backoff_factor ** attempt)
            
        # Add jitter to avoid thundering herd
        jitter = random.uniform(-self.config.jitter_factor, self.config.jitter_factor)
        delay = base_delay * (1 + jitter)
        
        return min(delay, self.config.max_delay)


class CircuitBreaker:
    """Circuit breaker implementation for failing services"""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig = None):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState(
            service_name=service_name,
            state=CircuitState.CLOSED,
            failure_count=0,
            last_failure_time=None,
            last_success_time=None,
            state_change_time=datetime.now()
        )
        self._lock = threading.Lock()
        
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self.config.enabled:
            return await func(*args, **kwargs)
            
        with self._lock:
            if self.state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise RuntimeError(f"Circuit breaker open for service: {self.service_name}")
                    
            elif self.state.state == CircuitState.HALF_OPEN:
                if self.state.half_open_calls >= self.config.half_open_max_calls:
                    raise RuntimeError(f"Circuit breaker half-open limit reached for service: {self.service_name}")
                    
        try:
            if self.state.state == CircuitState.HALF_OPEN:
                with self._lock:
                    self.state.half_open_calls += 1
                    
            result = await func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection (sync)"""
        if not self.config.enabled:
            return func(*args, **kwargs)
            
        with self._lock:
            if self.state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise RuntimeError(f"Circuit breaker open for service: {self.service_name}")
                    
            elif self.state.state == CircuitState.HALF_OPEN:
                if self.state.half_open_calls >= self.config.half_open_max_calls:
                    raise RuntimeError(f"Circuit breaker half-open limit reached for service: {self.service_name}")
                    
        try:
            if self.state.state == CircuitState.HALF_OPEN:
                with self._lock:
                    self.state.half_open_calls += 1
                    
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self.state.last_success_time = datetime.now()
            
            if self.state.state == CircuitState.HALF_OPEN:
                self.state.half_open_calls = 0
                if self.state.failure_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                self.state.failure_count = max(0, self.state.failure_count - 1)
    
    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = datetime.now()
            
            if (self.state.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
                self.state.failure_count >= self.config.failure_threshold):
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.state.last_failure_time:
            return False
            
        elapsed = datetime.now() - self.state.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state.state = CircuitState.OPEN
        self.state.state_change_time = datetime.now()
        logger.warning(f"Circuit breaker opened for service: {self.service_name}")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state.state = CircuitState.HALF_OPEN
        self.state.state_change_time = datetime.now()
        self.state.half_open_calls = 0
        logger.info(f"Circuit breaker half-open for service: {self.service_name}")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state.state = CircuitState.CLOSED
        self.state.state_change_time = datetime.now()
        self.state.failure_count = 0
        logger.info(f"Circuit breaker closed for service: {self.service_name}")


class ErrorRecoveryManager:
    """Main error recovery and retry management system"""
    
    def __init__(self, storage_path: str = "data/recovery"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.classifier = ErrorClassifier()
        self.retry_manager = RetryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = MetricsCollector()
        
        # Error tracking
        self.error_events: Dict[str, ErrorEvent] = {}
        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = {}
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="recovery")
        self._shutdown_event = asyncio.Event()
        
        # Configuration
        self.config = {
            'max_error_history': 10000,
            'cleanup_interval_hours': 24,
            'metrics_flush_interval': 300,
            'background_recovery_enabled': True,
            'auto_fallback_enabled': True,
            'checkpoint_integration_enabled': True,
        }
        
        logger.info("Error Recovery Manager initialized")
    
    async def start(self):
        """Start the error recovery manager"""
        await self._load_persistent_state()
        
        if self.config['background_recovery_enabled']:
            self.background_tasks.add(
                asyncio.create_task(self._background_recovery_worker())
            )
            
        self.background_tasks.add(
            asyncio.create_task(self._cleanup_worker())
        )
        
        self.background_tasks.add(
            asyncio.create_task(self._metrics_worker())
        )
        
        logger.info("Error Recovery Manager started")
    
    async def shutdown(self):
        """Shutdown the error recovery manager"""
        logger.info("Shutting down Error Recovery Manager...")
        
        self._shutdown_event.set()
        
        # Wait for background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save state
        await self._save_persistent_state()
        
        logger.info("Error Recovery Manager shutdown complete")
    
    async def handle_error(self, error: Exception, 
                          context: Dict[str, Any] = None,
                          operation_func: Callable = None,
                          operation_args: tuple = None,
                          operation_kwargs: dict = None) -> Optional[Any]:
        """
        Handle an error with automatic recovery
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            operation_func: The function that failed (for retry)
            operation_args: Arguments for the function
            operation_kwargs: Keyword arguments for the function
            
        Returns:
            Result if recovery successful, None otherwise
        """
        context = context or {}
        operation_args = operation_args or ()
        operation_kwargs = operation_kwargs or {}
        
        # Create error event
        error_event = self._create_error_event(error, context)
        
        # Classify error
        classification = self.classifier.classify_error(error, context)
        error_event.classification = classification
        
        # Store error event
        self.error_events[error_event.error_id] = error_event
        self.recovery_attempts[error_event.error_id] = []
        
        logger.error(f"Error occurred: {error_event.error_id} - {error_event.error_type}: {error_event.error_message}")
        
        # Attempt recovery
        try:
            result = await self._attempt_recovery(
                error_event, classification, 
                operation_func, operation_args, operation_kwargs
            )
            
            # Mark as resolved
            error_event.resolved = True
            error_event.resolution_method = classification.strategy.value
            
            # Update metrics
            self.metrics.increment('errors_recovered_total')
            self.metrics.increment(f'recovery_strategy_{classification.strategy.value}')
            
            return result
            
        except Exception as recovery_error:
            logger.error(f"Recovery failed for error {error_event.error_id}: {recovery_error}")
            
            # Update metrics
            self.metrics.increment('recovery_failed_total')
            
            # Handle manual intervention if required
            if classification.requires_manual:
                await self._trigger_manual_intervention(error_event, recovery_error)
            
            raise recovery_error
    
    async def _attempt_recovery(self, error_event: ErrorEvent, 
                               classification: ErrorClassification,
                               operation_func: Callable = None,
                               operation_args: tuple = None,
                               operation_kwargs: dict = None) -> Optional[Any]:
        """Attempt recovery based on classification"""
        strategy = classification.strategy
        
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY and operation_func:
            return await self._immediate_retry_recovery(
                error_event, classification, operation_func, operation_args, operation_kwargs
            )
            
        elif strategy in [RecoveryStrategy.EXPONENTIAL_BACKOFF, RecoveryStrategy.LINEAR_BACKOFF] and operation_func:
            return await self._backoff_retry_recovery(
                error_event, classification, operation_func, operation_args, operation_kwargs
            )
            
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER and operation_func:
            return await self._circuit_breaker_recovery(
                error_event, classification, operation_func, operation_args, operation_kwargs
            )
            
        elif strategy == RecoveryStrategy.FALLBACK_SERVICE:
            return await self._fallback_service_recovery(error_event, classification)
            
        elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
            return await self._checkpoint_restore_recovery(error_event, classification)
            
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_recovery(error_event, classification)
            
        elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            return await self._skip_and_continue_recovery(error_event, classification)
            
        elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            await self._trigger_manual_intervention(error_event, None)
            raise RuntimeError(f"Manual intervention required for error: {error_event.error_id}")
            
        else:
            raise RuntimeError(f"Unknown recovery strategy: {strategy}")
    
    async def _immediate_retry_recovery(self, error_event: ErrorEvent, 
                                      classification: ErrorClassification,
                                      operation_func: Callable,
                                      operation_args: tuple,
                                      operation_kwargs: dict) -> Any:
        """Immediate retry recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.IMMEDIATE_RETRY)
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*operation_args, **operation_kwargs)
            else:
                result = operation_func(*operation_args, **operation_kwargs)
                
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['retry_error'] = str(e)
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _backoff_retry_recovery(self, error_event: ErrorEvent,
                                     classification: ErrorClassification,
                                     operation_func: Callable,
                                     operation_args: tuple,
                                     operation_kwargs: dict) -> Any:
        """Exponential/linear backoff retry recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, classification.strategy)
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(operation_func):
                result = await self.retry_manager.retry_async(
                    operation_func, *operation_args,
                    classification=classification,
                    context={'error_event': error_event},
                    **operation_kwargs
                )
            else:
                result = self.retry_manager.retry_sync(
                    operation_func, *operation_args,
                    classification=classification,
                    context={'error_event': error_event},
                    **operation_kwargs
                )
                
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['retry_error'] = str(e)
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _circuit_breaker_recovery(self, error_event: ErrorEvent,
                                      classification: ErrorClassification,
                                      operation_func: Callable,
                                      operation_args: tuple,
                                      operation_kwargs: dict) -> Any:
        """Circuit breaker recovery"""
        service_name = error_event.context.get('service_name', 'default')
        
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(service_name)
            
        circuit_breaker = self.circuit_breakers[service_name]
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.CIRCUIT_BREAKER)
        
        try:
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(operation_func):
                result = await circuit_breaker.call_async(operation_func, *operation_args, **operation_kwargs)
            else:
                result = circuit_breaker.call_sync(operation_func, *operation_args, **operation_kwargs)
                
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['circuit_state'] = circuit_breaker.state.state.value
            
            return result
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['circuit_error'] = str(e)
            attempt.details['circuit_state'] = circuit_breaker.state.state.value
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _fallback_service_recovery(self, error_event: ErrorEvent,
                                       classification: ErrorClassification) -> Optional[Any]:
        """Fallback service recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.FALLBACK_SERVICE)
        
        try:
            start_time = time.time()
            
            # Get fallback service from context
            fallback_func = error_event.context.get('fallback_function')
            fallback_args = error_event.context.get('fallback_args', ())
            fallback_kwargs = error_event.context.get('fallback_kwargs', {})
            
            if not fallback_func:
                raise RuntimeError("No fallback service configured")
                
            if asyncio.iscoroutinefunction(fallback_func):
                result = await fallback_func(*fallback_args, **fallback_kwargs)
            else:
                result = fallback_func(*fallback_args, **fallback_kwargs)
                
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['fallback_used'] = True
            
            return result
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['fallback_error'] = str(e)
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _checkpoint_restore_recovery(self, error_event: ErrorEvent,
                                         classification: ErrorClassification) -> Optional[Any]:
        """Checkpoint restore recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.CHECKPOINT_RESTORE)
        
        try:
            start_time = time.time()
            
            # This would integrate with the checkpoint manager
            checkpoint_id = error_event.context.get('checkpoint_id')
            if not checkpoint_id:
                raise RuntimeError("No checkpoint available for restore")
                
            # TODO: Integrate with CheckpointManager
            # from .checkpoint_manager import CheckpointManager
            # checkpoint_manager = CheckpointManager()
            # result = await checkpoint_manager.restore_checkpoint(checkpoint_id)
            
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['checkpoint_restored'] = checkpoint_id
            
            logger.info(f"Checkpoint restore recovery completed for error: {error_event.error_id}")
            return None  # Checkpoint restore doesn't return a value
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['restore_error'] = str(e)
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _graceful_degradation_recovery(self, error_event: ErrorEvent,
                                           classification: ErrorClassification) -> Optional[Any]:
        """Graceful degradation recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.GRACEFUL_DEGRADATION)
        
        try:
            start_time = time.time()
            
            # Return degraded service or cached result
            degraded_result = error_event.context.get('degraded_result')
            cached_result = error_event.context.get('cached_result')
            
            result = degraded_result or cached_result
            
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['degraded_service'] = result is not None
            
            logger.warning(f"Graceful degradation recovery for error: {error_event.error_id}")
            return result
            
        except Exception as e:
            attempt.success = False
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['degradation_error'] = str(e)
            raise e
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _skip_and_continue_recovery(self, error_event: ErrorEvent,
                                        classification: ErrorClassification) -> Optional[Any]:
        """Skip and continue recovery"""
        attempt = self._create_recovery_attempt(error_event.error_id, RecoveryStrategy.SKIP_AND_CONTINUE)
        
        try:
            start_time = time.time()
            
            # Log the skip and continue
            logger.warning(f"Skipping failed operation for error: {error_event.error_id}")
            
            attempt.success = True
            attempt.duration_ms = (time.time() - start_time) * 1000
            attempt.details['skipped'] = True
            
            return None  # Skip returns no value
            
        finally:
            self.recovery_attempts[error_event.error_id].append(attempt)
    
    async def _trigger_manual_intervention(self, error_event: ErrorEvent, recovery_error: Exception):
        """Trigger manual intervention for critical errors"""
        logger.critical(f"Manual intervention required for error: {error_event.error_id}")
        logger.critical(f"Original error: {error_event.error_message}")
        
        if recovery_error:
            logger.critical(f"Recovery error: {str(recovery_error)}")
            
        # TODO: Integrate with alerting system
        # This could send notifications, create tickets, etc.
        
        # Update metrics
        self.metrics.increment('manual_intervention_required')
    
    def _create_error_event(self, error: Exception, context: Dict[str, Any]) -> ErrorEvent:
        """Create an error event from an exception"""
        timestamp = datetime.now()
        error_id = hashlib.md5(
            f"{timestamp.isoformat()}:{type(error).__name__}:{str(error)}".encode()
        ).hexdigest()[:16]
        
        return ErrorEvent(
            error_id=error_id,
            timestamp=timestamp,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context.copy(),
            classification=None,  # Will be set later
            session_id=context.get('session_id'),
            operation_id=context.get('operation_id')
        )
    
    def _create_recovery_attempt(self, error_id: str, strategy: RecoveryStrategy) -> RecoveryAttempt:
        """Create a recovery attempt record"""
        return RecoveryAttempt(
            attempt_id=f"{error_id}_{strategy.value}_{int(time.time())}",
            error_id=error_id,
            strategy=strategy,
            timestamp=datetime.now(),
            success=False,
            duration_ms=0.0
        )
    
    async def _background_recovery_worker(self):
        """Background worker for automatic recovery"""
        logger.info("Starting background recovery worker")
        
        while not self._shutdown_event.is_set():
            try:
                await self._process_background_recovery()
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background recovery worker error: {e}")
                await asyncio.sleep(60)
                
        logger.info("Background recovery worker stopped")
    
    async def _process_background_recovery(self):
        """Process background recovery for unresolved errors"""
        unresolved_errors = [
            event for event in self.error_events.values()
            if not event.resolved and 
            event.classification and
            event.classification.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        ]
        
        for error_event in unresolved_errors[-10:]:  # Process last 10 unresolved
            try:
                # Attempt background recovery for suitable errors
                if (error_event.classification.strategy in [
                    RecoveryStrategy.EXPONENTIAL_BACKOFF,
                    RecoveryStrategy.LINEAR_BACKOFF,
                    RecoveryStrategy.GRACEFUL_DEGRADATION
                ] and error_event.retry_count < 3):
                    
                    error_event.retry_count += 1
                    logger.info(f"Background recovery attempt for error: {error_event.error_id}")
                    
                    # This would need operation context to actually retry
                    # For now, just mark some as potentially recoverable
                    if error_event.classification.category == ErrorCategory.RATE_LIMIT:
                        error_event.resolved = True
                        error_event.resolution_method = "background_recovery"
                        
            except Exception as e:
                logger.error(f"Background recovery failed for {error_event.error_id}: {e}")
    
    async def _cleanup_worker(self):
        """Background cleanup worker"""
        logger.info("Starting cleanup worker")
        
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_errors()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(3600)
                
        logger.info("Cleanup worker stopped")
    
    async def _cleanup_old_errors(self):
        """Clean up old error events and recovery attempts"""
        cutoff_time = datetime.now() - timedelta(hours=self.config['cleanup_interval_hours'])
        
        # Clean up old resolved errors
        old_errors = [
            error_id for error_id, event in self.error_events.items()
            if event.resolved and event.timestamp < cutoff_time
        ]
        
        for error_id in old_errors:
            del self.error_events[error_id]
            if error_id in self.recovery_attempts:
                del self.recovery_attempts[error_id]
                
        if old_errors:
            logger.info(f"Cleaned up {len(old_errors)} old error events")
            
        # Limit total error history
        if len(self.error_events) > self.config['max_error_history']:
            # Keep most recent errors
            sorted_events = sorted(self.error_events.items(), key=lambda x: x[1].timestamp, reverse=True)
            keep_events = dict(sorted_events[:self.config['max_error_history']])
            
            removed_count = len(self.error_events) - len(keep_events)
            self.error_events = keep_events
            
            logger.info(f"Trimmed error history: removed {removed_count} old events")
    
    async def _metrics_worker(self):
        """Background metrics collection worker"""
        logger.info("Starting metrics worker")
        
        while not self._shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config['metrics_flush_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
                await asyncio.sleep(self.config['metrics_flush_interval'])
                
        logger.info("Metrics worker stopped")
    
    async def _collect_metrics(self):
        """Collect and flush metrics"""
        # Count errors by severity
        severity_counts = {}
        for event in self.error_events.values():
            if event.classification:
                severity = event.classification.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
        for severity, count in severity_counts.items():
            self.metrics.gauge(f'errors_by_severity_{severity}', count)
            
        # Count circuit breaker states
        for service_name, breaker in self.circuit_breakers.items():
            self.metrics.gauge(f'circuit_breaker_state_{service_name}', 
                             1 if breaker.state.state == CircuitState.OPEN else 0)
            
        # Resolution rate
        total_errors = len(self.error_events)
        resolved_errors = len([e for e in self.error_events.values() if e.resolved])
        
        if total_errors > 0:
            resolution_rate = resolved_errors / total_errors
            self.metrics.gauge('error_resolution_rate', resolution_rate)
            
        logger.debug(f"Metrics collected: {total_errors} total errors, {resolved_errors} resolved")
    
    async def _load_persistent_state(self):
        """Load persistent state from storage"""
        try:
            state_file = self.storage_path / "recovery_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                # Load error events (recent ones only)
                if 'error_events' in state:
                    cutoff_time = datetime.now() - timedelta(days=1)
                    for event_data in state['error_events']:
                        event = ErrorEvent(**event_data)
                        if event.timestamp > cutoff_time:
                            self.error_events[event.error_id] = event
                            
                logger.info(f"Loaded {len(self.error_events)} error events from persistent state")
                
        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save persistent state to storage"""
        try:
            state_file = self.storage_path / "recovery_state.json"
            
            # Only save recent, important errors
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_events = [
                asdict(event) for event in self.error_events.values()
                if event.timestamp > cutoff_time and
                event.classification and
                event.classification.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
            ]
            
            state = {
                'error_events': recent_events,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info(f"Saved {len(recent_events)} error events to persistent state")
            
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_events)
        resolved_errors = len([e for e in self.error_events.values() if e.resolved])
        
        # Group by severity
        severity_stats = {}
        for event in self.error_events.values():
            if event.classification:
                severity = event.classification.severity.value
                if severity not in severity_stats:
                    severity_stats[severity] = {'total': 0, 'resolved': 0}
                severity_stats[severity]['total'] += 1
                if event.resolved:
                    severity_stats[severity]['resolved'] += 1
        
        # Group by category
        category_stats = {}
        for event in self.error_events.values():
            if event.classification:
                category = event.classification.category.value
                if category not in category_stats:
                    category_stats[category] = {'total': 0, 'resolved': 0}
                category_stats[category]['total'] += 1
                if event.resolved:
                    category_stats[category]['resolved'] += 1
        
        # Recovery strategy effectiveness
        strategy_stats = {}
        for attempts_list in self.recovery_attempts.values():
            for attempt in attempts_list:
                strategy = attempt.strategy.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'attempts': 0, 'successes': 0}
                strategy_stats[strategy]['attempts'] += 1
                if attempt.success:
                    strategy_stats[strategy]['successes'] += 1
        
        return {
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'resolution_rate': resolved_errors / total_errors if total_errors > 0 else 0,
            'severity_breakdown': severity_stats,
            'category_breakdown': category_stats,
            'strategy_effectiveness': strategy_stats,
            'circuit_breakers': {
                name: {
                    'state': breaker.state.state.value,
                    'failure_count': breaker.state.failure_count,
                    'last_failure': breaker.state.last_failure_time.isoformat() if breaker.state.last_failure_time else None
                }
                for name, breaker in self.circuit_breakers.items()
            }
        }


def with_error_recovery(recovery_manager: ErrorRecoveryManager = None):
    """Decorator for automatic error recovery"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = recovery_manager or ErrorRecoveryManager()
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': func.__name__,
                    'function_module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                return await manager.handle_error(
                    e, context, func, args, kwargs
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = recovery_manager or ErrorRecoveryManager()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': func.__name__,
                    'function_module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                # For sync functions, we can't use async recovery
                # Just log and re-raise for now
                logger.error(f"Error in {func.__name__}: {e}")
                raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize recovery manager
        recovery_manager = ErrorRecoveryManager()
        await recovery_manager.start()
        
        try:
            # Example error handling
            try:
                # Simulate a network error
                raise ConnectionError("Failed to connect to API")
            except Exception as e:
                result = await recovery_manager.handle_error(
                    e, 
                    context={
                        'service_name': 'api_client',
                        'operation': 'fetch_data',
                        'fallback_function': lambda: "cached_data",
                        'fallback_args': (),
                        'fallback_kwargs': {}
                    }
                )
                print(f"Recovery result: {result}")
            
            # Print statistics
            stats = recovery_manager.get_error_statistics()
            print(f"Error statistics: {json.dumps(stats, indent=2)}")
            
        finally:
            await recovery_manager.shutdown()
    
    asyncio.run(main())