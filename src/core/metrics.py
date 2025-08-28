"""Metrics collection and monitoring for EU Parliament scraper."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock
from datetime import datetime, timedelta

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetric:
    """Individual request metric data."""
    timestamp: float
    service: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_type: Optional[str] = None


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service."""
    service_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)
    status_counts: Dict[int, int] = field(default_factory=dict)
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate recent requests per minute."""
        if not self.recent_requests:
            return 0.0
        
        now = time.time()
        minute_ago = now - 60
        recent_count = sum(1 for req in self.recent_requests if req.timestamp > minute_ago)
        return recent_count


class MetricsCollector:
    """Collects and aggregates metrics for API clients."""
    
    def __init__(self, max_history_size: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history_size: Maximum number of metrics to retain
        """
        self.max_history_size = max_history_size
        self.services: Dict[str, ServiceMetrics] = {}
        self.all_requests: deque = deque(maxlen=max_history_size)
        self._lock = Lock()
        
        logger.info("Metrics collector initialized", max_history_size=max_history_size)
    
    def record_request(self, service: str, endpoint: str, method: str,
                      status_code: int, response_time: float, 
                      error_type: Optional[str] = None) -> None:
        """
        Record an API request metric.
        
        Args:
            service: Service name (e.g., 'opendata', 'eurlex')
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            response_time: Response time in seconds
            error_type: Type of error if any
        """
        timestamp = time.time()
        success = 200 <= status_code < 300
        
        metric = RequestMetric(
            timestamp=timestamp,
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            success=success,
            error_type=error_type
        )
        
        with self._lock:
            # Add to global history
            self.all_requests.append(metric)
            
            # Initialize service metrics if needed
            if service not in self.services:
                self.services[service] = ServiceMetrics(service_name=service)
            
            service_metrics = self.services[service]
            
            # Update service metrics
            service_metrics.total_requests += 1
            service_metrics.total_response_time += response_time
            service_metrics.recent_requests.append(metric)
            
            if success:
                service_metrics.successful_requests += 1
            else:
                service_metrics.failed_requests += 1
                
            # Update response time bounds
            service_metrics.min_response_time = min(service_metrics.min_response_time, response_time)
            service_metrics.max_response_time = max(service_metrics.max_response_time, response_time)
            
            # Update status code counts
            service_metrics.status_counts[status_code] = service_metrics.status_counts.get(status_code, 0) + 1
            
            # Update error counts
            if error_type:
                service_metrics.error_counts[error_type] = service_metrics.error_counts.get(error_type, 0) + 1
        
        logger.debug(
            "Request metric recorded",
            service=service,
            endpoint=endpoint,
            status_code=status_code,
            response_time=response_time,
            success=success
        )
    
    def get_service_metrics(self, service: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific service."""
        with self._lock:
            if service not in self.services:
                return None
            
            metrics = self.services[service]
            return {
                'service_name': metrics.service_name,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'success_rate': round(metrics.success_rate, 2),
                'average_response_time': round(metrics.average_response_time, 4),
                'min_response_time': round(metrics.min_response_time, 4) if metrics.min_response_time != float('inf') else 0,
                'max_response_time': round(metrics.max_response_time, 4),
                'requests_per_minute': round(metrics.requests_per_minute, 2),
                'status_counts': dict(metrics.status_counts),
                'error_counts': dict(metrics.error_counts)
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all services."""
        with self._lock:
            return {
                service_name: self.get_service_metrics(service_name)
                for service_name in self.services.keys()
            }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global metrics across all services."""
        with self._lock:
            if not self.all_requests:
                return {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'success_rate': 0.0,
                    'average_response_time': 0.0
                }
            
            total = len(self.all_requests)
            successful = sum(1 for req in self.all_requests if req.success)
            failed = total - successful
            
            avg_response_time = sum(req.response_time for req in self.all_requests) / total
            
            return {
                'total_requests': total,
                'successful_requests': successful,
                'failed_requests': failed,
                'success_rate': round((successful / total) * 100, 2),
                'average_response_time': round(avg_response_time, 4),
                'services': list(self.services.keys()),
                'time_range_minutes': round((time.time() - self.all_requests[0].timestamp) / 60, 2) if self.all_requests else 0
            }
    
    def get_recent_failures(self, service: Optional[str] = None, 
                           minutes: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed requests."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            failures = []
            for req in reversed(self.all_requests):
                if req.timestamp < cutoff_time:
                    break
                
                if not req.success and (service is None or req.service == service):
                    failures.append({
                        'timestamp': datetime.fromtimestamp(req.timestamp).isoformat(),
                        'service': req.service,
                        'endpoint': req.endpoint,
                        'status_code': req.status_code,
                        'error_type': req.error_type,
                        'response_time': req.response_time
                    })
            
            return failures
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self._lock:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'global_metrics': self.get_global_metrics(),
                'service_metrics': self.get_all_metrics(),
                'recent_failures': self.get_recent_failures(minutes=5),
                'health_status': 'healthy'
            }
            
            # Determine overall health
            global_success_rate = report['global_metrics']['success_rate']
            if global_success_rate < 90:
                report['health_status'] = 'degraded'
            if global_success_rate < 70:
                report['health_status'] = 'unhealthy'
            
            # Add service health details
            service_health = {}
            for service_name, metrics in report['service_metrics'].items():
                if not metrics:
                    continue
                    
                status = 'healthy'
                if metrics['success_rate'] < 90:
                    status = 'degraded'
                if metrics['success_rate'] < 70:
                    status = 'unhealthy'
                
                service_health[service_name] = {
                    'status': status,
                    'success_rate': metrics['success_rate'],
                    'requests_per_minute': metrics['requests_per_minute'],
                    'average_response_time': metrics['average_response_time']
                }
            
            report['service_health'] = service_health
            return report
    
    def reset_metrics(self, service: Optional[str] = None) -> None:
        """Reset metrics for a service or all services."""
        with self._lock:
            if service:
                if service in self.services:
                    del self.services[service]
                    # Remove requests for this service from global history
                    self.all_requests = deque(
                        (req for req in self.all_requests if req.service != service),
                        maxlen=self.max_history_size
                    )
                    logger.info("Metrics reset for service", service=service)
            else:
                self.services.clear()
                self.all_requests.clear()
                logger.info("All metrics reset")


# Global metrics collector instance
metrics_collector = MetricsCollector()