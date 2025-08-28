"""European Parliament Open Data Portal API client."""

import requests
from typing import Dict, List, Optional, Any
import time
from urllib.parse import urljoin

from ..core.config import APIConfig
from ..core.rate_limiter import RateLimiter, ExponentialBackoff
from ..core.exceptions import APIError, RateLimitExceededError
from ..core.logging import get_logger, log_api_request
from ..core.circuit_breaker import circuit_registry, CircuitBreakerConfig, CircuitBreakerError
from ..core.metrics import metrics_collector

logger = get_logger(__name__)


class OpenDataClient:
    """Client for European Parliament Open Data Portal API."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize Open Data Portal client.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.session = requests.Session()
        
        # Set headers as specified in implementation plan
        self.session.headers.update({
            'User-Agent': 'EU-Parliament-Research-Tool/1.0',
            'Accept': 'application/json',
            'Accept-Language': 'en'
        })
        
        # Initialize circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,  # 2 minutes for EU APIs
            success_threshold=3,
            timeout=config.timeout
        )
        self.circuit_breaker = circuit_registry.get_breaker('opendata_api', cb_config)
        
        # Service name for metrics
        self.service_name = 'opendata'
        
        logger.info(
            "OpenData client initialized",
            base_url=config.base_url,
            rate_limit=config.rate_limit,
            circuit_breaker_enabled=True
        )
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make rate-limited API request with circuit breaker and metrics.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            APIError: On API errors
            RateLimitExceededError: On rate limiting
            CircuitBreakerError: When circuit breaker is open
        """
        return self.circuit_breaker.call(self._execute_request, endpoint, params)
    
    def _execute_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the actual HTTP request with metrics collection."""
        self.rate_limiter.wait_if_needed()
        
        url = urljoin(self.config.base_url + "/", endpoint.lstrip('/'))
        start_time = time.time()
        response = None
        error_type = None
        
        try:
            backoff = ExponentialBackoff()
            while backoff.wait():
                request_start = time.time()
                
                try:
                    logger.debug("Making API request", url=url, params=params)
                    
                    response = self.session.get(
                        url,
                        params=params or {},
                        timeout=self.config.timeout
                    )
                    
                    response_time = time.time() - request_start
                    
                    logger.info(
                        "API request completed",
                        url=url,
                        method="GET",
                        response_time=response_time,
                        status_code=response.status_code
                    )
                    
                    
                    # Check for rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(
                            "Rate limit exceeded",
                            retry_after=retry_after,
                            attempt=backoff.attempt
                        )
                        error_type = "rate_limit"
                        time.sleep(retry_after)
                        continue
                    
                    # Check for other errors
                    if response.status_code >= 400:
                        error_msg = f"API request failed: {response.status_code}"
                        logger.error(
                            error_msg,
                            status_code=response.status_code,
                            response_text=response.text[:500]
                        )
                        
                        if response.status_code >= 500:
                            # Server error - retry
                            error_type = "server_error"
                            continue
                        else:
                            # Client error - don't retry
                            error_type = "client_error"
                            raise APIError(
                                error_msg,
                                status_code=response.status_code,
                                response_data=response.text
                            )
                    
                    # Success - parse and return JSON
                    try:
                        return response.json()
                    except ValueError as e:
                        error_type = "json_parse_error"
                        logger.error("Failed to parse JSON response", error=str(e))
                        raise APIError(f"Invalid JSON response: {e}")
                    
                except requests.exceptions.Timeout as e:
                    error_type = "timeout"
                    logger.warning("Request timeout", attempt=backoff.attempt)
                    continue
                    
                except requests.exceptions.ConnectionError as e:
                    error_type = "connection_error"
                    logger.warning("Connection error", error=str(e), attempt=backoff.attempt)
                    continue
                    
                except requests.exceptions.RequestException as e:
                    error_type = "request_error"
                    logger.error("Request exception", error=str(e))
                    raise APIError(f"Request failed: {e}")
            
            # Maximum retries exceeded
            error_type = "max_retries_exceeded"
            raise APIError("Maximum retries exceeded")
            
        finally:
            # Always record metrics, even on final failure
            if response is not None:
                total_time = time.time() - start_time
                metrics_collector.record_request(
                    service=self.service_name,
                    endpoint=endpoint,
                    method="GET",
                    status_code=getattr(response, 'status_code', 0),
                    response_time=total_time,
                    error_type=error_type
                )
            elif error_type:
                # Record failed request without response
                total_time = time.time() - start_time
                metrics_collector.record_request(
                    service=self.service_name,
                    endpoint=endpoint,
                    method="GET",
                    status_code=0,
                    response_time=total_time,
                    error_type=error_type
                )
    
    def get_plenary_sessions(self, start_date: str, end_date: str, 
                           limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve plenary session metadata for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)  
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata
        """
        logger.info(
            "Fetching plenary sessions",
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json',
            'type': 'plenary'
        }
        
        if limit:
            params['limit'] = limit
        
        # Note: The actual API endpoint structure may differ
        # This is based on the implementation plan specifications
        try:
            response = self._make_request('plenary-session-documents', params)
            
            sessions = response.get('results', []) if isinstance(response, dict) else response
            
            logger.info(
                "Retrieved plenary sessions",
                count=len(sessions),
                start_date=start_date,
                end_date=end_date
            )
            
            return sessions
            
        except APIError as e:
            logger.error(
                "Failed to fetch plenary sessions",
                error=str(e),
                start_date=start_date,
                end_date=end_date
            )
            raise
    
    def get_mep_data(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve Member of European Parliament information.
        
        Args:
            active_only: Only return currently active MEPs
            
        Returns:
            List of MEP data
        """
        logger.info("Fetching MEP data", active_only=active_only)
        
        params = {
            'format': 'json'
        }
        
        if active_only:
            params['status'] = 'active'
        
        try:
            response = self._make_request('members-european-parliament', params)
            
            meps = response.get('results', []) if isinstance(response, dict) else response
            
            logger.info("Retrieved MEP data", count=len(meps))
            
            return meps
            
        except APIError as e:
            logger.error("Failed to fetch MEP data", error=str(e))
            raise
    
    def get_session_documents(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve documents for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session document data
        """
        logger.info("Fetching session documents", session_id=session_id)
        
        try:
            response = self._make_request(f'sessions/{session_id}/documents')
            
            logger.info("Retrieved session documents", session_id=session_id)
            
            return response
            
        except APIError as e:
            logger.error(
                "Failed to fetch session documents",
                error=str(e),
                session_id=session_id
            )
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the OpenData API.
        
        Returns:
            Health status information
        """
        health_info = {
            'service': self.service_name,
            'base_url': self.config.base_url,
            'healthy': False,
            'response_time': 0.0,
            'circuit_breaker_state': self.circuit_breaker.get_state(),
            'metrics': metrics_collector.get_service_metrics(self.service_name),
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Simple health check - get basic API info
            response = self._make_request('', {'format': 'json', 'limit': 1})
            
            health_info['response_time'] = time.time() - start_time
            health_info['healthy'] = True
            
            logger.info(
                "Health check passed",
                service=self.service_name,
                response_time=health_info['response_time']
            )
            
        except Exception as e:
            health_info['error'] = str(e)
            health_info['response_time'] = time.time() - start_time
            
            logger.warning(
                "Health check failed",
                service=self.service_name,
                error=str(e)
            )
        
        return health_info
    
    def get_client_metrics(self) -> Dict[str, Any]:
        """Get comprehensive client metrics."""
        return {
            'service_name': self.service_name,
            'circuit_breaker': self.circuit_breaker.get_state(),
            'metrics': metrics_collector.get_service_metrics(self.service_name),
            'rate_limiter': {
                'requests_per_second': self.rate_limiter.requests_per_second,
                'min_interval': self.rate_limiter.min_interval,
                'last_request_time': getattr(self.rate_limiter, 'last_request_time', 0)
            },
            'configuration': {
                'base_url': self.config.base_url,
                'timeout': self.config.timeout,
                'rate_limit': self.config.rate_limit
            }
        }
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self.circuit_breaker.reset()
        logger.info("Circuit breaker reset", service=self.service_name)
    
    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.info("OpenData client closed")