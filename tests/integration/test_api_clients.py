"""Integration tests for API clients with circuit breaker and metrics."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.clients.opendata_client import OpenDataClient
from src.core.config import APIConfig
from src.core.circuit_breaker import CircuitState, CircuitBreakerError, circuit_registry
from src.core.metrics import metrics_collector
from src.core.exceptions import APIError


class TestOpenDataClientIntegration:
    """Integration tests for enhanced OpenData client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        circuit_registry.reset_all()
        metrics_collector.reset_metrics()
        
        # Create test config
        self.config = APIConfig(
            base_url="https://data.europarl.europa.eu/api",
            timeout=10,
            rate_limit=2.0  # Higher for testing
        )
        
        self.client = OpenDataClient(self.config)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.client.close()
        circuit_registry.reset_all()
        metrics_collector.reset_metrics()
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker is properly initialized."""
        assert self.client.circuit_breaker is not None
        assert self.client.service_name == 'opendata'
        
        state = self.client.circuit_breaker.get_state()
        assert state['service_name'] == 'opendata_api'
        assert state['state'] == CircuitState.CLOSED.value
        assert state['failure_count'] == 0
    
    def test_metrics_collection_on_success(self):
        """Test metrics collection on successful requests."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'results': [{'id': '123'}]}
            mock_get.return_value = mock_response
            
            # Make request
            result = self.client._make_request('test-endpoint')
            
            # Verify response
            assert result == {'results': [{'id': '123'}]}
            
            # Check metrics were recorded
            metrics = metrics_collector.get_service_metrics('opendata')
            assert metrics is not None
            assert metrics['total_requests'] == 1
            assert metrics['successful_requests'] == 1
            assert metrics['failed_requests'] == 0
            assert metrics['success_rate'] == 100.0
            assert metrics['status_counts'][200] == 1
    
    def test_metrics_collection_on_failure(self):
        """Test metrics collection on failed requests."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock failed response
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_get.return_value = mock_response
            
            # Make request that should retry and eventually fail
            with pytest.raises(APIError):
                self.client._make_request('test-endpoint')
            
            # Check metrics were recorded
            metrics = metrics_collector.get_service_metrics('opendata')
            assert metrics is not None
            assert metrics['total_requests'] > 0
            assert metrics['failed_requests'] > 0
            assert metrics['success_rate'] < 100.0
            assert 500 in metrics['status_counts']
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after repeated failures."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock consistent failures
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Server Error"
            mock_get.return_value = mock_response
            
            # Make enough failed requests to open circuit breaker
            failure_count = 0
            for i in range(10):  # More than the threshold
                try:
                    self.client._make_request(f'test-endpoint-{i}')
                except APIError:
                    failure_count += 1
                except CircuitBreakerError:
                    # Circuit breaker opened
                    break
            
            # Verify circuit breaker is open
            state = self.client.circuit_breaker.get_state()
            assert state['state'] == CircuitState.OPEN.value
            assert state['failure_count'] >= 5  # Threshold
            
            # Verify subsequent requests are blocked
            with pytest.raises(CircuitBreakerError):
                self.client._make_request('blocked-endpoint')
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after failures."""
        # First, force circuit breaker to open
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock failures
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Server Error"
            mock_get.return_value = mock_response
            
            # Force failures to open circuit
            for i in range(6):
                try:
                    self.client._make_request(f'fail-{i}')
                except (APIError, CircuitBreakerError):
                    pass
        
        # Verify circuit is open
        assert self.client.circuit_breaker.get_state()['state'] == CircuitState.OPEN.value
        
        # Reset circuit breaker manually to test recovery
        self.client.reset_circuit_breaker()
        
        # Verify circuit is closed
        assert self.client.circuit_breaker.get_state()['state'] == CircuitState.CLOSED.value
    
    def test_rate_limiting_integration(self):
        """Test rate limiting works with circuit breaker."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_get.return_value = mock_response
            
            # Make rapid requests
            start_time = time.time()
            for i in range(3):
                self.client._make_request(f'rate-test-{i}')
            end_time = time.time()
            
            # Should have taken at least some time due to rate limiting
            # (2.0 req/sec = 0.5 seconds between requests minimum)
            min_expected_time = 1.0  # 2 intervals of 0.5 seconds
            assert end_time - start_time >= min_expected_time * 0.8  # Allow some tolerance
    
    def test_health_check(self):
        """Test health check functionality."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock successful health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'healthy'}
            mock_get.return_value = mock_response
            
            health = self.client.health_check()
            
            assert health['service'] == 'opendata'
            assert health['healthy'] is True
            assert health['response_time'] > 0
            assert health['error'] is None
            assert 'circuit_breaker_state' in health
            assert 'metrics' in health
    
    def test_health_check_failure(self):
        """Test health check handles failures."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock health check failure
            mock_get.side_effect = Exception("Connection failed")
            
            health = self.client.health_check()
            
            assert health['service'] == 'opendata'
            assert health['healthy'] is False
            assert health['response_time'] > 0
            assert health['error'] == "Connection failed"
    
    def test_client_metrics(self):
        """Test comprehensive client metrics."""
        metrics = self.client.get_client_metrics()
        
        assert metrics['service_name'] == 'opendata'
        assert 'circuit_breaker' in metrics
        assert 'rate_limiter' in metrics
        assert 'configuration' in metrics
        
        # Check configuration is included
        config = metrics['configuration']
        assert config['base_url'] == self.config.base_url
        assert config['timeout'] == self.config.timeout
        assert config['rate_limit'] == self.config.rate_limit
    
    @pytest.mark.api
    def test_real_api_connectivity(self):
        """Test real API connectivity (requires network)."""
        # This test requires actual network connectivity
        # Skip if running in CI or without network
        try:
            health = self.client.health_check()
            
            # If we get here, the API is reachable
            if health['healthy']:
                assert health['response_time'] > 0
                assert health['error'] is None
            else:
                # API might be down or rate limited, but test framework works
                assert health['error'] is not None
                
        except Exception as e:
            # Network issues expected in some environments
            pytest.skip(f"Network connectivity required: {e}")
    
    def test_error_type_classification(self):
        """Test proper error type classification in metrics."""
        test_cases = [
            # (status_code, expected_error_type)
            (400, 'client_error'),
            (429, 'rate_limit'),
            (500, 'server_error'),
            (502, 'server_error'),
            (503, 'server_error'),
        ]
        
        for status_code, expected_error_type in test_cases:
            # Reset metrics for clean test
            metrics_collector.reset_metrics('opendata')
            
            with patch.object(self.client.session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = f"Error {status_code}"
                mock_get.return_value = mock_response
                
                try:
                    self.client._make_request(f'test-{status_code}')
                except (APIError, CircuitBreakerError):
                    pass
                
                # Check error was classified correctly
                metrics = metrics_collector.get_service_metrics('opendata')
                if status_code != 429:  # Rate limit gets retried
                    assert expected_error_type in metrics['error_counts']
                    assert metrics['error_counts'][expected_error_type] >= 1
    
    def test_json_parsing_error_handling(self):
        """Test JSON parsing error handling."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock response with invalid JSON
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response
            
            with pytest.raises(APIError, match="Invalid JSON response"):
                self.client._make_request('invalid-json')
            
            # Check error was recorded in metrics
            metrics = metrics_collector.get_service_metrics('opendata')
            assert 'json_parse_error' in metrics['error_counts']
            assert metrics['error_counts']['json_parse_error'] >= 1
    
    def test_timeout_handling(self):
        """Test timeout error handling."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock timeout
            import requests
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(APIError, match="Maximum retries exceeded"):
                self.client._make_request('timeout-test')
            
            # Check timeout was recorded in metrics
            metrics = metrics_collector.get_service_metrics('opendata')
            assert 'timeout' in metrics['error_counts']
    
    def test_connection_error_handling(self):
        """Test connection error handling."""
        with patch.object(self.client.session, 'get') as mock_get:
            # Mock connection error
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            with pytest.raises(APIError, match="Maximum retries exceeded"):
                self.client._make_request('connection-test')
            
            # Check connection error was recorded
            metrics = metrics_collector.get_service_metrics('opendata')
            assert 'connection_error' in metrics['error_counts']


class TestMetricsIntegration:
    """Integration tests for metrics collection across multiple clients."""
    
    def setup_method(self):
        """Set up test fixtures."""
        metrics_collector.reset_metrics()
    
    def test_global_metrics_aggregation(self):
        """Test global metrics across multiple services."""
        # Simulate metrics from different services
        metrics_collector.record_request('opendata', '/sessions', 'GET', 200, 0.5)
        metrics_collector.record_request('opendata', '/meps', 'GET', 200, 0.3)
        metrics_collector.record_request('eurlex', '/documents', 'GET', 200, 1.2)
        metrics_collector.record_request('eurlex', '/search', 'GET', 500, 2.0, 'server_error')
        
        global_metrics = metrics_collector.get_global_metrics()
        
        assert global_metrics['total_requests'] == 4
        assert global_metrics['successful_requests'] == 3
        assert global_metrics['failed_requests'] == 1
        assert global_metrics['success_rate'] == 75.0
        assert 'opendata' in global_metrics['services']
        assert 'eurlex' in global_metrics['services']
    
    def test_health_report_generation(self):
        """Test comprehensive health report generation."""
        # Add some test metrics
        metrics_collector.record_request('opendata', '/test', 'GET', 200, 0.5)
        metrics_collector.record_request('opendata', '/test', 'GET', 500, 1.0, 'server_error')
        
        health_report = metrics_collector.generate_health_report()
        
        assert 'timestamp' in health_report
        assert 'global_metrics' in health_report
        assert 'service_metrics' in health_report
        assert 'recent_failures' in health_report
        assert 'health_status' in health_report
        assert 'service_health' in health_report
        
        # Check service health classification
        assert 'opendata' in health_report['service_health']
        opendata_health = health_report['service_health']['opendata']
        assert 'status' in opendata_health
        assert opendata_health['success_rate'] == 50.0  # 1 success, 1 failure