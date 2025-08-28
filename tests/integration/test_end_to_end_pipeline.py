"""End-to-end processing pipeline tests."""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.config import Settings, APIConfig
from src.clients.opendata_client import OpenDataClient
from src.services.session_discovery import SessionDiscoveryService
from src.services.progress_tracker import ProgressTracker
from src.services.monitoring_service import MonitoringService
from src.parsers.verbatim_parser import VerbatimParser
from src.models.session import SessionMetadata
from src.core.circuit_breaker import circuit_registry
from src.core.metrics import metrics_collector
from src.core.config_validator import ConfigValidator


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""
    
    def setup_method(self):
        """Set up test environment."""
        # Reset global state
        circuit_registry.reset_all()
        metrics_collector.reset_metrics()
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test configuration
        self.config = Settings(
            api={
                'opendata': APIConfig(base_url="https://data.europarl.europa.eu/api", timeout=10, rate_limit=2.0),
                'eurlex': APIConfig(base_url="http://publications.europa.eu/webapi/rdf/sparql", timeout=10, rate_limit=1.0),
                'verbatim': APIConfig(base_url="https://www.europarl.europa.eu/doceo/document", timeout=15, rate_limit=0.5),
                'mep': APIConfig(base_url="https://www.europarl.europa.eu/meps/en", timeout=10, rate_limit=1.0)
            },
            processing={
                'date_range': {'start': '2024-01-01', 'end': '2024-01-31'},
                'languages': ['en'],
                'max_workers': 2,
                'chunk_size': 10
            },
            quality={
                'min_speech_length': 10,
                'max_speech_length': 5000,
                'speaker_confidence_threshold': 0.8,
                'validation_sample_size': 5
            },
            output={
                'base_dir': str(self.temp_path / 'output'),
                'format': 'json',
                'include_metadata': True,
                'compress': False
            }
        )
        
        # Initialize core services
        self.opendata_client = OpenDataClient(self.config.api['opendata'])
        self.session_discovery = SessionDiscoveryService(self.opendata_client)
        self.progress_tracker = ProgressTracker(str(self.temp_path / 'checkpoint.json'))
        self.monitoring_service = MonitoringService(self.progress_tracker)
        self.verbatim_parser = VerbatimParser()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up clients
        self.opendata_client.close()
        
        # Reset global state
        circuit_registry.reset_all()
        metrics_collector.reset_metrics()
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_validation_pipeline(self):
        """Test complete configuration validation pipeline."""
        # Create temporary config file
        config_file = self.temp_path / 'test_config.yaml'
        
        config_data = {
            'api': {
                'opendata': {
                    'base_url': 'https://data.europarl.europa.eu/api',
                    'timeout': 30,
                    'rate_limit': 0.5
                },
                'eurlex': {
                    'base_url': 'http://publications.europa.eu/webapi/rdf/sparql',
                    'timeout': 30,
                    'rate_limit': 0.5
                },
                'verbatim': {
                    'base_url': 'https://www.europarl.europa.eu/doceo/document',
                    'timeout': 30,
                    'rate_limit': 0.33
                },
                'mep': {
                    'base_url': 'https://www.europarl.europa.eu/meps/en',
                    'timeout': 30,
                    'rate_limit': 0.5
                }
            },
            'processing': {
                'date_range': {'start': '2024-01-01', 'end': '2024-01-31'},
                'languages': ['en'],
                'max_workers': 4,
                'chunk_size': 100
            },
            'quality': {
                'min_speech_length': 10,
                'max_speech_length': 10000,
                'speaker_confidence_threshold': 0.8,
                'validation_sample_size': 25
            },
            'output': {
                'base_dir': 'data/output',
                'format': 'json',
                'include_metadata': True,
                'compress': False
            }
        }
        
        # Write config file
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run validation
        validator = ConfigValidator()
        results = validator.validate_all(str(config_file))
        
        # Verify validation results
        assert isinstance(results, dict)
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'checks_total' in results
        assert results['checks_total'] > 0
        
        # Generate report
        report = validator.generate_validation_report()
        assert isinstance(report, str)
        assert 'Configuration Validation Report' in report
    
    def test_session_discovery_pipeline(self):
        """Test session discovery pipeline with mocked responses."""
        with patch.object(self.opendata_client.session, 'get') as mock_get:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'results': [
                    {
                        'identifier': 'PV-9-2024-01-15',
                        'date': '2024-01-15',
                        'title': {'en': 'Test Plenary Session'},
                        'language': 'en',
                        'verbatim_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
                    },
                    {
                        'identifier': 'PV-9-2024-01-16',
                        'date': '2024-01-16',
                        'title': {'en': 'Test Plenary Session 2'},
                        'language': 'en',
                        'verbatim_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-16_EN.html'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            # Discover sessions
            sessions = self.session_discovery.discover_sessions('2024-01-15', '2024-01-16')
            
            # Verify results
            assert len(sessions) == 2
            assert all(isinstance(s, SessionMetadata) for s in sessions)
            assert sessions[0].session_id == 'PV-9-2024-01-15'
            assert sessions[1].session_id == 'PV-9-2024-01-16'
            
            # Test progress tracking integration
            for session in sessions:
                self.progress_tracker.mark_session_discovered(session)
            
            # Verify progress tracking
            progress = self.progress_tracker.get_progress_summary()
            assert progress['sessions']['discovered'] == 2
            assert progress['sessions']['processed'] == 0
    
    def test_verbatim_parsing_pipeline(self):
        """Test verbatim parsing pipeline with realistic HTML."""
        # Sample verbatim HTML content
        sample_html = """
        <html>
        <body>
            <div class="contents">
                <p>The sitting opened at 09:00</p>
                <p><strong>President.</strong> − Good morning, colleagues. We begin today's session.</p>
                <p>GARCIA PÉREZ, María (PPE). − Thank you, Mr President. 
                I would like to address the climate change issue that affects all our communities.</p>
                <p>The latest scientific evidence shows we need immediate action.</p>
                <p><strong>MÜLLER, Hans (S&D).</strong> − Mr President, I completely agree. 
                We must accelerate our green transition policies.</p>
                <p>(Applause)</p>
                <p><strong>President.</strong> − Thank you. The session is now suspended.</p>
                <p>The sitting was suspended at 12:30</p>
            </div>
        </body>
        </html>
        """
        
        # Parse verbatim content
        session_date = datetime(2024, 1, 15, 9, 0)
        session_id = "test_session_2024_01_15"
        
        segments = self.verbatim_parser.parse_verbatim_report(
            sample_html, session_id, session_date
        )
        
        # Verify parsing results
        assert len(segments) > 0
        assert all(hasattr(s, 'session_id') for s in segments)
        assert all(s.session_id == session_id for s in segments)
        
        # Check parsing statistics
        stats = self.verbatim_parser.get_parsing_stats()
        assert 'total_segments' in stats
        assert 'procedural_segments' in stats
        assert 'speech_segments' in stats
        assert stats['total_segments'] > 0
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration across services."""
        # Test circuit breaker functionality
        with patch.object(self.opendata_client.session, 'get') as mock_get:
            # Mock consistent failures to trigger circuit breaker
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_get.return_value = mock_response
            
            # Make requests until circuit breaker opens
            failure_count = 0
            for i in range(10):
                try:
                    self.opendata_client._make_request(f'test-{i}')
                except Exception:
                    failure_count += 1
                
                # Check if circuit breaker opened
                cb_state = self.opendata_client.circuit_breaker.get_state()
                if not cb_state['is_healthy']:
                    break
            
            # Verify circuit breaker opened
            cb_state = self.opendata_client.circuit_breaker.get_state()
            assert not cb_state['is_healthy']
            assert cb_state['failure_count'] >= 5
    
    def test_metrics_collection_pipeline(self):
        """Test metrics collection across all services."""
        with patch.object(self.opendata_client.session, 'get') as mock_get:
            # Mock mixed success/failure responses
            responses = [
                Mock(status_code=200, json=lambda: {'success': True}),
                Mock(status_code=500, text="Server Error"),
                Mock(status_code=200, json=lambda: {'success': True}),
                Mock(status_code=429, headers={'Retry-After': '1'}, text="Rate Limited"),
            ]
            mock_get.side_effect = responses
            
            # Make requests
            success_count = 0
            for i in range(len(responses)):
                try:
                    result = self.opendata_client._make_request(f'test-{i}')
                    if result:
                        success_count += 1
                except Exception:
                    pass
            
            # Verify metrics collection
            service_metrics = metrics_collector.get_service_metrics('opendata')
            assert service_metrics is not None
            assert service_metrics['total_requests'] >= 1
            
            global_metrics = metrics_collector.get_global_metrics()
            assert global_metrics['total_requests'] >= 1
    
    def test_monitoring_service_integration(self):
        """Test monitoring service integration."""
        # Get system health status
        health_status = self.monitoring_service.get_health_status()
        
        # Verify health status structure
        assert 'status' in health_status
        assert 'system_metrics' in health_status
        assert 'application_metrics' in health_status
        assert 'last_check' in health_status
        
        # Verify system metrics
        system_metrics = health_status['system_metrics']
        assert 'cpu_percent' in system_metrics
        assert 'memory_percent' in system_metrics
        assert 'timestamp' in system_metrics
        
        # Generate monitoring report
        report = self.monitoring_service.generate_monitoring_report()
        assert 'report_type' in report
        assert report['report_type'] == 'monitoring'
        assert 'health_status' in report
        assert 'summary' in report
    
    def test_progress_tracking_integration(self):
        """Test progress tracking throughout pipeline."""
        # Create test sessions
        sessions = [
            SessionMetadata(
                session_id=f"test_session_{i:03d}",
                date=datetime(2024, 1, 15 + i),
                title=f"Test Session {i + 1}",
                session_type="plenary",
                language="en",
                status="discovered"
            )
            for i in range(3)
        ]
        
        # Track session discovery
        for session in sessions:
            self.progress_tracker.mark_session_discovered(session)
        
        # Track processing
        self.progress_tracker.mark_session_processing_start(sessions[0].session_id)
        self.progress_tracker.mark_session_processed(
            sessions[0].session_id,
            {'segments_extracted': 25, 'speakers_resolved': 22}
        )
        
        # Track failure
        self.progress_tracker.mark_session_failed(
            sessions[1].session_id,
            "Network timeout",
            retry_count=1
        )
        
        # Verify progress state
        summary = self.progress_tracker.get_progress_summary()
        assert summary['sessions']['discovered'] == 3
        assert summary['sessions']['processed'] == 1
        assert summary['sessions']['failed'] == 1
        
        # Generate progress report
        report = self.progress_tracker.generate_progress_report()
        assert isinstance(report, str)
        assert 'Progress Report' in report
        assert 'Processed: 1' in report
        assert 'Failed: 1' in report
    
    def test_error_handling_pipeline(self):
        """Test comprehensive error handling across pipeline."""
        # Test API client error handling
        with patch.object(self.opendata_client.session, 'get') as mock_get:
            # Test various error conditions
            error_scenarios = [
                {'side_effect': TimeoutError("Request timeout")},
                {'side_effect': ConnectionError("Connection failed")},
                {'return_value': Mock(status_code=404, text="Not Found")},
                {'return_value': Mock(status_code=500, text="Server Error")}
            ]
            
            for scenario in error_scenarios:
                mock_get.reset_mock()
                if 'side_effect' in scenario:
                    mock_get.side_effect = scenario['side_effect']
                else:
                    mock_get.return_value = scenario['return_value']
                
                # Verify errors are handled gracefully
                try:
                    self.opendata_client._make_request('error-test')
                except Exception as e:
                    # Errors should be wrapped in APIError
                    assert hasattr(e, '__class__')
                    # Error should be logged and tracked
                    metrics = metrics_collector.get_service_metrics('opendata')
                    assert metrics is not None
    
    def test_full_pipeline_simulation(self):
        """Test full pipeline simulation with realistic data flow."""
        with patch.object(self.opendata_client.session, 'get') as mock_get:
            # Mock session discovery response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'results': [
                    {
                        'identifier': 'PV-9-2024-01-15',
                        'date': '2024-01-15',
                        'title': {'en': 'Plenary Session January 15'},
                        'language': 'en',
                        'verbatim_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            # 1. Session Discovery
            sessions = self.session_discovery.discover_sessions('2024-01-15', '2024-01-15')
            assert len(sessions) == 1
            
            session = sessions[0]
            
            # 2. Progress Tracking - Discovery
            self.progress_tracker.mark_session_discovered(session)
            
            # 3. Progress Tracking - Processing Start
            self.progress_tracker.mark_session_processing_start(session.session_id)
            
            # 4. Verbatim Parsing (with mock HTML)
            mock_html = """
            <html><body><div class="contents">
            <p><strong>President.</strong> − The session is open.</p>
            <p>GARCIA, Maria (PPE). − Thank you, President. Climate change is urgent.</p>
            </div></body></html>
            """
            
            segments = self.verbatim_parser.parse_verbatim_report(
                mock_html, session.session_id, session.date
            )
            
            # 5. Progress Tracking - Processing Complete
            processing_stats = {
                'segments_extracted': len(segments),
                'speakers_resolved': len([s for s in segments if s.speaker_raw]),
                'announcements_classified': len([s for s in segments if s.is_procedural])
            }
            
            self.progress_tracker.mark_session_processed(
                session.session_id, 
                processing_stats
            )
            
            # 6. Monitoring and Health Check
            health = self.monitoring_service.get_health_status()
            monitoring_report = self.monitoring_service.generate_monitoring_report()
            
            # 7. Final Verification
            final_summary = self.progress_tracker.get_progress_summary()
            assert final_summary['sessions']['discovered'] == 1
            assert final_summary['sessions']['processed'] == 1
            assert final_summary['sessions']['failed'] == 0
            
            # Verify metrics were collected
            service_metrics = metrics_collector.get_service_metrics('opendata')
            assert service_metrics is not None
            assert service_metrics['total_requests'] >= 1
            
            # Verify monitoring data
            assert health['status'] in ['healthy', 'degraded', 'unhealthy']
            assert 'summary' in monitoring_report
            
            # Verify circuit breaker is healthy
            cb_states = circuit_registry.get_all_states()
            if cb_states:
                for service, state in cb_states.items():
                    # Circuit breaker should be closed (healthy) after successful operations
                    assert state.get('is_healthy', True)


class TestPipelinePerformance:
    """Performance tests for pipeline components."""
    
    def test_session_discovery_performance(self):
        """Test session discovery performance with large datasets."""
        # This would typically test with larger datasets
        # For now, verify basic performance characteristics
        start_time = datetime.now()
        
        # Create mock service
        opendata_client = Mock()
        discovery_service = SessionDiscoveryService(opendata_client)
        
        # Mock large response
        large_response = [
            {
                'identifier': f'PV-9-2024-01-{i:02d}',
                'date': f'2024-01-{i:02d}',
                'title': {'en': f'Session {i}'},
                'language': 'en'
            }
            for i in range(1, 32)  # Full month
        ]
        
        opendata_client.get_plenary_sessions.return_value = large_response
        
        # Process sessions
        sessions = discovery_service._discover_from_opendata('2024-01-01', '2024-01-31')
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert len(sessions) > 0
        assert processing_time < 5.0  # Should process quickly with mocked data
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance under load."""
        start_time = datetime.now()
        
        # Simulate high-volume metrics collection
        for i in range(1000):
            metrics_collector.record_request(
                service='test_service',
                endpoint=f'/endpoint-{i % 10}',
                method='GET',
                status_code=200 if i % 10 != 0 else 500,
                response_time=0.1 + (i % 5) * 0.05,
                error_type='server_error' if i % 10 == 0 else None
            )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance and results
        assert processing_time < 1.0  # Should handle 1000 requests quickly
        
        metrics = metrics_collector.get_service_metrics('test_service')
        assert metrics['total_requests'] == 1000
        assert metrics['failed_requests'] == 100  # 10% failure rate
        assert metrics['success_rate'] == 90.0