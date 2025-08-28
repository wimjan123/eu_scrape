"""Pytest configuration and shared fixtures for EU Parliament scraper tests."""

import os
import sys
import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.core.config import Settings
from src.core.logging import setup_logging, get_logger


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Set up logging for tests
    setup_logging()
    logger = get_logger("test_setup")
    logger.info("Test environment initialized")
    
    # Set test environment variables
    os.environ["EU_SCRAPE_ENV"] = "test"
    os.environ["EU_SCRAPE_LOG_LEVEL"] = "INFO"
    
    yield
    
    # Cleanup after tests
    logger.info("Test environment cleanup")


@pytest.fixture
def test_config() -> Settings:
    """Provide test configuration."""
    config_path = project_root / "config" / "settings.yaml"
    return Settings.load_from_file(str(config_path))


@pytest.fixture
def sample_session_metadata() -> Dict[str, Any]:
    """Sample session metadata for testing."""
    return {
        'session_id': 'test_session_2024_01_15',
        'date': '2024-01-15',
        'title': 'Test Plenary Session - January 15, 2024',
        'session_type': 'plenary',
        'language': 'en',
        'verbatim_url': 'https://www.europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html',
        'agenda_url': 'https://www.europarl.europa.eu/doceo/document/OJ-2024-01-15_EN.html',
        'status': 'discovered'
    }


@pytest.fixture
def sample_verbatim_html() -> str:
    """Sample verbatim report HTML for parser testing."""
    return """
    <html>
    <body>
        <div class="contents">
            <p>The sitting opened at 09:00</p>
            <p><strong>President.</strong> − Good morning, colleagues.</p>
            <p>GARCIA PÉREZ, María (PPE). − Thank you, Mr President. 
            I would like to address the issue of climate change and its impact on our communities.</p>
            <p>The Commission's recent report shows that we need immediate action.</p>
            <p><strong>MÜLLER, Hans (S&D).</strong> − Mr President, I completely agree with my colleague. 
            We must act now to protect our environment for future generations.</p>
            <p>(The sitting was suspended at 12:30)</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_raw_speech_data() -> Dict[str, Any]:
    """Sample raw speech segment data."""
    return {
        'session_id': 'test_session_2024_01_15',
        'sequence_number': 1,
        'speaker_raw': 'GARCIA PÉREZ, María (PPE)',
        'speech_text': 'Thank you, Mr President. I would like to address the issue of climate change.',
        'timestamp_hint': '09:15',
        'is_procedural': False,
        'confidence_score': 0.9,
        'parsing_metadata': {
            'parser_version': '1.0',
            'position': 0,
            'timestamp_extracted': True
        }
    }


@pytest.fixture
def sample_mep_data() -> Dict[str, Any]:
    """Sample MEP data for speaker resolution testing."""
    return {
        'full_name': 'María García Pérez',
        'first_name': 'María',
        'last_name': 'García Pérez',
        'country': 'Spain',
        'political_group': 'Group of the European People\'s Party (Christian Democrats)',
        'political_group_code': 'PPE',
        'national_party': 'Partido Popular',
        'mep_id': 'ESP_12345',
        'period_start': '2019-07-01',
        'period_end': '2024-07-01',
        'is_active': True
    }


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Provide temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "cache").mkdir()
    (data_dir / "progress").mkdir()
    (data_dir / "output").mkdir()
    (data_dir / "meps").mkdir()
    
    return data_dir


@pytest.fixture
def mock_api_response() -> Dict[str, Any]:
    """Mock API response for testing."""
    return {
        'status': 'success',
        'data': [
            {
                'identifier': 'PV-9-2024-01-15',
                'date': '2024-01-15',
                'title': {'en': 'Plenary Session January 15, 2024'},
                'language': 'en',
                'type': 'plenary',
                'documents': [
                    {
                        'type': 'verbatim',
                        'url': 'https://www.europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
                    }
                ]
            }
        ],
        'total': 1,
        'page': 1,
        'per_page': 50
    }


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = {'Content-Type': 'application/json'}
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")
    
    @property
    def text(self):
        return str(self.json_data)


@pytest.fixture
def mock_requests_session(monkeypatch):
    """Mock requests session for API testing."""
    def mock_get(url, **kwargs):
        # Return different responses based on URL
        if "plenary-session-documents" in url:
            return MockResponse({
                'results': [
                    {
                        'identifier': 'PV-9-2024-01-15',
                        'date': '2024-01-15',
                        'title': 'Test Session'
                    }
                ]
            })
        return MockResponse({'error': 'Not found'}, 404)
    
    def mock_post(url, **kwargs):
        return MockResponse({'message': 'Success'})
    
    import requests
    monkeypatch.setattr(requests.Session, 'get', mock_get)
    monkeypatch.setattr(requests.Session, 'post', mock_post)


# Pytest markers for test categorization
pytest.register_marker = pytest.mark


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test requiring network"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )