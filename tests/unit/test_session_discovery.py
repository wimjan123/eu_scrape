"""Unit tests for session discovery service."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.services.session_discovery import SessionDiscoveryService
from src.clients.opendata_client import OpenDataClient
from src.clients.eurlex_client import EURLexClient
from src.models.session import SessionMetadata


class TestSessionDiscoveryService:
    """Test cases for SessionDiscoveryService."""
    
    def test_service_initialization(self):
        """Test service initializes correctly."""
        opendata_client = Mock(spec=OpenDataClient)
        eurlex_client = Mock(spec=EURLexClient)
        
        service = SessionDiscoveryService(opendata_client, eurlex_client)
        
        assert service.opendata_client == opendata_client
        assert service.eurlex_client == eurlex_client
        assert service.sessions_cache == {}
        assert service.discovery_cache_file.name == "session_discovery.json"
    
    def test_opendata_session_conversion(self, sample_session_metadata):
        """Test OpenData session conversion."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        raw_session = {
            'identifier': 'PV-9-2024-01-15',
            'date': '2024-01-15',
            'title': {'en': 'Test Plenary Session'},
            'language': 'en',
            'verbatim_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
        }
        
        session = service._convert_opendata_session(raw_session)
        
        assert session is not None
        assert session.session_id == 'PV-9-2024-01-15'
        assert session.date == datetime(2024, 1, 15)
        assert session.title == 'Test Plenary Session'
        assert session.session_type == 'plenary'
        assert session.language == 'en'
        assert session.verbatim_url.endswith('_EN.html')
    
    def test_opendata_session_conversion_variations(self):
        """Test various OpenData session formats."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        test_cases = [
            # String title instead of dict
            {
                'identifier': 'test-001',
                'date': '2024-01-15',
                'title': 'Simple String Title',
                'language': 'en'
            },
            # Missing optional fields
            {
                'identifier': 'test-002',
                'date': '2024-01-16',
                'language': 'en'
            },
            # Complex language field
            {
                'identifier': 'test-003',
                'date': '2024-01-17',
                'title': {'en': 'English Title', 'fr': 'Titre Français'},
                'language': {'en': 'en'}
            }
        ]
        
        for raw_session in test_cases:
            session = service._convert_opendata_session(raw_session)
            assert session is not None
            assert session.session_id == raw_session['identifier']
            assert session.language == 'en'
    
    def test_eurlex_document_conversion(self):
        """Test EUR-Lex document conversion."""
        opendata_client = Mock(spec=OpenDataClient)
        eurlex_client = Mock(spec=EURLexClient)
        service = SessionDiscoveryService(opendata_client, eurlex_client)
        
        document = {
            'identifier': 'CELEX:52024IP0015',
            'date': '2024-01-15',
            'title': 'Test EUR-Lex Document',
            'language': 'en',
            'document_uri': 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52024IP0015'
        }
        
        session = service._convert_eurlex_document(document)
        
        assert session is not None
        assert session.session_id.startswith('eurlex_')
        assert session.date == datetime(2024, 1, 15)
        assert session.title == 'Test EUR-Lex Document'
        assert session.verbatim_url == document['document_uri']
    
    def test_verbatim_url_extraction(self):
        """Test verbatim URL extraction from session data."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        test_cases = [
            # Direct URL field
            {
                'verbatim_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html',
                'expected': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
            },
            # Document URL with CRE indicator
            {
                'document_url': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html',
                'expected': 'https://europarl.europa.eu/doceo/document/CRE-9-2024-01-15_EN.html'
            },
            # Documents array structure
            {
                'documents': [
                    {'type': 'agenda', 'url': 'https://example.com/agenda'},
                    {'type': 'verbatim', 'url': 'https://example.com/verbatim'}
                ],
                'expected': 'https://example.com/verbatim'
            },
            # No verbatim URL found
            {
                'other_field': 'value',
                'expected': None
            }
        ]
        
        for case in test_cases:
            result = service._extract_verbatim_url(case)
            if case['expected']:
                assert result == case['expected']
            else:
                assert result is None
    
    def test_agenda_url_extraction(self):
        """Test agenda URL extraction from session data."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        test_cases = [
            # Direct agenda URL
            {
                'agenda_url': 'https://europarl.europa.eu/agenda/2024-01-15',
                'expected': 'https://europarl.europa.eu/agenda/2024-01-15'
            },
            # Documents array with agenda
            {
                'documents': [
                    {'type': 'agenda', 'url': 'https://example.com/agenda'},
                    {'type': 'other', 'url': 'https://example.com/other'}
                ],
                'expected': 'https://example.com/agenda'
            },
            # No agenda URL
            {
                'verbatim_url': 'https://example.com/verbatim',
                'expected': None
            }
        ]
        
        for case in test_cases:
            result = service._extract_agenda_url(case)
            if case['expected']:
                assert result == case['expected']
            else:
                assert result is None
    
    def test_session_data_merging(self):
        """Test merging session data from multiple sources."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        # Primary sessions (from OpenData)
        primary_sessions = [
            SessionMetadata(
                session_id="session_001",
                date=datetime(2024, 1, 15),
                title="Primary Session",
                session_type="plenary",
                language="en",
                status="discovered"
            )
        ]
        
        # Secondary sessions (from EUR-Lex) - one overlapping, one new
        secondary_sessions = [
            SessionMetadata(
                session_id="session_001",  # Same as primary
                date=datetime(2024, 1, 15),
                title="Enhanced Session",
                session_type="plenary",
                language="en",
                verbatim_url="https://eur-lex.europa.eu/enhanced",
                status="discovered"
            ),
            SessionMetadata(
                session_id="session_002",  # New session
                date=datetime(2024, 1, 16),
                title="EUR-Lex Only Session",
                session_type="plenary",
                language="en",
                status="discovered"
            )
        ]
        
        merged = service._merge_session_data(primary_sessions, secondary_sessions)
        
        assert len(merged) == 2
        
        # Find the merged session
        session_001 = next(s for s in merged if s.session_id == "session_001")
        assert session_001.verbatim_url == "https://eur-lex.europa.eu/enhanced"
        
        # Find the new session
        session_002 = next(s for s in merged if s.session_id == "session_002")
        assert session_002.title == "EUR-Lex Only Session"
    
    def test_session_metadata_validation(self):
        """Test session metadata validation."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        # Valid session
        valid_session = SessionMetadata(
            session_id="valid_session_001",
            date=datetime(2024, 1, 15),
            title="Valid Session",
            session_type="plenary",
            language="en",
            status="discovered"
        )
        assert service._validate_session_metadata(valid_session)
        
        # Invalid sessions
        invalid_cases = [
            # Short session ID
            SessionMetadata(
                session_id="x",
                date=datetime(2024, 1, 15),
                title="Short ID",
                session_type="plenary",
                language="en",
                status="discovered"
            ),
            # Future date
            SessionMetadata(
                session_id="future_session",
                date=datetime(2025, 12, 31),
                title="Future Session",
                session_type="plenary",
                language="en",
                status="discovered"
            ),
            # Wrong session type
            SessionMetadata(
                session_id="wrong_type",
                date=datetime(2024, 1, 15),
                title="Wrong Type",
                session_type="committee",  # Not plenary
                language="en",
                status="discovered"
            )
        ]
        
        for invalid_session in invalid_cases:
            assert not service._validate_session_metadata(invalid_session)
    
    def test_session_metadata_enrichment(self):
        """Test session metadata enrichment."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        
        # Session without verbatim URL
        session = SessionMetadata(
            session_id="enrich_test",
            date=datetime(2024, 1, 15),
            title="",  # Empty title
            session_type="plenary",
            language="en",
            status="discovered"
        )
        
        enriched = service._enrich_session_metadata(session)
        
        # Should have generated probable URL
        assert enriched.verbatim_url is not None
        assert "2024-01-15" in enriched.verbatim_url
        assert "CRE-9" in enriched.verbatim_url
        
        # Should have enhanced title
        assert enriched.title != ""
        assert "January 15, 2024" in enriched.title
    
    @patch('src.services.session_discovery.json.load')
    @patch('src.services.session_discovery.Path.exists')
    def test_cache_loading(self, mock_exists, mock_json_load, temp_data_dir):
        """Test cache loading functionality."""
        opendata_client = Mock(spec=OpenDataClient)
        service = SessionDiscoveryService(opendata_client)
        service.discovery_cache_file = temp_data_dir / "test_cache.json"
        
        # Mock cache exists and is recent
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "2024-01-15_2024-01-16": {
                "timestamp": datetime.now().isoformat(),
                "sessions": [
                    {
                        "session_id": "cached_session",
                        "date": "2024-01-15T00:00:00",
                        "title": "Cached Session",
                        "session_type": "plenary",
                        "language": "en",
                        "verbatim_url": None,
                        "agenda_url": None,
                        "status": "discovered"
                    }
                ]
            }
        }
        
        # Load from cache
        cached_sessions = service._load_from_cache("2024-01-15_2024-01-16")
        
        assert cached_sessions is not None
        assert len(cached_sessions) == 1
        assert cached_sessions[0].session_id == "cached_session"