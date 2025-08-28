#!/usr/bin/env python3
"""
Phase 1 Day 1 Integration Test

This test validates that all core infrastructure components are working
as specified in the comprehensive implementation plan.

SUCCESS CRITERIA:
- Complete project directory structure ✓
- Working configuration management system ✓  
- Basic logging infrastructure ✓
- API connectivity confirmation ✓
- All API clients implemented and functional
- Core data models working
- Utility modules functional
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_project_structure():
    """Test that project structure is complete."""
    print("Testing project structure...")
    
    required_dirs = [
        'src/core', 'src/clients', 'src/models', 'src/parsers',
        'src/services', 'src/utils', 'config', 'tests', 'data'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure complete")
        return True


def test_configuration_system():
    """Test configuration management."""
    print("Testing configuration system...")
    
    try:
        from core.config import Settings
        
        # Test loading configuration
        settings = Settings.load_from_file('config/settings.yaml')
        
        # Validate API configs
        api_config = settings.get_api_config('opendata')
        assert api_config.base_url
        assert api_config.rate_limit > 0
        
        # Validate processing config
        assert settings.processing.max_workers > 0
        assert len(settings.processing.languages) > 0
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False


def test_logging_system():
    """Test logging infrastructure."""
    print("Testing logging system...")
    
    try:
        from core.logging import setup_logging, get_logger
        
        # Setup logging
        setup_logging()
        
        # Get logger and test
        logger = get_logger('test')
        logger.info("Test log message")
        
        print("✅ Logging system working")
        return True
        
    except Exception as e:
        print(f"❌ Logging system failed: {e}")
        return False


def test_data_models():
    """Test data models."""
    print("Testing data models...")
    
    try:
        from models.speech import SpeechSegment, RawSpeechSegment
        from models.speaker import MEPData, SpeakerResolution
        from models.session import SessionMetadata
        from datetime import datetime
        
        # Test SpeechSegment
        segment = SpeechSegment(
            speaker_name="Test Speaker",
            speaker_country="Germany", 
            speaker_party_or_group="EPP",
            segment_start_ts="2024-01-15T14:30:00Z",
            segment_end_ts="2024-01-15T14:32:00Z",
            speech_text="This is a test speech content.",
            is_announcement=False,
            announcement_label=""
        )
        assert segment.speaker_name == "Test Speaker"
        
        # Test MEP data
        mep = MEPData(
            mep_id="test-123",
            full_name="Test MEP",
            first_name="Test",
            family_name="MEP",
            country="Germany",
            political_group="EPP"
        )
        assert mep.full_name == "Test MEP"
        
        print("✅ Data models working")
        return True
        
    except Exception as e:
        print(f"❌ Data models failed: {e}")
        return False


def test_api_clients():
    """Test API client implementations."""
    print("Testing API clients...")
    
    try:
        from clients.opendata_client import OpenDataClient
        from clients.eurlex_client import EURLexClient
        from clients.verbatim_client import VerbatimClient
        from clients.mep_client import MEPClient
        from core.config import APIConfig
        
        # Create test configs
        config = APIConfig(
            base_url="https://test.example.com",
            timeout=30,
            rate_limit=0.5
        )
        
        # Test client initialization
        opendata_client = OpenDataClient(config)
        eurlex_client = EURLexClient(config)
        verbatim_client = VerbatimClient(config)
        mep_client = MEPClient(opendata_client)
        
        # Test basic functionality (without actual API calls)
        assert opendata_client.config.base_url == config.base_url
        assert eurlex_client.config.rate_limit == config.rate_limit
        
        print("✅ API clients initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ API clients failed: {e}")
        return False


def test_utility_modules():
    """Test utility modules."""
    print("Testing utility modules...")
    
    try:
        from utils.text_utils import clean_html_text, normalize_speaker_name, extract_speaker_info
        from utils.time_utils import parse_timestamp, format_iso8601_utc, estimate_speech_duration
        from utils.validation import validate_speech_segment, validate_iso8601_timestamp
        from datetime import datetime, timedelta
        
        # Test text utilities
        html_text = "<p>Test <b>content</b> with HTML</p>"
        clean_text = clean_html_text(html_text)
        assert "Test content with HTML" in clean_text
        
        # Test speaker name normalization
        speaker_name = "Mr. John Smith (EPP-DE)"
        normalized = normalize_speaker_name(speaker_name)
        assert "John Smith" in normalized
        
        # Test time utilities
        dt = datetime(2024, 1, 15, 14, 30)
        iso_time = format_iso8601_utc(dt)
        assert "2024-01-15T14:30:00Z" in iso_time
        
        # Test duration estimation
        text = "This is a short speech with about ten words total."
        duration = estimate_speech_duration(text)
        assert isinstance(duration, timedelta)
        
        # Test validation
        valid_timestamp = "2024-01-15T14:30:00Z"
        is_valid, parsed = validate_iso8601_timestamp(valid_timestamp)
        assert is_valid
        
        print("✅ Utility modules working")
        return True
        
    except Exception as e:
        print(f"❌ Utility modules failed: {e}")
        return False


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("Testing rate limiting...")
    
    try:
        from core.rate_limiter import RateLimiter, ExponentialBackoff
        import time
        
        # Test rate limiter
        limiter = RateLimiter(2.0)  # 2 requests per second
        
        start_time = time.time()
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time
        
        # Should have waited at least 0.5 seconds (1/2.0)
        assert elapsed >= 0.4  # Allow some tolerance
        
        # Test exponential backoff
        backoff = ExponentialBackoff()
        can_continue = backoff.wait()
        assert can_continue is True
        
        print("✅ Rate limiting working")
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting failed: {e}")
        return False


def main():
    """Run all Phase 1 Day 1 tests."""
    print("🔍 EU Parliament Scraper - Phase 1 Day 1 Integration Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration System", test_configuration_system), 
        ("Logging System", test_logging_system),
        ("Data Models", test_data_models),
        ("API Clients", test_api_clients),
        ("Utility Modules", test_utility_modules),
        ("Rate Limiting", test_rate_limiting),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\\n🧪 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - Exception: {e}")
            failed += 1
    
    print(f"\\n{'='*60}")
    print(f"📊 PHASE 1 DAY 1 TEST RESULTS")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print(f"\\n🎉 SUCCESS: Phase 1 Day 1 Complete!")
        print(f"✅ All core infrastructure components are functional.")
        print(f"✅ Ready to proceed to Phase 1 Day 2.")
        return 0
    else:
        print(f"\\n⚠️  ISSUES DETECTED: {failed} test(s) failed.")
        print(f"❌ Address issues before proceeding to Phase 1 Day 2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())