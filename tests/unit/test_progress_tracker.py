"""Unit tests for progress tracker."""

import pytest
import json
from datetime import datetime
from pathlib import Path
from src.services.progress_tracker import ProgressTracker
from src.models.session import SessionMetadata


class TestProgressTracker:
    """Test cases for ProgressTracker."""
    
    def test_tracker_initialization(self, temp_data_dir):
        """Test tracker initializes correctly."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        assert tracker.checkpoint_file == checkpoint_file
        assert 'session_id' in tracker.progress
        assert 'sessions_processed' in tracker.progress
        assert tracker.session_start_time is not None
    
    def test_empty_progress_creation(self, temp_data_dir):
        """Test empty progress structure creation."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        progress = tracker._create_empty_progress()
        
        required_keys = [
            'session_id', 'start_time', 'current_phase',
            'sessions_discovered', 'sessions_processed', 'failed_sessions',
            'processing_stats', 'quality_metrics', 'performance_metrics'
        ]
        
        for key in required_keys:
            assert key in progress
    
    def test_session_discovery_tracking(self, temp_data_dir):
        """Test session discovery tracking."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        # Create test session
        session = SessionMetadata(
            session_id="test_session_001",
            date=datetime(2024, 1, 15),
            title="Test Session",
            session_type="plenary",
            language="en",
            status="discovered"
        )
        
        # Mark as discovered
        tracker.mark_session_discovered(session)
        
        assert len(tracker.progress['sessions_discovered']) == 1
        discovered = tracker.progress['sessions_discovered'][0]
        assert discovered['session_id'] == "test_session_001"
        assert 'discovered_at' in discovered
    
    def test_session_processing_lifecycle(self, temp_data_dir):
        """Test complete session processing lifecycle."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        session_id = "test_session_processing"
        
        # Start processing
        tracker.mark_session_processing_start(session_id)
        assert len(tracker.progress.get('sessions_in_progress', [])) == 1
        
        # Mark as processed
        stats = {
            'segments_extracted': 25,
            'speakers_resolved': 22,
            'announcements_classified': 3
        }
        tracker.mark_session_processed(session_id, stats)
        
        # Verify processing completed
        assert len(tracker.progress['sessions_processed']) == 1
        assert len(tracker.progress.get('sessions_in_progress', [])) == 0
        
        processed = tracker.progress['sessions_processed'][0]
        assert processed['session_id'] == session_id
        assert processed['status'] == 'completed'
        assert 'stats' in processed
    
    def test_session_failure_tracking(self, temp_data_dir):
        """Test session failure tracking."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        session_id = "test_session_failure"
        error_message = "Network timeout during verbatim download"
        
        # Mark as failed
        tracker.mark_session_failed(session_id, error_message, retry_count=1)
        
        assert len(tracker.progress['failed_sessions']) == 1
        failed = tracker.progress['failed_sessions'][0]
        assert failed['session_id'] == session_id
        assert failed['error_message'] == error_message
        assert failed['retry_count'] == 1
        assert failed['status'] == 'failed'
    
    def test_pending_sessions_identification(self, temp_data_dir):
        """Test pending sessions identification."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        # Create test sessions
        all_sessions = [
            SessionMetadata(
                session_id=f"session_{i:03d}",
                date=datetime(2024, 1, i + 1),
                title=f"Session {i + 1}",
                session_type="plenary",
                language="en",
                status="discovered"
            )
            for i in range(5)
        ]
        
        # Mark some as processed and failed
        tracker.mark_session_processed("session_001")
        tracker.mark_session_processed("session_002")
        tracker.mark_session_failed("session_003", "Test error")
        
        # Get pending sessions
        pending = tracker.get_pending_sessions(all_sessions)
        
        assert len(pending) == 2  # sessions 004 and 005 should be pending
        pending_ids = {s.session_id for s in pending}
        assert "session_004" in pending_ids
        assert "session_005" in pending_ids
    
    def test_retry_sessions_identification(self, temp_data_dir):
        """Test retry sessions identification."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        # Add failed sessions with different retry counts
        tracker.mark_session_failed("session_retry_1", "Error 1", retry_count=1)
        tracker.mark_session_failed("session_retry_2", "Error 2", retry_count=2)
        tracker.mark_session_failed("session_retry_max", "Error Max", retry_count=3)
        
        # Get sessions for retry (max_retries=3)
        retry_sessions = tracker.get_failed_sessions_for_retry(max_retries=3)
        
        assert len(retry_sessions) == 2
        assert "session_retry_1" in retry_sessions
        assert "session_retry_2" in retry_sessions
        assert "session_retry_max" not in retry_sessions  # Already at max retries
    
    def test_checkpoint_save_and_load(self, temp_data_dir):
        """Test checkpoint saving and loading."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        
        # Create tracker and add some data
        tracker1 = ProgressTracker(str(checkpoint_file))
        tracker1.mark_session_processed("test_session")
        tracker1.set_current_phase("testing")
        tracker1.save_checkpoint({'test_data': 'test_value'})
        
        # Create new tracker and verify data loaded
        tracker2 = ProgressTracker(str(checkpoint_file))
        
        assert len(tracker2.progress['sessions_processed']) == 1
        assert tracker2.progress['current_phase'] == "testing"
        assert tracker2.progress.get('test_data') == 'test_value'
        assert 'last_checkpoint' in tracker2.progress
    
    def test_progress_summary(self, temp_data_dir):
        """Test progress summary generation."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        # Add some test data
        session = SessionMetadata(
            session_id="summary_test",
            date=datetime(2024, 1, 15),
            title="Summary Test Session",
            session_type="plenary",
            language="en",
            status="discovered"
        )
        tracker.mark_session_discovered(session)
        tracker.mark_session_processed("summary_test")
        
        summary = tracker.get_progress_summary()
        
        assert 'session_id' in summary
        assert 'current_phase' in summary
        assert 'sessions' in summary
        assert summary['sessions']['discovered'] == 1
        assert summary['sessions']['processed'] == 1
        assert summary['sessions']['failed'] == 0
    
    def test_progress_report_generation(self, temp_data_dir):
        """Test human-readable progress report generation."""
        checkpoint_file = temp_data_dir / "test_checkpoint.json"
        tracker = ProgressTracker(str(checkpoint_file))
        
        # Add test data
        tracker.mark_session_processed("report_test_1")
        tracker.mark_session_processed("report_test_2")
        tracker.set_current_phase("testing_reports")
        
        report = tracker.generate_progress_report()
        
        assert isinstance(report, str)
        assert "EU Parliament Scraper - Progress Report" in report
        assert "testing_reports" in report
        assert "Processed: 2" in report
        assert "Generated:" in report