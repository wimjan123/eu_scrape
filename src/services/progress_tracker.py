"""Progress tracking and checkpoint system for EU Parliament scraper."""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from ..models.session import SessionMetadata
from ..core.exceptions import CheckpointError
from ..core.logging import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """Comprehensive progress tracking with checkpoint/resume capability."""
    
    def __init__(self, checkpoint_file: str = "data/progress/checkpoint.json"):
        """
        Initialize progress tracker.
        
        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.progress = self._load_progress()
        self.session_start_time = datetime.utcnow()
        
        logger.info(
            "Progress tracker initialized",
            checkpoint_file=str(self.checkpoint_file),
            existing_progress=bool(self.progress.get('sessions_processed'))
        )
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load existing progress from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    progress = json.load(f)
                
                logger.info(
                    "Loaded existing checkpoint",
                    sessions_processed=len(progress.get('sessions_processed', [])),
                    failed_sessions=len(progress.get('failed_sessions', [])),
                    last_checkpoint=progress.get('last_checkpoint')
                )
                
                return progress
                
            except Exception as e:
                logger.error("Failed to load checkpoint", error=str(e))
                return self._create_empty_progress()
        
        return self._create_empty_progress()
    
    def _create_empty_progress(self) -> Dict[str, Any]:
        """Create empty progress structure."""
        return {
            'session_id': f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.utcnow().isoformat(),
            'last_checkpoint': None,
            'current_phase': 'initialization',
            'sessions_discovered': [],
            'sessions_processed': [],
            'failed_sessions': [],
            'processing_stats': {
                'total_segments_extracted': 0,
                'total_speakers_resolved': 0,
                'total_announcements_classified': 0,
                'processing_errors': 0
            },
            'quality_metrics': {
                'session_success_rate': 0.0,
                'segment_extraction_rate': 0.0,
                'speaker_resolution_rate': 0.0,
                'classification_accuracy': 0.0
            },
            'performance_metrics': {
                'sessions_per_hour': 0.0,
                'average_session_processing_time': 0.0,
                'total_processing_time': 0.0
            }
        }
    
    def save_checkpoint(self, additional_data: Dict[str, Any] = None) -> None:
        """Save current progress to checkpoint file."""
        try:
            self.progress['last_checkpoint'] = datetime.utcnow().isoformat()
            
            # Add additional data if provided
            if additional_data:
                self.progress.update(additional_data)
            
            # Calculate performance metrics
            self._update_performance_metrics()
            
            # Atomic write to prevent corruption
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.rename(self.checkpoint_file)
            
            logger.info(
                "Checkpoint saved",
                sessions_processed=len(self.progress['sessions_processed']),
                current_phase=self.progress['current_phase']
            )
            
        except Exception as e:
            logger.error("Failed to save checkpoint", error=str(e))
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def mark_session_discovered(self, session: SessionMetadata) -> None:
        """Mark session as discovered."""
        session_entry = {
            'session_id': session.session_id,
            'date': session.date.isoformat(),
            'title': session.title,
            'discovered_at': datetime.utcnow().isoformat()
        }
        
        # Avoid duplicates
        if not any(s['session_id'] == session.session_id for s in self.progress['sessions_discovered']):
            self.progress['sessions_discovered'].append(session_entry)
            
            logger.debug("Session marked as discovered", session_id=session.session_id)
    
    def mark_session_processing_start(self, session_id: str) -> None:
        """Mark session processing start."""
        processing_entry = {
            'session_id': session_id,
            'processing_start': datetime.utcnow().isoformat(),
            'status': 'processing'
        }
        
        # Add to processing list or update existing
        self.progress.setdefault('sessions_in_progress', [])
        
        # Remove from in_progress if already there
        self.progress['sessions_in_progress'] = [
            s for s in self.progress['sessions_in_progress'] 
            if s['session_id'] != session_id
        ]
        
        self.progress['sessions_in_progress'].append(processing_entry)
        
        logger.info("Session processing started", session_id=session_id)
    
    def mark_session_processed(self, session_id: str, processing_stats: Dict[str, Any] = None) -> None:
        """Mark session as successfully processed."""
        processing_entry = {
            'session_id': session_id,
            'processed_at': datetime.utcnow().isoformat(),
            'status': 'completed'
        }
        
        if processing_stats:
            processing_entry['stats'] = processing_stats
            
            # Update global stats
            stats = self.progress['processing_stats']
            stats['total_segments_extracted'] += processing_stats.get('segments_extracted', 0)
            stats['total_speakers_resolved'] += processing_stats.get('speakers_resolved', 0)
            stats['total_announcements_classified'] += processing_stats.get('announcements_classified', 0)
        
        # Move from in_progress to processed
        if session_id not in [s['session_id'] for s in self.progress['sessions_processed']]:
            self.progress['sessions_processed'].append(processing_entry)
        
        # Remove from in_progress
        if hasattr(self.progress, 'sessions_in_progress'):
            self.progress['sessions_in_progress'] = [
                s for s in self.progress.get('sessions_in_progress', [])
                if s['session_id'] != session_id
            ]
        
        self.save_checkpoint()
        
        logger.info(
            "Session marked as processed",
            session_id=session_id,
            total_processed=len(self.progress['sessions_processed'])
        )
    
    def mark_session_failed(self, session_id: str, error_message: str, 
                          retry_count: int = 0) -> None:
        """Mark session as failed."""
        failure_entry = {
            'session_id': session_id,
            'failed_at': datetime.utcnow().isoformat(),
            'error_message': error_message,
            'retry_count': retry_count,
            'status': 'failed'
        }
        
        # Remove existing failure entry for same session
        self.progress['failed_sessions'] = [
            s for s in self.progress['failed_sessions'] 
            if s['session_id'] != session_id
        ]
        
        self.progress['failed_sessions'].append(failure_entry)
        
        # Remove from in_progress
        if hasattr(self.progress, 'sessions_in_progress'):
            self.progress['sessions_in_progress'] = [
                s for s in self.progress.get('sessions_in_progress', [])
                if s['session_id'] != session_id
            ]
        
        # Update error count
        self.progress['processing_stats']['processing_errors'] += 1
        
        self.save_checkpoint()
        
        logger.warning(
            "Session marked as failed",
            session_id=session_id,
            error_message=error_message,
            retry_count=retry_count
        )
    
    def get_pending_sessions(self, all_sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Get sessions that still need processing."""
        processed_ids = set(s['session_id'] for s in self.progress['sessions_processed'])
        failed_ids = set(s['session_id'] for s in self.progress['failed_sessions'])
        
        pending_sessions = []
        for session in all_sessions:
            if session.session_id not in processed_ids and session.session_id not in failed_ids:
                pending_sessions.append(session)
        
        logger.info(
            "Identified pending sessions",
            total_sessions=len(all_sessions),
            processed=len(processed_ids),
            failed=len(failed_ids),
            pending=len(pending_sessions)
        )
        
        return pending_sessions
    
    def get_failed_sessions_for_retry(self, max_retries: int = 3) -> List[str]:
        """Get failed sessions that can be retried."""
        retry_sessions = []
        
        for failure in self.progress['failed_sessions']:
            if failure.get('retry_count', 0) < max_retries:
                retry_sessions.append(failure['session_id'])
        
        logger.info("Sessions available for retry", count=len(retry_sessions))
        return retry_sessions
    
    def set_current_phase(self, phase: str) -> None:
        """Set current processing phase."""
        self.progress['current_phase'] = phase
        self.progress[f'phase_{phase}_start'] = datetime.utcnow().isoformat()
        
        logger.info("Processing phase updated", phase=phase)
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            current_time = datetime.utcnow()
            session_start = datetime.fromisoformat(self.progress['start_time'])
            total_time_hours = (current_time - session_start).total_seconds() / 3600
            
            processed_count = len(self.progress['sessions_processed'])
            
            if total_time_hours > 0:
                self.progress['performance_metrics']['sessions_per_hour'] = processed_count / total_time_hours
            
            if processed_count > 0:
                self.progress['performance_metrics']['average_session_processing_time'] = (
                    total_time_hours / processed_count * 60  # Convert to minutes
                )
            
            self.progress['performance_metrics']['total_processing_time'] = total_time_hours
            
            # Update quality metrics
            self._update_quality_metrics()
            
        except Exception as e:
            logger.warning("Failed to update performance metrics", error=str(e))
    
    def _update_quality_metrics(self) -> None:
        """Update quality metrics."""
        try:
            total_sessions = len(self.progress['sessions_discovered'])
            processed_sessions = len(self.progress['sessions_processed'])
            failed_sessions = len(self.progress['failed_sessions'])
            
            if total_sessions > 0:
                self.progress['quality_metrics']['session_success_rate'] = processed_sessions / total_sessions
            
            # Calculate other quality metrics from processing stats
            stats = self.progress['processing_stats']
            if stats['total_segments_extracted'] > 0:
                self.progress['quality_metrics']['speaker_resolution_rate'] = (
                    stats['total_speakers_resolved'] / stats['total_segments_extracted']
                )
            
        except Exception as e:
            logger.warning("Failed to update quality metrics", error=str(e))
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary."""
        return {
            'session_id': self.progress.get('session_id'),
            'current_phase': self.progress.get('current_phase'),
            'start_time': self.progress.get('start_time'),
            'last_checkpoint': self.progress.get('last_checkpoint'),
            'sessions': {
                'discovered': len(self.progress.get('sessions_discovered', [])),
                'processed': len(self.progress.get('sessions_processed', [])),
                'failed': len(self.progress.get('failed_sessions', [])),
                'in_progress': len(self.progress.get('sessions_in_progress', []))
            },
            'processing_stats': self.progress.get('processing_stats', {}),
            'quality_metrics': self.progress.get('quality_metrics', {}),
            'performance_metrics': self.progress.get('performance_metrics', {})
        }
    
    def generate_progress_report(self) -> str:
        """Generate human-readable progress report."""
        summary = self.get_progress_summary()
        
        report_lines = [
            f"EU Parliament Scraper - Progress Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"",
            f"Session: {summary['session_id']}",
            f"Current Phase: {summary['current_phase']}",
            f"Started: {summary['start_time']}",
            f"",
            f"Sessions Status:",
            f"  Discovered: {summary['sessions']['discovered']}",
            f"  Processed: {summary['sessions']['processed']}",
            f"  Failed: {summary['sessions']['failed']}",
            f"  In Progress: {summary['sessions']['in_progress']}",
            f"",
            f"Processing Statistics:",
            f"  Total Segments: {summary['processing_stats'].get('total_segments_extracted', 0)}",
            f"  Speakers Resolved: {summary['processing_stats'].get('total_speakers_resolved', 0)}",
            f"  Announcements Classified: {summary['processing_stats'].get('total_announcements_classified', 0)}",
            f"  Processing Errors: {summary['processing_stats'].get('processing_errors', 0)}",
            f"",
            f"Quality Metrics:",
            f"  Session Success Rate: {summary['quality_metrics'].get('session_success_rate', 0):.2%}",
            f"  Speaker Resolution Rate: {summary['quality_metrics'].get('speaker_resolution_rate', 0):.2%}",
            f"",
            f"Performance:",
            f"  Sessions/Hour: {summary['performance_metrics'].get('sessions_per_hour', 0):.2f}",
            f"  Avg Processing Time: {summary['performance_metrics'].get('average_session_processing_time', 0):.2f} min",
            f"  Total Runtime: {summary['performance_metrics'].get('total_processing_time', 0):.2f} hours"
        ]
        
        return "\\n".join(report_lines)
    
    def cleanup_old_checkpoints(self, keep_count: int = 10) -> None:
        """Clean up old checkpoint files."""
        try:
            checkpoint_dir = self.checkpoint_file.parent
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
            
            if len(checkpoint_files) > keep_count:
                # Sort by modification time
                checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Remove oldest files
                for old_file in checkpoint_files[keep_count:]:
                    old_file.unlink()
                    logger.info("Removed old checkpoint", file=str(old_file))
        
        except Exception as e:
            logger.warning("Failed to cleanup old checkpoints", error=str(e))