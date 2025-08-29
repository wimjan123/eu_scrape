"""
Comprehensive session lifecycle management system for EU Parliament data collection.
Manages session states, priorities, dependencies, and orchestration.
"""

from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import asyncio
import json
import uuid
from pathlib import Path
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict

from .data_collection_pipeline import DataCollectionPipeline, PipelineResult, PipelineConfig
from ..models.session import SessionMetadata
from ..core.exceptions import ProcessingError, DataValidationError
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..core.intelligent_cache import IntelligentCacheManager, CacheType

logger = get_logger(__name__)


class SessionState(Enum):
    """Session processing states."""
    DISCOVERED = "discovered"
    QUEUED = "queued"  
    PRIORITIZED = "prioritized"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_PENDING = "retry_pending"


class SessionPriority(Enum):
    """Session priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class DependencyType(Enum):
    """Types of session dependencies."""
    PREREQUISITE = "prerequisite"    # Must complete before this session
    RELATED = "related"             # Prefer to process together  
    EXCLUSIVE = "exclusive"         # Cannot process simultaneously
    RESOURCE = "resource"           # Shares limited resources


@dataclass
class SessionDependency:
    """Session dependency specification."""
    session_id: str
    dependency_type: DependencyType
    target_session_id: str
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSchedule:
    """Session scheduling information."""
    scheduled_time: Optional[datetime] = None
    earliest_start: Optional[datetime] = None
    latest_start: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scheduling_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManagedSession:
    """Complete session management information."""
    session_id: str
    session_metadata: SessionMetadata
    state: SessionState
    priority: SessionPriority
    
    # Timing information
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Dependencies and relationships
    dependencies: List[SessionDependency] = field(default_factory=list)
    dependents: Set[str] = field(default_factory=set)  # Sessions that depend on this
    
    # Scheduling
    schedule: Optional[SessionSchedule] = None
    
    # Processing information
    pipeline_config: Optional[PipelineConfig] = None
    pipeline_result: Optional[PipelineResult] = None
    processing_attempts: int = 0
    last_error: Optional[str] = None
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_stage: Optional[str] = None
    stage_progress: Dict[str, float] = field(default_factory=dict)
    
    # Quality and performance metrics
    quality_score: float = 0.0
    processing_time: Optional[timedelta] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Management metadata
    tags: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    management_metadata: Dict[str, Any] = field(default_factory=dict)


class SessionFilter:
    """Filter criteria for session queries."""
    
    def __init__(self):
        self.states: Optional[Set[SessionState]] = None
        self.priorities: Optional[Set[SessionPriority]] = None
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.tags: Optional[Set[str]] = None
        self.min_quality_score: Optional[float] = None
        self.has_errors: Optional[bool] = None
        self.processing_time_range: Optional[Tuple[timedelta, timedelta]] = None


class SessionScheduler(ABC):
    """Abstract base class for session scheduling strategies."""
    
    @abstractmethod
    async def schedule_sessions(self, sessions: List[ManagedSession], 
                              resources: Dict[str, Any]) -> List[ManagedSession]:
        """Schedule sessions based on strategy."""
        pass


class PriorityScheduler(SessionScheduler):
    """Priority-based session scheduler."""
    
    async def schedule_sessions(self, sessions: List[ManagedSession], 
                              resources: Dict[str, Any]) -> List[ManagedSession]:
        """Schedule sessions by priority and dependencies."""
        available_sessions = []
        
        # Filter sessions that are ready to process
        for session in sessions:
            if session.state in [SessionState.QUEUED, SessionState.PRIORITIZED]:
                # Check if dependencies are satisfied
                if await self._dependencies_satisfied(session, sessions):
                    available_sessions.append(session)
        
        # Sort by priority and creation time
        available_sessions.sort(key=lambda s: (s.priority.value, s.created_at))
        
        return available_sessions
    
    async def _dependencies_satisfied(self, session: ManagedSession, 
                                    all_sessions: List[ManagedSession]) -> bool:
        """Check if all required dependencies are satisfied."""
        session_states = {s.session_id: s.state for s in all_sessions}
        
        for dependency in session.dependencies:
            if not dependency.required:
                continue
            
            target_state = session_states.get(dependency.target_session_id)
            
            if dependency.dependency_type == DependencyType.PREREQUISITE:
                if target_state != SessionState.COMPLETED:
                    return False
            elif dependency.dependency_type == DependencyType.EXCLUSIVE:
                if target_state == SessionState.PROCESSING:
                    return False
        
        return True


class TimeBasedScheduler(SessionScheduler):
    """Time-based session scheduler with scheduling windows."""
    
    async def schedule_sessions(self, sessions: List[ManagedSession], 
                              resources: Dict[str, Any]) -> List[ManagedSession]:
        """Schedule sessions based on time constraints and resource availability."""
        now = datetime.now()
        available_sessions = []
        
        for session in sessions:
            if session.state not in [SessionState.QUEUED, SessionState.PRIORITIZED]:
                continue
            
            # Check time constraints
            if session.schedule:
                if session.schedule.earliest_start and now < session.schedule.earliest_start:
                    continue
                if session.schedule.latest_start and now > session.schedule.latest_start:
                    # Mark as overdue - could trigger alerts
                    session.management_metadata['overdue'] = True
            
            # Check resource availability
            if await self._resources_available(session, resources):
                available_sessions.append(session)
        
        # Sort by scheduled time, then priority
        available_sessions.sort(key=lambda s: (
            s.schedule.scheduled_time if s.schedule and s.schedule.scheduled_time else now,
            s.priority.value,
            s.created_at
        ))
        
        return available_sessions
    
    async def _resources_available(self, session: ManagedSession, 
                                 resources: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        if not session.schedule or not session.schedule.resource_requirements:
            return True
        
        for resource, required_amount in session.schedule.resource_requirements.items():
            available_amount = resources.get(resource, 0)
            if available_amount < required_amount:
                return False
        
        return True


class SessionManager:
    """Comprehensive session lifecycle management system."""
    
    def __init__(self,
                 data_pipeline: DataCollectionPipeline,
                 cache_manager: IntelligentCacheManager,
                 metrics_collector: Optional[MetricsCollector] = None,
                 scheduler: Optional[SessionScheduler] = None):
        """
        Initialize session manager.
        
        Args:
            data_pipeline: Data collection pipeline
            cache_manager: Intelligent cache manager
            metrics_collector: Optional metrics collection
            scheduler: Session scheduling strategy
        """
        self.data_pipeline = data_pipeline
        self.cache_manager = cache_manager
        self.metrics = metrics_collector or MetricsCollector()
        self.scheduler = scheduler or PriorityScheduler()
        
        # Session storage
        self.sessions: Dict[str, ManagedSession] = {}
        self.session_index: Dict[str, Set[str]] = defaultdict(set)  # Index by state, priority, etc.
        
        # Processing management
        self.active_sessions: Dict[str, asyncio.Task] = {}
        self.processing_semaphore = asyncio.Semaphore(5)  # Max concurrent sessions
        self.max_concurrent_sessions = 5
        
        # Persistence
        self.persistence_dir = Path("data/session_management")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.persistence_dir / "sessions.json"
        
        # Configuration
        self.auto_retry_failed = True
        self.max_retry_attempts = 3
        self.retry_delay_base = timedelta(minutes=5)
        self.cleanup_completed_after = timedelta(days=7)
        
        # Background tasks
        self._scheduler_task = None
        self._cleanup_task = None
        self._start_background_tasks()
        
        logger.info("Session manager initialized")
    
    async def add_session(self, session_metadata: SessionMetadata,
                         priority: SessionPriority = SessionPriority.NORMAL,
                         pipeline_config: Optional[PipelineConfig] = None,
                         schedule: Optional[SessionSchedule] = None,
                         dependencies: List[SessionDependency] = None,
                         tags: Set[str] = None) -> str:
        """
        Add a new session to management.
        
        Args:
            session_metadata: Session metadata
            priority: Session priority level
            pipeline_config: Pipeline configuration
            schedule: Session scheduling information
            dependencies: List of session dependencies
            tags: Session tags for categorization
            
        Returns:
            Session ID
        """
        session_id = session_metadata.session_id
        now = datetime.now()
        
        # Create managed session
        managed_session = ManagedSession(
            session_id=session_id,
            session_metadata=session_metadata,
            state=SessionState.DISCOVERED,
            priority=priority,
            created_at=now,
            updated_at=now,
            dependencies=dependencies or [],
            schedule=schedule,
            pipeline_config=pipeline_config,
            tags=tags or set()
        )
        
        # Store session
        self.sessions[session_id] = managed_session
        self._update_session_index(managed_session)
        
        # Update dependent relationships
        for dependency in managed_session.dependencies:
            if dependency.target_session_id in self.sessions:
                self.sessions[dependency.target_session_id].dependents.add(session_id)
        
        # Persist changes
        await self._persist_sessions()
        
        logger.info(f"Added session to management: {session_id} (priority: {priority.value})")
        
        # Queue session for processing
        await self.queue_session(session_id)
        
        return session_id
    
    async def add_sessions_batch(self, sessions_metadata: List[SessionMetadata],
                               priority: SessionPriority = SessionPriority.NORMAL,
                               pipeline_config: Optional[PipelineConfig] = None,
                               tags: Set[str] = None) -> List[str]:
        """Add multiple sessions to management in batch."""
        session_ids = []
        
        for session_metadata in sessions_metadata:
            try:
                session_id = await self.add_session(
                    session_metadata, priority, pipeline_config, tags=tags
                )
                session_ids.append(session_id)
            except Exception as e:
                logger.error(f"Failed to add session {session_metadata.session_id}: {e}")
                continue
        
        logger.info(f"Added {len(session_ids)} sessions in batch")
        return session_ids
    
    async def queue_session(self, session_id: str) -> bool:
        """Queue a session for processing."""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False
        
        if session.state != SessionState.DISCOVERED:
            logger.warning(f"Session {session_id} not in DISCOVERED state: {session.state}")
            return False
        
        # Update session state
        await self._update_session_state(session, SessionState.QUEUED)
        
        logger.info(f"Queued session for processing: {session_id}")
        return True
    
    async def prioritize_session(self, session_id: str, 
                               new_priority: SessionPriority) -> bool:
        """Change session priority."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        old_priority = session.priority
        session.priority = new_priority
        session.updated_at = datetime.now()
        
        self._update_session_index(session)
        await self._persist_sessions()
        
        logger.info(f"Updated session priority: {session_id} ({old_priority.value} → {new_priority.value})")
        return True
    
    async def process_session(self, session_id: str) -> bool:
        """Process a single session through the pipeline."""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False
        
        if session.state not in [SessionState.QUEUED, SessionState.PRIORITIZED, SessionState.RETRY_PENDING]:
            logger.warning(f"Session {session_id} not ready for processing: {session.state}")
            return False
        
        # Update state and timing
        await self._update_session_state(session, SessionState.PROCESSING)
        session.started_at = datetime.now()
        session.processing_attempts += 1
        
        try:
            # Create pipeline configuration
            config = session.pipeline_config or PipelineConfig()
            
            # Process through pipeline
            pipeline_result = await self.data_pipeline.process_session(session_id, config)
            
            # Update session with results
            session.pipeline_result = pipeline_result
            session.quality_score = pipeline_result.quality_score
            session.progress_percentage = 100.0
            session.completed_at = datetime.now()
            session.processing_time = session.completed_at - session.started_at
            
            # Determine final state based on pipeline result
            if pipeline_result.status.value in ["completed", "partial"]:
                await self._update_session_state(session, SessionState.COMPLETED)
                
                # Notify dependents that this session is complete
                await self._notify_dependents(session_id)
                
                logger.info(f"Session processing completed: {session_id} "
                           f"(quality: {session.quality_score:.2f})")
                
                # Record success metrics
                self.metrics.record_request(
                    service="session_manager",
                    success=True,
                    response_time=session.processing_time.total_seconds()
                )
                
                return True
            else:
                raise ProcessingError(f"Pipeline processing failed: {pipeline_result.status}")
            
        except Exception as e:
            # Handle processing failure
            session.last_error = str(e)
            session.completed_at = datetime.now()
            session.processing_time = session.completed_at - session.started_at
            
            logger.error(f"Session processing failed: {session_id} - {e}")
            
            # Determine if retry is appropriate
            if (self.auto_retry_failed and 
                session.processing_attempts < self.max_retry_attempts):
                await self._schedule_retry(session)
            else:
                await self._update_session_state(session, SessionState.FAILED)
            
            # Record failure metrics
            self.metrics.record_request(
                service="session_manager",
                success=False,
                response_time=session.processing_time.total_seconds() if session.processing_time else 0,
                error_type="processing_error"
            )
            
            return False
        
        finally:
            # Clean up active session tracking
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            await self._persist_sessions()
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a processing session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if session.state != SessionState.PROCESSING:
            logger.warning(f"Cannot pause session {session_id} in state {session.state}")
            return False
        
        # Cancel active processing task if exists
        if session_id in self.active_sessions:
            task = self.active_sessions[session_id]
            task.cancel()
            del self.active_sessions[session_id]
        
        await self._update_session_state(session, SessionState.PAUSED)
        logger.info(f"Paused session: {session_id}")
        
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        if session.state != SessionState.PAUSED:
            logger.warning(f"Cannot resume session {session_id} in state {session.state}")
            return False
        
        await self._update_session_state(session, SessionState.QUEUED)
        logger.info(f"Resumed session: {session_id}")
        
        return True
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Cancel active processing if running
        if session_id in self.active_sessions:
            task = self.active_sessions[session_id]
            task.cancel()
            del self.active_sessions[session_id]
        
        await self._update_session_state(session, SessionState.CANCELLED)
        logger.info(f"Cancelled session: {session_id}")
        
        return True
    
    def get_session(self, session_id: str) -> Optional[ManagedSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def query_sessions(self, filter_criteria: Optional[SessionFilter] = None) -> List[ManagedSession]:
        """Query sessions with filtering."""
        sessions = list(self.sessions.values())
        
        if not filter_criteria:
            return sessions
        
        filtered_sessions = []
        for session in sessions:
            if self._session_matches_filter(session, filter_criteria):
                filtered_sessions.append(session)
        
        return filtered_sessions
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        total_sessions = len(self.sessions)
        
        if total_sessions == 0:
            return {"total_sessions": 0}
        
        # State distribution
        state_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        
        # Performance metrics
        processing_times = []
        quality_scores = []
        
        for session in self.sessions.values():
            state_counts[session.state.value] += 1
            priority_counts[session.priority.value] += 1
            
            if session.processing_time:
                processing_times.append(session.processing_time.total_seconds())
            
            if session.quality_score > 0:
                quality_scores.append(session.quality_score)
        
        # Calculate averages
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Success rates
        completed_count = state_counts.get("completed", 0)
        failed_count = state_counts.get("failed", 0)
        processed_count = completed_count + failed_count
        success_rate = (completed_count / processed_count * 100) if processed_count > 0 else 0
        
        return {
            "total_sessions": total_sessions,
            "state_distribution": dict(state_counts),
            "priority_distribution": dict(priority_counts),
            "active_sessions": len(self.active_sessions),
            "avg_processing_time_seconds": avg_processing_time,
            "avg_quality_score": avg_quality_score,
            "success_rate_percent": success_rate,
            "retry_attempts": sum(s.processing_attempts for s in self.sessions.values()),
        }
    
    async def _update_session_state(self, session: ManagedSession, new_state: SessionState):
        """Update session state and maintain indices."""
        old_state = session.state
        session.state = new_state
        session.updated_at = datetime.now()
        
        # Update indices
        self.session_index[f"state:{old_state.value}"].discard(session.session_id)
        self.session_index[f"state:{new_state.value}"].add(session.session_id)
        
        logger.debug(f"Session {session.session_id}: {old_state.value} → {new_state.value}")
    
    def _update_session_index(self, session: ManagedSession):
        """Update session indices."""
        session_id = session.session_id
        
        # Update state index
        self.session_index[f"state:{session.state.value}"].add(session_id)
        
        # Update priority index
        self.session_index[f"priority:{session.priority.value}"].add(session_id)
        
        # Update tag indices
        for tag in session.tags:
            self.session_index[f"tag:{tag}"].add(session_id)
    
    async def _schedule_retry(self, session: ManagedSession):
        """Schedule a session for retry."""
        retry_delay = self.retry_delay_base * (2 ** (session.processing_attempts - 1))
        retry_time = datetime.now() + retry_delay
        
        # Update schedule
        if not session.schedule:
            session.schedule = SessionSchedule()
        session.schedule.earliest_start = retry_time
        
        await self._update_session_state(session, SessionState.RETRY_PENDING)
        
        logger.info(f"Scheduled session retry: {session.session_id} "
                   f"(attempt {session.processing_attempts}, delay: {retry_delay})")
    
    async def _notify_dependents(self, session_id: str):
        """Notify dependent sessions that a prerequisite is complete."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        for dependent_id in session.dependents:
            dependent = self.sessions.get(dependent_id)
            if dependent and dependent.state in [SessionState.QUEUED, SessionState.PRIORITIZED]:
                # Check if all dependencies are now satisfied
                if await self.scheduler._dependencies_satisfied(dependent, list(self.sessions.values())):
                    await self._update_session_state(dependent, SessionState.PRIORITIZED)
                    logger.info(f"Dependencies satisfied for session: {dependent_id}")
    
    def _session_matches_filter(self, session: ManagedSession, 
                               filter_criteria: SessionFilter) -> bool:
        """Check if session matches filter criteria."""
        if filter_criteria.states and session.state not in filter_criteria.states:
            return False
        
        if filter_criteria.priorities and session.priority not in filter_criteria.priorities:
            return False
        
        if filter_criteria.date_range:
            start_date, end_date = filter_criteria.date_range
            if not (start_date <= session.created_at <= end_date):
                return False
        
        if filter_criteria.tags and not filter_criteria.tags.intersection(session.tags):
            return False
        
        if filter_criteria.min_quality_score and session.quality_score < filter_criteria.min_quality_score:
            return False
        
        if filter_criteria.has_errors is not None:
            has_errors = bool(session.last_error)
            if has_errors != filter_criteria.has_errors:
                return False
        
        return True
    
    async def _persist_sessions(self):
        """Persist session data to disk."""
        try:
            session_data = {}
            for session_id, session in self.sessions.items():
                # Convert to serializable format
                session_dict = asdict(session)
                
                # Handle datetime serialization
                session_dict['created_at'] = session.created_at.isoformat()
                session_dict['updated_at'] = session.updated_at.isoformat()
                if session.started_at:
                    session_dict['started_at'] = session.started_at.isoformat()
                if session.completed_at:
                    session_dict['completed_at'] = session.completed_at.isoformat()
                
                # Handle timedelta serialization
                if session.processing_time:
                    session_dict['processing_time'] = session.processing_time.total_seconds()
                
                session_data[session_id] = session_dict
            
            # Write to file
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Failed to persist sessions: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        async def scheduler_task():
            """Background session scheduling task."""
            while True:
                try:
                    await asyncio.sleep(30)  # Run every 30 seconds
                    await self._schedule_ready_sessions()
                except Exception as e:
                    logger.error(f"Scheduler task error: {e}")
        
        async def cleanup_task():
            """Background cleanup task."""
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_old_sessions()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._scheduler_task = asyncio.create_task(scheduler_task())
        self._cleanup_task = asyncio.create_task(cleanup_task())
    
    async def _schedule_ready_sessions(self):
        """Schedule ready sessions for processing."""
        # Get queued and prioritized sessions
        ready_sessions = []
        for session in self.sessions.values():
            if session.state in [SessionState.QUEUED, SessionState.PRIORITIZED, SessionState.RETRY_PENDING]:
                ready_sessions.append(session)
        
        if not ready_sessions:
            return
        
        # Apply scheduling strategy
        scheduled_sessions = await self.scheduler.schedule_sessions(
            ready_sessions, {"concurrent_slots": self.max_concurrent_sessions}
        )
        
        # Process scheduled sessions up to capacity
        available_slots = self.max_concurrent_sessions - len(self.active_sessions)
        sessions_to_process = scheduled_sessions[:available_slots]
        
        for session in sessions_to_process:
            # Create processing task
            task = asyncio.create_task(self.process_session(session.session_id))
            self.active_sessions[session.session_id] = task
    
    async def _cleanup_old_sessions(self):
        """Clean up old completed sessions."""
        cutoff_time = datetime.now() - self.cleanup_completed_after
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if (session.state == SessionState.COMPLETED and 
                session.completed_at and 
                session.completed_at < cutoff_time):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            # Clean up indices
            for index_key, session_set in self.session_index.items():
                session_set.discard(session_id)
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old completed sessions")
            await self._persist_sessions()
    
    async def close(self):
        """Clean up resources."""
        # Cancel background tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Cancel active processing tasks
        for task in self.active_sessions.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_sessions:
            await asyncio.gather(*self.active_sessions.values(), return_exceptions=True)
        
        logger.info("Session manager closed")