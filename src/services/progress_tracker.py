"""
Advanced progress tracking system with real-time analytics and monitoring.
Provides granular progress visibility, performance analytics, and trend analysis.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import json
import uuid
from pathlib import Path
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod

from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..core.intelligent_cache import IntelligentCacheManager, CacheType

logger = get_logger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    MILESTONE_REACHED = "milestone_reached"
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"
    CHECKPOINT_CREATED = "checkpoint_created"
    OPERATION_PAUSED = "operation_paused"
    OPERATION_RESUMED = "operation_resumed"


class ProgressStatus(Enum):
    """Overall progress status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """Individual progress event."""
    event_id: str
    operation_id: str
    event_type: ProgressEventType
    timestamp: datetime
    stage: Optional[str] = None
    progress_percentage: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[timedelta] = None
    error: Optional[str] = None


@dataclass
class StageProgress:
    """Progress information for a specific stage."""
    stage_name: str
    status: ProgressStatus
    progress_percentage: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    items_processed: int = 0
    items_total: Optional[int] = None
    throughput_per_second: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressMilestone:
    """Important milestone in progress tracking."""
    milestone_id: str
    name: str
    description: str
    target_percentage: float
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for progress tracking."""
    operations_per_second: float
    average_stage_duration: float
    peak_throughput: float
    error_rate: float
    retry_rate: float
    resource_utilization: Dict[str, float]
    bottleneck_stages: List[str]
    trend_analysis: Dict[str, float]


@dataclass
class ProgressSnapshot:
    """Point-in-time progress snapshot."""
    operation_id: str
    timestamp: datetime
    overall_progress: float
    status: ProgressStatus
    current_stage: Optional[str]
    stages: Dict[str, StageProgress]
    milestones: List[ProgressMilestone]
    performance_metrics: PerformanceMetrics
    estimated_completion: Optional[datetime]
    quality_score: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressEstimator(ABC):
    """Abstract base class for progress estimation algorithms."""
    
    @abstractmethod
    async def estimate_completion_time(self, operation_id: str, 
                                     current_progress: float,
                                     historical_data: List[ProgressSnapshot]) -> Optional[datetime]:
        """Estimate completion time based on current progress and history."""
        pass
    
    @abstractmethod
    async def estimate_stage_duration(self, stage_name: str,
                                    historical_data: List[StageProgress]) -> Optional[timedelta]:
        """Estimate duration for a specific stage."""
        pass


class LinearProgressEstimator(ProgressEstimator):
    """Simple linear progress estimation based on current rate."""
    
    async def estimate_completion_time(self, operation_id: str, 
                                     current_progress: float,
                                     historical_data: List[ProgressSnapshot]) -> Optional[datetime]:
        """Estimate completion using linear extrapolation."""
        if current_progress <= 0 or not historical_data:
            return None
        
        # Calculate average progress rate from recent snapshots
        recent_snapshots = sorted(historical_data[-10:], key=lambda x: x.timestamp)
        if len(recent_snapshots) < 2:
            return None
        
        # Calculate rate of progress
        time_span = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
        progress_change = recent_snapshots[-1].overall_progress - recent_snapshots[0].overall_progress
        
        if time_span <= 0 or progress_change <= 0:
            return None
        
        progress_rate = progress_change / time_span  # percentage per second
        remaining_progress = 100.0 - current_progress
        estimated_seconds = remaining_progress / progress_rate
        
        return datetime.now() + timedelta(seconds=estimated_seconds)
    
    async def estimate_stage_duration(self, stage_name: str,
                                    historical_data: List[StageProgress]) -> Optional[timedelta]:
        """Estimate stage duration based on historical averages."""
        completed_stages = [
            stage for stage in historical_data 
            if stage.stage_name == stage_name and stage.actual_duration
        ]
        
        if not completed_stages:
            return None
        
        durations = [stage.actual_duration.total_seconds() for stage in completed_stages]
        average_duration = statistics.mean(durations)
        
        return timedelta(seconds=average_duration)


class AdaptiveProgressEstimator(ProgressEstimator):
    """Adaptive progress estimation that considers trends and variations."""
    
    async def estimate_completion_time(self, operation_id: str, 
                                     current_progress: float,
                                     historical_data: List[ProgressSnapshot]) -> Optional[datetime]:
        """Adaptive estimation considering acceleration/deceleration trends."""
        if current_progress <= 0 or len(historical_data) < 3:
            return None
        
        recent_snapshots = sorted(historical_data[-15:], key=lambda x: x.timestamp)
        
        # Calculate progress rates over different time windows
        rates = []
        for i in range(1, min(6, len(recent_snapshots))):
            time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[-i].timestamp).total_seconds()
            progress_diff = recent_snapshots[-1].overall_progress - recent_snapshots[-i].overall_progress
            
            if time_diff > 0 and progress_diff > 0:
                rate = progress_diff / time_diff
                rates.append(rate)
        
        if not rates:
            return None
        
        # Use weighted average favoring more recent rates
        weights = [2**i for i in range(len(rates))]
        weighted_rate = sum(r * w for r, w in zip(rates, weights)) / sum(weights)
        
        remaining_progress = 100.0 - current_progress
        estimated_seconds = remaining_progress / weighted_rate
        
        return datetime.now() + timedelta(seconds=estimated_seconds)
    
    async def estimate_stage_duration(self, stage_name: str,
                                    historical_data: List[StageProgress]) -> Optional[timedelta]:
        """Adaptive stage estimation considering recent performance trends."""
        completed_stages = [
            stage for stage in historical_data 
            if stage.stage_name == stage_name and stage.actual_duration
        ]
        
        if len(completed_stages) < 2:
            return None
        
        # Sort by completion time (most recent first)
        completed_stages.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
        
        # Weight recent stages more heavily
        durations = []
        weights = []
        for i, stage in enumerate(completed_stages[:10]):  # Use up to 10 recent stages
            weight = 2 ** (10 - i)  # Exponentially decreasing weight
            durations.append(stage.actual_duration.total_seconds())
            weights.append(weight)
        
        weighted_average = sum(d * w for d, w in zip(durations, weights)) / sum(weights)
        
        return timedelta(seconds=weighted_average)


class ProgressTracker:
    """Advanced progress tracking system with real-time analytics."""
    
    def __init__(self,
                 cache_manager: Optional[IntelligentCacheManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 estimator: Optional[ProgressEstimator] = None):
        """
        Initialize progress tracker.
        
        Args:
            cache_manager: Optional cache manager for persistence
            metrics_collector: Optional metrics collection
            estimator: Progress estimation algorithm
        """
        self.cache_manager = cache_manager
        self.metrics = metrics_collector or MetricsCollector()
        self.estimator = estimator or AdaptiveProgressEstimator()
        
        # Progress storage
        self.operations: Dict[str, ProgressSnapshot] = {}
        self.events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Recent events per operation
        self.historical_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Subscribers for real-time updates
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.stage_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Persistence
        self.persistence_dir = Path("data/progress_tracking")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.snapshot_interval = timedelta(seconds=30)
        self.performance_analysis_interval = timedelta(minutes=5)
        
        # Background tasks
        self._snapshot_task = None
        self._analysis_task = None
        self._start_background_tasks()
        
        logger.info("Advanced progress tracker initialized")
    
    async def start_operation(self, operation_id: str,
                            stages: List[str],
                            milestones: List[ProgressMilestone] = None,
                            total_items: Optional[int] = None,
                            metadata: Dict[str, Any] = None) -> str:
        """
        Start tracking progress for a new operation.
        
        Args:
            operation_id: Unique operation identifier
            stages: List of stage names in processing order
            milestones: Optional progress milestones
            total_items: Optional total number of items to process
            metadata: Additional operation metadata
            
        Returns:
            Operation ID
        """
        now = datetime.now()
        
        # Initialize stage progress
        stage_progress = {}
        for stage in stages:
            stage_progress[stage] = StageProgress(
                stage_name=stage,
                status=ProgressStatus.NOT_STARTED,
                progress_percentage=0.0,
                items_total=total_items
            )
        
        # Create initial performance metrics
        performance_metrics = PerformanceMetrics(
            operations_per_second=0.0,
            average_stage_duration=0.0,
            peak_throughput=0.0,
            error_rate=0.0,
            retry_rate=0.0,
            resource_utilization={},
            bottleneck_stages=[],
            trend_analysis={}
        )
        
        # Create progress snapshot
        snapshot = ProgressSnapshot(
            operation_id=operation_id,
            timestamp=now,
            overall_progress=0.0,
            status=ProgressStatus.IN_PROGRESS,
            current_stage=stages[0] if stages else None,
            stages=stage_progress,
            milestones=milestones or [],
            performance_metrics=performance_metrics,
            estimated_completion=None,
            metadata=metadata or {}
        )
        
        self.operations[operation_id] = snapshot
        
        # Record start event
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.STAGE_STARTED,
            stage=stages[0] if stages else None,
            message="Operation started"
        )
        
        logger.info(f"Started progress tracking for operation: {operation_id}")
        await self._notify_subscribers(operation_id, snapshot)
        
        return operation_id
    
    async def update_stage_progress(self, operation_id: str,
                                  stage: str,
                                  progress_percentage: float,
                                  items_processed: Optional[int] = None,
                                  message: Optional[str] = None,
                                  metadata: Dict[str, Any] = None) -> bool:
        """Update progress for a specific stage."""
        if operation_id not in self.operations:
            logger.error(f"Operation not found: {operation_id}")
            return False
        
        snapshot = self.operations[operation_id]
        
        if stage not in snapshot.stages:
            logger.error(f"Stage not found: {stage} in operation {operation_id}")
            return False
        
        stage_info = snapshot.stages[stage]
        now = datetime.now()
        
        # Update stage progress
        old_progress = stage_info.progress_percentage
        stage_info.progress_percentage = min(100.0, max(0.0, progress_percentage))
        stage_info.status = ProgressStatus.IN_PROGRESS
        
        if stage_info.started_at is None:
            stage_info.started_at = now
        
        if items_processed is not None:
            stage_info.items_processed = items_processed
            
            # Calculate throughput
            if stage_info.started_at:
                elapsed = (now - stage_info.started_at).total_seconds()
                if elapsed > 0:
                    stage_info.throughput_per_second = items_processed / elapsed
        
        if metadata:
            stage_info.metadata.update(metadata)
        
        # Update current stage in operation
        if snapshot.current_stage != stage:
            # Mark previous stage as completed if it was at 100%
            if (snapshot.current_stage and 
                snapshot.current_stage in snapshot.stages and
                snapshot.stages[snapshot.current_stage].progress_percentage >= 100.0):
                await self._complete_stage(operation_id, snapshot.current_stage)
            
            snapshot.current_stage = stage
        
        # Calculate overall progress
        await self._calculate_overall_progress(operation_id)
        
        # Record progress event
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.STAGE_PROGRESS,
            stage=stage,
            progress_percentage=progress_percentage,
            message=message or f"Stage progress updated: {progress_percentage:.1f}%"
        )
        
        # Check milestones
        await self._check_milestones(operation_id)
        
        # Update snapshot timestamp
        snapshot.timestamp = now
        
        # Notify subscribers
        await self._notify_subscribers(operation_id, snapshot)
        
        return True
    
    async def complete_stage(self, operation_id: str, stage: str,
                           final_progress: float = 100.0,
                           message: Optional[str] = None) -> bool:
        """Mark a stage as completed."""
        return await self._complete_stage(operation_id, stage, final_progress, message)
    
    async def fail_stage(self, operation_id: str, stage: str,
                        error: str, retry_possible: bool = True) -> bool:
        """Mark a stage as failed."""
        if operation_id not in self.operations:
            return False
        
        snapshot = self.operations[operation_id]
        
        if stage not in snapshot.stages:
            return False
        
        stage_info = snapshot.stages[stage]
        stage_info.status = ProgressStatus.FAILED
        stage_info.error_count += 1
        
        # Record failure event
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.STAGE_FAILED,
            stage=stage,
            message=f"Stage failed: {error}",
            error=error
        )
        
        if not retry_possible:
            # Mark entire operation as failed
            snapshot.status = ProgressStatus.FAILED
        
        await self._notify_subscribers(operation_id, snapshot)
        logger.error(f"Stage failed: {stage} in operation {operation_id} - {error}")
        
        return True
    
    async def retry_stage(self, operation_id: str, stage: str) -> bool:
        """Retry a failed stage."""
        if operation_id not in self.operations:
            return False
        
        snapshot = self.operations[operation_id]
        
        if stage not in snapshot.stages:
            return False
        
        stage_info = snapshot.stages[stage]
        stage_info.status = ProgressStatus.IN_PROGRESS
        stage_info.retry_count += 1
        stage_info.progress_percentage = 0.0  # Reset progress
        
        # Record retry event
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.RETRY_ATTEMPTED,
            stage=stage,
            message=f"Retrying stage (attempt {stage_info.retry_count + 1})"
        )
        
        await self._notify_subscribers(operation_id, snapshot)
        logger.info(f"Retrying stage: {stage} in operation {operation_id}")
        
        return True
    
    async def pause_operation(self, operation_id: str) -> bool:
        """Pause an operation."""
        if operation_id not in self.operations:
            return False
        
        snapshot = self.operations[operation_id]
        snapshot.status = ProgressStatus.PAUSED
        snapshot.timestamp = datetime.now()
        
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.OPERATION_PAUSED,
            message="Operation paused"
        )
        
        await self._notify_subscribers(operation_id, snapshot)
        logger.info(f"Paused operation: {operation_id}")
        
        return True
    
    async def resume_operation(self, operation_id: str) -> bool:
        """Resume a paused operation."""
        if operation_id not in self.operations:
            return False
        
        snapshot = self.operations[operation_id]
        if snapshot.status != ProgressStatus.PAUSED:
            return False
        
        snapshot.status = ProgressStatus.IN_PROGRESS
        snapshot.timestamp = datetime.now()
        
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.OPERATION_RESUMED,
            message="Operation resumed"
        )
        
        await self._notify_subscribers(operation_id, snapshot)
        logger.info(f"Resumed operation: {operation_id}")
        
        return True
    
    async def complete_operation(self, operation_id: str,
                               quality_score: float = 0.0,
                               metadata: Dict[str, Any] = None) -> bool:
        """Mark an operation as completed."""
        if operation_id not in self.operations:
            return False
        
        snapshot = self.operations[operation_id]
        snapshot.status = ProgressStatus.COMPLETED
        snapshot.overall_progress = 100.0
        snapshot.quality_score = quality_score
        snapshot.timestamp = datetime.now()
        
        if metadata:
            snapshot.metadata.update(metadata)
        
        # Complete any remaining stages
        for stage_name, stage_info in snapshot.stages.items():
            if stage_info.status == ProgressStatus.IN_PROGRESS:
                await self._complete_stage(operation_id, stage_name, 100.0)
        
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.STAGE_COMPLETED,
            message=f"Operation completed (quality: {quality_score:.2f})"
        )
        
        await self._notify_subscribers(operation_id, snapshot)
        logger.info(f"Completed operation: {operation_id} (quality: {quality_score:.2f})")
        
        return True
    
    def get_progress(self, operation_id: str) -> Optional[ProgressSnapshot]:
        """Get current progress snapshot for an operation."""
        return self.operations.get(operation_id)
    
    def get_all_operations(self) -> Dict[str, ProgressSnapshot]:
        """Get progress snapshots for all operations."""
        return self.operations.copy()
    
    async def get_performance_analytics(self, operation_id: Optional[str] = None,
                                      time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get performance analytics for operations."""
        if operation_id:
            # Analytics for specific operation
            if operation_id not in self.operations:
                return {}
            
            snapshot = self.operations[operation_id]
            events = list(self.events[operation_id])
            historical = list(self.historical_snapshots[operation_id])
            
            return await self._analyze_operation_performance(snapshot, events, historical)
        
        else:
            # Global analytics
            return await self._analyze_global_performance(time_range)
    
    async def subscribe_to_updates(self, operation_id: str, 
                                 callback: Callable[[str, ProgressSnapshot], None]) -> str:
        """Subscribe to real-time progress updates."""
        subscription_id = str(uuid.uuid4())
        self.subscribers[operation_id].append((subscription_id, callback))
        
        logger.debug(f"Added subscriber {subscription_id} for operation {operation_id}")
        return subscription_id
    
    async def unsubscribe(self, operation_id: str, subscription_id: str) -> bool:
        """Unsubscribe from progress updates."""
        if operation_id not in self.subscribers:
            return False
        
        self.subscribers[operation_id] = [
            (sid, callback) for sid, callback in self.subscribers[operation_id]
            if sid != subscription_id
        ]
        
        logger.debug(f"Removed subscriber {subscription_id} from operation {operation_id}")
        return True
    
    async def _complete_stage(self, operation_id: str, stage: str,
                            final_progress: float = 100.0,
                            message: Optional[str] = None) -> bool:
        """Internal method to complete a stage."""
        snapshot = self.operations[operation_id]
        stage_info = snapshot.stages[stage]
        
        now = datetime.now()
        stage_info.status = ProgressStatus.COMPLETED
        stage_info.progress_percentage = final_progress
        stage_info.completed_at = now
        
        if stage_info.started_at:
            stage_info.actual_duration = now - stage_info.started_at
            
            # Update stage performance history
            self.stage_performance[stage].append(stage_info)
        
        await self._record_event(
            operation_id=operation_id,
            event_type=ProgressEventType.STAGE_COMPLETED,
            stage=stage,
            progress_percentage=final_progress,
            message=message or f"Stage completed: {stage}",
            duration=stage_info.actual_duration
        )
        
        return True
    
    async def _calculate_overall_progress(self, operation_id: str):
        """Calculate overall operation progress based on stage progress."""
        snapshot = self.operations[operation_id]
        
        if not snapshot.stages:
            return
        
        # Simple average for now - could be weighted by importance
        total_progress = sum(stage.progress_percentage for stage in snapshot.stages.values())
        snapshot.overall_progress = total_progress / len(snapshot.stages)
        
        # Update estimated completion time
        historical = list(self.historical_snapshots[operation_id])
        snapshot.estimated_completion = await self.estimator.estimate_completion_time(
            operation_id, snapshot.overall_progress, historical
        )
    
    async def _check_milestones(self, operation_id: str):
        """Check if any milestones have been reached."""
        snapshot = self.operations[operation_id]
        
        for milestone in snapshot.milestones:
            if not milestone.achieved and snapshot.overall_progress >= milestone.target_percentage:
                milestone.achieved = True
                milestone.achieved_at = datetime.now()
                
                await self._record_event(
                    operation_id=operation_id,
                    event_type=ProgressEventType.MILESTONE_REACHED,
                    message=f"Milestone reached: {milestone.name}",
                    details={"milestone_id": milestone.milestone_id}
                )
                
                logger.info(f"Milestone reached: {milestone.name} in operation {operation_id}")
    
    async def _record_event(self, operation_id: str, event_type: ProgressEventType,
                          stage: Optional[str] = None,
                          progress_percentage: Optional[float] = None,
                          message: Optional[str] = None,
                          details: Dict[str, Any] = None,
                          duration: Optional[timedelta] = None,
                          error: Optional[str] = None):
        """Record a progress event."""
        event = ProgressEvent(
            event_id=str(uuid.uuid4()),
            operation_id=operation_id,
            event_type=event_type,
            timestamp=datetime.now(),
            stage=stage,
            progress_percentage=progress_percentage,
            message=message,
            details=details or {},
            duration=duration,
            error=error
        )
        
        self.events[operation_id].append(event)
        
        # Record metrics
        self.metrics.record_request(
            service="progress_tracker",
            success=(error is None),
            response_time=0.001,
            metadata={"event_type": event_type.value}
        )
    
    async def _notify_subscribers(self, operation_id: str, snapshot: ProgressSnapshot):
        """Notify subscribers of progress updates."""
        if operation_id not in self.subscribers:
            return
        
        for subscription_id, callback in self.subscribers[operation_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(operation_id, snapshot)
                else:
                    callback(operation_id, snapshot)
            except Exception as e:
                logger.error(f"Subscriber notification failed: {e}")
    
    async def _analyze_operation_performance(self, snapshot: ProgressSnapshot,
                                           events: List[ProgressEvent],
                                           historical: List[ProgressSnapshot]) -> Dict[str, Any]:
        """Analyze performance for a specific operation."""
        analysis = {
            "operation_id": snapshot.operation_id,
            "current_status": snapshot.status.value,
            "overall_progress": snapshot.overall_progress,
            "stages_analysis": {},
            "event_summary": {},
            "performance_trends": {}
        }
        
        # Stage analysis
        for stage_name, stage_info in snapshot.stages.items():
            stage_analysis = {
                "status": stage_info.status.value,
                "progress": stage_info.progress_percentage,
                "throughput": stage_info.throughput_per_second,
                "error_count": stage_info.error_count,
                "retry_count": stage_info.retry_count
            }
            
            if stage_info.actual_duration:
                stage_analysis["actual_duration"] = stage_info.actual_duration.total_seconds()
            
            if stage_info.estimated_duration:
                stage_analysis["estimated_duration"] = stage_info.estimated_duration.total_seconds()
            
            analysis["stages_analysis"][stage_name] = stage_analysis
        
        # Event summary
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type.value] += 1
        analysis["event_summary"] = dict(event_counts)
        
        # Performance trends
        if len(historical) >= 2:
            progress_rates = []
            for i in range(1, len(historical)):
                time_diff = (historical[i].timestamp - historical[i-1].timestamp).total_seconds()
                progress_diff = historical[i].overall_progress - historical[i-1].overall_progress
                if time_diff > 0:
                    progress_rates.append(progress_diff / time_diff)
            
            if progress_rates:
                analysis["performance_trends"]["average_progress_rate"] = statistics.mean(progress_rates)
                analysis["performance_trends"]["progress_rate_trend"] = progress_rates[-5:]  # Recent trend
        
        return analysis
    
    async def _analyze_global_performance(self, time_range: Optional[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        """Analyze global performance across all operations."""
        analysis = {
            "total_operations": len(self.operations),
            "status_distribution": {},
            "stage_performance": {},
            "global_trends": {},
            "bottleneck_analysis": {}
        }
        
        # Status distribution
        status_counts = defaultdict(int)
        for snapshot in self.operations.values():
            status_counts[snapshot.status.value] += 1
        analysis["status_distribution"] = dict(status_counts)
        
        # Stage performance analysis
        for stage_name, stage_history in self.stage_performance.items():
            if not stage_history:
                continue
            
            completed_stages = [s for s in stage_history if s.actual_duration]
            if completed_stages:
                durations = [s.actual_duration.total_seconds() for s in completed_stages]
                throughputs = [s.throughput_per_second for s in completed_stages if s.throughput_per_second > 0]
                
                stage_stats = {
                    "average_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "total_executions": len(completed_stages),
                }
                
                if throughputs:
                    stage_stats["average_throughput"] = statistics.mean(throughputs)
                    stage_stats["peak_throughput"] = max(throughputs)
                
                analysis["stage_performance"][stage_name] = stage_stats
        
        return analysis
    
    def _start_background_tasks(self):
        """Start background tasks for snapshot and analysis."""
        async def snapshot_task():
            """Periodic snapshot creation."""
            while True:
                try:
                    await asyncio.sleep(self.snapshot_interval.total_seconds())
                    await self._create_snapshots()
                except Exception as e:
                    logger.error(f"Snapshot task error: {e}")
        
        async def analysis_task():
            """Periodic performance analysis."""
            while True:
                try:
                    await asyncio.sleep(self.performance_analysis_interval.total_seconds())
                    await self._update_performance_metrics()
                except Exception as e:
                    logger.error(f"Analysis task error: {e}")
        
        self._snapshot_task = asyncio.create_task(snapshot_task())
        self._analysis_task = asyncio.create_task(analysis_task())
    
    async def _create_snapshots(self):
        """Create historical snapshots for all active operations."""
        for operation_id, snapshot in self.operations.items():
            if snapshot.status == ProgressStatus.IN_PROGRESS:
                # Create snapshot copy
                snapshot_copy = ProgressSnapshot(
                    operation_id=snapshot.operation_id,
                    timestamp=datetime.now(),
                    overall_progress=snapshot.overall_progress,
                    status=snapshot.status,
                    current_stage=snapshot.current_stage,
                    stages=snapshot.stages.copy(),
                    milestones=snapshot.milestones.copy(),
                    performance_metrics=snapshot.performance_metrics,
                    estimated_completion=snapshot.estimated_completion,
                    quality_score=snapshot.quality_score,
                    resource_usage=snapshot.resource_usage.copy(),
                    metadata=snapshot.metadata.copy()
                )
                
                self.historical_snapshots[operation_id].append(snapshot_copy)
    
    async def _update_performance_metrics(self):
        """Update performance metrics for all operations."""
        for operation_id, snapshot in self.operations.items():
            if snapshot.status == ProgressStatus.IN_PROGRESS:
                # Update estimated completion
                historical = list(self.historical_snapshots[operation_id])
                snapshot.estimated_completion = await self.estimator.estimate_completion_time(
                    operation_id, snapshot.overall_progress, historical
                )
                
                # Update performance metrics
                snapshot.performance_metrics = await self._calculate_performance_metrics(
                    operation_id, historical
                )
    
    async def _calculate_performance_metrics(self, operation_id: str,
                                           historical: List[ProgressSnapshot]) -> PerformanceMetrics:
        """Calculate performance metrics for an operation."""
        # This is a simplified calculation - could be much more sophisticated
        metrics = PerformanceMetrics(
            operations_per_second=0.0,
            average_stage_duration=0.0,
            peak_throughput=0.0,
            error_rate=0.0,
            retry_rate=0.0,
            resource_utilization={},
            bottleneck_stages=[],
            trend_analysis={}
        )
        
        if len(historical) < 2:
            return metrics
        
        # Calculate progress rate
        recent_snapshots = historical[-5:]
        if len(recent_snapshots) >= 2:
            time_span = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
            progress_change = recent_snapshots[-1].overall_progress - recent_snapshots[0].overall_progress
            
            if time_span > 0:
                metrics.operations_per_second = progress_change / time_span
        
        return metrics
    
    async def close(self):
        """Clean up resources."""
        if self._snapshot_task:
            self._snapshot_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()
        
        # Clear subscribers
        self.subscribers.clear()
        
        logger.info("Progress tracker closed")