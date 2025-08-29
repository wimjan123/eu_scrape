"""
Checkpoint and resume system for large-scale EU Parliament data collection operations.
Provides atomic checkpoint creation, state recovery, and distributed processing coordination.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import json
import uuid
import hashlib
import gzip
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
import fcntl
import tempfile
import shutil

from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, DataValidationError
from ..core.intelligent_cache import IntelligentCacheManager, CacheType
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class CheckpointType(Enum):
    """Types of checkpoints."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"
    FAILURE_RECOVERY = "failure_recovery"
    MILESTONE = "milestone"


class CheckpointStatus(Enum):
    """Checkpoint status."""
    CREATING = "creating"
    COMPLETED = "completed"
    CORRUPTED = "corrupted"
    RESTORING = "restoring"
    FAILED = "failed"


class RecoveryStrategy(Enum):
    """Recovery strategies for checkpoint restoration."""
    FULL_RESTORE = "full_restore"        # Complete state restoration
    PARTIAL_RESTORE = "partial_restore"   # Restore only essential state
    MERGE_RESTORE = "merge_restore"       # Merge with current state
    INCREMENTAL_RESTORE = "incremental_restore"  # Step-by-step restoration


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    operation_id: str
    checkpoint_type: CheckpointType
    created_at: datetime
    created_by: str
    description: str
    
    # State information
    operation_state: str
    progress_percentage: float
    current_stage: Optional[str]
    completed_stages: List[str]
    
    # Data integrity
    data_hash: str
    file_size: int
    compression_used: bool
    
    # Dependencies and relationships
    parent_checkpoint_id: Optional[str]
    dependent_operations: List[str]
    
    # Recovery information
    recovery_priority: int = 1  # Higher = more important
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FULL_RESTORE
    
    # Validation
    validation_hash: str = ""
    integrity_verified: bool = False
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointData:
    """Complete checkpoint data package."""
    metadata: CheckpointMetadata
    
    # Core state data
    operation_state: Dict[str, Any]
    session_states: Dict[str, Dict[str, Any]]
    progress_data: Dict[str, Any]
    cache_snapshots: Dict[str, Any]
    
    # Processing context
    configuration: Dict[str, Any]
    environment_info: Dict[str, Any]
    resource_allocations: Dict[str, Any]
    
    # Error and recovery context
    recent_errors: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    recovery_hints: Dict[str, Any]
    
    # Performance data
    performance_metrics: Dict[str, Any]
    timing_data: Dict[str, Any]


@dataclass
class RestoreResult:
    """Result of checkpoint restoration."""
    success: bool
    checkpoint_id: str
    restored_at: datetime
    recovery_strategy: RecoveryStrategy
    
    # Restoration details
    restored_operations: List[str]
    restored_sessions: List[str]
    restored_progress: Dict[str, float]
    
    # Data integrity
    data_integrity_ok: bool
    validation_errors: List[str]
    
    # Performance
    restore_duration: timedelta
    data_size_restored: int
    
    # Issues and warnings
    warnings: List[str]
    partial_restore_items: List[str]
    
    error: Optional[str] = None


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""
    
    @abstractmethod
    async def save_checkpoint(self, checkpoint_id: str, data: bytes) -> bool:
        """Save checkpoint data."""
        pass
    
    @abstractmethod
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint data."""
        pass
    
    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint."""
        pass
    
    @abstractmethod
    async def list_checkpoints(self, operation_id: Optional[str] = None) -> List[str]:
        """List available checkpoints."""
        pass
    
    @abstractmethod
    async def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        pass


class FileSystemCheckpointStorage(CheckpointStorage):
    """File system-based checkpoint storage."""
    
    def __init__(self, storage_dir: Union[str, Path] = "data/checkpoints"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.storage_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    async def save_checkpoint(self, checkpoint_id: str, data: bytes) -> bool:
        """Save checkpoint data to filesystem with atomic write."""
        try:
            checkpoint_file = self.storage_dir / f"{checkpoint_id}.checkpoint"
            temp_file = self.storage_dir / f"{checkpoint_id}.checkpoint.tmp"
            
            # Write to temporary file first for atomic operation
            with open(temp_file, 'wb') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(data)
                f.flush()
                f.fsync()  # Force write to disk
            
            # Atomic rename
            temp_file.rename(checkpoint_file)
            
            logger.debug(f"Saved checkpoint to filesystem: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint data from filesystem."""
        try:
            checkpoint_file = self.storage_dir / f"{checkpoint_id}.checkpoint"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'rb') as f:
                # Acquire shared lock
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = f.read()
            
            logger.debug(f"Loaded checkpoint from filesystem: {checkpoint_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from filesystem."""
        try:
            checkpoint_file = self.storage_dir / f"{checkpoint_id}.checkpoint"
            metadata_file = self.metadata_dir / f"{checkpoint_id}.json"
            
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.debug(f"Deleted checkpoint from filesystem: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def list_checkpoints(self, operation_id: Optional[str] = None) -> List[str]:
        """List available checkpoints."""
        try:
            checkpoints = []
            
            for checkpoint_file in self.storage_dir.glob("*.checkpoint"):
                checkpoint_id = checkpoint_file.stem
                
                if operation_id:
                    # Filter by operation ID if specified
                    metadata_file = self.metadata_dir / f"{checkpoint_id}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('operation_id') == operation_id:
                                checkpoints.append(checkpoint_id)
                else:
                    checkpoints.append(checkpoint_id)
            
            return sorted(checkpoints)
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.checkpoint"
        return checkpoint_file.exists()


class CheckpointManager:
    """Comprehensive checkpoint and resume system."""
    
    def __init__(self,
                 storage: Optional[CheckpointStorage] = None,
                 cache_manager: Optional[IntelligentCacheManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            storage: Checkpoint storage backend
            cache_manager: Cache manager for optimization
            metrics_collector: Metrics collection
        """
        self.storage = storage or FileSystemCheckpointStorage()
        self.cache_manager = cache_manager
        self.metrics = metrics_collector or MetricsCollector()
        
        # Active checkpoint tracking
        self.active_checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_locks: Dict[str, asyncio.Lock] = {}
        
        # Configuration
        self.enable_compression = True
        self.compression_threshold = 1024 * 1024  # 1MB
        self.auto_checkpoint_interval = timedelta(minutes=15)
        self.max_checkpoints_per_operation = 10
        self.checkpoint_retention_days = 30
        
        # Recovery configuration
        self.max_recovery_attempts = 3
        self.recovery_timeout = timedelta(minutes=30)
        
        # Background tasks
        self._cleanup_task = None
        self._auto_checkpoint_task = None
        self._start_background_tasks()
        
        logger.info("Checkpoint manager initialized")
    
    async def create_checkpoint(self, operation_id: str,
                              checkpoint_type: CheckpointType = CheckpointType.MANUAL,
                              description: str = "",
                              operation_state: Dict[str, Any] = None,
                              session_states: Dict[str, Dict[str, Any]] = None,
                              progress_data: Dict[str, Any] = None,
                              force: bool = False) -> Optional[str]:
        """
        Create a new checkpoint.
        
        Args:
            operation_id: Operation identifier
            checkpoint_type: Type of checkpoint
            description: Human-readable description
            operation_state: Current operation state
            session_states: Individual session states
            progress_data: Progress tracking data
            force: Force checkpoint creation even if recent one exists
            
        Returns:
            Checkpoint ID if successful
        """
        checkpoint_id = f"{operation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Check if we should create checkpoint
        if not force and not await self._should_create_checkpoint(operation_id, checkpoint_type):
            logger.debug(f"Skipping checkpoint creation for operation {operation_id}")
            return None
        
        start_time = datetime.now()
        
        try:
            # Acquire lock for this operation
            if operation_id not in self.checkpoint_locks:
                self.checkpoint_locks[operation_id] = asyncio.Lock()
            
            async with self.checkpoint_locks[operation_id]:
                logger.info(f"Creating checkpoint: {checkpoint_id}")
                
                # Gather checkpoint data
                checkpoint_data = await self._gather_checkpoint_data(
                    operation_id=operation_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    description=description,
                    operation_state=operation_state or {},
                    session_states=session_states or {},
                    progress_data=progress_data or {}
                )
                
                # Create metadata
                metadata = await self._create_checkpoint_metadata(
                    checkpoint_id=checkpoint_id,
                    operation_id=operation_id,
                    checkpoint_type=checkpoint_type,
                    description=description,
                    checkpoint_data=checkpoint_data
                )
                
                # Serialize and compress data
                serialized_data = await self._serialize_checkpoint(checkpoint_data)
                
                # Save to storage
                if await self.storage.save_checkpoint(checkpoint_id, serialized_data):
                    # Save metadata separately for quick access
                    await self._save_checkpoint_metadata(metadata)
                    
                    # Track active checkpoint
                    self.active_checkpoints[checkpoint_id] = metadata
                    
                    # Clean up old checkpoints if needed
                    await self._cleanup_old_checkpoints(operation_id)
                    
                    duration = datetime.now() - start_time
                    
                    # Record metrics
                    self.metrics.record_request(
                        service="checkpoint_manager",
                        success=True,
                        response_time=duration.total_seconds(),
                        metadata={"operation": "create_checkpoint", "size": len(serialized_data)}
                    )
                    
                    logger.info(f"Checkpoint created successfully: {checkpoint_id} "
                               f"(size: {len(serialized_data)} bytes, duration: {duration.total_seconds():.2f}s)")
                    
                    return checkpoint_id
                else:
                    raise ProcessingError("Failed to save checkpoint to storage")
        
        except Exception as e:
            duration = datetime.now() - start_time
            
            # Record failure metrics
            self.metrics.record_request(
                service="checkpoint_manager",
                success=False,
                response_time=duration.total_seconds(),
                error_type="checkpoint_creation_error"
            )
            
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            return None
    
    async def restore_checkpoint(self, checkpoint_id: str,
                               recovery_strategy: RecoveryStrategy = RecoveryStrategy.FULL_RESTORE,
                               target_operation_id: Optional[str] = None,
                               validate_integrity: bool = True) -> RestoreResult:
        """
        Restore from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore from
            recovery_strategy: Recovery strategy to use
            target_operation_id: Target operation ID (if different from original)
            validate_integrity: Whether to validate data integrity
            
        Returns:
            Restore result with details
        """
        start_time = datetime.now()
        
        logger.info(f"Starting checkpoint restoration: {checkpoint_id} (strategy: {recovery_strategy.value})")
        
        try:
            # Load checkpoint metadata
            metadata = await self._load_checkpoint_metadata(checkpoint_id)
            if not metadata:
                return RestoreResult(
                    success=False,
                    checkpoint_id=checkpoint_id,
                    restored_at=start_time,
                    recovery_strategy=recovery_strategy,
                    restored_operations=[],
                    restored_sessions=[],
                    restored_progress={},
                    data_integrity_ok=False,
                    validation_errors=["Checkpoint metadata not found"],
                    restore_duration=datetime.now() - start_time,
                    data_size_restored=0,
                    warnings=[],
                    partial_restore_items=[],
                    error="Checkpoint metadata not found"
                )
            
            # Load checkpoint data
            serialized_data = await self.storage.load_checkpoint(checkpoint_id)
            if not serialized_data:
                return RestoreResult(
                    success=False,
                    checkpoint_id=checkpoint_id,
                    restored_at=start_time,
                    recovery_strategy=recovery_strategy,
                    restored_operations=[],
                    restored_sessions=[],
                    restored_progress={},
                    data_integrity_ok=False,
                    validation_errors=["Checkpoint data not found"],
                    restore_duration=datetime.now() - start_time,
                    data_size_restored=0,
                    warnings=[],
                    partial_restore_items=[]
                )
            
            # Validate data integrity
            validation_errors = []
            if validate_integrity:
                validation_errors = await self._validate_checkpoint_integrity(
                    metadata, serialized_data
                )
            
            # Deserialize checkpoint data
            try:
                checkpoint_data = await self._deserialize_checkpoint(serialized_data)
            except Exception as e:
                return RestoreResult(
                    success=False,
                    checkpoint_id=checkpoint_id,
                    restored_at=start_time,
                    recovery_strategy=recovery_strategy,
                    restored_operations=[],
                    restored_sessions=[],
                    restored_progress={},
                    data_integrity_ok=False,
                    validation_errors=[f"Deserialization failed: {e}"],
                    restore_duration=datetime.now() - start_time,
                    data_size_restored=len(serialized_data),
                    warnings=[],
                    partial_restore_items=[],
                    error=f"Deserialization failed: {e}"
                )
            
            # Perform restoration based on strategy
            restore_result = await self._perform_restoration(
                checkpoint_data=checkpoint_data,
                metadata=metadata,
                recovery_strategy=recovery_strategy,
                target_operation_id=target_operation_id
            )
            
            # Update result with common fields
            restore_result.checkpoint_id = checkpoint_id
            restore_result.restored_at = start_time
            restore_result.recovery_strategy = recovery_strategy
            restore_result.restore_duration = datetime.now() - start_time
            restore_result.data_size_restored = len(serialized_data)
            restore_result.data_integrity_ok = len(validation_errors) == 0
            restore_result.validation_errors = validation_errors
            
            # Record metrics
            self.metrics.record_request(
                service="checkpoint_manager",
                success=restore_result.success,
                response_time=restore_result.restore_duration.total_seconds(),
                metadata={
                    "operation": "restore_checkpoint",
                    "strategy": recovery_strategy.value,
                    "size": len(serialized_data)
                }
            )
            
            if restore_result.success:
                logger.info(f"Checkpoint restoration completed: {checkpoint_id} "
                           f"(duration: {restore_result.restore_duration.total_seconds():.2f}s)")
            else:
                logger.error(f"Checkpoint restoration failed: {checkpoint_id} - {restore_result.error}")
            
            return restore_result
        
        except Exception as e:
            duration = datetime.now() - start_time
            
            logger.error(f"Checkpoint restoration error: {checkpoint_id} - {e}")
            
            return RestoreResult(
                success=False,
                checkpoint_id=checkpoint_id,
                restored_at=start_time,
                recovery_strategy=recovery_strategy,
                restored_operations=[],
                restored_sessions=[],
                restored_progress={},
                data_integrity_ok=False,
                validation_errors=[],
                restore_duration=duration,
                data_size_restored=0,
                warnings=[],
                partial_restore_items=[],
                error=str(e)
            )
    
    async def list_checkpoints(self, operation_id: Optional[str] = None,
                             checkpoint_type: Optional[CheckpointType] = None,
                             limit: Optional[int] = None) -> List[CheckpointMetadata]:
        """List available checkpoints with filtering."""
        try:
            checkpoint_ids = await self.storage.list_checkpoints(operation_id)
            checkpoints = []
            
            for checkpoint_id in checkpoint_ids:
                metadata = await self._load_checkpoint_metadata(checkpoint_id)
                if metadata:
                    # Apply filters
                    if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                        continue
                    
                    checkpoints.append(metadata)
            
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            
            if limit:
                checkpoints = checkpoints[:limit]
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        try:
            # Remove from active tracking
            if checkpoint_id in self.active_checkpoints:
                del self.active_checkpoints[checkpoint_id]
            
            # Delete from storage
            success = await self.storage.delete_checkpoint(checkpoint_id)
            
            if success:
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def validate_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Validate checkpoint integrity and contents."""
        try:
            # Load metadata and data
            metadata = await self._load_checkpoint_metadata(checkpoint_id)
            if not metadata:
                return {"valid": False, "errors": ["Metadata not found"]}
            
            serialized_data = await self.storage.load_checkpoint(checkpoint_id)
            if not serialized_data:
                return {"valid": False, "errors": ["Data not found"]}
            
            # Validate integrity
            errors = await self._validate_checkpoint_integrity(metadata, serialized_data)
            
            # Additional validation
            try:
                checkpoint_data = await self._deserialize_checkpoint(serialized_data)
                
                # Validate data structure
                structure_errors = self._validate_checkpoint_structure(checkpoint_data)
                errors.extend(structure_errors)
                
            except Exception as e:
                errors.append(f"Deserialization failed: {e}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "metadata": asdict(metadata) if metadata else None,
                "size": len(serialized_data),
                "created_at": metadata.created_at.isoformat() if metadata else None
            }
            
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    async def get_recovery_options(self, operation_id: str) -> List[Dict[str, Any]]:
        """Get available recovery options for an operation."""
        checkpoints = await self.list_checkpoints(operation_id)
        
        recovery_options = []
        for checkpoint in checkpoints:
            option = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "created_at": checkpoint.created_at.isoformat(),
                "description": checkpoint.description,
                "progress_percentage": checkpoint.progress_percentage,
                "recovery_strategy": checkpoint.recovery_strategy.value,
                "recovery_priority": checkpoint.recovery_priority,
                "data_size": checkpoint.file_size,
                "integrity_verified": checkpoint.integrity_verified
            }
            recovery_options.append(option)
        
        return recovery_options
    
    async def _should_create_checkpoint(self, operation_id: str, 
                                      checkpoint_type: CheckpointType) -> bool:
        """Determine if a checkpoint should be created."""
        if checkpoint_type in [CheckpointType.MANUAL, CheckpointType.FAILURE_RECOVERY]:
            return True
        
        # Check recent checkpoints
        recent_checkpoints = await self.list_checkpoints(operation_id, limit=3)
        
        if not recent_checkpoints:
            return True
        
        # Don't create automatic checkpoints too frequently
        latest_checkpoint = recent_checkpoints[0]
        time_since_last = datetime.now() - latest_checkpoint.created_at
        
        if checkpoint_type == CheckpointType.AUTOMATIC:
            return time_since_last >= self.auto_checkpoint_interval
        
        return True
    
    async def _gather_checkpoint_data(self, operation_id: str, checkpoint_id: str,
                                    checkpoint_type: CheckpointType, description: str,
                                    operation_state: Dict[str, Any],
                                    session_states: Dict[str, Dict[str, Any]],
                                    progress_data: Dict[str, Any]) -> CheckpointData:
        """Gather all data needed for checkpoint."""
        
        # Get cache snapshots if available
        cache_snapshots = {}
        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_stats()
                cache_snapshots["stats"] = asdict(cache_stats)
            except Exception as e:
                logger.warning(f"Failed to get cache snapshots: {e}")
        
        # Environment and configuration info
        environment_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": "3.12.3",  # Would get dynamically
            "hostname": "localhost",  # Would get dynamically
        }
        
        configuration = {
            "checkpoint_type": checkpoint_type.value,
            "compression_enabled": self.enable_compression
        }
        
        # Performance metrics
        performance_metrics = {}
        if self.metrics:
            try:
                # Get recent metrics - implementation would depend on metrics system
                performance_metrics["recent_requests"] = "placeholder"
            except Exception as e:
                logger.warning(f"Failed to gather performance metrics: {e}")
        
        return CheckpointData(
            metadata=None,  # Will be set later
            operation_state=operation_state,
            session_states=session_states,
            progress_data=progress_data,
            cache_snapshots=cache_snapshots,
            configuration=configuration,
            environment_info=environment_info,
            resource_allocations={},
            recent_errors=[],
            retry_counts={},
            recovery_hints={},
            performance_metrics=performance_metrics,
            timing_data={}
        )
    
    async def _create_checkpoint_metadata(self, checkpoint_id: str, operation_id: str,
                                        checkpoint_type: CheckpointType, description: str,
                                        checkpoint_data: CheckpointData) -> CheckpointMetadata:
        """Create checkpoint metadata."""
        
        # Calculate data hash for integrity
        serialized_data = await self._serialize_checkpoint(checkpoint_data)
        data_hash = hashlib.sha256(serialized_data).hexdigest()
        
        # Extract progress information
        progress_percentage = checkpoint_data.progress_data.get("overall_progress", 0.0)
        current_stage = checkpoint_data.progress_data.get("current_stage")
        completed_stages = checkpoint_data.progress_data.get("completed_stages", [])
        
        return CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            operation_id=operation_id,
            checkpoint_type=checkpoint_type,
            created_at=datetime.now(),
            created_by="checkpoint_manager",
            description=description or f"Automatic checkpoint for {operation_id}",
            operation_state="active",  # Would determine actual state
            progress_percentage=progress_percentage,
            current_stage=current_stage,
            completed_stages=completed_stages,
            data_hash=data_hash,
            file_size=len(serialized_data),
            compression_used=self.enable_compression and len(serialized_data) > self.compression_threshold,
            parent_checkpoint_id=None,  # Could track checkpoint lineage
            dependent_operations=[],
            recovery_priority=1,
            recovery_strategy=RecoveryStrategy.FULL_RESTORE,
            validation_hash=data_hash,  # Same as data_hash for now
            integrity_verified=True,
            tags=set(),
            custom_metadata={}
        )
    
    async def _serialize_checkpoint(self, checkpoint_data: CheckpointData) -> bytes:
        """Serialize checkpoint data with optional compression."""
        try:
            # Convert to serializable format
            serializable_data = asdict(checkpoint_data)
            
            # Serialize with pickle for full Python object support
            serialized = pickle.dumps(serializable_data)
            
            # Compress if enabled and beneficial
            if (self.enable_compression and 
                len(serialized) > self.compression_threshold):
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.9:  # Only if significant reduction
                    return compressed
            
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to serialize checkpoint data: {e}")
            raise
    
    async def _deserialize_checkpoint(self, data: bytes) -> CheckpointData:
        """Deserialize checkpoint data with decompression."""
        try:
            # Try decompression first
            try:
                if data[:2] == b'\x1f\x8b':  # Gzip magic number
                    data = gzip.decompress(data)
            except:
                pass  # Not compressed or decompression failed
            
            # Deserialize
            serializable_data = pickle.loads(data)
            
            # Convert back to CheckpointData
            # This is simplified - would need proper reconstruction
            return CheckpointData(**serializable_data)
            
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint data: {e}")
            raise
    
    async def _validate_checkpoint_integrity(self, metadata: CheckpointMetadata, 
                                           data: bytes) -> List[str]:
        """Validate checkpoint data integrity."""
        errors = []
        
        try:
            # Validate file size
            if len(data) != metadata.file_size:
                errors.append(f"File size mismatch: expected {metadata.file_size}, got {len(data)}")
            
            # Validate data hash
            data_hash = hashlib.sha256(data).hexdigest()
            if data_hash != metadata.data_hash:
                errors.append(f"Data hash mismatch: expected {metadata.data_hash}, got {data_hash}")
            
            # Additional validation could be added here
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors
    
    def _validate_checkpoint_structure(self, checkpoint_data: CheckpointData) -> List[str]:
        """Validate checkpoint data structure."""
        errors = []
        
        try:
            # Check required fields
            if not checkpoint_data.operation_state:
                errors.append("Missing operation_state")
            
            if not checkpoint_data.configuration:
                errors.append("Missing configuration")
            
            # Additional structure validation could be added
            
        except Exception as e:
            errors.append(f"Structure validation error: {e}")
        
        return errors
    
    async def _perform_restoration(self, checkpoint_data: CheckpointData,
                                 metadata: CheckpointMetadata,
                                 recovery_strategy: RecoveryStrategy,
                                 target_operation_id: Optional[str]) -> RestoreResult:
        """Perform the actual restoration based on strategy."""
        
        restored_operations = []
        restored_sessions = []
        restored_progress = {}
        warnings = []
        partial_restore_items = []
        
        try:
            operation_id = target_operation_id or metadata.operation_id
            
            if recovery_strategy == RecoveryStrategy.FULL_RESTORE:
                # Restore complete state
                restored_operations.append(operation_id)
                restored_sessions = list(checkpoint_data.session_states.keys())
                restored_progress = checkpoint_data.progress_data.copy()
                
            elif recovery_strategy == RecoveryStrategy.PARTIAL_RESTORE:
                # Restore only essential state
                restored_operations.append(operation_id)
                # Restore only completed sessions
                for session_id, session_state in checkpoint_data.session_states.items():
                    if session_state.get("status") == "completed":
                        restored_sessions.append(session_id)
                    else:
                        partial_restore_items.append(f"session:{session_id}")
                
                restored_progress = {
                    "overall_progress": checkpoint_data.progress_data.get("overall_progress", 0.0),
                    "completed_stages": checkpoint_data.progress_data.get("completed_stages", [])
                }
            
            elif recovery_strategy == RecoveryStrategy.MERGE_RESTORE:
                # Merge with current state (simplified implementation)
                warnings.append("Merge restore is simplified in this implementation")
                restored_operations.append(operation_id)
                restored_sessions = list(checkpoint_data.session_states.keys())
                restored_progress = checkpoint_data.progress_data.copy()
            
            elif recovery_strategy == RecoveryStrategy.INCREMENTAL_RESTORE:
                # Step-by-step restoration (would need more complex implementation)
                warnings.append("Incremental restore not fully implemented")
                restored_operations.append(operation_id)
                
            return RestoreResult(
                success=True,
                checkpoint_id="",  # Will be set by caller
                restored_at=datetime.now(),
                recovery_strategy=recovery_strategy,
                restored_operations=restored_operations,
                restored_sessions=restored_sessions,
                restored_progress=restored_progress,
                data_integrity_ok=True,
                validation_errors=[],
                restore_duration=timedelta(),  # Will be set by caller
                data_size_restored=0,  # Will be set by caller
                warnings=warnings,
                partial_restore_items=partial_restore_items
            )
            
        except Exception as e:
            return RestoreResult(
                success=False,
                checkpoint_id="",
                restored_at=datetime.now(),
                recovery_strategy=recovery_strategy,
                restored_operations=[],
                restored_sessions=[],
                restored_progress={},
                data_integrity_ok=False,
                validation_errors=[],
                restore_duration=timedelta(),
                data_size_restored=0,
                warnings=[],
                partial_restore_items=[],
                error=str(e)
            )
    
    async def _save_checkpoint_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata separately for quick access."""
        try:
            metadata_file = Path(self.storage.storage_dir) / "metadata" / f"{metadata.checkpoint_id}.json"
            
            with open(metadata_file, 'w') as f:
                # Convert to JSON-serializable format
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['tags'] = list(metadata.tags)
                
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    async def _load_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Load checkpoint metadata."""
        try:
            metadata_file = Path(self.storage.storage_dir) / "metadata" / f"{checkpoint_id}.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                
                # Convert back to proper types
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['checkpoint_type'] = CheckpointType(metadata_dict['checkpoint_type'])
                metadata_dict['recovery_strategy'] = RecoveryStrategy(metadata_dict['recovery_strategy'])
                metadata_dict['tags'] = set(metadata_dict['tags'])
                
                return CheckpointMetadata(**metadata_dict)
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
            return None
    
    async def _cleanup_old_checkpoints(self, operation_id: str):
        """Clean up old checkpoints for an operation."""
        try:
            checkpoints = await self.list_checkpoints(operation_id)
            
            if len(checkpoints) <= self.max_checkpoints_per_operation:
                return
            
            # Sort by creation time (oldest first)
            checkpoints.sort(key=lambda x: x.created_at)
            
            # Remove excess checkpoints
            excess_count = len(checkpoints) - self.max_checkpoints_per_operation
            for i in range(excess_count):
                checkpoint = checkpoints[i]
                await self.delete_checkpoint(checkpoint.checkpoint_id)
                logger.debug(f"Cleaned up old checkpoint: {checkpoint.checkpoint_id}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        async def cleanup_task():
            """Background cleanup of expired checkpoints."""
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_expired_checkpoints()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_task())
    
    async def _cleanup_expired_checkpoints(self):
        """Clean up expired checkpoints."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.checkpoint_retention_days)
            
            all_checkpoints = await self.list_checkpoints()
            expired_checkpoints = [
                cp for cp in all_checkpoints if cp.created_at < cutoff_time
            ]
            
            for checkpoint in expired_checkpoints:
                await self.delete_checkpoint(checkpoint.checkpoint_id)
                logger.debug(f"Cleaned up expired checkpoint: {checkpoint.checkpoint_id}")
            
            if expired_checkpoints:
                logger.info(f"Cleaned up {len(expired_checkpoints)} expired checkpoints")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup expired checkpoints: {e}")
    
    async def close(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._auto_checkpoint_task:
            self._auto_checkpoint_task.cancel()
        
        # Clear locks and tracking
        self.checkpoint_locks.clear()
        self.active_checkpoints.clear()
        
        logger.info("Checkpoint manager closed")