"""
Comprehensive data collection pipeline for EU Parliament scraping.
Orchestrates session discovery, data integration, document resolution, and caching.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
from pathlib import Path

from .session_discovery import SessionDiscoveryService
from .data_integration_service import DataIntegrationService, IntegrationResult
from .document_resolver import DocumentResolver, DocumentCollection
from ..core.intelligent_cache import IntelligentCacheManager, CacheType
from ..parsers.verbatim_parser import VerbatimParser
from ..parsers.speech_classifier import SpeechClassifier
from ..validators.parsing_validator import ParsingValidator
from ..models.session import SessionMetadata, SessionConfig
from ..models.speech import RawSpeechSegment
from ..core.exceptions import ProcessingError, DataValidationError
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Stages of the data collection pipeline."""
    DISCOVERY = "discovery"
    INTEGRATION = "integration"
    DOCUMENT_RESOLUTION = "document_resolution"
    CONTENT_PARSING = "content_parsing"
    CLASSIFICATION = "classification"
    VALIDATION = "validation"
    CACHING = "caching"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Overall pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: PipelineStage
    status: PipelineStatus
    data: Optional[Any]
    error: Optional[str]
    processing_time: float
    timestamp: datetime


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    session_id: str
    status: PipelineStatus
    stages: Dict[PipelineStage, StageResult]
    final_data: Optional[Dict[str, Any]]
    total_processing_time: float
    quality_score: float
    completeness_score: float
    document_count: int
    speech_segment_count: int
    created_at: datetime


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    max_concurrent_sessions: int = 5
    stage_timeout_minutes: int = 10
    quality_threshold: float = 0.7
    enable_document_resolution: bool = True
    enable_content_parsing: bool = True
    enable_classification: bool = True
    enable_validation: bool = True
    force_refresh: bool = False


class DataCollectionPipeline:
    """
    Enhanced comprehensive data collection pipeline with session management integration.
    Orchestrates the complete workflow from session discovery to validated speech segments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core services - enhanced with session integration
        self.session_discovery = SessionDiscoveryService()
        self.data_integration = DataIntegrationService()
        self.document_resolver = DocumentResolver()
        self.cache_manager = IntelligentCacheManager()
        
        # Parsing and validation services - enhanced
        self.verbatim_parser = VerbatimParser()
        self.speech_classifier = SpeechClassifier()
        self.parsing_validator = ParsingValidator()
        
        # Session management integration
        from .session_manager import SessionManager
        from .progress_tracker import AdvancedProgressTracker
        from .checkpoint_manager import CheckpointManager
        from .error_recovery_manager import ErrorRecoveryManager
        
        self.session_manager = SessionManager()
        self.progress_tracker = AdvancedProgressTracker()
        self.checkpoint_manager = CheckpointManager()
        self.error_recovery = ErrorRecoveryManager()
        
        # Performance monitoring
        self.metrics = MetricsCollector()
        
        # Pipeline state
        self.current_executions: Dict[str, PipelineExecution] = {}
        self.execution_history: List[PipelineExecution] = []
        
        # Enhanced configuration
        self.pipeline_config = {
            'max_concurrent_executions': 10,
            'session_batch_size': 50,
            'content_extraction_timeout': 300,
            'quality_threshold': 0.6,
            'enable_intelligent_caching': True,
            'enable_background_processing': True,
            'checkpoint_interval_minutes': 30,
            'retry_failed_sessions': True,
            'session_discovery_interval_hours': 6,
            **self.config
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        logger.info("Enhanced Data Collection Pipeline initialized with session management")

    async def start(self):
        """Start the enhanced pipeline with all integrated services"""
        logger.info("Starting Enhanced Data Collection Pipeline...")
        
        # Start integrated services
        await self.session_manager.start()
        await self.progress_tracker.start()
        await self.checkpoint_manager.start()
        await self.error_recovery.start()
        
        # Start cache manager
        await self.cache_manager.start()
        
        # Start background tasks
        if self.pipeline_config['enable_background_processing']:
            self._background_tasks.add(
                asyncio.create_task(self._background_session_discovery())
            )
            self._background_tasks.add(
                asyncio.create_task(self._background_pipeline_monitoring())
            )
            self._background_tasks.add(
                asyncio.create_task(self._background_quality_analysis())
            )
        
        logger.info("Enhanced Data Collection Pipeline started successfully")

    async def shutdown(self):
        """Shutdown pipeline and all services"""
        logger.info("Shutting down Enhanced Data Collection Pipeline...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for background tasks
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown services
        await self.error_recovery.shutdown()
        await self.checkpoint_manager.shutdown()
        await self.progress_tracker.shutdown()
        await self.session_manager.shutdown()
        await self.cache_manager.shutdown()
        
        logger.info("Enhanced Data Collection Pipeline shutdown complete")

    async def execute_comprehensive_collection(
        self,
        date_range: Tuple[datetime, datetime],
        session_types: List[str] = None,
        quality_threshold: float = None,
        enable_checkpointing: bool = True
    ) -> PipelineResult:
        """
        Execute comprehensive data collection with enhanced session management
        
        Args:
            date_range: Start and end dates for collection
            session_types: Types of sessions to collect (plenary, committee, etc.)
            quality_threshold: Minimum quality score for acceptance
            enable_checkpointing: Enable automatic checkpointing
            
        Returns:
            Complete pipeline result with quality metrics
        """
        start_time = datetime.now()
        execution_id = f"collection_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create pipeline execution record
        execution = PipelineExecution(
            execution_id=execution_id,
            start_time=start_time,
            date_range=date_range,
            session_types=session_types or ['plenary', 'committee'],
            status=PipelineStatus.RUNNING,
            stage=PipelineStage.DISCOVERY,
            quality_threshold=quality_threshold or self.pipeline_config['quality_threshold']
        )
        
        self.current_executions[execution_id] = execution
        
        # Initialize progress tracking
        progress_tracker_id = await self.progress_tracker.start_tracking(
            operation_id=execution_id,
            operation_type="comprehensive_collection",
            total_stages=7,
            context={
                'date_range': [date_range[0].isoformat(), date_range[1].isoformat()],
                'session_types': session_types,
                'quality_threshold': quality_threshold
            }
        )
        
        try:
            # Stage 1: Enhanced Session Discovery
            logger.info(f"Starting comprehensive collection: {execution_id}")
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "discovery", 0.0, "Starting session discovery"
            )
            
            if enable_checkpointing:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    operation_id=execution_id,
                    checkpoint_type="automatic",
                    state={
                        'stage': 'discovery_start',
                        'execution': asdict(execution)
                    }
                )
            
            discovered_sessions = await self._execute_enhanced_discovery(
                execution, progress_tracker_id
            )
            
            # Stage 2: Data Integration and Resolution
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "integration", 0.0, "Starting data integration"
            )
            
            integrated_data = await self._execute_data_integration(
                discovered_sessions, execution, progress_tracker_id
            )
            
            # Stage 3: Document Resolution and Collection
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "document_resolution", 0.0, "Resolving documents"
            )
            
            document_collections = await self._execute_document_resolution(
                integrated_data, execution, progress_tracker_id
            )
            
            # Stage 4: Enhanced Content Parsing
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "content_parsing", 0.0, "Parsing content"
            )
            
            parsed_content = await self._execute_enhanced_parsing(
                document_collections, execution, progress_tracker_id
            )
            
            # Stage 5: Speech Classification and Enhancement
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "classification", 0.0, "Classifying speeches"
            )
            
            classified_content = await self._execute_enhanced_classification(
                parsed_content, execution, progress_tracker_id
            )
            
            # Stage 6: Quality Validation and Scoring
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "validation", 0.0, "Validating quality"
            )
            
            validated_content = await self._execute_enhanced_validation(
                classified_content, execution, progress_tracker_id
            )
            
            # Stage 7: Caching and Storage
            await self.progress_tracker.update_stage_progress(
                progress_tracker_id, "caching", 0.0, "Caching results"
            )
            
            cached_results = await self._execute_intelligent_caching(
                validated_content, execution, progress_tracker_id
            )
            
            # Complete execution
            execution.status = PipelineStatus.COMPLETED
            execution.stage = PipelineStage.COMPLETED
            execution.end_time = datetime.now()
            execution.duration_minutes = (execution.end_time - execution.start_time).total_seconds() / 60
            
            await self.progress_tracker.complete_tracking(
                progress_tracker_id, "Comprehensive collection completed successfully"
            )
            
            # Create final result
            result = PipelineResult(
                execution_id=execution_id,
                status=PipelineStatus.COMPLETED,
                sessions_discovered=len(discovered_sessions),
                sessions_processed=len(validated_content),
                total_speeches=sum(len(session.speeches) for session in validated_content),
                quality_metrics=self._calculate_quality_metrics(validated_content),
                performance_metrics=self._calculate_performance_metrics(execution),
                execution_summary=execution,
                detailed_results=cached_results
            )
            
            # Store result
            self.execution_history.append(execution)
            if execution_id in self.current_executions:
                del self.current_executions[execution_id]
            
            # Update metrics
            self.metrics.increment('pipeline_executions_completed')
            self.metrics.histogram('pipeline_execution_duration', execution.duration_minutes)
            self.metrics.gauge('pipeline_sessions_processed', len(validated_content))
            self.metrics.gauge('pipeline_speeches_extracted', result.total_speeches)
            
            logger.info(f"Comprehensive collection completed: {execution_id} - "
                       f"{result.sessions_processed} sessions, {result.total_speeches} speeches")
            
            return result
            
        except Exception as e:
            # Handle execution failure with error recovery
            logger.error(f"Pipeline execution failed: {execution_id} - {e}")
            
            execution.status = PipelineStatus.FAILED
            execution.stage = PipelineStage.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)
            
            await self.progress_tracker.mark_failed(
                progress_tracker_id, f"Pipeline execution failed: {str(e)}"
            )
            
            # Attempt error recovery
            try:
                recovery_result = await self.error_recovery.handle_error(
                    e,
                    context={
                        'execution_id': execution_id,
                        'stage': execution.stage.value,
                        'operation': 'comprehensive_collection'
                    }
                )
                
                if recovery_result:
                    logger.info(f"Error recovery successful for execution: {execution_id}")
                    
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
            
            # Update metrics
            self.metrics.increment('pipeline_executions_failed')
            
            raise ProcessingError(f"Pipeline execution failed: {str(e)}") from e

    async def _execute_enhanced_discovery(
        self,
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[SessionMetadata]:
        """Execute enhanced session discovery with progress tracking"""
        
        try:
            # Use session manager for discovery
            discovery_result = await self.session_discovery.discover_sessions_batch(
                start_date=execution.date_range[0],
                end_date=execution.date_range[1],
                session_types=execution.session_types,
                batch_size=self.pipeline_config['session_batch_size']
            )
            
            # Register discovered sessions with session manager
            registered_sessions = []
            for i, session in enumerate(discovery_result):
                # Register with session manager
                session_id = await self.session_manager.register_session(
                    session_metadata=session,
                    priority='NORMAL',
                    dependencies=[],
                    context={
                        'execution_id': execution.execution_id,
                        'discovery_batch': i // self.pipeline_config['session_batch_size']
                    }
                )
                
                registered_sessions.append(session)
                
                # Update progress
                progress = (i + 1) / len(discovery_result)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id, 
                    "discovery", 
                    progress, 
                    f"Discovered {i + 1}/{len(discovery_result)} sessions"
                )
            
            # Mark discovery stage complete
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "discovery", f"Discovered {len(registered_sessions)} sessions"
            )
            
            execution.sessions_discovered = len(registered_sessions)
            execution.stage = PipelineStage.INTEGRATION
            
            return registered_sessions
            
        except Exception as e:
            logger.error(f"Enhanced discovery failed: {e}")
            raise ProcessingError(f"Session discovery failed: {str(e)}") from e

    async def _execute_data_integration(
        self,
        sessions: List[SessionMetadata],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[IntegrationResult]:
        """Execute data integration with multiple sources"""
        
        try:
            integrated_results = []
            
            # Process sessions in batches for better performance
            batch_size = 10
            session_batches = [sessions[i:i + batch_size] for i in range(0, len(sessions), batch_size)]
            
            for batch_index, session_batch in enumerate(session_batches):
                # Create batch integration tasks
                batch_tasks = []
                for session in session_batch:
                    task = self.data_integration.integrate_session_data(
                        session_metadata=session,
                        data_sources=['opendata', 'eurlex', 'verbatim'],
                        quality_threshold=execution.quality_threshold
                    )
                    batch_tasks.append(task)
                
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results and handle errors
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Integration failed for session: {result}")
                        # Use error recovery
                        try:
                            recovery_result = await self.error_recovery.handle_error(
                                result,
                                context={
                                    'operation': 'data_integration',
                                    'execution_id': execution.execution_id
                                }
                            )
                            if recovery_result:
                                integrated_results.append(recovery_result)
                        except Exception:
                            pass  # Skip failed session
                    else:
                        integrated_results.append(result)
                
                # Update progress
                progress = (batch_index + 1) / len(session_batches)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "integration",
                    progress,
                    f"Integrated {len(integrated_results)}/{len(sessions)} sessions"
                )
            
            # Mark integration stage complete
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "integration", f"Integrated {len(integrated_results)} sessions"
            )
            
            execution.sessions_integrated = len(integrated_results)
            execution.stage = PipelineStage.DOCUMENT_RESOLUTION
            
            return integrated_results
            
        except Exception as e:
            logger.error(f"Data integration failed: {e}")
            raise ProcessingError(f"Data integration failed: {str(e)}") from e

    async def _execute_document_resolution(
        self,
        integration_results: List[IntegrationResult],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[DocumentCollection]:
        """Execute document resolution with enhanced error handling"""
        
        try:
            document_collections = []
            
            for i, integration_result in enumerate(integration_results):
                try:
                    # Resolve documents for each integration result
                    document_collection = await self.document_resolver.resolve_session_documents(
                        session_metadata=integration_result.session_metadata,
                        integration_data=integration_result,
                        include_transcripts=True,
                        include_amendments=True,
                        include_voting_records=True
                    )
                    
                    document_collections.append(document_collection)
                    
                except Exception as e:
                    logger.warning(f"Document resolution failed for session {integration_result.session_metadata.session_id}: {e}")
                    
                    # Try error recovery
                    try:
                        recovery_result = await self.error_recovery.handle_error(
                            e,
                            context={
                                'operation': 'document_resolution',
                                'session_id': integration_result.session_metadata.session_id,
                                'execution_id': execution.execution_id
                            }
                        )
                        if recovery_result:
                            document_collections.append(recovery_result)
                    except Exception:
                        pass  # Skip failed session
                
                # Update progress
                progress = (i + 1) / len(integration_results)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "document_resolution",
                    progress,
                    f"Resolved documents for {i + 1}/{len(integration_results)} sessions"
                )
            
            # Mark document resolution stage complete
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "document_resolution", 
                f"Resolved documents for {len(document_collections)} sessions"
            )
            
            execution.documents_resolved = sum(len(collection.documents) for collection in document_collections)
            execution.stage = PipelineStage.CONTENT_PARSING
            
            return document_collections
            
        except Exception as e:
            logger.error(f"Document resolution failed: {e}")
            raise ProcessingError(f"Document resolution failed: {str(e)}") from e

    async def _execute_enhanced_parsing(
        self,
        document_collections: List[DocumentCollection],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[SessionParsingResult]:
        """Execute enhanced content parsing with quality validation"""
        
        try:
            parsing_results = []
            
            for i, doc_collection in enumerate(document_collections):
                try:
                    session_parsing_result = SessionParsingResult(
                        session_id=doc_collection.session_metadata.session_id,
                        session_metadata=doc_collection.session_metadata,
                        speeches=[],
                        parsing_quality=0.0,
                        parsing_metadata={}
                    )
                    
                    # Parse each document in the collection
                    for document in doc_collection.documents:
                        if document.document_type == 'verbatim':
                            # Parse verbatim content
                            speeches = self.verbatim_parser.parse_verbatim_report(
                                html_content=document.content,
                                session_id=doc_collection.session_metadata.session_id,
                                session_date=doc_collection.session_metadata.date
                            )
                            session_parsing_result.speeches.extend(speeches)
                    
                    # Calculate parsing quality
                    if session_parsing_result.speeches:
                        valid_speeches = [s for s in session_parsing_result.speeches if len(s.speech_text.strip()) > 10]
                        session_parsing_result.parsing_quality = len(valid_speeches) / len(session_parsing_result.speeches)
                    
                    # Add metadata
                    session_parsing_result.parsing_metadata = {
                        'total_documents': len(doc_collection.documents),
                        'parsing_timestamp': datetime.now().isoformat(),
                        'parser_version': '2.0',
                        'execution_id': execution.execution_id
                    }
                    
                    parsing_results.append(session_parsing_result)
                    
                except Exception as e:
                    logger.warning(f"Parsing failed for session {doc_collection.session_metadata.session_id}: {e}")
                    
                    # Error recovery
                    try:
                        recovery_result = await self.error_recovery.handle_error(
                            e,
                            context={
                                'operation': 'content_parsing',
                                'session_id': doc_collection.session_metadata.session_id,
                                'execution_id': execution.execution_id
                            }
                        )
                        if recovery_result:
                            parsing_results.append(recovery_result)
                    except Exception:
                        pass
                
                # Update progress
                progress = (i + 1) / len(document_collections)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "content_parsing",
                    progress,
                    f"Parsed {i + 1}/{len(document_collections)} sessions"
                )
            
            # Mark parsing stage complete
            total_speeches = sum(len(result.speeches) for result in parsing_results)
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "content_parsing",
                f"Parsed {total_speeches} speeches from {len(parsing_results)} sessions"
            )
            
            execution.speeches_parsed = total_speeches
            execution.stage = PipelineStage.CLASSIFICATION
            
            return parsing_results
            
        except Exception as e:
            logger.error(f"Enhanced parsing failed: {e}")
            raise ProcessingError(f"Content parsing failed: {str(e)}") from e

    async def _execute_enhanced_classification(
        self,
        parsing_results: List[SessionParsingResult],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[SessionParsingResult]:
        """Execute enhanced speech classification with quality tracking"""
        
        try:
            classified_results = []
            
            for i, parsing_result in enumerate(parsing_results):
                try:
                    # Classify each speech in the session
                    for speech in parsing_result.speeches:
                        classification = self.speech_classifier.classify_segment(speech)
                        
                        # Add classification metadata
                        speech.parsing_metadata['classification'] = {
                            'content_type': classification.content_type.value,
                            'confidence': classification.confidence,
                            'announcement_type': classification.announcement_type.value if classification.announcement_type else None,
                            'classifier_version': '2.0',
                            'classification_timestamp': datetime.now().isoformat()
                        }
                    
                    # Update parsing result quality based on classification
                    high_confidence_speeches = [
                        s for s in parsing_result.speeches 
                        if s.parsing_metadata.get('classification', {}).get('confidence', 0) >= 0.4
                    ]
                    
                    if parsing_result.speeches:
                        classification_quality = len(high_confidence_speeches) / len(parsing_result.speeches)
                        parsing_result.parsing_quality = (parsing_result.parsing_quality + classification_quality) / 2
                    
                    classified_results.append(parsing_result)
                    
                except Exception as e:
                    logger.warning(f"Classification failed for session {parsing_result.session_id}: {e}")
                    
                    # Error recovery
                    try:
                        recovery_result = await self.error_recovery.handle_error(
                            e,
                            context={
                                'operation': 'speech_classification',
                                'session_id': parsing_result.session_id,
                                'execution_id': execution.execution_id
                            }
                        )
                        if recovery_result:
                            classified_results.append(recovery_result)
                        else:
                            # Add without classification as fallback
                            classified_results.append(parsing_result)
                    except Exception:
                        classified_results.append(parsing_result)
                
                # Update progress
                progress = (i + 1) / len(parsing_results)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "classification",
                    progress,
                    f"Classified speeches in {i + 1}/{len(parsing_results)} sessions"
                )
            
            # Mark classification stage complete
            total_classified_speeches = sum(
                len([s for s in result.speeches if 'classification' in s.parsing_metadata])
                for result in classified_results
            )
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "classification",
                f"Classified {total_classified_speeches} speeches"
            )
            
            execution.speeches_classified = total_classified_speeches
            execution.stage = PipelineStage.VALIDATION
            
            return classified_results
            
        except Exception as e:
            logger.error(f"Enhanced classification failed: {e}")
            raise ProcessingError(f"Speech classification failed: {str(e)}") from e

    async def _execute_enhanced_validation(
        self,
        classified_results: List[SessionParsingResult],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> List[SessionParsingResult]:
        """Execute enhanced quality validation with comprehensive scoring"""
        
        try:
            validated_results = []
            
            for i, session_result in enumerate(classified_results):
                try:
                    # Validate the entire session's speeches
                    validation_result = self.parsing_validator.validate_batch(session_result.speeches)
                    
                    # Update session with validation results
                    session_result.parsing_metadata.update({
                        'validation': {
                            'quality_score': validation_result.quality_score,
                            'segment_count': validation_result.segment_count,
                            'issues_count': len(validation_result.issues),
                            'validation_timestamp': datetime.now().isoformat(),
                            'validator_version': '2.0'
                        }
                    })
                    
                    # Only include sessions that meet quality threshold
                    if validation_result.quality_score >= execution.quality_threshold:
                        validated_results.append(session_result)
                    else:
                        logger.warning(f"Session {session_result.session_id} failed quality threshold: "
                                     f"{validation_result.quality_score} < {execution.quality_threshold}")
                    
                except Exception as e:
                    logger.warning(f"Validation failed for session {session_result.session_id}: {e}")
                    
                    # Error recovery
                    try:
                        recovery_result = await self.error_recovery.handle_error(
                            e,
                            context={
                                'operation': 'quality_validation',
                                'session_id': session_result.session_id,
                                'execution_id': execution.execution_id
                            }
                        )
                        if recovery_result:
                            validated_results.append(recovery_result)
                    except Exception:
                        pass
                
                # Update progress
                progress = (i + 1) / len(classified_results)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "validation",
                    progress,
                    f"Validated {i + 1}/{len(classified_results)} sessions"
                )
            
            # Mark validation stage complete
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "validation",
                f"Validated {len(validated_results)} sessions (quality threshold: {execution.quality_threshold})"
            )
            
            execution.sessions_validated = len(validated_results)
            execution.stage = PipelineStage.CACHING
            
            return validated_results
            
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            raise ProcessingError(f"Quality validation failed: {str(e)}") from e

    async def _execute_intelligent_caching(
        self,
        validated_results: List[SessionParsingResult],
        execution: PipelineExecution,
        progress_tracker_id: str
    ) -> Dict[str, Any]:
        """Execute intelligent caching with multiple cache strategies"""
        
        try:
            cache_results = {
                'sessions_cached': 0,
                'speeches_cached': 0,
                'cache_strategies_used': [],
                'cache_efficiency': 0.0
            }
            
            for i, session_result in enumerate(validated_results):
                try:
                    # Cache session metadata
                    session_cache_key = f"session:{session_result.session_id}"
                    await self.cache_manager.set(
                        key=session_cache_key,
                        value=asdict(session_result.session_metadata),
                        cache_type=CacheType.METADATA,
                        ttl_hours=24
                    )
                    
                    # Cache speeches with content-based strategy
                    for speech in session_result.speeches:
                        speech_cache_key = f"speech:{session_result.session_id}:{speech.sequence_number}"
                        await self.cache_manager.set(
                            key=speech_cache_key,
                            value=asdict(speech),
                            cache_type=CacheType.CONTENT,
                            ttl_hours=168  # 1 week
                        )
                    
                    cache_results['sessions_cached'] += 1
                    cache_results['speeches_cached'] += len(session_result.speeches)
                    
                    # Update session manager with completion
                    await self.session_manager.update_session_status(
                        session_result.session_id,
                        'COMPLETED',
                        {
                            'speeches_count': len(session_result.speeches),
                            'quality_score': session_result.parsing_metadata.get('validation', {}).get('quality_score', 0.0),
                            'completion_timestamp': datetime.now().isoformat()
                        }
                    )
                    
                except Exception as e:
                    logger.warning(f"Caching failed for session {session_result.session_id}: {e}")
                    
                    # Error recovery
                    try:
                        await self.error_recovery.handle_error(
                            e,
                            context={
                                'operation': 'intelligent_caching',
                                'session_id': session_result.session_id,
                                'execution_id': execution.execution_id
                            }
                        )
                    except Exception:
                        pass
                
                # Update progress
                progress = (i + 1) / len(validated_results)
                await self.progress_tracker.update_stage_progress(
                    progress_tracker_id,
                    "caching",
                    progress,
                    f"Cached {i + 1}/{len(validated_results)} sessions"
                )
            
            # Calculate cache efficiency
            if validated_results:
                cache_results['cache_efficiency'] = cache_results['sessions_cached'] / len(validated_results)
            
            # Mark caching stage complete
            await self.progress_tracker.complete_stage(
                progress_tracker_id, "caching",
                f"Cached {cache_results['sessions_cached']} sessions, {cache_results['speeches_cached']} speeches"
            )
            
            execution.sessions_cached = cache_results['sessions_cached']
            execution.stage = PipelineStage.COMPLETED
            
            return cache_results
            
        except Exception as e:
            logger.error(f"Intelligent caching failed: {e}")
            raise ProcessingError(f"Caching failed: {str(e)}") from e

    async def _background_session_discovery(self):
        """Background task for continuous session discovery"""
        logger.info("Starting background session discovery")
        
        while not self._shutdown_event.is_set():
            try:
                # Discover new sessions periodically
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  # Look back 1 week
                
                new_sessions = await self.session_discovery.discover_sessions_batch(
                    start_date=start_date,
                    end_date=end_date,
                    session_types=['plenary', 'committee'],
                    batch_size=20
                )
                
                # Register new sessions with session manager
                for session in new_sessions:
                    try:
                        await self.session_manager.register_session(
                            session_metadata=session,
                            priority='LOW',
                            dependencies=[],
                            context={'source': 'background_discovery'}
                        )
                    except Exception as e:
                        logger.warning(f"Failed to register session {session.session_id}: {e}")
                
                if new_sessions:
                    logger.info(f"Background discovery found {len(new_sessions)} new sessions")
                
                # Wait for next discovery interval
                await asyncio.sleep(self.pipeline_config['session_discovery_interval_hours'] * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background session discovery error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
        
        logger.info("Background session discovery stopped")

    async def _background_pipeline_monitoring(self):
        """Background task for pipeline monitoring and optimization"""
        logger.info("Starting background pipeline monitoring")
        
        while not self._shutdown_event.is_set():
            try:
                # Monitor current executions
                for execution_id, execution in list(self.current_executions.items()):
                    # Check for stuck executions
                    if execution.start_time and datetime.now() - execution.start_time > timedelta(hours=6):
                        logger.warning(f"Execution {execution_id} appears stuck, investigating...")
                        
                        # Attempt to recover stuck execution
                        try:
                            await self.error_recovery.handle_error(
                                RuntimeError(f"Execution timeout: {execution_id}"),
                                context={
                                    'operation': 'pipeline_monitoring',
                                    'execution_id': execution_id,
                                    'stage': execution.stage.value
                                }
                            )
                        except Exception as e:
                            logger.error(f"Failed to recover stuck execution {execution_id}: {e}")
                
                # Clean up completed executions
                completed_executions = [
                    exec_id for exec_id, execution in self.current_executions.items()
                    if execution.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
                    and execution.end_time
                    and datetime.now() - execution.end_time > timedelta(hours=1)
                ]
                
                for exec_id in completed_executions:
                    if exec_id in self.current_executions:
                        del self.current_executions[exec_id]
                
                if completed_executions:
                    logger.info(f"Cleaned up {len(completed_executions)} completed executions")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background pipeline monitoring error: {e}")
                await asyncio.sleep(1800)
        
        logger.info("Background pipeline monitoring stopped")

    async def _background_quality_analysis(self):
        """Background task for quality analysis and optimization"""
        logger.info("Starting background quality analysis")
        
        while not self._shutdown_event.is_set():
            try:
                # Analyze recent executions for quality trends
                recent_executions = [
                    exec for exec in self.execution_history[-100:]  # Last 100 executions
                    if exec.end_time and datetime.now() - exec.end_time < timedelta(days=7)
                ]
                
                if len(recent_executions) >= 10:
                    # Calculate quality metrics
                    quality_scores = []
                    performance_metrics = []
                    
                    for execution in recent_executions:
                        if hasattr(execution, 'quality_metrics') and execution.quality_metrics:
                            quality_scores.append(execution.quality_metrics.get('average_quality', 0.0))
                        
                        if execution.duration_minutes:
                            performance_metrics.append(execution.duration_minutes)
                    
                    if quality_scores:
                        avg_quality = sum(quality_scores) / len(quality_scores)
                        logger.info(f"Recent quality analysis: Average quality score = {avg_quality:.3f}")
                        
                        # Alert on quality degradation
                        if avg_quality < 0.5:
                            logger.warning(f"Quality degradation detected: {avg_quality:.3f}")
                    
                    if performance_metrics:
                        avg_duration = sum(performance_metrics) / len(performance_metrics)
                        logger.info(f"Recent performance analysis: Average duration = {avg_duration:.1f} minutes")
                
                await asyncio.sleep(3600)  # Analyze every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background quality analysis error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Background quality analysis stopped")

    def _calculate_quality_metrics(self, validated_results: List[SessionParsingResult]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for the pipeline execution"""
        if not validated_results:
            return {}
        
        quality_scores = []
        classification_confidences = []
        total_speeches = 0
        valid_speeches = 0
        
        for session_result in validated_results:
            # Session quality score
            if 'validation' in session_result.parsing_metadata:
                quality_scores.append(session_result.parsing_metadata['validation']['quality_score'])
            
            # Speech-level metrics
            for speech in session_result.speeches:
                total_speeches += 1
                
                if len(speech.speech_text.strip()) > 10:
                    valid_speeches += 1
                
                # Classification confidence
                if 'classification' in speech.parsing_metadata:
                    classification_confidences.append(
                        speech.parsing_metadata['classification']['confidence']
                    )
        
        metrics = {}
        
        if quality_scores:
            metrics['average_quality'] = sum(quality_scores) / len(quality_scores)
            metrics['min_quality'] = min(quality_scores)
            metrics['max_quality'] = max(quality_scores)
        
        if classification_confidences:
            metrics['average_classification_confidence'] = sum(classification_confidences) / len(classification_confidences)
        
        if total_speeches > 0:
            metrics['speech_validity_rate'] = valid_speeches / total_speeches
        
        metrics['total_sessions'] = len(validated_results)
        metrics['total_speeches'] = total_speeches
        metrics['valid_speeches'] = valid_speeches
        
        return metrics

    def _calculate_performance_metrics(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Calculate performance metrics for the pipeline execution"""
        metrics = {
            'execution_duration_minutes': execution.duration_minutes or 0.0,
            'sessions_per_minute': 0.0,
            'speeches_per_minute': 0.0
        }
        
        if execution.duration_minutes and execution.duration_minutes > 0:
            if execution.sessions_validated:
                metrics['sessions_per_minute'] = execution.sessions_validated / execution.duration_minutes
            
            if execution.speeches_parsed:
                metrics['speeches_per_minute'] = execution.speeches_parsed / execution.duration_minutes
        
        return metrics

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        current_count = len(self.current_executions)
        completed_count = len([e for e in self.execution_history if e.status == PipelineStatus.COMPLETED])
        failed_count = len([e for e in self.execution_history if e.status == PipelineStatus.FAILED])
        
        return {
            'current_executions': current_count,
            'completed_executions': completed_count,
            'failed_executions': failed_count,
            'total_executions': len(self.execution_history),
            'success_rate': completed_count / len(self.execution_history) if self.execution_history else 0.0,
            'current_execution_details': [
                {
                    'execution_id': execution.execution_id,
                    'status': execution.status.value,
                    'stage': execution.stage.value,
                    'start_time': execution.start_time.isoformat() if execution.start_time else None,
                    'duration_minutes': execution.duration_minutes
                }
                for execution in self.current_executions.values()
            ]
        }