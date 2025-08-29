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
    """Comprehensive data collection pipeline for EU Parliament data."""
    
    def __init__(self,
                 session_discovery: SessionDiscoveryService,
                 data_integration: DataIntegrationService,
                 document_resolver: DocumentResolver,
                 cache_manager: IntelligentCacheManager,
                 verbatim_parser: Optional[VerbatimParser] = None,
                 speech_classifier: Optional[SpeechClassifier] = None,
                 parsing_validator: Optional[ParsingValidator] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize data collection pipeline.
        
        Args:
            session_discovery: Session discovery service
            data_integration: Multi-source data integration service
            document_resolver: Document URL resolution service
            cache_manager: Intelligent caching system
            verbatim_parser: Optional verbatim content parser
            speech_classifier: Optional speech classification
            parsing_validator: Optional validation framework
            metrics_collector: Optional metrics collection
        """
        self.session_discovery = session_discovery
        self.data_integration = data_integration
        self.document_resolver = document_resolver
        self.cache_manager = cache_manager
        self.verbatim_parser = verbatim_parser
        self.speech_classifier = speech_classifier
        self.parsing_validator = parsing_validator
        self.metrics = metrics_collector or MetricsCollector()
        
        # Pipeline state
        self.active_sessions: Dict[str, PipelineResult] = {}
        self.completed_sessions: Dict[str, PipelineResult] = {}
        
        # Results storage
        self.results_dir = Path("data/pipeline_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data collection pipeline initialized")
    
    async def process_session(self, session_id: str, 
                            config: Optional[PipelineConfig] = None) -> PipelineResult:
        """
        Process a single session through the complete pipeline.
        
        Args:
            session_id: Session identifier
            config: Pipeline configuration
            
        Returns:
            Complete pipeline result
        """
        config = config or PipelineConfig()
        start_time = datetime.now()
        
        logger.info(f"Starting pipeline processing for session {session_id}")
        
        # Initialize pipeline result
        result = PipelineResult(
            session_id=session_id,
            status=PipelineStatus.RUNNING,
            stages={},
            final_data=None,
            total_processing_time=0.0,
            quality_score=0.0,
            completeness_score=0.0,
            document_count=0,
            speech_segment_count=0,
            created_at=start_time
        )
        
        self.active_sessions[session_id] = result
        
        try:
            # Check cache for complete pipeline result
            if config.enable_caching and not config.force_refresh:
                cached_result = await self._get_cached_pipeline_result(session_id)
                if cached_result:
                    logger.info(f"Retrieved complete pipeline result from cache for session {session_id}")
                    self.completed_sessions[session_id] = cached_result
                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]
                    return cached_result
            
            # Stage 1: Session Discovery & Integration
            integration_result = await self._execute_discovery_integration(session_id, config)
            result.stages[PipelineStage.DISCOVERY] = self._create_stage_result(
                PipelineStage.DISCOVERY, integration_result
            )
            result.stages[PipelineStage.INTEGRATION] = self._create_stage_result(
                PipelineStage.INTEGRATION, integration_result
            )
            
            if not integration_result or not integration_result.merged_data:
                raise ProcessingError("Session discovery and integration failed")
            
            session_metadata = integration_result.merged_data
            
            # Stage 2: Document Resolution
            document_collection = None
            if config.enable_document_resolution:
                document_collection = await self._execute_document_resolution(session_metadata, config)
                result.stages[PipelineStage.DOCUMENT_RESOLUTION] = self._create_stage_result(
                    PipelineStage.DOCUMENT_RESOLUTION, document_collection
                )
                
                if document_collection:
                    result.document_count = document_collection.total_documents
            
            # Stage 3: Content Parsing
            parsed_segments = []
            if config.enable_content_parsing and self.verbatim_parser and document_collection:
                parsed_segments = await self._execute_content_parsing(
                    session_metadata, document_collection, config
                )
                result.stages[PipelineStage.CONTENT_PARSING] = self._create_stage_result(
                    PipelineStage.CONTENT_PARSING, parsed_segments
                )
                
                result.speech_segment_count = len(parsed_segments)
            
            # Stage 4: Speech Classification
            classified_segments = parsed_segments
            if config.enable_classification and self.speech_classifier and parsed_segments:
                classified_segments = await self._execute_classification(parsed_segments, config)
                result.stages[PipelineStage.CLASSIFICATION] = self._create_stage_result(
                    PipelineStage.CLASSIFICATION, classified_segments
                )
            
            # Stage 5: Validation
            validation_result = None
            if config.enable_validation and self.parsing_validator and classified_segments:
                validation_result = await self._execute_validation(classified_segments, config)
                result.stages[PipelineStage.VALIDATION] = self._create_stage_result(
                    PipelineStage.VALIDATION, validation_result
                )
                
                if validation_result:
                    result.quality_score = validation_result.quality_score
            
            # Calculate completeness score
            result.completeness_score = self._calculate_completeness_score(result)
            
            # Compile final data
            result.final_data = {
                'session_metadata': session_metadata.dict() if session_metadata else None,
                'document_collection': asdict(document_collection) if document_collection else None,
                'parsed_segments': [seg.dict() for seg in classified_segments] if classified_segments else [],
                'validation_result': validation_result.dict() if validation_result else None,
                'integration_result': {
                    'quality_score': integration_result.quality_score,
                    'completeness_score': integration_result.completeness_score,
                    'sources': list(integration_result.sources.keys())
                } if integration_result else None
            }
            
            # Determine final status
            if result.quality_score >= config.quality_threshold:
                result.status = PipelineStatus.COMPLETED
            else:
                result.status = PipelineStatus.PARTIAL
            
            # Cache the complete result
            if config.enable_caching:
                await self._cache_pipeline_result(result)
            
            # Record metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            result.total_processing_time = processing_time
            
            self.metrics.record_request(
                service="pipeline",
                success=(result.status == PipelineStatus.COMPLETED),
                response_time=processing_time
            )
            
            logger.info(f"Pipeline processing completed for session {session_id}: "
                       f"status={result.status.value}, quality={result.quality_score:.2f}, "
                       f"documents={result.document_count}, segments={result.speech_segment_count}")
            
            # Move to completed
            self.completed_sessions[session_id] = result
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return result
            
        except Exception as e:
            # Record failure
            processing_time = (datetime.now() - start_time).total_seconds()
            result.status = PipelineStatus.FAILED
            result.total_processing_time = processing_time
            
            # Add failure stage
            result.stages[PipelineStage.FAILED] = StageResult(
                stage=PipelineStage.FAILED,
                status=PipelineStatus.FAILED,
                data=None,
                error=str(e),
                processing_time=0.0,
                timestamp=datetime.now()
            )
            
            logger.error(f"Pipeline processing failed for session {session_id}: {e}")
            
            # Record failed metrics
            self.metrics.record_request(
                service="pipeline",
                success=False,
                response_time=processing_time,
                error_type="pipeline_error"
            )
            
            # Move to completed (as failed)
            self.completed_sessions[session_id] = result
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return result
    
    async def process_sessions_batch(self, session_ids: List[str],
                                   config: Optional[PipelineConfig] = None,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, PipelineResult]:
        """
        Process multiple sessions through the pipeline in batches.
        
        Args:
            session_ids: List of session identifiers
            config: Pipeline configuration
            progress_callback: Progress reporting callback
            
        Returns:
            Dictionary mapping session IDs to pipeline results
        """
        config = config or PipelineConfig()
        logger.info(f"Starting batch pipeline processing for {len(session_ids)} sessions")
        
        results = {}
        batch_size = config.max_concurrent_sessions
        
        for i in range(0, len(session_ids), batch_size):
            batch = session_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: sessions {i+1}-{min(i+batch_size, len(session_ids))}")
            
            # Process batch concurrently
            batch_tasks = [
                self.process_session(session_id, config)
                for session_id in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for session_id, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing failed for session {session_id}: {result}")
                        # Create failed result
                        results[session_id] = PipelineResult(
                            session_id=session_id,
                            status=PipelineStatus.FAILED,
                            stages={},
                            final_data=None,
                            total_processing_time=0.0,
                            quality_score=0.0,
                            completeness_score=0.0,
                            document_count=0,
                            speech_segment_count=0,
                            created_at=datetime.now()
                        )
                    else:
                        results[session_id] = result
                
                if progress_callback:
                    progress_callback(i + len(batch), len(session_ids), len(batch))
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                continue
        
        successful = sum(1 for r in results.values() if r.status == PipelineStatus.COMPLETED)
        logger.info(f"Batch pipeline processing completed: {successful}/{len(session_ids)} successful")
        
        return results
    
    async def discover_and_process_sessions(self, start_date: str, end_date: str,
                                          config: Optional[PipelineConfig] = None,
                                          progress_callback: Optional[Callable] = None) -> Dict[str, PipelineResult]:
        """
        Discover sessions in date range and process them through the pipeline.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            config: Pipeline configuration
            progress_callback: Progress reporting callback
            
        Returns:
            Dictionary mapping session IDs to pipeline results
        """
        config = config or PipelineConfig()
        logger.info(f"Discovering and processing sessions from {start_date} to {end_date}")
        
        try:
            # Discover sessions
            sessions = await self.session_discovery.discover_sessions_batch(
                start_date, end_date, progress_callback=lambda current, total, batch: None
            )
            
            if not sessions:
                logger.warning(f"No sessions discovered for date range {start_date} to {end_date}")
                return {}
            
            session_ids = [session.session_id for session in sessions]
            logger.info(f"Discovered {len(session_ids)} sessions, starting pipeline processing")
            
            # Process discovered sessions
            return await self.process_sessions_batch(session_ids, config, progress_callback)
            
        except Exception as e:
            logger.error(f"Discover and process failed for date range {start_date}-{end_date}: {e}")
            raise ProcessingError(f"Failed to discover and process sessions: {e}")
    
    async def _execute_discovery_integration(self, session_id: str, 
                                           config: PipelineConfig) -> Optional[IntegrationResult]:
        """Execute session discovery and data integration stage."""
        try:
            stage_timeout = timedelta(minutes=config.stage_timeout_minutes)
            
            # Use data integration service to get comprehensive session data
            result = await asyncio.wait_for(
                self.data_integration.integrate_session_data(
                    session_id, force_refresh=config.force_refresh
                ),
                timeout=stage_timeout.total_seconds()
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Discovery/integration stage timeout for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"Discovery/integration stage failed for session {session_id}: {e}")
            return None
    
    async def _execute_document_resolution(self, session_metadata: SessionMetadata,
                                         config: PipelineConfig) -> Optional[DocumentCollection]:
        """Execute document resolution stage."""
        try:
            stage_timeout = timedelta(minutes=config.stage_timeout_minutes)
            
            # Get all known URLs from session metadata
            known_urls = []
            known_urls.extend(session_metadata.verbatim_urls)
            known_urls.extend(session_metadata.agenda_urls)
            
            result = await asyncio.wait_for(
                self.document_resolver.resolve_session_documents(
                    session_metadata.session_id,
                    known_urls,
                    force_refresh=config.force_refresh
                ),
                timeout=stage_timeout.total_seconds()
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Document resolution stage timeout for session {session_metadata.session_id}")
            return None
        except Exception as e:
            logger.error(f"Document resolution stage failed for session {session_metadata.session_id}: {e}")
            return None
    
    async def _execute_content_parsing(self, session_metadata: SessionMetadata,
                                     document_collection: DocumentCollection,
                                     config: PipelineConfig) -> List[RawSpeechSegment]:
        """Execute content parsing stage."""
        try:
            if not self.verbatim_parser:
                return []
            
            parsed_segments = []
            
            # Parse verbatim documents
            for doc_link in document_collection.verbatim_documents:
                try:
                    # In a real implementation, would fetch and parse document content
                    # For now, create placeholder segments
                    # This would be: content = fetch_document_content(doc_link.url)
                    # segments = self.verbatim_parser.parse_verbatim_report(content, ...)
                    pass
                except Exception as e:
                    logger.warning(f"Failed to parse document {doc_link.url}: {e}")
                    continue
            
            return parsed_segments
            
        except Exception as e:
            logger.error(f"Content parsing stage failed for session {session_metadata.session_id}: {e}")
            return []
    
    async def _execute_classification(self, segments: List[RawSpeechSegment],
                                    config: PipelineConfig) -> List[RawSpeechSegment]:
        """Execute speech classification stage."""
        try:
            if not self.speech_classifier:
                return segments
            
            classified_segments = []
            
            for segment in segments:
                try:
                    classification = self.speech_classifier.classify_segment(segment)
                    
                    # Add classification to segment metadata
                    segment.parsing_metadata = segment.parsing_metadata or {}
                    segment.parsing_metadata['classification'] = {
                        'content_type': classification.content_type.value,
                        'confidence': classification.confidence,
                        'announcement_type': classification.announcement_type.value if classification.announcement_type else None
                    }
                    
                    classified_segments.append(segment)
                    
                except Exception as e:
                    logger.warning(f"Classification failed for segment {segment.segment_id}: {e}")
                    classified_segments.append(segment)  # Keep unclassified
            
            return classified_segments
            
        except Exception as e:
            logger.error(f"Classification stage failed: {e}")
            return segments
    
    async def _execute_validation(self, segments: List[RawSpeechSegment],
                                config: PipelineConfig):
        """Execute validation stage."""
        try:
            if not self.parsing_validator:
                return None
            
            validation_result = self.parsing_validator.validate_batch(segments)
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation stage failed: {e}")
            return None
    
    def _create_stage_result(self, stage: PipelineStage, data: Any) -> StageResult:
        """Create a stage result from stage output."""
        return StageResult(
            stage=stage,
            status=PipelineStatus.COMPLETED if data is not None else PipelineStatus.FAILED,
            data=data,
            error=None if data is not None else "Stage produced no output",
            processing_time=0.0,  # Would be measured in real implementation
            timestamp=datetime.now()
        )
    
    def _calculate_completeness_score(self, result: PipelineResult) -> float:
        """Calculate overall pipeline completeness score."""
        total_stages = len(PipelineStage) - 2  # Exclude COMPLETED and FAILED
        completed_stages = sum(
            1 for stage_result in result.stages.values()
            if stage_result.status == PipelineStatus.COMPLETED
        )
        
        return completed_stages / total_stages if total_stages > 0 else 0.0
    
    async def _get_cached_pipeline_result(self, session_id: str) -> Optional[PipelineResult]:
        """Get complete pipeline result from cache."""
        try:
            cached_data = await self.cache_manager.get(f"pipeline:{session_id}", CacheType.INTEGRATION_RESULTS)
            
            if cached_data:
                # Deserialize pipeline result
                # In production would need proper deserialization
                return PipelineResult(**cached_data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get cached pipeline result for session {session_id}: {e}")
            return None
    
    async def _cache_pipeline_result(self, result: PipelineResult):
        """Cache complete pipeline result."""
        try:
            # Serialize result for caching
            cache_data = asdict(result)
            
            # Cache with dependencies
            dependencies = [
                f"session:{result.session_id}",
                f"documents:{result.session_id}"
            ]
            
            await self.cache_manager.set(
                f"pipeline:{result.session_id}",
                cache_data,
                CacheType.INTEGRATION_RESULTS,
                dependencies=dependencies,
                metadata={'session_id': result.session_id, 'stage_count': len(result.stages)}
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache pipeline result for session {result.session_id}: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'total_processed': len(self.completed_sessions),
            'success_rate': self._calculate_success_rate(),
            'average_processing_time': self._calculate_average_processing_time(),
            'average_quality_score': self._calculate_average_quality_score(),
            'total_documents_processed': sum(r.document_count for r in self.completed_sessions.values()),
            'total_speech_segments': sum(r.speech_segment_count for r in self.completed_sessions.values())
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate pipeline success rate."""
        if not self.completed_sessions:
            return 0.0
        
        successful = sum(
            1 for result in self.completed_sessions.values()
            if result.status == PipelineStatus.COMPLETED
        )
        
        return successful / len(self.completed_sessions)
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.completed_sessions:
            return 0.0
        
        total_time = sum(result.total_processing_time for result in self.completed_sessions.values())
        return total_time / len(self.completed_sessions)
    
    def _calculate_average_quality_score(self) -> float:
        """Calculate average quality score."""
        if not self.completed_sessions:
            return 0.0
        
        total_score = sum(result.quality_score for result in self.completed_sessions.values())
        return total_score / len(self.completed_sessions)