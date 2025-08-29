"""
Multi-source data integration service for EU Parliament data collection.
Coordinates data from OpenData Portal, EUR-Lex, and Verbatim reports.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..clients.opendata_client import OpenDataClient
from ..clients.eurlex_client import EURLexClient
from ..clients.verbatim_client import VerbatimClient
from ..models.session import SessionMetadata, SessionConfig
from ..core.exceptions import APIError, ProcessingError, DataValidationError
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..utils.validation import validate_session_completeness

logger = get_logger(__name__)


class DataSource(Enum):
    """Available data sources for integration."""
    OPENDATA = "opendata"
    EURLEX = "eurlex"
    VERBATIM = "verbatim"
    CACHED = "cached"


class IntegrationStatus(Enum):
    """Integration status for session data."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DataSourceResult:
    """Result from a specific data source."""
    source: DataSource
    status: IntegrationStatus
    data: Optional[Any]
    error: Optional[str]
    fetch_time: datetime
    processing_time: float


@dataclass
class IntegrationResult:
    """Complete integration result for a session."""
    session_id: str
    status: IntegrationStatus
    sources: Dict[DataSource, DataSourceResult]
    merged_data: Optional[SessionMetadata]
    quality_score: float
    completeness_score: float
    total_processing_time: float
    created_at: datetime


class DataIntegrationService:
    """Service for integrating data from multiple EU Parliament sources."""
    
    def __init__(self, 
                 opendata_client: OpenDataClient,
                 eurlex_client: Optional[EURLexClient] = None,
                 verbatim_client: Optional[VerbatimClient] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize data integration service.
        
        Args:
            opendata_client: OpenData Portal client
            eurlex_client: Optional EUR-Lex client
            verbatim_client: Optional Verbatim reports client
            metrics_collector: Optional metrics collection
        """
        self.opendata_client = opendata_client
        self.eurlex_client = eurlex_client
        self.verbatim_client = verbatim_client
        self.metrics = metrics_collector or MetricsCollector()
        
        # Configuration
        self.max_concurrent_sources = 3
        self.source_timeout = timedelta(minutes=5)
        self.quality_threshold = 0.7
        self.completeness_threshold = 0.8
        
        # Caching
        self.integration_cache = Path("data/cache/integration_results.json")
        self.integration_cache.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data integration service initialized")
    
    async def integrate_session_data(self, session_id: str, 
                                   sources: List[DataSource] = None,
                                   force_refresh: bool = False) -> IntegrationResult:
        """
        Integrate data for a single session from multiple sources.
        
        Args:
            session_id: Session identifier
            sources: List of sources to use (default: all available)
            force_refresh: Force refresh of cached data
            
        Returns:
            Integration result with merged data and quality metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting data integration for session {session_id}")
        
        try:
            # Check cache first
            if not force_refresh:
                cached_result = await self._get_cached_result(session_id)
                if cached_result:
                    logger.info(f"Retrieved session {session_id} from integration cache")
                    return cached_result
            
            # Determine sources to use
            available_sources = self._get_available_sources()
            target_sources = sources or available_sources
            
            # Fetch data from all sources concurrently
            source_results = await self._fetch_from_sources(session_id, target_sources)
            
            # Merge data from successful sources
            merged_data = await self._merge_source_data(source_results)
            
            # Calculate quality and completeness scores
            quality_score = self._calculate_quality_score(source_results, merged_data)
            completeness_score = self._calculate_completeness_score(merged_data)
            
            # Determine overall status
            status = self._determine_integration_status(source_results, quality_score, completeness_score)
            
            # Create integration result
            processing_time = (datetime.now() - start_time).total_seconds()
            result = IntegrationResult(
                session_id=session_id,
                status=status,
                sources=source_results,
                merged_data=merged_data,
                quality_score=quality_score,
                completeness_score=completeness_score,
                total_processing_time=processing_time,
                created_at=start_time
            )
            
            # Cache the result
            await self._cache_result(result)
            
            # Record metrics
            self.metrics.record_request(
                service="data_integration",
                success=(status == IntegrationStatus.COMPLETED),
                response_time=processing_time
            )
            
            logger.info(f"Integration completed for session {session_id}: "
                       f"status={status.value}, quality={quality_score:.2f}, "
                       f"completeness={completeness_score:.2f}")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Integration failed for session {session_id}: {e}")
            
            # Record failed metrics
            self.metrics.record_request(
                service="data_integration",
                success=False,
                response_time=processing_time,
                error_type="integration_error"
            )
            
            raise ProcessingError(f"Data integration failed for session {session_id}: {e}")
    
    async def integrate_sessions_batch(self, session_ids: List[str],
                                     sources: List[DataSource] = None,
                                     progress_callback: Callable = None) -> Dict[str, IntegrationResult]:
        """
        Integrate data for multiple sessions in batch.
        
        Args:
            session_ids: List of session identifiers
            sources: List of sources to use
            progress_callback: Progress reporting callback
            
        Returns:
            Dictionary mapping session IDs to integration results
        """
        logger.info(f"Starting batch integration for {len(session_ids)} sessions")
        
        results = {}
        batch_size = 10  # Process in smaller batches to avoid overwhelming sources
        
        for i in range(0, len(session_ids), batch_size):
            batch = session_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: sessions {i+1}-{min(i+batch_size, len(session_ids))}")
            
            # Process batch concurrently
            batch_tasks = [
                self.integrate_session_data(session_id, sources)
                for session_id in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for session_id, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch integration failed for session {session_id}: {result}")
                        # Create failed result
                        results[session_id] = IntegrationResult(
                            session_id=session_id,
                            status=IntegrationStatus.FAILED,
                            sources={},
                            merged_data=None,
                            quality_score=0.0,
                            completeness_score=0.0,
                            total_processing_time=0.0,
                            created_at=datetime.now()
                        )
                    else:
                        results[session_id] = result
                
                if progress_callback:
                    progress_callback(i + len(batch), len(session_ids), len(batch))
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Continue with remaining batches
                continue
        
        successful = sum(1 for r in results.values() if r.status == IntegrationStatus.COMPLETED)
        logger.info(f"Batch integration completed: {successful}/{len(session_ids)} successful")
        
        return results
    
    async def _fetch_from_sources(self, session_id: str, 
                                sources: List[DataSource]) -> Dict[DataSource, DataSourceResult]:
        """Fetch data from multiple sources concurrently."""
        tasks = {}
        
        # Create tasks for each source
        if DataSource.OPENDATA in sources and self.opendata_client:
            tasks[DataSource.OPENDATA] = self._fetch_opendata(session_id)
        
        if DataSource.EURLEX in sources and self.eurlex_client:
            tasks[DataSource.EURLEX] = self._fetch_eurlex(session_id)
        
        if DataSource.VERBATIM in sources and self.verbatim_client:
            tasks[DataSource.VERBATIM] = self._fetch_verbatim(session_id)
        
        # Execute tasks with timeout
        results = {}
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for source, result in zip(tasks.keys(), completed_tasks):
            if isinstance(result, Exception):
                logger.warning(f"Source {source.value} failed for session {session_id}: {result}")
                results[source] = DataSourceResult(
                    source=source,
                    status=IntegrationStatus.FAILED,
                    data=None,
                    error=str(result),
                    fetch_time=datetime.now(),
                    processing_time=0.0
                )
            else:
                results[source] = result
        
        return results
    
    async def _fetch_opendata(self, session_id: str) -> DataSourceResult:
        """Fetch data from OpenData Portal."""
        start_time = datetime.now()
        
        try:
            # Get session metadata from OpenData Portal
            session_data = self.opendata_client.get_session_details(session_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DataSourceResult(
                source=DataSource.OPENDATA,
                status=IntegrationStatus.COMPLETED,
                data=session_data,
                error=None,
                fetch_time=start_time,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"OpenData fetch failed for session {session_id}: {e}")
            
            return DataSourceResult(
                source=DataSource.OPENDATA,
                status=IntegrationStatus.FAILED,
                data=None,
                error=str(e),
                fetch_time=start_time,
                processing_time=processing_time
            )
    
    async def _fetch_eurlex(self, session_id: str) -> DataSourceResult:
        """Fetch data from EUR-Lex."""
        start_time = datetime.now()
        
        try:
            # Get session metadata from EUR-Lex
            eurlex_data = self.eurlex_client.get_session_metadata(session_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DataSourceResult(
                source=DataSource.EURLEX,
                status=IntegrationStatus.COMPLETED,
                data=eurlex_data,
                error=None,
                fetch_time=start_time,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"EUR-Lex fetch failed for session {session_id}: {e}")
            
            return DataSourceResult(
                source=DataSource.EURLEX,
                status=IntegrationStatus.FAILED,
                data=None,
                error=str(e),
                fetch_time=start_time,
                processing_time=processing_time
            )
    
    async def _fetch_verbatim(self, session_id: str) -> DataSourceResult:
        """Fetch data from Verbatim reports."""
        start_time = datetime.now()
        
        try:
            # Get verbatim report URLs and metadata
            verbatim_data = self.verbatim_client.get_session_documents(session_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DataSourceResult(
                source=DataSource.VERBATIM,
                status=IntegrationStatus.COMPLETED,
                data=verbatim_data,
                error=None,
                fetch_time=start_time,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"Verbatim fetch failed for session {session_id}: {e}")
            
            return DataSourceResult(
                source=DataSource.VERBATIM,
                status=IntegrationStatus.FAILED,
                data=None,
                error=str(e),
                fetch_time=start_time,
                processing_time=processing_time
            )
    
    async def _merge_source_data(self, source_results: Dict[DataSource, DataSourceResult]) -> Optional[SessionMetadata]:
        """Merge data from multiple sources into unified session metadata."""
        try:
            merged_data = {}
            verbatim_urls = []
            agenda_urls = []
            additional_metadata = {}
            
            # Process each successful source
            for source, result in source_results.items():
                if result.status != IntegrationStatus.COMPLETED or not result.data:
                    continue
                
                data = result.data
                
                if source == DataSource.OPENDATA:
                    # OpenData is primary source for basic metadata
                    merged_data.update({
                        'session_id': data.get('session_id'),
                        'session_date': data.get('session_date'),
                        'session_type': data.get('session_type', 'plenary'),
                        'location': data.get('location', 'Unknown'),
                        'title': data.get('title')
                    })
                    
                    if 'agenda_urls' in data:
                        agenda_urls.extend(data['agenda_urls'])
                    if 'verbatim_urls' in data:
                        verbatim_urls.extend(data['verbatim_urls'])
                
                elif source == DataSource.EURLEX:
                    # EUR-Lex provides additional official metadata
                    additional_metadata['eurlex'] = data
                    
                    if 'document_urls' in data:
                        verbatim_urls.extend(data['document_urls'])
                    if 'official_journal' in data:
                        additional_metadata['official_journal'] = data['official_journal']
                
                elif source == DataSource.VERBATIM:
                    # Verbatim provides document URLs and content metadata
                    additional_metadata['verbatim'] = data
                    
                    if 'document_urls' in data:
                        verbatim_urls.extend(data['document_urls'])
                    if 'content_summary' in data:
                        additional_metadata['content_summary'] = data['content_summary']
            
            # Check if we have minimum required data
            if not merged_data.get('session_id'):
                logger.warning("No session ID found in merged data")
                return None
            
            # Create SessionMetadata object
            session = SessionMetadata(
                session_id=merged_data['session_id'],
                session_date=merged_data.get('session_date'),
                session_type=merged_data.get('session_type', 'plenary'),
                location=merged_data.get('location', 'Unknown'),
                title=merged_data.get('title', f"Session {merged_data['session_id']}"),
                agenda_urls=list(set(agenda_urls)),  # Remove duplicates
                verbatim_urls=list(set(verbatim_urls)),  # Remove duplicates
                additional_metadata=additional_metadata
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Data merging failed: {e}")
            return None
    
    def _calculate_quality_score(self, source_results: Dict[DataSource, DataSourceResult],
                               merged_data: Optional[SessionMetadata]) -> float:
        """Calculate quality score based on source success and data completeness."""
        if not merged_data:
            return 0.0
        
        # Base score from successful sources
        successful_sources = sum(1 for result in source_results.values() 
                               if result.status == IntegrationStatus.COMPLETED)
        total_sources = len(source_results)
        source_score = successful_sources / total_sources if total_sources > 0 else 0.0
        
        # Data completeness score
        completeness_score = validate_session_completeness(merged_data)
        
        # Combined quality score (weighted average)
        quality_score = (source_score * 0.4 + completeness_score * 0.6)
        
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_completeness_score(self, merged_data: Optional[SessionMetadata]) -> float:
        """Calculate completeness score for merged session data."""
        if not merged_data:
            return 0.0
        
        return validate_session_completeness(merged_data)
    
    def _determine_integration_status(self, source_results: Dict[DataSource, DataSourceResult],
                                    quality_score: float, completeness_score: float) -> IntegrationStatus:
        """Determine overall integration status."""
        successful_sources = sum(1 for result in source_results.values() 
                               if result.status == IntegrationStatus.COMPLETED)
        
        if successful_sources == 0:
            return IntegrationStatus.FAILED
        
        if quality_score >= self.quality_threshold and completeness_score >= self.completeness_threshold:
            return IntegrationStatus.COMPLETED
        
        if successful_sources == len(source_results):
            return IntegrationStatus.PARTIAL
        
        return IntegrationStatus.PARTIAL
    
    def _get_available_sources(self) -> List[DataSource]:
        """Get list of available data sources."""
        sources = []
        
        if self.opendata_client:
            sources.append(DataSource.OPENDATA)
        
        if self.eurlex_client:
            sources.append(DataSource.EURLEX)
        
        if self.verbatim_client:
            sources.append(DataSource.VERBATIM)
        
        return sources
    
    async def _get_cached_result(self, session_id: str) -> Optional[IntegrationResult]:
        """Get integration result from cache."""
        try:
            if not self.integration_cache.exists():
                return None
            
            with open(self.integration_cache, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if session_id not in cache_data:
                return None
            
            cached_entry = cache_data[session_id]
            
            # Check cache age (4 hours for integration results)
            cache_time = datetime.fromisoformat(cached_entry['created_at'])
            if datetime.now() - cache_time > timedelta(hours=4):
                return None
            
            # Deserialize result
            # This is simplified - in production would need proper deserialization
            return IntegrationResult(**cached_entry)
            
        except Exception as e:
            logger.debug(f"Failed to load cached integration result: {e}")
            return None
    
    async def _cache_result(self, result: IntegrationResult):
        """Cache integration result."""
        try:
            # Load existing cache
            cache_data = {}
            if self.integration_cache.exists():
                with open(self.integration_cache, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Serialize result (simplified)
            cache_data[result.session_id] = {
                'session_id': result.session_id,
                'status': result.status.value,
                'quality_score': result.quality_score,
                'completeness_score': result.completeness_score,
                'total_processing_time': result.total_processing_time,
                'created_at': result.created_at.isoformat()
            }
            
            # Write cache
            with open(self.integration_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Failed to cache integration result: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration service statistics."""
        available_sources = self._get_available_sources()
        
        stats = {
            'available_sources': [source.value for source in available_sources],
            'total_sources': len(available_sources),
            'cache_file_exists': self.integration_cache.exists(),
            'quality_threshold': self.quality_threshold,
            'completeness_threshold': self.completeness_threshold,
            'max_concurrent_sources': self.max_concurrent_sources
        }
        
        # Add cache statistics
        if self.integration_cache.exists():
            try:
                with open(self.integration_cache, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                stats['cached_sessions'] = len(cache_data)
            except:
                stats['cached_sessions'] = 0
        else:
            stats['cached_sessions'] = 0
        
        return stats