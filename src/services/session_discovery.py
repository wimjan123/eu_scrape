"""Session discovery service for EU Parliament scraper."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..clients.opendata_client import OpenDataClient
from ..clients.eurlex_client import EURLexClient
from ..models.session import SessionMetadata, SessionConfig
from ..core.exceptions import APIError, ProcessingError
from ..core.logging import get_logger
from ..utils.time_utils import parse_session_date

logger = get_logger(__name__)


class SessionDiscoveryService:
    """Enhanced service for discovering and cataloging plenary sessions with batch processing."""
    
    def __init__(self, opendata_client: OpenDataClient, eurlex_client: EURLexClient = None):
        """
        Initialize session discovery service.
        
        Args:
            opendata_client: OpenData API client
            eurlex_client: Optional EUR-Lex client for additional metadata
        """
        self.opendata_client = opendata_client
        self.eurlex_client = eurlex_client
        self.sessions_cache: Dict[str, SessionMetadata] = {}
        self.discovery_cache_file = Path("data/cache/session_discovery.json")
        self.batch_cache_file = Path("data/cache/batch_discovery.json")
        
        # Ensure cache directory exists
        self.discovery_cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Batch processing configuration
        self.batch_size = 50  # Sessions per batch
        self.max_concurrent_requests = 5
        self.cache_ttl = timedelta(hours=24)
        
        logger.info("Enhanced session discovery service initialized with batch processing")
    
    async def discover_sessions_batch(self, start_date: str, end_date: str, 
                                    batch_size: int = None,
                                    progress_callback: callable = None) -> List[SessionMetadata]:
        """
        Discover plenary sessions in date range with batch processing.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            batch_size: Override default batch size
            progress_callback: Optional progress reporting callback
            
        Returns:
            List of discovered session metadata
        """
        batch_size = batch_size or self.batch_size
        logger.info(f"Starting batch session discovery: {start_date} to {end_date}, batch_size={batch_size}")
        
        try:
            # Check cache first
            cache_key = f"{start_date}_{end_date}"
            if cached_result := await self._get_batch_cache(cache_key):
                logger.info(f"Retrieved {len(cached_result)} sessions from batch cache")
                return cached_result
            
            # Discover sessions in batches
            all_sessions = []
            date_ranges = self._split_date_range(start_date, end_date, batch_size)
            
            for i, (batch_start, batch_end) in enumerate(date_ranges):
                logger.info(f"Processing batch {i+1}/{len(date_ranges)}: {batch_start} to {batch_end}")
                
                try:
                    batch_sessions = await self._discover_batch(batch_start, batch_end)
                    all_sessions.extend(batch_sessions)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(date_ranges), len(batch_sessions))
                    
                    logger.info(f"Batch {i+1} completed: {len(batch_sessions)} sessions discovered")
                    
                except Exception as e:
                    logger.error(f"Batch {i+1} failed: {e}")
                    # Continue with other batches
                    continue
            
            # Enrich sessions with additional metadata
            enriched_sessions = await self._enrich_sessions_batch(all_sessions)
            
            # Cache the results
            await self._set_batch_cache(cache_key, enriched_sessions)
            
            logger.info(f"Batch discovery completed: {len(enriched_sessions)} total sessions")
            return enriched_sessions
            
        except Exception as e:
            logger.error(f"Batch session discovery failed: {e}")
            raise ProcessingError(f"Failed to discover sessions in batch: {e}")
    
    def discover_sessions(self, start_date: str, end_date: str, 
                         force_refresh: bool = False) -> List[SessionMetadata]:
        """
        Discover plenary sessions in date range (legacy sync method).
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            force_refresh: Force refresh of cached data
            
        Returns:
            List of discovered session metadata
        """
        logger.info(f"Discovering sessions from {start_date} to {end_date}")
        
        try:
            # Check cache first
            cache_key = f"{start_date}_{end_date}"
            if not force_refresh:
                cached_sessions = self._load_from_cache(cache_key)
                if cached_sessions:
                    logger.info(f"Found {len(cached_sessions)} sessions in cache")
                    return cached_sessions
            
            # Fetch session data from OpenData Portal
            opendata_sessions = self._fetch_opendata_sessions(start_date, end_date)
            logger.info(f"Retrieved {len(opendata_sessions)} sessions from OpenData Portal")
            
            # Enrich with EUR-Lex data if client available
            if self.eurlex_client:
                enriched_sessions = self._enrich_with_eurlex(opendata_sessions)
                logger.info(f"Enhanced {len(enriched_sessions)} sessions with EUR-Lex data")
            else:
                enriched_sessions = opendata_sessions
            
            # Validate and clean session data
            validated_sessions = self._validate_sessions(enriched_sessions)
            logger.info(f"Validated {len(validated_sessions)} sessions")
            
            # Cache the results
            self._save_to_cache(cache_key, validated_sessions)
            
            return validated_sessions
            
        except APIError as e:
            logger.error(f"API error during session discovery: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during session discovery: {e}")
            raise ProcessingError(f"Failed to discover sessions: {e}")
    
    async def _discover_batch(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Discover sessions for a specific date batch."""
        try:
            # Fetch from OpenData Portal with error handling
            opendata_sessions = await self._fetch_opendata_sessions_async(start_date, end_date)
            
            # Add basic validation
            validated_sessions = []
            for session in opendata_sessions:
                try:
                    if self._validate_session_basic(session):
                        validated_sessions.append(session)
                except Exception as e:
                    logger.warning(f"Session validation failed, skipping: {e}")
                    continue
            
            return validated_sessions
            
        except Exception as e:
            logger.error(f"Batch discovery failed for {start_date}-{end_date}: {e}")
            return []
    
    async def _enrich_sessions_batch(self, sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Enrich sessions with additional metadata in batches."""
        if not self.eurlex_client:
            return sessions
        
        logger.info(f"Enriching {len(sessions)} sessions with EUR-Lex metadata")
        enriched = []
        
        # Process in smaller batches to avoid overwhelming EUR-Lex
        eurlex_batch_size = 10
        for i in range(0, len(sessions), eurlex_batch_size):
            batch = sessions[i:i + eurlex_batch_size]
            
            try:
                enriched_batch = await self._enrich_eurlex_batch(batch)
                enriched.extend(enriched_batch)
                logger.debug(f"Enriched batch {i//eurlex_batch_size + 1}: {len(enriched_batch)} sessions")
                
            except Exception as e:
                logger.warning(f"EUR-Lex enrichment failed for batch {i//eurlex_batch_size + 1}: {e}")
                # Add original sessions without enrichment
                enriched.extend(batch)
        
        return enriched
    
    async def _fetch_opendata_sessions_async(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Async version of OpenData Portal session fetching."""
        # Simulate async operation for now - can be enhanced with actual async HTTP later
        return self._fetch_opendata_sessions(start_date, end_date)
    
    async def _enrich_eurlex_batch(self, sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Enrich a batch of sessions with EUR-Lex data."""
        enriched = []
        for session in sessions:
            try:
                # Add EUR-Lex enrichment logic here
                enriched_session = self._enrich_single_session_eurlex(session)
                enriched.append(enriched_session)
            except Exception as e:
                logger.warning(f"EUR-Lex enrichment failed for session {session.session_id}: {e}")
                enriched.append(session)  # Keep original
        return enriched
    
    def _enrich_single_session_eurlex(self, session: SessionMetadata) -> SessionMetadata:
        """Enrich single session with EUR-Lex metadata."""
        try:
            # Query EUR-Lex for additional session information
            eurlex_data = self.eurlex_client.get_session_metadata(session.session_id)
            
            if eurlex_data:
                # Merge EUR-Lex data into session metadata
                session.additional_metadata = session.additional_metadata or {}
                session.additional_metadata.update(eurlex_data)
                
                # Extract specific fields if available
                if 'document_urls' in eurlex_data:
                    session.verbatim_urls.extend(eurlex_data['document_urls'])
                
            return session
            
        except Exception as e:
            logger.debug(f"EUR-Lex enrichment failed for session {session.session_id}: {e}")
            return session
    
    def _split_date_range(self, start_date: str, end_date: str, batch_size: int) -> List[tuple]:
        """Split date range into batches for processing."""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate days per batch (roughly batch_size sessions / ~2 sessions per week)
        days_per_batch = max(7, batch_size // 2)  # At least 1 week per batch
        
        date_ranges = []
        current_start = start
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=days_per_batch), end)
            date_ranges.append((
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d")
            ))
            current_start = current_end + timedelta(days=1)
        
        return date_ranges
    
    async def _get_batch_cache(self, cache_key: str) -> Optional[List[SessionMetadata]]:
        """Get batch results from cache."""
        try:
            if not self.batch_cache_file.exists():
                return None
            
            with open(self.batch_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_key not in cache_data:
                return None
            
            cached_entry = cache_data[cache_key]
            
            # Check cache age
            cache_time = datetime.fromisoformat(cached_entry['timestamp'])
            if datetime.now() - cache_time > self.cache_ttl:
                logger.debug(f"Batch cache expired for key: {cache_key}")
                return None
            
            # Deserialize sessions
            sessions = []
            for session_data in cached_entry['sessions']:
                try:
                    session = SessionMetadata(**session_data)
                    sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached session: {e}")
                    continue
            
            return sessions
            
        except Exception as e:
            logger.debug(f"Failed to load batch cache: {e}")
            return None
    
    async def _set_batch_cache(self, cache_key: str, sessions: List[SessionMetadata]):
        """Save batch results to cache."""
        try:
            # Load existing cache
            cache_data = {}
            if self.batch_cache_file.exists():
                with open(self.batch_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Serialize sessions
            session_data = []
            for session in sessions:
                try:
                    session_dict = session.dict()
                    session_data.append(session_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize session for cache: {e}")
                    continue
            
            # Store in cache
            cache_data[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'sessions': session_data
            }
            
            # Write cache file
            with open(self.batch_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(sessions)} sessions to batch cache with key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save batch cache: {e}")
    
    def _validate_session_basic(self, session: SessionMetadata) -> bool:
        """Basic session validation for batch processing."""
        try:
            # Check required fields
            if not session.session_id or not session.session_date:
                return False
            
            # Check date format
            if not isinstance(session.session_date, datetime):
                return False
            
            # Check session type
            if session.session_type not in ['plenary', 'committee', 'other']:
                session.session_type = 'plenary'  # Default
            
            return True
            
        except Exception as e:
            logger.debug(f"Session validation failed: {e}")
            return False
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery service statistics."""
        stats = {
            'sessions_cached': len(self.sessions_cache),
            'cache_file_exists': self.discovery_cache_file.exists(),
            'batch_cache_exists': self.batch_cache_file.exists(),
            'batch_size': self.batch_size,
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600
        }
        
        # Add cache file sizes if they exist
        if self.discovery_cache_file.exists():
            stats['cache_size_kb'] = self.discovery_cache_file.stat().st_size // 1024
        
        if self.batch_cache_file.exists():
            stats['batch_cache_size_kb'] = self.batch_cache_file.stat().st_size // 1024
        
        return stats

    
    # ===== Phase 1 Compatible Methods (preserved for backward compatibility) =====
    
    def _fetch_opendata_sessions(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Fetch session data from OpenData Portal."""
        try:
            sessions_data = self.opendata_client.get_plenary_sessions(start_date, end_date)
            
            sessions = []
            for session_data in sessions_data:
                try:
                    session = self._parse_opendata_session(session_data)
                    if session:
                        sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to parse session data: {e}")
                    continue
            
            return sessions
            
        except APIError as e:
            logger.error(f"OpenData API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching OpenData sessions: {e}")
            raise ProcessingError(f"Failed to fetch OpenData sessions: {e}")
    
    def _parse_opendata_session(self, session_data: Dict[str, Any]) -> Optional[SessionMetadata]:
        """Parse session data from OpenData Portal format."""
        try:
            # Extract basic session information
            session_id = session_data.get('session_id') or session_data.get('id')
            session_date = session_data.get('session_date') or session_data.get('date')
            
            if not session_id or not session_date:
                logger.debug(f"Missing required fields in session data: {session_data}")
                return None
            
            # Parse date if it's a string
            if isinstance(session_date, str):
                session_date = parse_session_date(session_date)
            
            # Create session metadata
            session = SessionMetadata(
                session_id=str(session_id),
                session_date=session_date,
                session_type=session_data.get('session_type', 'plenary'),
                location=session_data.get('location', 'Unknown'),
                title=session_data.get('title', f"Session {session_id}"),
                agenda_urls=session_data.get('agenda_urls', []),
                verbatim_urls=session_data.get('verbatim_urls', []),
                additional_metadata=session_data.get('additional_metadata', {})
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Error parsing session data: {e}")
            return None
    
    def _enrich_with_eurlex(self, sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Enrich sessions with EUR-Lex data."""
        enriched_sessions = []
        
        for session in sessions:
            try:
                enriched_session = self._enrich_single_session_eurlex(session)
                enriched_sessions.append(enriched_session)
                
            except Exception as e:
                logger.warning(f"EUR-Lex enrichment failed for session {session.session_id}: {e}")
                enriched_sessions.append(session)  # Keep original
        
        return enriched_sessions
    
    def _validate_sessions(self, sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Validate and filter session data."""
        validated_sessions = []
        
        for session in sessions:
            try:
                # Basic validation
                if not session.session_id or not session.session_date:
                    logger.debug(f"Skipping session with missing required fields")
                    continue
                
                # Date validation
                if not isinstance(session.session_date, datetime):
                    logger.debug(f"Skipping session with invalid date: {session.session_date}")
                    continue
                
                # Additional validation can be added here
                validated_sessions.append(session)
                
            except Exception as e:
                logger.warning(f"Session validation error: {e}")
                continue
        
        return validated_sessions
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[SessionMetadata]]:
        """Load sessions from local cache."""
        try:
            if not self.discovery_cache_file.exists():
                return None
            
            with open(self.discovery_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_key not in cache_data:
                return None
            
            cached_entry = cache_data[cache_key]
            
            # Check cache age (24 hours)
            cache_time = datetime.fromisoformat(cached_entry['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=24):
                logger.debug(f"Cache expired for key: {cache_key}")
                return None
            
            # Deserialize sessions
            sessions = []
            for session_data in cached_entry['sessions']:
                try:
                    session = SessionMetadata(**session_data)
                    sessions.append(session)
                    self.sessions_cache[session.session_id] = session
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached session: {e}")
                    continue
            
            return sessions
            
        except Exception as e:
            logger.debug(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, sessions: List[SessionMetadata]):
        """Save sessions to local cache."""
        try:
            # Load existing cache
            cache_data = {}
            if self.discovery_cache_file.exists():
                with open(self.discovery_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Serialize sessions
            session_data = []
            for session in sessions:
                try:
                    session_dict = session.dict()
                    session_data.append(session_dict)
                    # Update memory cache
                    self.sessions_cache[session.session_id] = session
                except Exception as e:
                    logger.warning(f"Failed to serialize session for cache: {e}")
                    continue
            
            # Store in cache
            cache_data[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'sessions': session_data
            }
            
            # Write cache file
            with open(self.discovery_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(sessions)} sessions to cache with key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def get_session_by_id(self, session_id: str) -> Optional[SessionMetadata]:
        """Get specific session by ID."""
        return self.sessions_cache.get(session_id)
    
    def clear_cache(self) -> bool:
        """Clear discovery cache."""
        try:
            if self.discovery_cache_file.exists():
                self.discovery_cache_file.unlink()
            
            if self.batch_cache_file.exists():
                self.batch_cache_file.unlink()
            
            self.sessions_cache.clear()
            logger.info("Discovery cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
