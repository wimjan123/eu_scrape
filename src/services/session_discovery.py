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
    """Service for discovering and cataloging plenary sessions."""
    
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
        
        # Ensure cache directory exists
        self.discovery_cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Session discovery service initialized")
    
    def discover_sessions(self, start_date: str, end_date: str, 
                         force_refresh: bool = False) -> List[SessionMetadata]:
        """
        Discover plenary sessions in date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            force_refresh: Force refresh of cached data
            
        Returns:
            List of discovered session metadata
        """
        cache_key = f"{start_date}_{end_date}"
        
        logger.info(
            "Starting session discovery",
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Check cache first
        if not force_refresh:
            cached_sessions = self._load_from_cache(cache_key)
            if cached_sessions:
                logger.info("Using cached session data", count=len(cached_sessions))
                return cached_sessions
        
        try:
            # Discover sessions from multiple sources
            sessions = []
            
            # Primary source: Open Data Portal
            opendata_sessions = self._discover_from_opendata(start_date, end_date)
            sessions.extend(opendata_sessions)
            
            # Secondary source: EUR-Lex (if available)
            if self.eurlex_client:
                eurlex_sessions = self._discover_from_eurlex(start_date, end_date)
                sessions = self._merge_session_data(sessions, eurlex_sessions)
            
            # Validate and enrich session data
            validated_sessions = []
            for session in sessions:
                if self._validate_session_metadata(session):
                    enriched_session = self._enrich_session_metadata(session)
                    validated_sessions.append(enriched_session)
                else:
                    logger.warning("Invalid session metadata", session_id=session.session_id)
            
            # Cache results
            self._save_to_cache(cache_key, validated_sessions)
            
            logger.info(
                "Session discovery completed",
                total_found=len(validated_sessions),
                start_date=start_date,
                end_date=end_date
            )
            
            return validated_sessions
            
        except Exception as e:
            logger.error("Session discovery failed", error=str(e))
            raise ProcessingError(f"Session discovery failed: {e}")
    
    def _discover_from_opendata(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Discover sessions from Open Data Portal."""
        logger.info("Discovering sessions from Open Data Portal")
        
        try:
            raw_sessions = self.opendata_client.get_plenary_sessions(start_date, end_date)
            
            sessions = []
            for raw_session in raw_sessions:
                try:
                    session = self._convert_opendata_session(raw_session)
                    if session:
                        sessions.append(session)
                except Exception as e:
                    logger.warning(
                        "Failed to convert OpenData session",
                        error=str(e),
                        session_data=raw_session
                    )
                    continue
            
            logger.info("OpenData sessions discovered", count=len(sessions))
            return sessions
            
        except APIError as e:
            logger.error("OpenData session discovery failed", error=str(e))
            return []
    
    def _discover_from_eurlex(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Discover sessions from EUR-Lex."""
        if not self.eurlex_client:
            return []
        
        logger.info("Discovering sessions from EUR-Lex")
        
        try:
            documents = self.eurlex_client.get_plenary_documents(start_date, end_date)
            
            sessions = []
            processed_sessions = set()
            
            for doc in documents:
                try:
                    session = self._convert_eurlex_document(doc)
                    if session and session.session_id not in processed_sessions:
                        sessions.append(session)
                        processed_sessions.add(session.session_id)
                except Exception as e:
                    logger.warning(
                        "Failed to convert EUR-Lex document",
                        error=str(e),
                        document=doc
                    )
                    continue
            
            logger.info("EUR-Lex sessions discovered", count=len(sessions))
            return sessions
            
        except APIError as e:
            logger.error("EUR-Lex session discovery failed", error=str(e))
            return []
    
    def _convert_opendata_session(self, raw_session: Dict[str, Any]) -> Optional[SessionMetadata]:
        """Convert OpenData session to SessionMetadata."""
        try:
            # Extract session ID
            session_id = raw_session.get('identifier') or raw_session.get('id', '')
            if not session_id:
                return None
            
            # Parse date
            date_str = raw_session.get('date', '')
            session_date = parse_session_date(date_str)
            if not session_date:
                logger.warning("Invalid session date", date_str=date_str)
                return None
            
            # Extract title
            title = ''
            if 'title' in raw_session:
                if isinstance(raw_session['title'], dict):
                    title = raw_session['title'].get('en', '') or str(raw_session['title'])
                else:
                    title = str(raw_session['title'])
            
            # Extract language
            language = raw_session.get('language', 'en')
            if isinstance(language, dict):
                language = language.get('en', 'en')
            
            # Extract URLs
            verbatim_url = self._extract_verbatim_url(raw_session)
            agenda_url = self._extract_agenda_url(raw_session)
            
            return SessionMetadata(
                session_id=session_id,
                date=session_date,
                title=title,
                session_type='plenary',
                language=language,
                verbatim_url=verbatim_url,
                agenda_url=agenda_url,
                status='discovered'
            )
            
        except Exception as e:
            logger.error("Failed to convert OpenData session", error=str(e))
            return None
    
    def _convert_eurlex_document(self, document: Dict[str, Any]) -> Optional[SessionMetadata]:
        """Convert EUR-Lex document to SessionMetadata."""
        try:
            # Extract identifier for session ID
            identifier = document.get('identifier', '')
            if not identifier:
                return None
            
            # Parse date
            date_str = document.get('date', '')
            session_date = parse_session_date(date_str)
            if not session_date:
                return None
            
            # Create session ID from identifier and date
            session_id = f"eurlex_{identifier}_{session_date.strftime('%Y-%m-%d')}"
            
            return SessionMetadata(
                session_id=session_id,
                date=session_date,
                title=document.get('title', ''),
                session_type='plenary',
                language=document.get('language', 'en'),
                verbatim_url=document.get('document_uri'),
                agenda_url=None,
                status='discovered'
            )
            
        except Exception as e:
            logger.error("Failed to convert EUR-Lex document", error=str(e))
            return None
    
    def _extract_verbatim_url(self, raw_session: Dict[str, Any]) -> Optional[str]:
        """Extract verbatim report URL from session data."""
        # Check for direct URL fields
        for url_field in ['verbatim_url', 'document_url', 'url', 'link']:
            if url_field in raw_session and raw_session[url_field]:
                url = raw_session[url_field]
                if 'CRE' in str(url) or 'verbatim' in str(url).lower():
                    return str(url)
        
        # Check in nested structures
        if 'documents' in raw_session:
            documents = raw_session['documents']
            if isinstance(documents, list):
                for doc in documents:
                    if isinstance(doc, dict) and doc.get('type') == 'verbatim':
                        return doc.get('url')
        
        return None
    
    def _extract_agenda_url(self, raw_session: Dict[str, Any]) -> Optional[str]:
        """Extract agenda URL from session data."""
        # Check for direct agenda URL fields
        for url_field in ['agenda_url', 'agenda', 'order_of_business']:
            if url_field in raw_session and raw_session[url_field]:
                return str(raw_session[url_field])
        
        # Check in documents
        if 'documents' in raw_session:
            documents = raw_session['documents']
            if isinstance(documents, list):
                for doc in documents:
                    if isinstance(doc, dict) and doc.get('type') == 'agenda':
                        return doc.get('url')
        
        return None
    
    def _merge_session_data(self, primary_sessions: List[SessionMetadata], 
                           secondary_sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Merge session data from multiple sources."""
        merged = {}
        
        # Add primary sessions
        for session in primary_sessions:
            merged[session.session_id] = session
        
        # Add/merge secondary sessions
        for session in secondary_sessions:
            if session.session_id in merged:
                # Merge data - enhance existing session
                existing = merged[session.session_id]
                if not existing.verbatim_url and session.verbatim_url:
                    existing.verbatim_url = session.verbatim_url
                if not existing.agenda_url and session.agenda_url:
                    existing.agenda_url = session.agenda_url
            else:
                # Add new session
                merged[session.session_id] = session
        
        return list(merged.values())
    
    def _validate_session_metadata(self, session: SessionMetadata) -> bool:
        """Validate session metadata quality."""
        if not session.session_id or len(session.session_id) < 3:
            return False
        
        if not session.date or session.date > datetime.now():
            return False
        
        if session.session_type != 'plenary':
            return False
        
        return True
    
    def _enrich_session_metadata(self, session: SessionMetadata) -> SessionMetadata:
        """Enrich session metadata with additional information."""
        # Add verbatim URL if missing
        if not session.verbatim_url:
            # Generate probable verbatim URL
            date_str = session.date.strftime('%Y-%m-%d')
            probable_url = f"https://www.europarl.europa.eu/doceo/document/CRE-9-{date_str}_EN.html"
            session.verbatim_url = probable_url
        
        # Enhance title if empty
        if not session.title or session.title.strip() == '':
            session.title = f"Plenary Session - {session.date.strftime('%B %d, %Y')}"
        
        return session
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[SessionMetadata]]:
        """Load sessions from cache."""
        if not self.discovery_cache_file.exists():
            return None
        
        try:
            with open(self.discovery_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if cache_key in cache_data:
                cached_sessions_data = cache_data[cache_key]
                
                # Check cache age (24 hours)
                cache_time = datetime.fromisoformat(cached_sessions_data['timestamp'])
                if (datetime.now() - cache_time).total_seconds() > 86400:  # 24 hours
                    logger.info("Cache expired", cache_key=cache_key)
                    return None
                
                # Convert back to SessionMetadata objects
                sessions = []
                for session_data in cached_sessions_data['sessions']:
                    session = SessionMetadata(
                        session_id=session_data['session_id'],
                        date=datetime.fromisoformat(session_data['date']),
                        title=session_data['title'],
                        session_type=session_data['session_type'],
                        language=session_data['language'],
                        verbatim_url=session_data.get('verbatim_url'),
                        agenda_url=session_data.get('agenda_url'),
                        status=session_data['status']
                    )
                    sessions.append(session)
                
                return sessions
        
        except Exception as e:
            logger.warning("Failed to load cache", error=str(e))
        
        return None
    
    def _save_to_cache(self, cache_key: str, sessions: List[SessionMetadata]) -> None:
        """Save sessions to cache."""
        try:
            # Load existing cache
            cache_data = {}
            if self.discovery_cache_file.exists():
                with open(self.discovery_cache_file, 'r') as f:
                    cache_data = json.load(f)
            
            # Convert sessions to serializable format
            serializable_sessions = []
            for session in sessions:
                session_data = {
                    'session_id': session.session_id,
                    'date': session.date.isoformat(),
                    'title': session.title,
                    'session_type': session.session_type,
                    'language': session.language,
                    'verbatim_url': session.verbatim_url,
                    'agenda_url': session.agenda_url,
                    'status': session.status
                }
                serializable_sessions.append(session_data)
            
            # Save to cache
            cache_data[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'sessions': serializable_sessions
            }
            
            with open(self.discovery_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info("Sessions saved to cache", cache_key=cache_key, count=len(sessions))
            
        except Exception as e:
            logger.warning("Failed to save cache", error=str(e))
    
    def get_session_by_id(self, session_id: str) -> Optional[SessionMetadata]:
        """Get specific session by ID."""
        if session_id in self.sessions_cache:
            return self.sessions_cache[session_id]
        
        # Search in cache file
        if self.discovery_cache_file.exists():
            try:
                with open(self.discovery_cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for cache_key, cached_data in cache_data.items():
                    for session_data in cached_data.get('sessions', []):
                        if session_data['session_id'] == session_id:
                            session = SessionMetadata(
                                session_id=session_data['session_id'],
                                date=datetime.fromisoformat(session_data['date']),
                                title=session_data['title'],
                                session_type=session_data['session_type'],
                                language=session_data['language'],
                                verbatim_url=session_data.get('verbatim_url'),
                                agenda_url=session_data.get('agenda_url'),
                                status=session_data['status']
                            )
                            return session
            
            except Exception as e:
                logger.warning("Failed to search cache for session", error=str(e))
        
        return None
    
    def get_sessions_count(self, start_date: str, end_date: str) -> int:
        """Get count of sessions in date range without full discovery."""
        try:
            sessions = self.discover_sessions(start_date, end_date)
            return len(sessions)
        except Exception as e:
            logger.error("Failed to get sessions count", error=str(e))
            return 0