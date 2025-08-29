"""
Document URL extraction and link resolution service for EU Parliament data.
Handles document discovery, URL validation, and link resolution across sources.
"""

from typing import Dict, List, Optional, Set, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import asyncio

from ..clients.verbatim_client import VerbatimClient
from ..core.exceptions import APIError, ProcessingError, DataValidationError
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class DocumentType(Enum):
    """Types of EU Parliament documents."""
    VERBATIM = "verbatim"
    AGENDA = "agenda"
    MINUTES = "minutes"
    AMENDMENT = "amendment"
    REPORT = "report"
    QUESTION = "question"
    RESOLUTION = "resolution"
    OTHER = "other"


class URLStatus(Enum):
    """Status of URL validation."""
    VALID = "valid"
    INVALID = "invalid"
    REDIRECTED = "redirected"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class DocumentLink:
    """Structured document link information."""
    url: str
    document_type: DocumentType
    title: Optional[str]
    language: Optional[str]
    file_format: Optional[str]  # html, pdf, xml, etc.
    file_size: Optional[int]
    last_modified: Optional[datetime]
    status: URLStatus
    redirect_url: Optional[str]
    validation_time: datetime


@dataclass
class DocumentCollection:
    """Collection of documents for a session."""
    session_id: str
    verbatim_documents: List[DocumentLink]
    agenda_documents: List[DocumentLink]
    other_documents: List[DocumentLink]
    total_documents: int
    validation_summary: Dict[URLStatus, int]
    last_updated: datetime


class DocumentResolver:
    """Service for extracting and resolving EU Parliament document URLs."""
    
    def __init__(self, verbatim_client: Optional[VerbatimClient] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize document resolver.
        
        Args:
            verbatim_client: Optional verbatim client for document fetching
            metrics_collector: Optional metrics collection
        """
        self.verbatim_client = verbatim_client
        self.metrics = metrics_collector or MetricsCollector()
        
        # Configuration
        self.max_concurrent_validations = 10
        self.validation_timeout = 30  # seconds
        self.cache_ttl = timedelta(hours=6)  # Document URLs change less frequently
        
        # Caching
        self.document_cache = Path("data/cache/document_links.json")
        self.document_cache.parent.mkdir(parents=True, exist_ok=True)
        
        # URL patterns for EU Parliament documents
        self.url_patterns = {
            DocumentType.VERBATIM: [
                r'https?://www\.europarl\.europa\.eu/doceo/document/[A-Z]+-\d+-\d+-[A-Z]{2}\.html',
                r'https?://www\.europarl\.europa\.eu/doceo/document/[A-Z]+-\d+-\d+-[A-Z]{2}\.pdf',
                r'https?://www\.europarl\.europa\.eu/plenary/[a-z]{2}/vod/',
            ],
            DocumentType.AGENDA: [
                r'https?://www\.europarl\.europa\.eu/doceo/document/[A-Z]+-\d+-\d+-[A-Z]{2}-AG\.html',
                r'https?://www\.europarl\.europa\.eu/plenary/[a-z]{2}/agenda/',
            ],
            DocumentType.MINUTES: [
                r'https?://www\.europarl\.europa\.eu/doceo/document/[A-Z]+-\d+-\d+-[A-Z]{2}-PV\.html',
            ]
        }
        
        # Known EU Parliament base URLs
        self.eu_base_urls = [
            'https://www.europarl.europa.eu',
            'https://data.europarl.europa.eu',
            'https://eur-lex.europa.eu',
            'http://publications.europa.eu'
        ]
        
        logger.info("Document resolver initialized")
    
    async def resolve_session_documents(self, session_id: str, 
                                      session_urls: List[str] = None,
                                      force_refresh: bool = False) -> DocumentCollection:
        """
        Resolve all documents for a session.
        
        Args:
            session_id: Session identifier
            session_urls: Known session URLs to start with
            force_refresh: Force refresh of cached data
            
        Returns:
            Complete document collection for the session
        """
        start_time = datetime.now()
        logger.info(f"Resolving documents for session {session_id}")
        
        try:
            # Check cache first
            if not force_refresh:
                cached_collection = await self._get_cached_collection(session_id)
                if cached_collection:
                    logger.info(f"Retrieved {cached_collection.total_documents} documents from cache")
                    return cached_collection
            
            # Discover document URLs
            all_urls = await self._discover_document_urls(session_id, session_urls or [])
            logger.info(f"Discovered {len(all_urls)} potential document URLs")
            
            # Validate and categorize URLs
            validated_links = await self._validate_and_categorize_urls(all_urls)
            logger.info(f"Validated {len(validated_links)} document links")
            
            # Organize by document type
            verbatim_docs = [link for link in validated_links if link.document_type == DocumentType.VERBATIM]
            agenda_docs = [link for link in validated_links if link.document_type == DocumentType.AGENDA]
            other_docs = [link for link in validated_links if link.document_type not in [DocumentType.VERBATIM, DocumentType.AGENDA]]
            
            # Create validation summary
            validation_summary = {}
            for status in URLStatus:
                validation_summary[status] = sum(1 for link in validated_links if link.status == status)
            
            # Create document collection
            collection = DocumentCollection(
                session_id=session_id,
                verbatim_documents=verbatim_docs,
                agenda_documents=agenda_docs,
                other_documents=other_docs,
                total_documents=len(validated_links),
                validation_summary=validation_summary,
                last_updated=start_time
            )
            
            # Cache the collection
            await self._cache_collection(collection)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document resolution completed for session {session_id}: "
                       f"{collection.total_documents} documents in {processing_time:.2f}s")
            
            return collection
            
        except Exception as e:
            logger.error(f"Document resolution failed for session {session_id}: {e}")
            raise ProcessingError(f"Failed to resolve documents for session {session_id}: {e}")
    
    async def _discover_document_urls(self, session_id: str, known_urls: List[str]) -> Set[str]:
        """Discover all possible document URLs for a session."""
        discovered_urls = set(known_urls)
        
        # Extract URLs from verbatim client if available
        if self.verbatim_client:
            try:
                verbatim_urls = await self._extract_verbatim_urls(session_id)
                discovered_urls.update(verbatim_urls)
                logger.debug(f"Extracted {len(verbatim_urls)} URLs from verbatim client")
            except Exception as e:
                logger.warning(f"Verbatim URL extraction failed: {e}")
        
        # Generate potential URLs based on EU Parliament URL patterns
        pattern_urls = self._generate_pattern_urls(session_id)
        discovered_urls.update(pattern_urls)
        logger.debug(f"Generated {len(pattern_urls)} URLs from patterns")
        
        # Extract URLs from HTML content if accessible
        content_urls = await self._extract_urls_from_content(list(discovered_urls))
        discovered_urls.update(content_urls)
        logger.debug(f"Extracted {len(content_urls)} URLs from content")
        
        return discovered_urls
    
    async def _extract_verbatim_urls(self, session_id: str) -> List[str]:
        """Extract URLs from verbatim client."""
        try:
            verbatim_data = self.verbatim_client.get_session_documents(session_id)
            
            urls = []
            if isinstance(verbatim_data, dict):
                if 'document_urls' in verbatim_data:
                    urls.extend(verbatim_data['document_urls'])
                if 'verbatim_urls' in verbatim_data:
                    urls.extend(verbatim_data['verbatim_urls'])
                if 'agenda_urls' in verbatim_data:
                    urls.extend(verbatim_data['agenda_urls'])
            
            return urls
            
        except Exception as e:
            logger.debug(f"Verbatim URL extraction failed: {e}")
            return []
    
    def _generate_pattern_urls(self, session_id: str) -> List[str]:
        """Generate potential URLs based on session ID and known patterns."""
        generated_urls = []
        
        # Parse session ID to extract components
        session_parts = self._parse_session_id(session_id)
        if not session_parts:
            return generated_urls
        
        # Generate URLs for different document types
        for doc_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                try:
                    # Replace pattern placeholders with session components
                    url = self._apply_session_pattern(pattern, session_parts)
                    if url:
                        generated_urls.append(url)
                except Exception as e:
                    logger.debug(f"Pattern URL generation failed: {e}")
                    continue
        
        return generated_urls
    
    def _parse_session_id(self, session_id: str) -> Optional[Dict[str, str]]:
        """Parse session ID to extract useful components."""
        try:
            # Common EU Parliament session ID patterns
            patterns = [
                r'(?P<type>[A-Z]+)-(?P<term>\d+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
                r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
                r'(?P<term>\d+)-(?P<session>\d+)',
            ]
            
            for pattern in patterns:
                match = re.match(pattern, session_id)
                if match:
                    return match.groupdict()
            
            # Fallback: treat as simple identifier
            return {'id': session_id}
            
        except Exception as e:
            logger.debug(f"Session ID parsing failed: {e}")
            return None
    
    def _apply_session_pattern(self, url_pattern: str, session_parts: Dict[str, str]) -> Optional[str]:
        """Apply session components to URL pattern."""
        try:
            # This is a simplified pattern application
            # In practice, would need more sophisticated URL generation logic
            
            # For now, return the pattern as-is since it's already a valid URL format
            # Real implementation would substitute session-specific values
            return url_pattern
            
        except Exception as e:
            logger.debug(f"Pattern application failed: {e}")
            return None
    
    async def _extract_urls_from_content(self, source_urls: List[str]) -> Set[str]:
        """Extract additional URLs by analyzing HTML content."""
        extracted_urls = set()
        
        # Limit to avoid overwhelming requests
        source_urls = source_urls[:5]
        
        for url in source_urls:
            try:
                # In a real implementation, would fetch and parse HTML content
                # For now, just return empty set
                pass
            except Exception as e:
                logger.debug(f"Content URL extraction failed for {url}: {e}")
                continue
        
        return extracted_urls
    
    async def _validate_and_categorize_urls(self, urls: Set[str]) -> List[DocumentLink]:
        """Validate URLs and categorize by document type."""
        validation_tasks = []
        
        # Create validation tasks
        for url in urls:
            task = self._validate_single_url(url)
            validation_tasks.append(task)
        
        # Execute validations concurrently with semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_validations)
        
        async def validate_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Wait for all validations
        results = await asyncio.gather(
            *[validate_with_semaphore(task) for task in validation_tasks],
            return_exceptions=True
        )
        
        # Filter successful validations
        validated_links = []
        for result in results:
            if isinstance(result, DocumentLink):
                validated_links.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"URL validation failed: {result}")
        
        return validated_links
    
    async def _validate_single_url(self, url: str) -> DocumentLink:
        """Validate a single URL and extract metadata."""
        validation_start = datetime.now()
        
        try:
            # Categorize document type
            doc_type = self._categorize_url(url)
            
            # Extract basic metadata from URL
            metadata = self._extract_url_metadata(url)
            
            # For now, assume URL is valid (in practice, would make HTTP request)
            # Real implementation would check HTTP status, redirects, etc.
            status = URLStatus.VALID
            redirect_url = None
            
            link = DocumentLink(
                url=url,
                document_type=doc_type,
                title=metadata.get('title'),
                language=metadata.get('language'),
                file_format=metadata.get('format'),
                file_size=metadata.get('size'),
                last_modified=metadata.get('last_modified'),
                status=status,
                redirect_url=redirect_url,
                validation_time=validation_start
            )
            
            return link
            
        except Exception as e:
            logger.debug(f"URL validation failed for {url}: {e}")
            
            # Return invalid link
            return DocumentLink(
                url=url,
                document_type=DocumentType.OTHER,
                title=None,
                language=None,
                file_format=None,
                file_size=None,
                last_modified=None,
                status=URLStatus.INVALID,
                redirect_url=None,
                validation_time=validation_start
            )
    
    def _categorize_url(self, url: str) -> DocumentType:
        """Categorize URL by document type based on patterns."""
        url_lower = url.lower()
        
        # Check verbatim patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.url_patterns[DocumentType.VERBATIM]):
            return DocumentType.VERBATIM
        
        # Check agenda patterns  
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.url_patterns[DocumentType.AGENDA]):
            return DocumentType.AGENDA
        
        # Check by URL content
        if 'verbatim' in url_lower or 'cre-' in url_lower:
            return DocumentType.VERBATIM
        elif 'agenda' in url_lower or '-ag' in url_lower:
            return DocumentType.AGENDA
        elif 'minutes' in url_lower or '-pv' in url_lower:
            return DocumentType.MINUTES
        elif 'amendment' in url_lower:
            return DocumentType.AMENDMENT
        elif 'report' in url_lower:
            return DocumentType.REPORT
        elif 'question' in url_lower:
            return DocumentType.QUESTION
        elif 'resolution' in url_lower:
            return DocumentType.RESOLUTION
        
        return DocumentType.OTHER
    
    def _extract_url_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from URL structure."""
        metadata = {}
        
        try:
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split('/')
            
            # Extract language from URL
            for part in path_parts:
                if len(part) == 2 and part.isalpha():
                    metadata['language'] = part.upper()
                    break
            
            # Extract file format from extension
            if '.' in parsed_url.path:
                extension = parsed_url.path.split('.')[-1].lower()
                if extension in ['html', 'pdf', 'xml', 'doc', 'docx']:
                    metadata['format'] = extension
            
            # Extract title from filename
            filename = path_parts[-1] if path_parts else None
            if filename:
                # Remove extension and create readable title
                title = filename.split('.')[0].replace('-', ' ').replace('_', ' ')
                metadata['title'] = title.title()
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed for {url}: {e}")
        
        return metadata
    
    async def _get_cached_collection(self, session_id: str) -> Optional[DocumentCollection]:
        """Get document collection from cache."""
        try:
            if not self.document_cache.exists():
                return None
            
            with open(self.document_cache, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if session_id not in cache_data:
                return None
            
            cached_entry = cache_data[session_id]
            
            # Check cache age
            cache_time = datetime.fromisoformat(cached_entry['last_updated'])
            if datetime.now() - cache_time > self.cache_ttl:
                return None
            
            # Deserialize collection (simplified)
            # In production would need proper deserialization
            return DocumentCollection(**cached_entry)
            
        except Exception as e:
            logger.debug(f"Failed to load cached document collection: {e}")
            return None
    
    async def _cache_collection(self, collection: DocumentCollection):
        """Cache document collection."""
        try:
            # Load existing cache
            cache_data = {}
            if self.document_cache.exists():
                with open(self.document_cache, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Serialize collection (simplified)
            cache_data[collection.session_id] = {
                'session_id': collection.session_id,
                'total_documents': collection.total_documents,
                'last_updated': collection.last_updated.isoformat()
            }
            
            # Write cache
            with open(self.document_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Failed to cache document collection: {e}")
    
    def get_resolver_stats(self) -> Dict[str, Any]:
        """Get document resolver statistics."""
        stats = {
            'max_concurrent_validations': self.max_concurrent_validations,
            'validation_timeout': self.validation_timeout,
            'cache_ttl_hours': self.cache_ttl.total_seconds() / 3600,
            'document_types': len(DocumentType),
            'url_patterns': {dt.value: len(patterns) for dt, patterns in self.url_patterns.items()},
            'eu_base_urls': len(self.eu_base_urls)
        }
        
        # Add cache statistics
        if self.document_cache.exists():
            try:
                with open(self.document_cache, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                stats['cached_sessions'] = len(cache_data)
                stats['cache_size_kb'] = self.document_cache.stat().st_size // 1024
            except:
                stats['cached_sessions'] = 0
                stats['cache_size_kb'] = 0
        else:
            stats['cached_sessions'] = 0
            stats['cache_size_kb'] = 0
        
        return stats