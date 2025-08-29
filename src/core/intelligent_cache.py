"""
Intelligent caching system with invalidation logic for EU Parliament data collection.
Provides hierarchical caching with smart invalidation, compression, and analytics.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import gzip
import pickle
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

from ..core.logging import get_logger
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class CacheType(Enum):
    """Types of cached data."""
    SESSION_DISCOVERY = "session_discovery"
    INTEGRATION_RESULTS = "integration_results" 
    DOCUMENT_LINKS = "document_links"
    VERBATIM_CONTENT = "verbatim_content"
    PARSED_SPEECHES = "parsed_speeches"
    VALIDATION_RESULTS = "validation_results"
    API_RESPONSES = "api_responses"


class InvalidationType(Enum):
    """Types of cache invalidation."""
    TIME_BASED = "time_based"
    CONTENT_BASED = "content_based"
    DEPENDENCY_BASED = "dependency_based"
    MANUAL = "manual"
    SIZE_BASED = "size_based"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    data: Any
    cache_type: CacheType
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl: timedelta
    size_bytes: int
    content_hash: str
    dependencies: List[str]
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    invalidation_count: int
    hit_rate: float
    average_access_count: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]


class CacheInvalidationStrategy(ABC):
    """Abstract base class for cache invalidation strategies."""
    
    @abstractmethod
    async def should_invalidate(self, entry: CacheEntry, context: Dict[str, Any]) -> bool:
        """Determine if cache entry should be invalidated."""
        pass


class TimeBasedInvalidation(CacheInvalidationStrategy):
    """Time-based cache invalidation strategy."""
    
    async def should_invalidate(self, entry: CacheEntry, context: Dict[str, Any]) -> bool:
        """Invalidate based on TTL and absolute age."""
        now = datetime.now()
        
        # Check TTL expiration
        if now - entry.created_at > entry.ttl:
            return True
        
        # Check absolute age limits
        max_age = context.get('max_age_hours', 168)  # 1 week default
        if now - entry.created_at > timedelta(hours=max_age):
            return True
        
        return False


class ContentBasedInvalidation(CacheInvalidationStrategy):
    """Content-based cache invalidation strategy."""
    
    async def should_invalidate(self, entry: CacheEntry, context: Dict[str, Any]) -> bool:
        """Invalidate based on content changes."""
        # Check if source data has changed
        current_hash = context.get('current_content_hash')
        if current_hash and current_hash != entry.content_hash:
            return True
        
        # Check modification time if available
        source_modified = context.get('source_modified_time')
        if source_modified and source_modified > entry.created_at:
            return True
        
        return False


class DependencyBasedInvalidation(CacheInvalidationStrategy):
    """Dependency-based cache invalidation strategy."""
    
    def __init__(self, cache_manager: 'IntelligentCacheManager'):
        self.cache_manager = cache_manager
    
    async def should_invalidate(self, entry: CacheEntry, context: Dict[str, Any]) -> bool:
        """Invalidate based on dependency changes."""
        # Check if any dependencies have been invalidated
        for dependency_key in entry.dependencies:
            if not await self.cache_manager.exists(dependency_key):
                return True
            
            # Check if dependency is newer than this entry
            dependency_entry = await self.cache_manager.get_entry_metadata(dependency_key)
            if dependency_entry and dependency_entry.created_at > entry.created_at:
                return True
        
        return False


class SizeBasedInvalidation(CacheInvalidationStrategy):
    """Size-based cache invalidation strategy (LRU eviction)."""
    
    async def should_invalidate(self, entry: CacheEntry, context: Dict[str, Any]) -> bool:
        """Invalidate based on cache size limits and access patterns."""
        max_cache_size = context.get('max_cache_size_mb', 1000) * 1024 * 1024
        current_cache_size = context.get('current_cache_size', 0)
        
        # If cache is over limit, prioritize eviction of least accessed items
        if current_cache_size > max_cache_size:
            min_access_count = context.get('min_access_count', 0)
            if entry.access_count <= min_access_count:
                return True
            
            # Also consider access recency
            last_access_threshold = context.get('last_access_hours', 72)
            if datetime.now() - entry.accessed_at > timedelta(hours=last_access_threshold):
                return True
        
        return False


class IntelligentCacheManager:
    """Intelligent cache manager with multiple invalidation strategies."""
    
    def __init__(self, cache_dir: Union[str, Path] = "data/cache",
                 metrics_collector: Optional[MetricsCollector] = None,
                 enable_compression: bool = True,
                 max_cache_size_mb: int = 1000):
        """
        Initialize intelligent cache manager.
        
        Args:
            cache_dir: Cache directory path
            metrics_collector: Optional metrics collection
            enable_compression: Enable data compression
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = metrics_collector or MetricsCollector()
        self.enable_compression = enable_compression
        self.max_cache_size_mb = max_cache_size_mb
        
        # Cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_index_file = self.cache_dir / "cache_index.json"
        
        # Invalidation strategies
        self.invalidation_strategies = {
            InvalidationType.TIME_BASED: TimeBasedInvalidation(),
            InvalidationType.CONTENT_BASED: ContentBasedInvalidation(),
            InvalidationType.DEPENDENCY_BASED: DependencyBasedInvalidation(self),
            InvalidationType.SIZE_BASED: SizeBasedInvalidation()
        }
        
        # Configuration
        self.default_ttl = {
            CacheType.SESSION_DISCOVERY: timedelta(hours=24),
            CacheType.INTEGRATION_RESULTS: timedelta(hours=4),
            CacheType.DOCUMENT_LINKS: timedelta(hours=6),
            CacheType.VERBATIM_CONTENT: timedelta(days=7),
            CacheType.PARSED_SPEECHES: timedelta(days=3),
            CacheType.VALIDATION_RESULTS: timedelta(hours=2),
            CacheType.API_RESPONSES: timedelta(minutes=30)
        }
        
        # Statistics
        self.stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            invalidation_count=0,
            hit_rate=0.0,
            average_access_count=0.0,
            oldest_entry=None,
            newest_entry=None
        )
        
        # Background tasks
        self._cleanup_task = None
        self._start_background_tasks()
        
        logger.info(f"Intelligent cache manager initialized: {self.cache_dir}")
    
    async def get(self, key: str, cache_type: CacheType) -> Optional[Any]:
        """
        Get cached data with intelligent access tracking.
        
        Args:
            key: Cache key
            cache_type: Type of cached data
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            entry = await self._get_entry(key)
            
            if not entry:
                self.stats.miss_count += 1
                self._update_hit_rate()
                return None
            
            # Check if entry should be invalidated
            if await self._should_invalidate_entry(entry):
                await self._remove_entry(key)
                self.stats.miss_count += 1
                self.stats.invalidation_count += 1
                self._update_hit_rate()
                return None
            
            # Update access tracking
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            await self._update_entry(entry)
            
            self.stats.hit_count += 1
            self._update_hit_rate()
            
            # Record metrics
            self.metrics.record_request(
                service="cache",
                success=True,
                response_time=0.001  # Cache hits are very fast
            )
            
            logger.debug(f"Cache hit for key: {key}")
            return entry.data
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.stats.miss_count += 1
            self._update_hit_rate()
            return None
    
    async def set(self, key: str, data: Any, cache_type: CacheType,
                 ttl: Optional[timedelta] = None,
                 dependencies: List[str] = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """
        Set cached data with intelligent storage and compression.
        
        Args:
            key: Cache key
            data: Data to cache
            cache_type: Type of cached data
            ttl: Time to live (uses default if not specified)
            dependencies: List of dependency keys
            metadata: Additional metadata
            
        Returns:
            True if successfully cached
        """
        try:
            # Calculate content hash
            content_hash = self._calculate_content_hash(data)
            
            # Determine TTL
            effective_ttl = ttl or self.default_ttl.get(cache_type, timedelta(hours=1))
            
            # Serialize and optionally compress data
            serialized_data, size_bytes = await self._serialize_data(data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,  # Keep uncompressed in memory
                cache_type=cache_type,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                ttl=effective_ttl,
                size_bytes=size_bytes,
                content_hash=content_hash,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            # Check cache size limits before adding
            await self._enforce_size_limits()
            
            # Store entry
            await self._store_entry(entry, serialized_data)
            
            # Update statistics
            self.stats.total_entries += 1
            self.stats.total_size_bytes += size_bytes
            self._update_entry_timestamps(entry.created_at)
            
            logger.debug(f"Cache set for key: {key}, size: {size_bytes} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def invalidate(self, key: str) -> bool:
        """Manually invalidate a cache entry."""
        try:
            if await self.exists(key):
                await self._remove_entry(key)
                self.stats.invalidation_count += 1
                logger.info(f"Cache entry invalidated: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Cache invalidation failed for key {key}: {e}")
            return False
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate multiple entries matching a pattern."""
        try:
            invalidated = 0
            keys_to_invalidate = []
            
            # Find matching keys
            for key in self.memory_cache.keys():
                if self._key_matches_pattern(key, pattern):
                    keys_to_invalidate.append(key)
            
            # Invalidate matching entries
            for key in keys_to_invalidate:
                if await self.invalidate(key):
                    invalidated += 1
            
            logger.info(f"Invalidated {invalidated} entries matching pattern: {pattern}")
            return invalidated
            
        except Exception as e:
            logger.error(f"Pattern invalidation failed for pattern {pattern}: {e}")
            return 0
    
    async def invalidate_dependencies(self, dependency_key: str) -> int:
        """Invalidate entries that depend on a specific key."""
        try:
            invalidated = 0
            keys_to_invalidate = []
            
            # Find entries with this dependency
            for key, entry in self.memory_cache.items():
                if dependency_key in entry.dependencies:
                    keys_to_invalidate.append(key)
            
            # Invalidate dependent entries
            for key in keys_to_invalidate:
                if await self.invalidate(key):
                    invalidated += 1
            
            logger.info(f"Invalidated {invalidated} dependent entries for: {dependency_key}")
            return invalidated
            
        except Exception as e:
            logger.error(f"Dependency invalidation failed for key {dependency_key}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if a cache entry exists and is valid."""
        try:
            entry = await self._get_entry(key)
            if not entry:
                return False
            
            # Check if entry should be invalidated
            return not await self._should_invalidate_entry(entry)
            
        except Exception as e:
            logger.debug(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def get_entry_metadata(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry metadata without loading data."""
        try:
            return await self._get_entry(key, load_data=False)
        except Exception as e:
            logger.debug(f"Metadata retrieval failed for key {key}: {e}")
            return None
    
    async def cleanup(self) -> Dict[str, int]:
        """Clean up expired and invalid cache entries."""
        logger.info("Starting cache cleanup")
        
        cleanup_stats = {
            'expired': 0,
            'invalid': 0,
            'size_evicted': 0,
            'errors': 0
        }
        
        try:
            keys_to_remove = []
            
            # Check all entries for invalidation
            for key, entry in self.memory_cache.items():
                try:
                    if await self._should_invalidate_entry(entry):
                        keys_to_remove.append(key)
                except Exception as e:
                    logger.warning(f"Cleanup check failed for key {key}: {e}")
                    cleanup_stats['errors'] += 1
            
            # Remove invalid entries
            for key in keys_to_remove:
                try:
                    entry = self.memory_cache.get(key)
                    if entry:
                        # Categorize removal reason
                        if datetime.now() - entry.created_at > entry.ttl:
                            cleanup_stats['expired'] += 1
                        else:
                            cleanup_stats['invalid'] += 1
                    
                    await self._remove_entry(key)
                    
                except Exception as e:
                    logger.warning(f"Entry removal failed for key {key}: {e}")
                    cleanup_stats['errors'] += 1
            
            # Enforce size limits
            evicted = await self._enforce_size_limits()
            cleanup_stats['size_evicted'] = evicted
            
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            cleanup_stats['errors'] += 1
            return cleanup_stats
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        # Update dynamic stats
        self.stats.total_entries = len(self.memory_cache)
        self.stats.total_size_bytes = sum(entry.size_bytes for entry in self.memory_cache.values())
        
        if self.memory_cache:
            self.stats.average_access_count = sum(entry.access_count for entry in self.memory_cache.values()) / len(self.memory_cache)
            entries_by_date = sorted(self.memory_cache.values(), key=lambda e: e.created_at)
            self.stats.oldest_entry = entries_by_date[0].created_at
            self.stats.newest_entry = entries_by_date[-1].created_at
        
        return self.stats
    
    async def _get_entry(self, key: str, load_data: bool = True) -> Optional[CacheEntry]:
        """Get cache entry from memory or disk."""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try to load from disk
        if load_data:
            return await self._load_entry_from_disk(key)
        
        return None
    
    async def _should_invalidate_entry(self, entry: CacheEntry) -> bool:
        """Check if entry should be invalidated using all strategies."""
        context = {
            'max_cache_size_mb': self.max_cache_size_mb,
            'current_cache_size': self.stats.total_size_bytes,
            'min_access_count': min(e.access_count for e in self.memory_cache.values()) if self.memory_cache else 0
        }
        
        # Check each invalidation strategy
        for invalidation_type, strategy in self.invalidation_strategies.items():
            try:
                if await strategy.should_invalidate(entry, context):
                    logger.debug(f"Entry {entry.key} invalidated by {invalidation_type.value}")
                    return True
            except Exception as e:
                logger.warning(f"Invalidation strategy {invalidation_type.value} failed: {e}")
        
        return False
    
    async def _serialize_data(self, data: Any) -> Tuple[bytes, int]:
        """Serialize and optionally compress data."""
        try:
            # Serialize with pickle
            serialized = pickle.dumps(data)
            
            # Compress if enabled and beneficial
            if self.enable_compression and len(serialized) > 1024:  # Only compress larger data
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.9:  # Only if significant compression
                    return compressed, len(compressed)
            
            return serialized, len(serialized)
            
        except Exception as e:
            logger.error(f"Data serialization failed: {e}")
            raise
    
    async def _store_entry(self, entry: CacheEntry, serialized_data: bytes):
        """Store cache entry in memory and disk."""
        # Store in memory
        self.memory_cache[entry.key] = entry
        
        # Store on disk for persistence
        cache_file = self.cache_dir / f"{entry.key}.cache"
        with open(cache_file, 'wb') as f:
            f.write(serialized_data)
        
        # Update index
        await self._update_cache_index()
    
    async def _load_entry_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk."""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            # Load serialized data
            with open(cache_file, 'rb') as f:
                serialized_data = f.read()
            
            # Try to decompress if it looks compressed
            try:
                if serialized_data[:2] == b'\x1f\x8b':  # Gzip magic number
                    serialized_data = gzip.decompress(serialized_data)
            except:
                pass  # Not compressed or decompression failed
            
            # Deserialize data
            data = pickle.loads(serialized_data)
            
            # Load entry metadata from index
            entry_metadata = await self._get_entry_metadata_from_index(key)
            if not entry_metadata:
                return None
            
            # Create entry object
            entry = CacheEntry(
                key=key,
                data=data,
                **entry_metadata
            )
            
            # Add to memory cache
            self.memory_cache[key] = entry
            
            return entry
            
        except Exception as e:
            logger.debug(f"Failed to load cache entry from disk for key {key}: {e}")
            return None
    
    async def _remove_entry(self, key: str):
        """Remove cache entry from memory and disk."""
        # Remove from memory
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.memory_cache[key]
        
        # Remove from disk
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()
        
        # Update index
        await self._update_cache_index()
    
    async def _enforce_size_limits(self) -> int:
        """Enforce cache size limits by evicting entries."""
        max_size = self.max_cache_size_mb * 1024 * 1024
        current_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        
        if current_size <= max_size:
            return 0
        
        # Sort by access patterns (LRU with access count weighting)
        entries = sorted(
            self.memory_cache.values(),
            key=lambda e: (e.access_count, e.accessed_at)
        )
        
        evicted_count = 0
        for entry in entries:
            if current_size <= max_size:
                break
            
            await self._remove_entry(entry.key)
            current_size -= entry.size_bytes
            evicted_count += 1
            self.stats.eviction_count += 1
        
        return evicted_count
    
    def _calculate_content_hash(self, data: Any) -> str:
        """Calculate hash of data content."""
        try:
            serialized = pickle.dumps(data)
            return hashlib.sha256(serialized).hexdigest()[:16]
        except:
            return "unknown"
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches a pattern (supports wildcards)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def _update_hit_rate(self):
        """Update cache hit rate statistic."""
        total_requests = self.stats.hit_count + self.stats.miss_count
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hit_count / total_requests
    
    def _update_entry_timestamps(self, timestamp: datetime):
        """Update oldest/newest entry timestamps."""
        if not self.stats.oldest_entry or timestamp < self.stats.oldest_entry:
            self.stats.oldest_entry = timestamp
        if not self.stats.newest_entry or timestamp > self.stats.newest_entry:
            self.stats.newest_entry = timestamp
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup()
                except Exception as e:
                    logger.error(f"Background cleanup task failed: {e}")
        
        # Don't start in test environments
        if not hasattr(self, '_testing'):
            self._cleanup_task = asyncio.create_task(cleanup_task())
    
    async def _update_cache_index(self):
        """Update cache index file."""
        # Simplified index update - in production would be more sophisticated
        pass
    
    async def _get_entry_metadata_from_index(self, key: str) -> Optional[Dict[str, Any]]:
        """Get entry metadata from index."""
        # Simplified - would load from actual index file
        return None
    
    async def _update_entry(self, entry: CacheEntry):
        """Update existing cache entry."""
        self.memory_cache[entry.key] = entry
        await self._update_cache_index()
    
    async def close(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Intelligent cache manager closed")