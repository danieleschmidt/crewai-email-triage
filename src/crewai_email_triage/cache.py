"""Intelligent caching system for CrewAI Email Triage."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from .metrics_export import get_metrics_collector

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""

    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache statistics and metrics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "size_bytes": self.size_bytes,
            "entry_count": self.entry_count
        }


class LRUCache:
    """Thread-safe LRU cache with TTL and size limits."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = 3600,
        max_memory_mb: float = 100.0
    ):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._metrics_collector = get_metrics_collector()

        logger.info("LRU cache initialized: max_size=%d, ttl=%s, max_memory=%.1fMB",
                   max_size, ttl_seconds, max_memory_mb)

    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            else:
                return 1024  # Default estimate

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        if self.ttl_seconds is None:
            return

        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.created_at > self.ttl_seconds
        ]

        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1
            logger.debug("Evicted expired cache entry: %s", key)

    def _evict_lru(self) -> None:
        """Evict least recently used entries to maintain size limits."""
        # Evict by count
        while len(self._cache) >= self.max_size:
            key, entry = self._cache.popitem(last=False)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1
            logger.debug("Evicted LRU cache entry (size): %s", key)

        # Evict by memory
        while self._stats.size_bytes > self.max_memory_bytes and self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
            self._stats.evictions += 1
            logger.debug("Evicted LRU cache entry (memory): %s", key)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()

            if key not in self._cache:
                self._stats.misses += 1
                self._metrics_collector.increment_counter("cache_misses")
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired:
                self._cache.pop(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                self._stats.misses += 1
                self._stats.evictions += 1
                self._metrics_collector.increment_counter("cache_misses")
                self._metrics_collector.increment_counter("cache_evictions")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._stats.hits += 1
            self._metrics_collector.increment_counter("cache_hits")

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._stats.size_bytes -= old_entry.size_bytes
                self._stats.entry_count -= 1

            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl or self.ttl_seconds,
                size_bytes=size_bytes
            )

            # Add to cache
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes
            self._stats.entry_count += 1

            # Evict if necessary
            self._evict_lru()

            self._metrics_collector.increment_counter("cache_sets")
            self._metrics_collector.set_gauge("cache_size_entries", len(self._cache))
            self._metrics_collector.set_gauge("cache_size_bytes", self._stats.size_bytes)

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                self._metrics_collector.increment_counter("cache_deletes")
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
            self._metrics_collector.increment_counter("cache_clears")
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.to_dict()
            stats.update({
                "max_size": self.max_size,
                "max_memory_bytes": self.max_memory_bytes,
                "ttl_seconds": self.ttl_seconds,
                "current_size": len(self._cache)
            })
            return stats


class SmartCache:
    """Intelligent multi-level caching system."""

    def __init__(self):
        # Different cache levels for different data types
        self.agent_results_cache = LRUCache(max_size=500, ttl_seconds=1800, max_memory_mb=50)
        self.content_analysis_cache = LRUCache(max_size=1000, ttl_seconds=3600, max_memory_mb=30)
        self.config_cache = LRUCache(max_size=100, ttl_seconds=7200, max_memory_mb=10)
        self.metrics_cache = LRUCache(max_size=200, ttl_seconds=300, max_memory_mb=5)

        self._metrics_collector = get_metrics_collector()
        logger.info("Smart cache system initialized")

    def _generate_key(self, prefix: str, content: str, config_hash: Optional[str] = None) -> str:
        """Generate cache key for content."""
        # Create hash of content for efficient key generation
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

        if config_hash:
            return f"{prefix}:{content_hash}:{config_hash}"
        else:
            return f"{prefix}:{content_hash}"

    def get_agent_result(self, agent_type: str, content: str, config_hash: str) -> Optional[Any]:
        """Get cached agent result."""
        key = self._generate_key(f"agent_{agent_type}", content, config_hash)
        result = self.agent_results_cache.get(key)

        if result:
            logger.debug("Cache hit for agent %s", agent_type)
            self._metrics_collector.increment_counter(f"cache_hits_agent_{agent_type}")
        else:
            logger.debug("Cache miss for agent %s", agent_type)
            self._metrics_collector.increment_counter(f"cache_misses_agent_{agent_type}")

        return result

    def set_agent_result(self, agent_type: str, content: str, config_hash: str, result: Any) -> None:
        """Cache agent result."""
        key = self._generate_key(f"agent_{agent_type}", content, config_hash)
        self.agent_results_cache.set(key, result)
        logger.debug("Cached result for agent %s", agent_type)

    def get_content_analysis(self, content: str) -> Optional[Any]:
        """Get cached content analysis (validation, sanitization)."""
        key = self._generate_key("analysis", content)
        return self.content_analysis_cache.get(key)

    def set_content_analysis(self, content: str, analysis: Any) -> None:
        """Cache content analysis."""
        key = self._generate_key("analysis", content)
        self.content_analysis_cache.set(key, analysis)

    def get_config(self, config_path: str) -> Optional[Any]:
        """Get cached configuration."""
        return self.config_cache.get(config_path)

    def set_config(self, config_path: str, config: Any) -> None:
        """Cache configuration."""
        self.config_cache.set(config_path, config)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "agent_results": self.agent_results_cache.get_stats(),
            "content_analysis": self.content_analysis_cache.get_stats(),
            "config": self.config_cache.get_stats(),
            "metrics": self.metrics_cache.get_stats()
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.agent_results_cache.clear()
        self.content_analysis_cache.clear()
        self.config_cache.clear()
        self.metrics_cache.clear()
        logger.info("All caches cleared")


# Global cache instance
_smart_cache: Optional[SmartCache] = None


def get_smart_cache() -> SmartCache:
    """Get the global smart cache instance."""
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache()
    return _smart_cache


def cached_agent_operation(agent_type: str, config_hash: str, ttl: Optional[float] = None):
    """Decorator for caching agent operations."""
    def decorator(func: Callable[[str], T]) -> Callable[[str], T]:
        @wraps(func)
        def wrapper(content: str) -> T:
            cache = get_smart_cache()

            # Try to get from cache
            cached_result = cache.get_agent_result(agent_type, content, config_hash)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            start_time = time.perf_counter()
            result = func(content)
            execution_time = time.perf_counter() - start_time

            # Cache the result
            cache.set_agent_result(agent_type, content, config_hash, result)

            # Update metrics
            cache._metrics_collector.record_histogram(f"agent_{agent_type}_execution_time", execution_time)

            return result

        return wrapper
    return decorator


def cache_aware_batch_processing(items: list, processor_func: Callable, batch_size: int = 10) -> list:
    """Process items in batches with intelligent caching."""
    cache = get_smart_cache()
    results = []
    uncached_items = []
    cached_results = {}

    # First pass: check cache for all items
    for i, item in enumerate(items):
        if hasattr(item, '__hash__'):
            key = str(hash(item))
            cached_result = cache.content_analysis_cache.get(f"batch_{key}")
            if cached_result is not None:
                cached_results[i] = cached_result
            else:
                uncached_items.append((i, item))

    # Second pass: process uncached items in batches
    for i in range(0, len(uncached_items), batch_size):
        batch = uncached_items[i:i + batch_size]
        batch_items = [item for _, item in batch]

        # Process batch
        batch_results = processor_func(batch_items)

        # Cache and store results
        for (original_index, item), result in zip(batch, batch_results):
            results.append((original_index, result))

            # Cache the result
            if hasattr(item, '__hash__'):
                key = str(hash(item))
                cache.content_analysis_cache.set(f"batch_{key}", result)

    # Merge cached and processed results
    final_results = [None] * len(items)

    # Add cached results
    for index, result in cached_results.items():
        final_results[index] = result

    # Add processed results
    for index, result in results:
        final_results[index] = result

    return final_results


class PersistentCache:
    """Persistent cache that survives application restarts."""

    def __init__(self, storage_path: str = "/tmp/crewai_cache"):
        self.storage_path = storage_path
        self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Ensure storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.storage_path, f"{safe_key}.cache")

    def get(self, key: str, max_age: Optional[float] = None) -> Optional[Any]:
        """Get value from persistent cache."""
        file_path = self._get_file_path(key)

        try:
            if not os.path.exists(file_path):
                return None

            # Check age if specified
            if max_age is not None:
                file_age = time.time() - os.path.getmtime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
                    return None

            with open(file_path, 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            logger.warning("Error reading persistent cache: %s", e)
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in persistent cache."""
        file_path = self._get_file_path(key)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning("Error writing persistent cache: %s", e)

    def delete(self, key: str) -> bool:
        """Delete value from persistent cache."""
        file_path = self._get_file_path(key)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            logger.warning("Error deleting persistent cache: %s", e)

        return False

    def cleanup(self, max_age: float = 86400) -> int:
        """Clean up old cache files."""
        cleaned = 0
        current_time = time.time()

        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.storage_path, filename)
                    if current_time - os.path.getmtime(file_path) > max_age:
                        os.remove(file_path)
                        cleaned += 1
        except Exception as e:
            logger.warning("Error during cache cleanup: %s", e)

        return cleaned


# Global persistent cache instance
_persistent_cache: Optional[PersistentCache] = None


def get_persistent_cache() -> PersistentCache:
    """Get the global persistent cache instance."""
    global _persistent_cache
    if _persistent_cache is None:
        _persistent_cache = PersistentCache()
    return _persistent_cache
