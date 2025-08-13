"""Intelligent caching system with adaptive strategies."""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1

class IntelligentCache:
    """High-performance cache with adaptive eviction strategies."""

    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }

        # Adaptive strategy parameters
        self._access_patterns = {}
        self._strategy_performance = {s: {"hits": 0, "misses": 0} for s in CacheStrategy}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                self._evict(key)
                self._stats["misses"] += 1
                return None

            # Update access information
            entry.touch()

            # Move to end for LRU
            if self.strategy in (CacheStrategy.LRU, CacheStrategy.ADAPTIVE):
                self._cache.move_to_end(key)

            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size estimate
            size_bytes = self._estimate_size(value)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats["size_bytes"] -= old_entry.size_bytes
                del self._cache[key]

            # Add new entry
            self._cache[key] = entry
            self._stats["size_bytes"] += size_bytes

            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_one()

    def _evict(self, key: str) -> None:
        """Evict specific key."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats["size_bytes"] -= entry.size_bytes
            del self._cache[key]
            self._stats["evictions"] += 1

    def _evict_one(self) -> None:
        """Evict one entry based on strategy."""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            key = min(self._cache.keys(),
                     key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Find expired entries first, then oldest
            expired_keys = [k for k, e in self._cache.items() if e.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self._cache.keys(),
                         key=lambda k: self._cache[k].created_at)
        else:  # ADAPTIVE
            key = self._adaptive_eviction()

        self._evict(key)

    def _adaptive_eviction(self) -> str:
        """Adaptive eviction strategy based on access patterns."""
        # Analyze access patterns and choose best strategy
        current_time = time.time()

        # Score entries based on multiple factors
        scores = {}
        for key, entry in self._cache.items():
            age_factor = current_time - entry.created_at
            recency_factor = current_time - entry.last_accessed
            frequency_factor = 1.0 / (entry.access_count + 1)

            # Composite score (lower is better for eviction)
            score = (age_factor * 0.3 + recency_factor * 0.5 + frequency_factor * 0.2)
            scores[key] = score

        # Return key with highest score (most eviction-worthy)
        return max(scores.keys(), key=lambda k: scores[k])

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v)
                          for k, v in value.items())
            else:
                return 1000  # Default estimate

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats["size_bytes"] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]) \
                      if (self._stats["hits"] + self._stats["misses"]) > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "size_bytes": self._stats["size_bytes"],
                "strategy": self.strategy.value
            }

class CacheManager:
    """Manages multiple caches with different strategies."""

    def __init__(self):
        self._caches = {}
        self._default_cache = IntelligentCache()

    def get_cache(self, name: str = "default", **kwargs) -> IntelligentCache:
        """Get or create named cache."""
        if name == "default":
            return self._default_cache

        if name not in self._caches:
            self._caches[name] = IntelligentCache(**kwargs)

        return self._caches[name]

    def cached_function(self, cache_name: str = "default", ttl: Optional[float] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            cache = self.get_cache(cache_name)

            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key_data = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

                # Try to get from cache
                result = cache.get(cache_key)
                if result is not None:
                    return result

                # Compute and cache result
                result = func(*args, **kwargs)
                cache.put(cache_key, result, ttl)
                return result

            wrapper._cache = cache
            wrapper._cache_key_func = lambda *args, **kwargs: \
                hashlib.md5(f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}".encode()).hexdigest()

            return wrapper
        return decorator

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {"default": self._default_cache.get_stats()}
        stats.update({name: cache.get_stats() for name, cache in self._caches.items()})
        return stats

# Global cache manager
_cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    return _cache_manager

def get_cache(name: str = "default") -> IntelligentCache:
    """Get cache by name."""
    return _cache_manager.get_cache(name)

def cached(cache_name: str = "default", ttl: Optional[float] = None):
    """Decorator for caching function results."""
    return _cache_manager.cached_function(cache_name, ttl)
