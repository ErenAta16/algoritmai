"""
Advanced Cache Optimizer and Database Performance Enhancement
"""

import redis
import pickle
import json
import hashlib
import zlib
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from threading import Lock, RLock
from collections import defaultdict, OrderedDict
import asyncio
import aioredis
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl: int
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl <= 0:
            return False
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def touch(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = datetime.now()

class MultiLevelCache:
    """
    Multi-level caching system with L1 (memory), L2 (Redis), and intelligent eviction
    """
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l2_redis_url: str = None,
                 compression_threshold: int = 1024,
                 default_ttl: int = 3600):
        
        # L1 Cache (In-Memory)
        self.l1_cache = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l1_lock = RLock()
        
        # L2 Cache (Redis)
        self.l2_enabled = False
        self.l2_client = None
        self.l2_async_client = None
        
        if l2_redis_url:
            try:
                self.l2_client = redis.from_url(l2_redis_url)
                self.l2_enabled = True
                logger.info("✅ L2 Cache (Redis) enabled")
            except Exception as e:
                logger.warning(f"⚠️ L2 Cache (Redis) failed to connect: {str(e)}")
        
        # Cache configuration
        self.compression_threshold = compression_threshold
        self.default_ttl = default_ttl
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'total_sets': 0,
            'total_gets': 0,
            'evictions': 0,
            'compressions': 0
        }
        
        # Prefetch patterns
        self.prefetch_patterns = {}
        self.access_patterns = defaultdict(list)
        
        # Background tasks
        self.cleanup_enabled = True
        self.cleanup_interval = 300  # 5 minutes
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background cleanup and optimization tasks"""
        import threading
        
        def cleanup_loop():
            while self.cleanup_enabled:
                try:
                    self._cleanup_expired_entries()
                    self._optimize_cache_layout()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"❌ Cache cleanup error: {str(e)}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        try:
            with self.l1_lock:
                expired_keys = []
                for key, entry in self.l1_cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.l1_cache[key]
                    self.stats['evictions'] += 1
                
                if expired_keys:
                    logger.info(f"🧹 Cleaned up {len(expired_keys)} expired L1 cache entries")
        
        except Exception as e:
            logger.error(f"❌ L1 cleanup error: {str(e)}")
    
    def _optimize_cache_layout(self):
        """Optimize cache layout based on access patterns"""
        try:
            with self.l1_lock:
                # Sort by access frequency and recency
                sorted_entries = sorted(
                    self.l1_cache.items(),
                    key=lambda x: (x[1].access_count, x[1].last_accessed or x[1].timestamp),
                    reverse=True
                )
                
                # Rebuild cache with optimized order
                self.l1_cache.clear()
                for key, entry in sorted_entries:
                    self.l1_cache[key] = entry
        
        except Exception as e:
            logger.error(f"❌ Cache optimization error: {str(e)}")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize and optionally compress value"""
        try:
            # Serialize
            serialized = pickle.dumps(value)
            
            # Compress if above threshold
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized):
                    self.stats['compressions'] += 1
                    return compressed, True
            
            return serialized, False
            
        except Exception as e:
            logger.error(f"❌ Serialization error: {str(e)}")
            return None, False
    
    def _deserialize_value(self, data: bytes, compressed: bool = False) -> Any:
        """Deserialize and decompress value"""
        try:
            if compressed:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"❌ Deserialization error: {str(e)}")
            return None
    
    def _generate_cache_key(self, key: str, namespace: str = None) -> str:
        """Generate cache key with namespace"""
        if namespace:
            return f"{namespace}:{key}"
        return key
    
    def get(self, key: str, namespace: str = None) -> Optional[Any]:
        """Get value from multi-level cache"""
        cache_key = self._generate_cache_key(key, namespace)
        self.stats['total_gets'] += 1
        
        # Try L1 cache first
        with self.l1_lock:
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats['l1_hits'] += 1
                    
                    # Move to end (LRU)
                    self.l1_cache.move_to_end(cache_key)
                    
                    # Record access pattern
                    self._record_access_pattern(cache_key)
                    
                    return entry.value
                else:
                    # Remove expired entry
                    del self.l1_cache[cache_key]
                    self.stats['evictions'] += 1
        
        self.stats['l1_misses'] += 1
        
        # Try L2 cache (Redis)
        if self.l2_enabled:
            try:
                l2_data = self.l2_client.get(cache_key)
                if l2_data:
                    # Deserialize from L2
                    entry_data = json.loads(l2_data)
                    value = self._deserialize_value(
                        entry_data['value'].encode('latin-1'),
                        entry_data['compressed']
                    )
                    
                    if value is not None:
                        self.stats['l2_hits'] += 1
                        
                        # Promote to L1
                        self._set_l1(cache_key, value, entry_data['ttl'])
                        
                        return value
            except Exception as e:
                logger.error(f"❌ L2 cache get error: {str(e)}")
        
        self.stats['l2_misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None, namespace: str = None):
        """Set value in multi-level cache"""
        cache_key = self._generate_cache_key(key, namespace)
        ttl = ttl or self.default_ttl
        self.stats['total_sets'] += 1
        
        # Set in L1
        self._set_l1(cache_key, value, ttl)
        
        # Set in L2 if enabled
        if self.l2_enabled:
            self._set_l2(cache_key, value, ttl)
    
    def _set_l1(self, key: str, value: Any, ttl: int):
        """Set value in L1 cache"""
        try:
            with self.l1_lock:
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=datetime.now(),
                    ttl=ttl,
                    last_accessed=datetime.now()
                )
                
                # Calculate size (approximate)
                try:
                    entry.size_bytes = len(pickle.dumps(value))
                except:
                    entry.size_bytes = 0
                
                # Add to cache
                self.l1_cache[key] = entry
                
                # Move to end (most recently used)
                self.l1_cache.move_to_end(key)
                
                # Evict if over capacity
                while len(self.l1_cache) > self.l1_max_size:
                    oldest_key = next(iter(self.l1_cache))
                    del self.l1_cache[oldest_key]
                    self.stats['evictions'] += 1
        
        except Exception as e:
            logger.error(f"❌ L1 cache set error: {str(e)}")
    
    def _set_l2(self, key: str, value: Any, ttl: int):
        """Set value in L2 cache"""
        try:
            serialized_data, compressed = self._serialize_value(value)
            if serialized_data:
                entry_data = {
                    'value': serialized_data.decode('latin-1'),
                    'compressed': compressed,
                    'ttl': ttl,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.l2_client.setex(
                    key,
                    ttl,
                    json.dumps(entry_data)
                )
        
        except Exception as e:
            logger.error(f"❌ L2 cache set error: {str(e)}")
    
    def _record_access_pattern(self, key: str):
        """Record access pattern for prefetching"""
        try:
            now = datetime.now()
            self.access_patterns[key].append(now)
            
            # Keep only last 100 accesses
            if len(self.access_patterns[key]) > 100:
                self.access_patterns[key] = self.access_patterns[key][-100:]
        
        except Exception as e:
            logger.error(f"❌ Access pattern recording error: {str(e)}")
    
    def delete(self, key: str, namespace: str = None):
        """Delete from all cache levels"""
        cache_key = self._generate_cache_key(key, namespace)
        
        # Delete from L1
        with self.l1_lock:
            if cache_key in self.l1_cache:
                del self.l1_cache[cache_key]
        
        # Delete from L2
        if self.l2_enabled:
            try:
                self.l2_client.delete(cache_key)
            except Exception as e:
                logger.error(f"❌ L2 cache delete error: {str(e)}")
    
    def clear(self, namespace: str = None):
        """Clear cache (optionally by namespace)"""
        if namespace:
            # Clear specific namespace
            pattern = f"{namespace}:*"
            keys_to_delete = []
            
            with self.l1_lock:
                for key in list(self.l1_cache.keys()):
                    if key.startswith(f"{namespace}:"):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.l1_cache[key]
            
            # Clear from L2
            if self.l2_enabled:
                try:
                    keys = self.l2_client.keys(pattern)
                    if keys:
                        self.l2_client.delete(*keys)
                except Exception as e:
                    logger.error(f"❌ L2 namespace clear error: {str(e)}")
        else:
            # Clear all
            with self.l1_lock:
                self.l1_cache.clear()
            
            if self.l2_enabled:
                try:
                    self.l2_client.flushdb()
                except Exception as e:
                    logger.error(f"❌ L2 full clear error: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.l1_lock:
            l1_size = len(self.l1_cache)
            l1_memory_usage = sum(entry.size_bytes for entry in self.l1_cache.values())
        
        total_requests = self.stats['total_gets']
        total_hits = self.stats['l1_hits'] + self.stats['l2_hits']
        
        return {
            'l1_size': l1_size,
            'l1_memory_usage_bytes': l1_memory_usage,
            'l1_hit_rate': (self.stats['l1_hits'] / total_requests * 100) if total_requests > 0 else 0,
            'l2_hit_rate': (self.stats['l2_hits'] / total_requests * 100) if total_requests > 0 else 0,
            'overall_hit_rate': (total_hits / total_requests * 100) if total_requests > 0 else 0,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'evictions': self.stats['evictions'],
            'compressions': self.stats['compressions'],
            'l2_enabled': self.l2_enabled,
            **self.stats
        }

class QueryOptimizer:
    """
    Database query optimization and caching
    """
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'avg_time': 0})
        self.slow_query_threshold = 1.0  # seconds
        self.slow_queries = []
        
    def cached_query(self, query_key: str, ttl: int = 3600):
        """Decorator for caching database queries"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"query:{query_key}:{hashlib.md5(str(args).encode()).hexdigest()}"
                
                # Try cache first
                cached_result = self.cache.get(cache_key, namespace="queries")
                if cached_result is not None:
                    return cached_result
                
                # Execute query
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update statistics
                    self.query_stats[query_key]['count'] += 1
                    self.query_stats[query_key]['total_time'] += execution_time
                    self.query_stats[query_key]['avg_time'] = (
                        self.query_stats[query_key]['total_time'] / 
                        self.query_stats[query_key]['count']
                    )
                    
                    # Check for slow queries
                    if execution_time > self.slow_query_threshold:
                        self.slow_queries.append({
                            'query_key': query_key,
                            'execution_time': execution_time,
                            'timestamp': datetime.now(),
                            'args': str(args)[:200]  # Limit size
                        })
                        
                        # Keep only last 100 slow queries
                        if len(self.slow_queries) > 100:
                            self.slow_queries = self.slow_queries[-100:]
                    
                    # Cache result
                    self.cache.set(cache_key, result, ttl, namespace="queries")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"❌ Query error for {query_key}: {str(e)} (took {execution_time:.3f}s)")
                    raise
            
            return wrapper
        return decorator
    
    def get_query_stats(self) -> Dict:
        """Get query performance statistics"""
        return {
            'query_statistics': dict(self.query_stats),
            'slow_queries': self.slow_queries[-10:],  # Last 10 slow queries
            'total_queries': sum(stats['count'] for stats in self.query_stats.values()),
            'average_query_time': sum(stats['avg_time'] for stats in self.query_stats.values()) / len(self.query_stats) if self.query_stats else 0
        }

# Global cache optimizer instance
cache_optimizer = MultiLevelCache(
    l1_max_size=1000,
    l2_redis_url=None,  # Set to Redis URL if available
    compression_threshold=1024,
    default_ttl=3600
)

# Global query optimizer instance
query_optimizer = QueryOptimizer(cache_optimizer) 