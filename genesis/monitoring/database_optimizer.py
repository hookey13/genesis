"""Database performance optimization with query analysis and caching for Project GENESIS."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a database query."""
    query: str
    execution_count: int
    total_time: float
    avg_time: float
    max_time: float
    min_time: float
    cache_hits: int
    cache_misses: int
    last_executed: datetime


@dataclass
class ExplainPlan:
    """Database query execution plan."""
    query: str
    plan_text: str
    estimated_cost: float
    actual_time: Optional[float]
    recommendations: List[str]


class QueryCache:
    """LRU cache for database query results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize query cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        logger.info(
            "QueryCache initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if key in self.cache:
            value, expiry_time = self.cache[key]
            
            if time.time() < expiry_time:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return value
            else:
                # Expired
                del self.cache[key]
                self.stats["expired"] += 1
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
        
        expiry_time = time.time() + self.ttl_seconds
        self.cache[key] = (value, expiry_time)
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys
        
        Returns:
            Number of invalidated entries
        """
        if pattern is None:
            count = len(self.cache)
            self.cache.clear()
            return count
        
        # Invalidate matching keys
        keys_to_remove = [
            k for k in self.cache.keys()
            if pattern in k
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )
        
        return {
            **self.stats,
            "size": len(self.cache),
            "hit_rate": hit_rate
        }


class RedisCache:
    """Redis-based cache for distributed caching."""
    
    def __init__(
        self,
        redis_client: Any,
        prefix: str = "genesis:",
        ttl_seconds: int = 300
    ):
        """Initialize Redis cache.
        
        Args:
            redis_client: Redis client instance
            prefix: Key prefix for namespacing
            ttl_seconds: Default TTL for cache entries
        """
        self.redis = redis_client
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.stats = defaultdict(int)
        
        logger.info(
            "RedisCache initialized",
            prefix=prefix,
            ttl_seconds=ttl_seconds
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        full_key = f"{self.prefix}{key}"
        
        try:
            value = await self.redis.get(full_key)
            if value:
                self.stats["hits"] += 1
                return json.loads(value)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error("Redis get error", key=full_key, error=str(e))
            self.stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        
        Returns:
            Success status
        """
        full_key = f"{self.prefix}{key}"
        ttl = ttl or self.ttl_seconds
        
        try:
            serialized = json.dumps(value)
            await self.redis.setex(full_key, ttl, serialized)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error("Redis set error", key=full_key, error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match keys
        
        Returns:
            Number of invalidated entries
        """
        full_pattern = f"{self.prefix}{pattern}"
        
        try:
            keys = await self.redis.keys(full_pattern)
            if keys:
                count = await self.redis.delete(*keys)
                self.stats["invalidations"] += count
                return count
            return 0
        except Exception as e:
            logger.error("Redis invalidate error", pattern=full_pattern, error=str(e))
            self.stats["errors"] += 1
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate
        }


class DatabaseOptimizer:
    """Database query optimization and performance management."""
    
    def __init__(
        self,
        engine: Engine,
        slow_query_threshold: float = 0.1,
        enable_query_cache: bool = True,
        enable_redis_cache: bool = False,
        redis_client: Optional[Any] = None
    ):
        """Initialize database optimizer.
        
        Args:
            engine: SQLAlchemy engine
            slow_query_threshold: Threshold for slow queries in seconds
            enable_query_cache: Enable local query caching
            enable_redis_cache: Enable Redis caching
            redis_client: Redis client for distributed caching
        """
        self.engine = engine
        self.slow_query_threshold = slow_query_threshold
        
        # Query metrics tracking
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.slow_queries: List[Tuple[str, float, datetime]] = []
        
        # Caching
        self.local_cache = QueryCache() if enable_query_cache else None
        self.redis_cache = (
            RedisCache(redis_client) 
            if enable_redis_cache and redis_client and REDIS_AVAILABLE 
            else None
        )
        
        # Hot data identification
        self.access_frequency = defaultdict(int)
        self.hot_data_threshold = 10  # Accesses per minute
        
        # Register event listeners
        self._register_listeners()
        
        logger.info(
            "DatabaseOptimizer initialized",
            slow_query_threshold=slow_query_threshold,
            local_cache=enable_query_cache,
            redis_cache=enable_redis_cache
        )
    
    def _register_listeners(self) -> None:
        """Register SQLAlchemy event listeners for query monitoring."""
        
        @event.listens_for(Engine, "before_execute")
        def before_execute(conn, clauseelement, multiparams, params, execution_options):
            """Track query start time."""
            conn.info.setdefault('query_start_time', []).append(time.time())
        
        @event.listens_for(Engine, "after_execute")
        def after_execute(conn, clauseelement, multiparams, params, execution_options, result):
            """Track query execution time and metrics."""
            start_times = conn.info.get('query_start_time', [])
            if start_times:
                execution_time = time.time() - start_times.pop()
                query_str = str(clauseelement)
                
                # Update metrics
                self._update_query_metrics(query_str, execution_time)
                
                # Check for slow queries
                if execution_time > self.slow_query_threshold:
                    self._record_slow_query(query_str, execution_time)
        
        # Register listeners for this optimizer's engine
        event.listen(self.engine, "before_execute", before_execute)
        event.listen(self.engine, "after_execute", after_execute)
    
    def _update_query_metrics(self, query: str, execution_time: float) -> None:
        """Update metrics for a query.
        
        Args:
            query: Query string
            execution_time: Execution time in seconds
        """
        # Normalize query for metrics (remove specific values)
        normalized_query = self._normalize_query(query)
        
        if normalized_query not in self.query_metrics:
            self.query_metrics[normalized_query] = QueryMetrics(
                query=normalized_query,
                execution_count=0,
                total_time=0,
                avg_time=0,
                max_time=0,
                min_time=float('inf'),
                cache_hits=0,
                cache_misses=0,
                last_executed=datetime.now()
            )
        
        metrics = self.query_metrics[normalized_query]
        metrics.execution_count += 1
        metrics.total_time += execution_time
        metrics.avg_time = metrics.total_time / metrics.execution_count
        metrics.max_time = max(metrics.max_time, execution_time)
        metrics.min_time = min(metrics.min_time, execution_time)
        metrics.last_executed = datetime.now()
        
        # Track access frequency for hot data identification
        self.access_frequency[normalized_query] += 1
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison.
        
        Args:
            query: Original query
        
        Returns:
            Normalized query
        """
        # Simple normalization - replace values with placeholders
        import re
        
        # Replace numbers
        normalized = re.sub(r'\b\d+\b', '?', query)
        # Replace quoted strings
        normalized = re.sub(r"'[^']*'", '?', normalized)
        normalized = re.sub(r'"[^"]*"', '?', normalized)
        
        return normalized.strip()
    
    def _record_slow_query(self, query: str, execution_time: float) -> None:
        """Record a slow query.
        
        Args:
            query: Query string
            execution_time: Execution time in seconds
        """
        self.slow_queries.append((query, execution_time, datetime.now()))
        
        # Keep only recent slow queries (last 100)
        if len(self.slow_queries) > 100:
            self.slow_queries = self.slow_queries[-100:]
        
        logger.warning(
            "Slow query detected",
            query=query[:200],
            execution_time=execution_time
        )
    
    async def get_with_cache(
        self,
        query: str,
        params: Optional[Dict] = None,
        cache_key: Optional[str] = None
    ) -> Any:
        """Execute query with caching.
        
        Args:
            query: Query to execute
            params: Query parameters
            cache_key: Optional custom cache key
        
        Returns:
            Query result
        """
        # Generate cache key
        if not cache_key:
            cache_key = self._generate_cache_key(query, params)
        
        # Check local cache
        if self.local_cache:
            cached = self.local_cache.get(cache_key)
            if cached is not None:
                self._update_cache_metrics(query, hit=True)
                return cached
        
        # Check Redis cache
        if self.redis_cache:
            cached = await self.redis_cache.get(cache_key)
            if cached is not None:
                self._update_cache_metrics(query, hit=True)
                # Update local cache
                if self.local_cache:
                    self.local_cache.set(cache_key, cached)
                return cached
        
        # Execute query
        self._update_cache_metrics(query, hit=False)
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            data = result.fetchall()
        
        # Cache result
        if self.local_cache:
            self.local_cache.set(cache_key, data)
        
        if self.redis_cache:
            await self.redis_cache.set(cache_key, data)
        
        return data
    
    def _generate_cache_key(self, query: str, params: Optional[Dict]) -> str:
        """Generate cache key for query.
        
        Args:
            query: Query string
            params: Query parameters
        
        Returns:
            Cache key
        """
        key_data = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_cache_metrics(self, query: str, hit: bool) -> None:
        """Update cache metrics for a query.
        
        Args:
            query: Query string
            hit: Whether it was a cache hit
        """
        normalized_query = self._normalize_query(query)
        
        if normalized_query in self.query_metrics:
            metrics = self.query_metrics[normalized_query]
            if hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1
    
    async def analyze_query(self, query: str) -> ExplainPlan:
        """Analyze query execution plan.
        
        Args:
            query: Query to analyze
        
        Returns:
            Execution plan analysis
        """
        recommendations = []
        
        # Get EXPLAIN plan
        with self.engine.connect() as conn:
            # SQLite EXPLAIN QUERY PLAN
            if 'sqlite' in self.engine.dialect.name:
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                result = conn.execute(text(explain_query))
                plan_rows = result.fetchall()
                plan_text = "\n".join(str(row) for row in plan_rows)
                
                # Parse for optimization opportunities
                if "SCAN" in plan_text and "INDEX" not in plan_text:
                    recommendations.append("Consider adding an index for better performance")
                
                if "TEMP B-TREE" in plan_text:
                    recommendations.append("Query requires sorting - consider an appropriate index")
            
            # PostgreSQL EXPLAIN ANALYZE
            elif 'postgresql' in self.engine.dialect.name:
                explain_query = f"EXPLAIN ANALYZE {query}"
                result = conn.execute(text(explain_query))
                plan_rows = result.fetchall()
                plan_text = "\n".join(str(row) for row in plan_rows)
                
                # Parse for optimization opportunities
                if "Seq Scan" in plan_text:
                    recommendations.append("Sequential scan detected - consider adding an index")
                
                if "Sort" in plan_text:
                    recommendations.append("Sort operation detected - consider an ordered index")
                
                if "Hash Join" in plan_text and "rows=" in plan_text:
                    # Check for large hash joins
                    import re
                    rows_match = re.search(r'rows=(\d+)', plan_text)
                    if rows_match and int(rows_match.group(1)) > 10000:
                        recommendations.append("Large hash join detected - consider join optimization")
        
        # Analyze from metrics
        normalized_query = self._normalize_query(query)
        if normalized_query in self.query_metrics:
            metrics = self.query_metrics[normalized_query]
            
            if metrics.avg_time > self.slow_query_threshold:
                recommendations.append(f"Query is slow (avg: {metrics.avg_time:.3f}s)")
            
            if metrics.cache_hits == 0 and metrics.execution_count > 10:
                recommendations.append("Frequently executed query - consider caching")
        
        return ExplainPlan(
            query=query,
            plan_text=plan_text,
            estimated_cost=0.0,  # Would need to parse from plan
            actual_time=None,
            recommendations=recommendations
        )
    
    def identify_hot_data(self, time_window_minutes: int = 5) -> List[str]:
        """Identify hot (frequently accessed) data.
        
        Args:
            time_window_minutes: Time window for analysis
        
        Returns:
            List of hot data queries
        """
        hot_queries = []
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        for query, metrics in self.query_metrics.items():
            if metrics.last_executed > cutoff_time:
                access_rate = metrics.execution_count / time_window_minutes
                if access_rate > self.hot_data_threshold:
                    hot_queries.append(query)
        
        return hot_queries
    
    async def optimize_schema(self) -> List[str]:
        """Generate schema optimization recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze slow queries for patterns
        slow_patterns = defaultdict(int)
        for query, _, _ in self.slow_queries:
            # Extract table names
            import re
            tables = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
            for table in tables:
                slow_patterns[table] += 1
        
        # Recommend indexes for frequently slow tables
        for table, count in slow_patterns.items():
            if count > 5:
                recommendations.append(
                    f"Table '{table}' appears in {count} slow queries - review indexes"
                )
        
        # Check cache effectiveness
        if self.local_cache:
            cache_stats = self.local_cache.get_stats()
            if cache_stats["hit_rate"] < 0.5 and cache_stats["hits"] + cache_stats["misses"] > 100:
                recommendations.append(
                    f"Low cache hit rate ({cache_stats['hit_rate']:.2%}) - review caching strategy"
                )
        
        # Identify queries that could benefit from caching
        for query, metrics in self.query_metrics.items():
            if metrics.execution_count > 100 and metrics.cache_hits == 0:
                recommendations.append(
                    f"Frequently executed query with no caching: {query[:100]}"
                )
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Returns:
            Performance report
        """
        # Calculate statistics
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        total_time = sum(m.total_time for m in self.query_metrics.values())
        
        # Get top slow queries
        top_slow = sorted(
            self.query_metrics.values(),
            key=lambda m: m.avg_time,
            reverse=True
        )[:10]
        
        # Get most frequent queries
        most_frequent = sorted(
            self.query_metrics.values(),
            key=lambda m: m.execution_count,
            reverse=True
        )[:10]
        
        report = {
            "summary": {
                "total_queries": total_queries,
                "total_time_seconds": total_time,
                "unique_queries": len(self.query_metrics),
                "slow_queries_recorded": len(self.slow_queries)
            },
            "top_slow_queries": [
                {
                    "query": q.query[:200],
                    "avg_time": q.avg_time,
                    "max_time": q.max_time,
                    "count": q.execution_count
                }
                for q in top_slow
            ],
            "most_frequent_queries": [
                {
                    "query": q.query[:200],
                    "count": q.execution_count,
                    "total_time": q.total_time,
                    "cache_hit_rate": (
                        q.cache_hits / (q.cache_hits + q.cache_misses)
                        if (q.cache_hits + q.cache_misses) > 0
                        else 0
                    )
                }
                for q in most_frequent
            ],
            "cache_stats": {
                "local": self.local_cache.get_stats() if self.local_cache else None,
                "redis": self.redis_cache.get_stats() if self.redis_cache else None
            },
            "hot_data": self.identify_hot_data()
        }
        
        return report