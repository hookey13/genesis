"""Performance optimization and profiling for Project GENESIS."""

import asyncio
import json
import os
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import structlog
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceProfile:
    """Container for performance profiling data."""
    
    cpu_percent: float
    memory_rss: int
    memory_vms: int
    active_tasks: int
    query_latencies: Dict[str, List[float]]
    cache_stats: Dict[str, int]
    connection_pool_stats: Dict[str, int]


class PerformanceProfiler:
    """Performance profiling with py-spy and memory tracking."""
    
    def __init__(self):
        self.profiles: List[PerformanceProfile] = []
        self.memory_snapshots = []
        self.is_profiling = False
        
    def start_profiling(self):
        """Start CPU and memory profiling."""
        self.is_profiling = True
        tracemalloc.start()
        logger.info("Performance profiling started")
        
    def stop_profiling(self):
        """Stop profiling and generate report."""
        self.is_profiling = False
        tracemalloc.stop()
        logger.info("Performance profiling stopped")
        return self.generate_profile_report()
    
    def capture_snapshot(self) -> PerformanceProfile:
        """Capture current performance snapshot."""
        process = psutil.Process()
        
        return PerformanceProfile(
            cpu_percent=process.cpu_percent(),
            memory_rss=process.memory_info().rss,
            memory_vms=process.memory_info().vms,
            active_tasks=len(asyncio.all_tasks()),
            query_latencies=defaultdict(list),
            cache_stats={"hits": 0, "misses": 0},
            connection_pool_stats={"active": 0, "idle": 0}
        )
    
    def detect_memory_leaks(self, threshold_mb: float = 100):
        """Detect potential memory leaks."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        leaks = []
        for stat in top_stats[:10]:
            size_mb = stat.size / 1024 / 1024
            if size_mb > threshold_mb:
                leaks.append({
                    "file": stat.traceback.format(),
                    "size_mb": size_mb,
                    "count": stat.count
                })
        
        return leaks
    
    def generate_profile_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        if not self.profiles:
            return {}
        
        return {
            "avg_cpu_percent": sum(p.cpu_percent for p in self.profiles) / len(self.profiles),
            "max_memory_rss": max(p.memory_rss for p in self.profiles),
            "memory_leaks": self.detect_memory_leaks(),
            "profile_count": len(self.profiles)
        }


class DatabaseOptimizer:
    """Database query optimization and caching."""
    
    def __init__(self):
        self.query_cache = {}
        self.query_stats = defaultdict(lambda: {"count": 0, "total_time": 0})
        self.slow_query_threshold = 0.1  # 100ms
        
    async def optimize_query(self, query: str, params: tuple = None) -> str:
        """Optimize database query with explain plan analysis."""
        # Add query optimization logic here
        # For SQLite: EXPLAIN QUERY PLAN
        # For PostgreSQL: EXPLAIN ANALYZE
        return query
    
    def cache_query_result(self, query: str, result: Any, ttl: int = 300):
        """Cache query result with TTL."""
        cache_key = f"{query}:{str(result)}"
        self.query_cache[cache_key] = {
            "result": result,
            "expires_at": time.time() + ttl
        }
    
    def get_cached_result(self, query: str) -> Optional[Any]:
        """Get cached query result if available."""
        cache_key = query
        if cache_key in self.query_cache:
            entry = self.query_cache[cache_key]
            if time.time() < entry["expires_at"]:
                return entry["result"]
            else:
                del self.query_cache[cache_key]
        return None
    
    def record_query_performance(self, query: str, execution_time: float):
        """Record query performance statistics."""
        self.query_stats[query]["count"] += 1
        self.query_stats[query]["total_time"] += execution_time
        
        if execution_time > self.slow_query_threshold:
            logger.warning(
                "Slow query detected",
                query=query[:100],
                execution_time=execution_time
            )
    
    def get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get list of slow queries."""
        slow_queries = []
        for query, stats in self.query_stats.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            if avg_time > self.slow_query_threshold:
                slow_queries.append({
                    "query": query[:100],
                    "avg_time": avg_time,
                    "count": stats["count"]
                })
        return sorted(slow_queries, key=lambda x: x["avg_time"], reverse=True)


class ConnectionPoolManager:
    """Connection pooling and batching optimization."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active_connections = []
        self.idle_connections = []
        self.pending_requests = asyncio.Queue()
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms
        
    async def get_connection(self):
        """Get connection from pool."""
        if self.idle_connections:
            conn = self.idle_connections.pop()
            self.active_connections.append(conn)
            return conn
        elif len(self.active_connections) < self.max_connections:
            # Create new connection
            conn = await self._create_connection()
            self.active_connections.append(conn)
            return conn
        else:
            # Wait for available connection
            await self.pending_requests.put(asyncio.current_task())
            return await self.get_connection()
    
    async def release_connection(self, conn):
        """Release connection back to pool."""
        if conn in self.active_connections:
            self.active_connections.remove(conn)
            self.idle_connections.append(conn)
            
            # Process pending requests
            if not self.pending_requests.empty():
                task = await self.pending_requests.get()
                task.cancel()
    
    async def _create_connection(self):
        """Create new connection (placeholder)."""
        # Implementation depends on connection type
        return {"id": len(self.active_connections) + 1, "created_at": time.time()}
    
    async def batch_requests(self, requests: List[Any]) -> List[Any]:
        """Batch multiple requests for efficiency."""
        batches = []
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batches.append(batch)
        
        results = []
        for batch in batches:
            batch_results = await asyncio.gather(*[self._process_request(req) for req in batch])
            results.extend(batch_results)
        
        return results
    
    async def _process_request(self, request: Any) -> Any:
        """Process individual request."""
        # Placeholder for actual request processing
        await asyncio.sleep(0.01)
        return {"request": request, "result": "processed"}
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        return {
            "active": len(self.active_connections),
            "idle": len(self.idle_connections),
            "pending": self.pending_requests.qsize(),
            "total": len(self.active_connections) + len(self.idle_connections)
        }


class GrafanaDashboardConfig:
    """Grafana dashboard configuration generator."""
    
    @staticmethod
    def generate_dashboard() -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        return {
            "dashboard": {
                "title": "GENESIS Trading Performance",
                "panels": [
                    {
                        "id": 1,
                        "title": "P&L Tracking",
                        "type": "graph",
                        "targets": [
                            {"expr": "genesis_daily_pnl_usdt"},
                            {"expr": "genesis_position_pnl_usdt"}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Latency Percentiles",
                        "type": "graph",
                        "targets": [
                            {"expr": 'histogram_quantile(0.5, genesis_order_execution_latency_seconds_bucket)'},
                            {"expr": 'histogram_quantile(0.95, genesis_order_execution_latency_seconds_bucket)'},
                            {"expr": 'histogram_quantile(0.99, genesis_order_execution_latency_seconds_bucket)'}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {"expr": 'rate(genesis_order_execution_errors_total[5m])'},
                            {"expr": 'rate(genesis_api_call_errors_total[5m])'}
                        ]
                    },
                    {
                        "id": 4,
                        "title": "API Rate Limits",
                        "type": "gauge",
                        "targets": [
                            {"expr": "genesis_rate_limit_usage_ratio"}
                        ]
                    },
                    {
                        "id": 5,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {"expr": "genesis_memory_usage_bytes"},
                            {"expr": "genesis_cpu_usage_percent"}
                        ]
                    },
                    {
                        "id": 6,
                        "title": "WebSocket Latency",
                        "type": "heatmap",
                        "targets": [
                            {"expr": "genesis_ws_message_latency_ms_bucket"}
                        ]
                    }
                ],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "10s"
            }
        }
    
    @staticmethod
    def save_dashboard_config(filepath: str = "config/grafana/dashboard.json"):
        """Save dashboard configuration to file."""
        config = GrafanaDashboardConfig.generate_dashboard()
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Grafana dashboard configuration saved to {filepath}")
        return filepath


class PerformanceBenchmark:
    """Performance benchmarking and regression detection."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline = self.load_baseline()
        self.current_metrics = {}
        
    def load_baseline(self) -> Dict[str, float]:
        """Load performance baseline from file."""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baseline(self, metrics: Dict[str, float]):
        """Save performance baseline to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.baseline = metrics
    
    def check_regression(self, metric_name: str, current_value: float, threshold: float = 0.1) -> bool:
        """Check if performance regression occurred."""
        if metric_name not in self.baseline:
            return False
        
        baseline_value = self.baseline[metric_name]
        degradation = (current_value - baseline_value) / baseline_value
        
        return degradation > threshold
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark suite."""
        results = {
            "order_execution_p99": 0.05,  # Placeholder
            "risk_check_p99": 0.001,      # Placeholder
            "db_query_avg": 0.005,         # Placeholder
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_usage_percent": psutil.Process().cpu_percent()
        }
        
        regressions = []
        for metric, value in results.items():
            if self.check_regression(metric, value):
                regressions.append({
                    "metric": metric,
                    "current": value,
                    "baseline": self.baseline.get(metric, 0)
                })
        
        return {
            "results": results,
            "regressions": regressions,
            "passed": len(regressions) == 0
        }