"""
Connection pool monitoring and alerting for PgBouncer.
Tracks pool metrics, detects issues, and provides performance insights.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .postgres_manager import PostgresManager

logger = logging.getLogger(__name__)


class PoolMetrics:
    """Container for connection pool metrics."""
    
    def __init__(self):
        self.timestamp = datetime.utcnow()
        self.total_connections = 0
        self.active_connections = 0
        self.idle_connections = 0
        self.waiting_clients = 0
        self.total_requests = 0
        self.total_received = 0
        self.total_sent = 0
        self.avg_query_time = 0.0
        self.max_query_time = 0.0
        self.pool_hit_rate = 0.0
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'idle_connections': self.idle_connections,
            'waiting_clients': self.waiting_clients,
            'total_requests': self.total_requests,
            'total_received': self.total_received,
            'total_sent': self.total_sent,
            'avg_query_time': self.avg_query_time,
            'max_query_time': self.max_query_time,
            'pool_hit_rate': self.pool_hit_rate,
            'connection_utilization': (
                self.active_connections / self.total_connections * 100
                if self.total_connections > 0 else 0
            )
        }


class PoolMonitor:
    """
    Monitors PgBouncer connection pool performance.
    Provides metrics collection, alerting, and optimization recommendations.
    """
    
    # Alert thresholds
    THRESHOLDS = {
        'connection_utilization': 80,  # Percentage
        'waiting_clients': 10,
        'avg_query_time': 100,  # Milliseconds
        'max_query_time': 5000,  # Milliseconds
        'pool_hit_rate': 95,  # Percentage
    }
    
    def __init__(
        self,
        db_manager: PostgresManager,
        pgbouncer_host: str = "localhost",
        pgbouncer_port: int = 6432
    ):
        """
        Initialize pool monitor.
        
        Args:
            db_manager: PostgreSQL connection manager
            pgbouncer_host: PgBouncer host
            pgbouncer_port: PgBouncer admin port
        """
        self.db = db_manager
        self.pgbouncer_host = pgbouncer_host
        self.pgbouncer_port = pgbouncer_port
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_history: List[PoolMetrics] = []
        self._max_history = 1440  # 24 hours at 1-minute intervals
        
    async def get_pgbouncer_stats(self) -> Dict[str, any]:
        """
        Fetch statistics from PgBouncer admin interface.
        
        Returns:
            Dictionary of PgBouncer statistics
        """
        try:
            # Try to use PgBouncer admin interface
            from .pgbouncer_admin import PgBouncerAdmin
            
            admin = PgBouncerAdmin(
                host=self.pgbouncer_host,
                port=self.pgbouncer_port
            )
            
            await admin.connect()
            stats_list = await admin.get_stats()
            await admin.disconnect()
            
            # Find stats for genesis_trading database
            for stats in stats_list:
                if stats['database'] == 'genesis_trading':
                    return stats
                    
            return {}
                
        except Exception as e:
            logger.warning(f"Could not fetch PgBouncer stats: {e}")
            # Fallback to application-level pool stats
            return await self.db.get_pool_stats()
            
    async def get_pool_metrics(self) -> PoolMetrics:
        """
        Collect current pool metrics.
        
        Returns:
            PoolMetrics object with current statistics
        """
        metrics = PoolMetrics()
        
        # Get PgBouncer stats
        stats = await self.get_pgbouncer_stats()
        
        # Get application pool stats
        app_stats = await self.db.get_pool_stats()
        
        # Combine metrics
        metrics.total_connections = app_stats.get('max_size', 0)
        metrics.active_connections = (
            app_stats.get('size', 0) - app_stats.get('free_size', 0)
        )
        metrics.idle_connections = app_stats.get('free_size', 0)
        
        # From PgBouncer stats
        if stats:
            metrics.total_requests = stats.get('total_query_count', 0)
            metrics.total_received = stats.get('total_received', 0)
            metrics.total_sent = stats.get('total_sent', 0)
            metrics.avg_query_time = stats.get('avg_query_time', 0)
            
        # Calculate hit rate
        if metrics.total_requests > 0:
            metrics.pool_hit_rate = (
                (metrics.total_requests - metrics.waiting_clients) /
                metrics.total_requests * 100
            )
            
        return metrics
        
    async def check_pool_health(self) -> Tuple[bool, List[str]]:
        """
        Check pool health against thresholds.
        
        Returns:
            Tuple of (is_healthy, list of issues)
        """
        metrics = await self.get_pool_metrics()
        issues = []
        
        # Check connection utilization
        utilization = (
            metrics.active_connections / metrics.total_connections * 100
            if metrics.total_connections > 0 else 0
        )
        if utilization > self.THRESHOLDS['connection_utilization']:
            issues.append(
                f"High connection utilization: {utilization:.1f}% "
                f"(threshold: {self.THRESHOLDS['connection_utilization']}%)"
            )
            
        # Check waiting clients
        if metrics.waiting_clients > self.THRESHOLDS['waiting_clients']:
            issues.append(
                f"Too many waiting clients: {metrics.waiting_clients} "
                f"(threshold: {self.THRESHOLDS['waiting_clients']})"
            )
            
        # Check query times
        if metrics.avg_query_time > self.THRESHOLDS['avg_query_time']:
            issues.append(
                f"High average query time: {metrics.avg_query_time:.1f}ms "
                f"(threshold: {self.THRESHOLDS['avg_query_time']}ms)"
            )
            
        if metrics.max_query_time > self.THRESHOLDS['max_query_time']:
            issues.append(
                f"Very high max query time: {metrics.max_query_time:.1f}ms "
                f"(threshold: {self.THRESHOLDS['max_query_time']}ms)"
            )
            
        # Check pool hit rate
        if metrics.pool_hit_rate < self.THRESHOLDS['pool_hit_rate']:
            issues.append(
                f"Low pool hit rate: {metrics.pool_hit_rate:.1f}% "
                f"(threshold: {self.THRESHOLDS['pool_hit_rate']}%)"
            )
            
        is_healthy = len(issues) == 0
        return is_healthy, issues
        
    async def get_connection_details(self) -> List[Dict]:
        """
        Get detailed information about active connections.
        
        Returns:
            List of connection details
        """
        query = """
        SELECT
            pid,
            usename,
            application_name,
            client_addr,
            state,
            query_start,
            state_change,
            wait_event_type,
            wait_event,
            backend_type,
            query
        FROM pg_stat_activity
        WHERE datname = 'genesis_trading'
        AND state != 'idle'
        ORDER BY query_start DESC
        """
        
        rows = await self.db.fetch(query)
        return [dict(row) for row in rows]
        
    async def get_slow_queries(
        self,
        threshold_ms: float = 1000
    ) -> List[Dict]:
        """
        Get queries running longer than threshold.
        
        Args:
            threshold_ms: Query time threshold in milliseconds
            
        Returns:
            List of slow queries
        """
        query = """
        SELECT
            pid,
            usename,
            application_name,
            state,
            query_start,
            EXTRACT(EPOCH FROM (NOW() - query_start)) * 1000 as duration_ms,
            query
        FROM pg_stat_activity
        WHERE datname = 'genesis_trading'
        AND state != 'idle'
        AND query_start < NOW() - INTERVAL '%s milliseconds'
        ORDER BY query_start
        """
        
        rows = await self.db.fetch(query % threshold_ms)
        return [dict(row) for row in rows]
        
    async def recommend_pool_size(self) -> Dict[str, int]:
        """
        Recommend optimal pool size based on usage patterns.
        
        Returns:
            Dictionary with recommended settings
        """
        if not self._metrics_history:
            return {
                'min_pool_size': 10,
                'default_pool_size': 50,
                'max_client_conn': 1000
            }
            
        # Analyze recent metrics
        recent_metrics = self._metrics_history[-60:]  # Last hour
        
        max_active = max(m.active_connections for m in recent_metrics)
        avg_active = sum(m.active_connections for m in recent_metrics) / len(recent_metrics)
        max_waiting = max(m.waiting_clients for m in recent_metrics)
        
        # Calculate recommendations
        recommended = {
            'min_pool_size': int(avg_active * 0.5),
            'default_pool_size': int(max_active * 1.2),
            'max_client_conn': int(max_active * 2)
        }
        
        # Apply safety limits
        recommended['min_pool_size'] = max(10, min(recommended['min_pool_size'], 100))
        recommended['default_pool_size'] = max(50, min(recommended['default_pool_size'], 200))
        recommended['max_client_conn'] = max(100, min(recommended['max_client_conn'], 2000))
        
        return recommended
        
    async def start_monitoring(
        self,
        interval_seconds: int = 60
    ) -> None:
        """
        Start continuous pool monitoring.
        
        Args:
            interval_seconds: Seconds between metric collections
        """
        async def monitoring_loop():
            while True:
                try:
                    # Collect metrics
                    metrics = await self.get_pool_metrics()
                    self._metrics_history.append(metrics)
                    
                    # Trim history
                    if len(self._metrics_history) > self._max_history:
                        self._metrics_history = self._metrics_history[-self._max_history:]
                        
                    # Check health
                    is_healthy, issues = await self.check_pool_health()
                    
                    if not is_healthy:
                        logger.warning(f"Pool health issues detected: {issues}")
                        
                    # Log metrics periodically
                    if len(self._metrics_history) % 10 == 0:
                        logger.info(f"Pool metrics: {metrics.to_dict()}")
                        
                except Exception as e:
                    logger.error(f"Pool monitoring error: {e}")
                    
                await asyncio.sleep(interval_seconds)
                
        self._monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info(f"Started pool monitoring (interval: {interval_seconds}s)")
        
    async def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Stopped pool monitoring")
            
    def get_metrics_summary(
        self,
        hours: int = 1
    ) -> Dict[str, any]:
        """
        Get summary of recent metrics.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Dictionary with metric summaries
        """
        if not self._metrics_history:
            return {}
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self._metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        return {
            'period_hours': hours,
            'samples': len(recent_metrics),
            'avg_active_connections': sum(m.active_connections for m in recent_metrics) / len(recent_metrics),
            'max_active_connections': max(m.active_connections for m in recent_metrics),
            'avg_waiting_clients': sum(m.waiting_clients for m in recent_metrics) / len(recent_metrics),
            'max_waiting_clients': max(m.waiting_clients for m in recent_metrics),
            'avg_query_time_ms': sum(m.avg_query_time for m in recent_metrics) / len(recent_metrics),
            'max_query_time_ms': max(m.max_query_time for m in recent_metrics),
            'avg_pool_hit_rate': sum(m.pool_hit_rate for m in recent_metrics) / len(recent_metrics),
            'total_requests': sum(m.total_requests for m in recent_metrics),
        }
        
    async def export_metrics(
        self,
        format: str = "prometheus"
    ) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('prometheus' or 'json')
            
        Returns:
            Formatted metrics string
        """
        metrics = await self.get_pool_metrics()
        
        if format == "prometheus":
            lines = [
                "# HELP pgbouncer_total_connections Total number of connections in pool",
                "# TYPE pgbouncer_total_connections gauge",
                f"pgbouncer_total_connections {metrics.total_connections}",
                "",
                "# HELP pgbouncer_active_connections Number of active connections",
                "# TYPE pgbouncer_active_connections gauge",
                f"pgbouncer_active_connections {metrics.active_connections}",
                "",
                "# HELP pgbouncer_idle_connections Number of idle connections",
                "# TYPE pgbouncer_idle_connections gauge",
                f"pgbouncer_idle_connections {metrics.idle_connections}",
                "",
                "# HELP pgbouncer_waiting_clients Number of waiting clients",
                "# TYPE pgbouncer_waiting_clients gauge",
                f"pgbouncer_waiting_clients {metrics.waiting_clients}",
                "",
                "# HELP pgbouncer_avg_query_time_ms Average query time in milliseconds",
                "# TYPE pgbouncer_avg_query_time_ms gauge",
                f"pgbouncer_avg_query_time_ms {metrics.avg_query_time}",
                "",
                "# HELP pgbouncer_pool_hit_rate Connection pool hit rate percentage",
                "# TYPE pgbouncer_pool_hit_rate gauge",
                f"pgbouncer_pool_hit_rate {metrics.pool_hit_rate}",
            ]
            return "\n".join(lines)
            
        else:  # JSON format
            import json
            return json.dumps(metrics.to_dict(), indent=2)