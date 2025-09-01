"""Performance monitoring with comprehensive Prometheus metrics for Project GENESIS."""

import asyncio
import functools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

import structlog
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.metrics import MetricWrapperBase

from genesis.core.exceptions import GenesisException

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Order execution metrics
    order_execution_latency: Histogram
    order_execution_counter: Counter
    order_execution_errors: Counter
    
    # API call metrics
    api_call_counter: Counter
    api_call_latency: Histogram
    api_call_errors: Counter
    api_rate_limit_hits: Counter
    
    # Position metrics
    active_positions_gauge: Gauge
    position_value_gauge: Gauge
    position_pnl_gauge: Gauge
    
    # System resource metrics
    memory_usage_gauge: Gauge
    cpu_usage_gauge: Gauge
    goroutine_count_gauge: Gauge
    
    # Risk metrics
    risk_check_latency: Histogram
    risk_check_counter: Counter
    risk_limit_breaches: Counter
    
    # Tilt detection metrics
    tilt_detection_latency: Histogram
    tilt_score_gauge: Gauge
    tilt_interventions_counter: Counter
    
    # Database metrics
    db_query_latency: Histogram
    db_query_counter: Counter
    db_connection_pool_gauge: Gauge
    db_transaction_counter: Counter
    
    # WebSocket metrics
    ws_message_latency: Histogram
    ws_connection_gauge: Gauge
    ws_reconnection_counter: Counter
    ws_message_counter: Counter
    
    # Circuit breaker metrics
    circuit_breaker_state_gauge: Gauge
    circuit_breaker_trips_counter: Counter
    circuit_breaker_success_rate: Gauge
    
    # Cache metrics (for future Redis integration)
    cache_hit_counter: Counter
    cache_miss_counter: Counter
    cache_eviction_counter: Counter
    cache_latency: Histogram
    
    # Custom business metrics
    tier_progression_gauge: Gauge
    daily_pnl_gauge: Gauge
    win_rate_gauge: Gauge
    sharpe_ratio_gauge: Gauge


class PerformanceMonitor:
    """Central performance monitoring system with Prometheus metrics."""
    
    def __init__(self, registry: CollectorRegistry | None = None):
        """Initialize performance monitor with metrics registry."""
        self.registry = registry or CollectorRegistry()
        self.metrics = self._initialize_metrics()
        self._latency_cache: dict[str, list[float]] = defaultdict(list)
        self._start_times: dict[str, float] = {}
        logger.info("PerformanceMonitor initialized with comprehensive metrics")
    
    def _initialize_metrics(self) -> PerformanceMetrics:
        """Initialize all Prometheus metrics."""
        return PerformanceMetrics(
            # Order execution metrics
            order_execution_latency=Histogram(
                'genesis_order_execution_latency_seconds',
                'Order execution latency in seconds',
                ['order_type', 'symbol', 'side', 'tier'],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                registry=self.registry
            ),
            order_execution_counter=Counter(
                'genesis_order_executions_total',
                'Total number of order executions',
                ['order_type', 'symbol', 'side', 'tier', 'status'],
                registry=self.registry
            ),
            order_execution_errors=Counter(
                'genesis_order_execution_errors_total',
                'Total number of order execution errors',
                ['order_type', 'symbol', 'error_type'],
                registry=self.registry
            ),
            
            # API call metrics
            api_call_counter=Counter(
                'genesis_api_calls_total',
                'Total number of API calls',
                ['exchange', 'endpoint', 'method', 'status_code'],
                registry=self.registry
            ),
            api_call_latency=Histogram(
                'genesis_api_call_latency_seconds',
                'API call latency in seconds',
                ['exchange', 'endpoint', 'method'],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
                registry=self.registry
            ),
            api_call_errors=Counter(
                'genesis_api_call_errors_total',
                'Total number of API call errors',
                ['exchange', 'endpoint', 'error_type'],
                registry=self.registry
            ),
            api_rate_limit_hits=Counter(
                'genesis_api_rate_limit_hits_total',
                'Total number of rate limit hits',
                ['exchange', 'endpoint'],
                registry=self.registry
            ),
            
            # Position metrics
            active_positions_gauge=Gauge(
                'genesis_active_positions',
                'Number of active positions',
                ['symbol', 'side', 'tier'],
                registry=self.registry
            ),
            position_value_gauge=Gauge(
                'genesis_position_value_usdt',
                'Total position value in USDT',
                ['symbol', 'side'],
                registry=self.registry
            ),
            position_pnl_gauge=Gauge(
                'genesis_position_pnl_usdt',
                'Position P&L in USDT',
                ['symbol', 'side', 'type'],  # type: realized/unrealized
                registry=self.registry
            ),
            
            # System resource metrics
            memory_usage_gauge=Gauge(
                'genesis_memory_usage_bytes',
                'Memory usage in bytes',
                ['type'],  # type: rss/vms/shared
                registry=self.registry
            ),
            cpu_usage_gauge=Gauge(
                'genesis_cpu_usage_percent',
                'CPU usage percentage',
                ['core'],
                registry=self.registry
            ),
            goroutine_count_gauge=Gauge(
                'genesis_goroutine_count',
                'Number of active goroutines/tasks',
                registry=self.registry
            ),
            
            # Risk metrics
            risk_check_latency=Histogram(
                'genesis_risk_check_latency_seconds',
                'Risk check latency in seconds',
                ['check_type', 'tier'],
                buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
                registry=self.registry
            ),
            risk_check_counter=Counter(
                'genesis_risk_checks_total',
                'Total number of risk checks',
                ['check_type', 'result'],  # result: passed/failed
                registry=self.registry
            ),
            risk_limit_breaches=Counter(
                'genesis_risk_limit_breaches_total',
                'Total number of risk limit breaches',
                ['limit_type', 'tier'],
                registry=self.registry
            ),
            
            # Tilt detection metrics
            tilt_detection_latency=Histogram(
                'genesis_tilt_detection_latency_seconds',
                'Tilt detection latency in seconds',
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
                registry=self.registry
            ),
            tilt_score_gauge=Gauge(
                'genesis_tilt_score',
                'Current tilt score (0-100)',
                ['indicator'],
                registry=self.registry
            ),
            tilt_interventions_counter=Counter(
                'genesis_tilt_interventions_total',
                'Total number of tilt interventions',
                ['intervention_type'],
                registry=self.registry
            ),
            
            # Database metrics
            db_query_latency=Histogram(
                'genesis_db_query_latency_seconds',
                'Database query latency in seconds',
                ['operation', 'table'],
                buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                registry=self.registry
            ),
            db_query_counter=Counter(
                'genesis_db_queries_total',
                'Total number of database queries',
                ['operation', 'table', 'status'],
                registry=self.registry
            ),
            db_connection_pool_gauge=Gauge(
                'genesis_db_connection_pool',
                'Database connection pool status',
                ['state'],  # state: active/idle/waiting
                registry=self.registry
            ),
            db_transaction_counter=Counter(
                'genesis_db_transactions_total',
                'Total number of database transactions',
                ['status'],  # status: committed/rolled_back
                registry=self.registry
            ),
            
            # WebSocket metrics
            ws_message_latency=Histogram(
                'genesis_ws_message_latency_ms',
                'WebSocket message latency in milliseconds',
                ['message_type', 'symbol'],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
                registry=self.registry
            ),
            ws_connection_gauge=Gauge(
                'genesis_ws_connections',
                'Number of WebSocket connections',
                ['state', 'exchange'],  # state: connected/disconnected/reconnecting
                registry=self.registry
            ),
            ws_reconnection_counter=Counter(
                'genesis_ws_reconnections_total',
                'Total number of WebSocket reconnections',
                ['exchange', 'reason'],
                registry=self.registry
            ),
            ws_message_counter=Counter(
                'genesis_ws_messages_total',
                'Total number of WebSocket messages',
                ['message_type', 'direction'],  # direction: inbound/outbound
                registry=self.registry
            ),
            
            # Circuit breaker metrics
            circuit_breaker_state_gauge=Gauge(
                'genesis_circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=open, 2=half-open)',
                ['service'],
                registry=self.registry
            ),
            circuit_breaker_trips_counter=Counter(
                'genesis_circuit_breaker_trips_total',
                'Total number of circuit breaker trips',
                ['service', 'reason'],
                registry=self.registry
            ),
            circuit_breaker_success_rate=Gauge(
                'genesis_circuit_breaker_success_rate',
                'Circuit breaker success rate',
                ['service'],
                registry=self.registry
            ),
            
            # Cache metrics
            cache_hit_counter=Counter(
                'genesis_cache_hits_total',
                'Total number of cache hits',
                ['cache_name', 'key_type'],
                registry=self.registry
            ),
            cache_miss_counter=Counter(
                'genesis_cache_misses_total',
                'Total number of cache misses',
                ['cache_name', 'key_type'],
                registry=self.registry
            ),
            cache_eviction_counter=Counter(
                'genesis_cache_evictions_total',
                'Total number of cache evictions',
                ['cache_name', 'reason'],
                registry=self.registry
            ),
            cache_latency=Histogram(
                'genesis_cache_operation_latency_seconds',
                'Cache operation latency in seconds',
                ['operation', 'cache_name'],
                buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005],
                registry=self.registry
            ),
            
            # Custom business metrics
            tier_progression_gauge=Gauge(
                'genesis_tier_progression',
                'Current tier progression (0=sniper, 1=hunter, 2=strategist)',
                registry=self.registry
            ),
            daily_pnl_gauge=Gauge(
                'genesis_daily_pnl_usdt',
                'Daily P&L in USDT',
                registry=self.registry
            ),
            win_rate_gauge=Gauge(
                'genesis_win_rate_percent',
                'Win rate percentage',
                ['timeframe'],  # timeframe: daily/weekly/monthly
                registry=self.registry
            ),
            sharpe_ratio_gauge=Gauge(
                'genesis_sharpe_ratio',
                'Sharpe ratio',
                ['timeframe'],
                registry=self.registry
            ),
        )
    
    # Context managers for timing operations
    def time_operation(self, operation_name: str) -> 'OperationTimer':
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    async def record_order_execution(
        self,
        order_type: str,
        symbol: str,
        side: str,
        tier: str,
        latency: float,
        status: str = "success"
    ) -> None:
        """Record order execution metrics."""
        labels = {
            'order_type': order_type,
            'symbol': symbol,
            'side': side,
            'tier': tier
        }
        
        self.metrics.order_execution_latency.labels(**labels).observe(latency)
        self.metrics.order_execution_counter.labels(**labels, status=status).inc()
        
        if status != "success":
            self.metrics.order_execution_errors.labels(
                order_type=order_type,
                symbol=symbol,
                error_type=status
            ).inc()
    
    async def record_api_call(
        self,
        exchange: str,
        endpoint: str,
        method: str,
        latency: float,
        status_code: int,
        error_type: str | None = None
    ) -> None:
        """Record API call metrics."""
        self.metrics.api_call_counter.labels(
            exchange=exchange,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        
        self.metrics.api_call_latency.labels(
            exchange=exchange,
            endpoint=endpoint,
            method=method
        ).observe(latency)
        
        if status_code == 429:
            self.metrics.api_rate_limit_hits.labels(
                exchange=exchange,
                endpoint=endpoint
            ).inc()
        
        if error_type:
            self.metrics.api_call_errors.labels(
                exchange=exchange,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
    
    async def update_position_metrics(
        self,
        symbol: str,
        side: str,
        tier: str,
        count: int,
        value_usdt: Decimal,
        pnl_realized: Decimal,
        pnl_unrealized: Decimal
    ) -> None:
        """Update position-related metrics."""
        self.metrics.active_positions_gauge.labels(
            symbol=symbol,
            side=side,
            tier=tier
        ).set(count)
        
        self.metrics.position_value_gauge.labels(
            symbol=symbol,
            side=side
        ).set(float(value_usdt))
        
        self.metrics.position_pnl_gauge.labels(
            symbol=symbol,
            side=side,
            type="realized"
        ).set(float(pnl_realized))
        
        self.metrics.position_pnl_gauge.labels(
            symbol=symbol,
            side=side,
            type="unrealized"
        ).set(float(pnl_unrealized))
    
    async def record_risk_check(
        self,
        check_type: str,
        tier: str,
        latency: float,
        passed: bool
    ) -> None:
        """Record risk check metrics."""
        self.metrics.risk_check_latency.labels(
            check_type=check_type,
            tier=tier
        ).observe(latency)
        
        self.metrics.risk_check_counter.labels(
            check_type=check_type,
            result="passed" if passed else "failed"
        ).inc()
        
        if not passed:
            self.metrics.risk_limit_breaches.labels(
                limit_type=check_type,
                tier=tier
            ).inc()
    
    async def record_database_operation(
        self,
        operation: str,
        table: str,
        latency: float,
        success: bool = True
    ) -> None:
        """Record database operation metrics."""
        self.metrics.db_query_latency.labels(
            operation=operation,
            table=table
        ).observe(latency)
        
        self.metrics.db_query_counter.labels(
            operation=operation,
            table=table,
            status="success" if success else "error"
        ).inc()
    
    async def record_websocket_message(
        self,
        message_type: str,
        symbol: str,
        latency_ms: float,
        direction: str = "inbound"
    ) -> None:
        """Record WebSocket message metrics."""
        self.metrics.ws_message_latency.labels(
            message_type=message_type,
            symbol=symbol
        ).observe(latency_ms)
        
        self.metrics.ws_message_counter.labels(
            message_type=message_type,
            direction=direction
        ).inc()
    
    async def update_circuit_breaker(
        self,
        service: str,
        state: str,
        success_rate: float
    ) -> None:
        """Update circuit breaker metrics."""
        state_map = {"closed": 0, "open": 1, "half-open": 2}
        self.metrics.circuit_breaker_state_gauge.labels(service=service).set(
            state_map.get(state, -1)
        )
        self.metrics.circuit_breaker_success_rate.labels(service=service).set(success_rate)
    
    async def record_cache_operation(
        self,
        cache_name: str,
        operation: str,
        key_type: str,
        hit: bool,
        latency: float
    ) -> None:
        """Record cache operation metrics."""
        if hit:
            self.metrics.cache_hit_counter.labels(
                cache_name=cache_name,
                key_type=key_type
            ).inc()
        else:
            self.metrics.cache_miss_counter.labels(
                cache_name=cache_name,
                key_type=key_type
            ).inc()
        
        self.metrics.cache_latency.labels(
            operation=operation,
            cache_name=cache_name
        ).observe(latency)
    
    async def update_system_metrics(
        self,
        memory_rss: int,
        memory_vms: int,
        cpu_percent: float,
        task_count: int
    ) -> None:
        """Update system resource metrics."""
        self.metrics.memory_usage_gauge.labels(type="rss").set(memory_rss)
        self.metrics.memory_usage_gauge.labels(type="vms").set(memory_vms)
        self.metrics.cpu_usage_gauge.labels(core="total").set(cpu_percent)
        self.metrics.goroutine_count_gauge.set(task_count)
    
    async def update_business_metrics(
        self,
        tier: int,
        daily_pnl: Decimal,
        win_rate_daily: float,
        win_rate_weekly: float,
        sharpe_daily: float,
        sharpe_weekly: float
    ) -> None:
        """Update business metrics."""
        self.metrics.tier_progression_gauge.set(tier)
        self.metrics.daily_pnl_gauge.set(float(daily_pnl))
        
        self.metrics.win_rate_gauge.labels(timeframe="daily").set(win_rate_daily)
        self.metrics.win_rate_gauge.labels(timeframe="weekly").set(win_rate_weekly)
        
        self.metrics.sharpe_ratio_gauge.labels(timeframe="daily").set(sharpe_daily)
        self.metrics.sharpe_ratio_gauge.labels(timeframe="weekly").set(sharpe_weekly)
    
    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def track_performance(self, 
                         operation_name: str | None = None,
                         labels: dict[str, str] | None = None) -> Callable:
        """Decorator for tracking function performance."""
        def decorator(func: Callable) -> Callable:
            name = operation_name or func.__name__
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    latency = time.perf_counter() - start_time
                    
                    # Record latency in cache for aggregation
                    self._latency_cache[name].append(latency)
                    if len(self._latency_cache[name]) > 1000:
                        self._latency_cache[name] = self._latency_cache[name][-1000:]
                    
                    logger.debug(
                        "Operation completed",
                        operation=name,
                        latency_seconds=latency,
                        labels=labels
                    )
                    return result
                except Exception as e:
                    latency = time.perf_counter() - start_time
                    logger.error(
                        "Operation failed",
                        operation=name,
                        latency_seconds=latency,
                        error=str(e),
                        labels=labels
                    )
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    latency = time.perf_counter() - start_time
                    
                    # Record latency in cache for aggregation
                    self._latency_cache[name].append(latency)
                    if len(self._latency_cache[name]) > 1000:
                        self._latency_cache[name] = self._latency_cache[name][-1000:]
                    
                    logger.debug(
                        "Operation completed",
                        operation=name,
                        latency_seconds=latency,
                        labels=labels
                    )
                    return result
                except Exception as e:
                    latency = time.perf_counter() - start_time
                    logger.error(
                        "Operation failed",
                        operation=name,
                        latency_seconds=latency,
                        error=str(e),
                        labels=labels
                    )
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time: float | None = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency = time.perf_counter() - self.start_time
            self.monitor._latency_cache[self.operation_name].append(latency)
            
            if exc_type:
                logger.error(
                    "Timed operation failed",
                    operation=self.operation_name,
                    latency_seconds=latency,
                    error=str(exc_val)
                )
            else:
                logger.debug(
                    "Timed operation completed",
                    operation=self.operation_name,
                    latency_seconds=latency
                )
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency = time.perf_counter() - self.start_time
            self.monitor._latency_cache[self.operation_name].append(latency)
            
            if exc_type:
                logger.error(
                    "Timed operation failed",
                    operation=self.operation_name,
                    latency_seconds=latency,
                    error=str(exc_val)
                )
            else:
                logger.debug(
                    "Timed operation completed",
                    operation=self.operation_name,
                    latency_seconds=latency
                )


# Global instance for easy access
_monitor_instance: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance