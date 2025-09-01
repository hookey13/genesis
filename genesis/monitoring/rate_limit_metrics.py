"""Rate limiting and circuit breaker metrics for Prometheus."""

from typing import Dict, Any, Optional
import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = structlog.get_logger(__name__)

# Rate Limiter Metrics
rate_limit_requests_total = Counter(
    'rate_limit_requests_total',
    'Total number of rate limit requests',
    ['priority', 'status']  # status: allowed, rejected, queued
)

rate_limit_tokens_used = Gauge(
    'rate_limit_tokens_used',
    'Current number of tokens used',
    ['bucket']
)

rate_limit_tokens_available = Gauge(
    'rate_limit_tokens_available',
    'Current number of tokens available',
    ['bucket']
)

rate_limit_utilization_percent = Gauge(
    'rate_limit_utilization_percent',
    'Rate limit utilization percentage',
    ['limiter']
)

rate_limit_queue_size = Gauge(
    'rate_limit_queue_size',
    'Number of requests in priority queue',
    ['priority']
)

rate_limit_wait_time_seconds = Histogram(
    'rate_limit_wait_time_seconds',
    'Time spent waiting for rate limit tokens',
    ['priority'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

rate_limit_critical_overrides = Counter(
    'rate_limit_critical_overrides_total',
    'Number of times critical priority overrode rate limits'
)

rate_limit_coalesced_requests = Counter(
    'rate_limit_coalesced_requests_total',
    'Number of requests coalesced'
)

rate_limit_adaptive_adjustments = Counter(
    'rate_limit_adaptive_adjustments_total',
    'Number of adaptive rate limit adjustments',
    ['adjustment_type']  # speed_up, slow_down
)

# Circuit Breaker Metrics
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
    ['circuit']
)

circuit_breaker_failures = Counter(
    'circuit_breaker_failures_total',
    'Total number of failures recorded',
    ['circuit']
)

circuit_breaker_successes = Counter(
    'circuit_breaker_successes_total',
    'Total number of successes recorded',
    ['circuit']
)

circuit_breaker_state_transitions = Counter(
    'circuit_breaker_state_transitions_total',
    'Number of state transitions',
    ['circuit', 'from_state', 'to_state']
)

circuit_breaker_call_duration_seconds = Histogram(
    'circuit_breaker_call_duration_seconds',
    'Duration of calls through circuit breaker',
    ['circuit', 'status'],  # status: success, failure, rejected
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

circuit_breaker_open_duration_seconds = Summary(
    'circuit_breaker_open_duration_seconds',
    'Duration circuit breaker remained open',
    ['circuit']
)

circuit_breaker_degradation_strategy_used = Counter(
    'circuit_breaker_degradation_strategy_used_total',
    'Number of times each degradation strategy was used',
    ['circuit', 'strategy']
)

circuit_breaker_cache_hits = Counter(
    'circuit_breaker_cache_hits_total',
    'Number of cache hits during degradation',
    ['circuit']
)

circuit_breaker_queued_requests = Gauge(
    'circuit_breaker_queued_requests',
    'Number of requests queued for recovery',
    ['circuit']
)

# Backpressure Metrics
backpressure_active = Gauge(
    'backpressure_active',
    'Whether backpressure is currently active (0=inactive, 1=active)',
    ['component']
)

backpressure_queue_utilization = Gauge(
    'backpressure_queue_utilization_percent',
    'Queue utilization percentage',
    ['component', 'priority']
)

backpressure_events_shed = Counter(
    'backpressure_events_shed_total',
    'Number of events shed due to backpressure',
    ['component', 'priority']
)

# Exchange Gateway Metrics
exchange_api_requests = Counter(
    'exchange_api_requests_total',
    'Total number of API requests to exchange',
    ['endpoint', 'method', 'status']
)

exchange_api_latency_seconds = Histogram(
    'exchange_api_latency_seconds',
    'API request latency',
    ['endpoint', 'method'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

exchange_weight_usage = Gauge(
    'exchange_weight_usage',
    'Current weight usage from exchange headers',
    ['time_window']  # 1m, 10s
)

exchange_order_count = Gauge(
    'exchange_order_count',
    'Current order count from exchange headers',
    ['time_window']  # 1m, 10s, daily
)


class RateLimitMetricsCollector:
    """Collector for rate limiting and circuit breaker metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = structlog.get_logger(__name__)
        
    def update_rate_limiter_metrics(self, limiter_name: str, metrics: Dict[str, Any]):
        """Update rate limiter metrics."""
        try:
            # Update request counters
            if 'requests_allowed' in metrics:
                rate_limit_requests_total.labels(
                    priority='normal',
                    status='allowed'
                ).inc(metrics.get('requests_allowed', 0))
            
            if 'requests_rejected' in metrics:
                rate_limit_requests_total.labels(
                    priority='normal',
                    status='rejected'
                ).inc(metrics.get('requests_rejected', 0))
            
            if 'requests_queued' in metrics:
                rate_limit_requests_total.labels(
                    priority='normal',
                    status='queued'
                ).inc(metrics.get('requests_queued', 0))
            
            # Update token gauges
            if 'token_bucket_tokens' in metrics:
                rate_limit_tokens_used.labels(bucket=limiter_name).set(
                    metrics['token_bucket_capacity'] - metrics['token_bucket_tokens']
                )
                rate_limit_tokens_available.labels(bucket=limiter_name).set(
                    metrics['token_bucket_tokens']
                )
            
            # Update utilization
            if 'token_bucket_capacity' in metrics and metrics['token_bucket_capacity'] > 0:
                utilization = (
                    (metrics['token_bucket_capacity'] - metrics['token_bucket_tokens']) /
                    metrics['token_bucket_capacity'] * 100
                )
                rate_limit_utilization_percent.labels(limiter=limiter_name).set(utilization)
            
            # Update critical overrides
            if 'critical_overrides' in metrics:
                rate_limit_critical_overrides.inc(metrics.get('critical_overrides', 0))
                
        except Exception as e:
            self.logger.error("Failed to update rate limiter metrics", error=str(e))
    
    def update_circuit_breaker_metrics(self, circuit_name: str, status: Dict[str, Any]):
        """Update circuit breaker metrics."""
        try:
            # Map state to numeric value
            state_map = {'closed': 0, 'open': 1, 'half_open': 2}
            state_value = state_map.get(status.get('state', 'closed'), 0)
            circuit_breaker_state.labels(circuit=circuit_name).set(state_value)
            
            # Update failure/success counts
            if 'failure_count' in status:
                circuit_breaker_failures.labels(circuit=circuit_name).inc(
                    status.get('failure_count', 0)
                )
            
            if 'success_count' in status:
                circuit_breaker_successes.labels(circuit=circuit_name).inc(
                    status.get('success_count', 0)
                )
            
            # Update open duration if circuit is open
            if 'circuit_open_duration' in status:
                circuit_breaker_open_duration_seconds.labels(
                    circuit=circuit_name
                ).observe(status['circuit_open_duration'])
                
        except Exception as e:
            self.logger.error("Failed to update circuit breaker metrics", error=str(e))
    
    def update_backpressure_metrics(self, component: str, metrics: Dict[str, Any]):
        """Update backpressure metrics."""
        try:
            # Update active state
            backpressure_active.labels(component=component).set(
                1 if metrics.get('backpressure_active', False) else 0
            )
            
            # Update queue utilization
            if 'queue_utilization_percent' in metrics:
                backpressure_queue_utilization.labels(
                    component=component,
                    priority='all'
                ).set(metrics['queue_utilization_percent'])
            
            # Update per-priority queue sizes
            if 'queue_sizes' in metrics:
                for priority, size in metrics['queue_sizes'].items():
                    rate_limit_queue_size.labels(priority=str(priority)).set(size)
            
            # Update events shed counter
            if 'events_dropped' in metrics:
                backpressure_events_shed.labels(
                    component=component,
                    priority='low'
                ).inc(metrics.get('events_dropped', 0))
                
        except Exception as e:
            self.logger.error("Failed to update backpressure metrics", error=str(e))
    
    def update_exchange_metrics(self, headers: Dict[str, str]):
        """Update exchange-specific metrics from response headers."""
        try:
            # Update weight usage
            if 'X-MBX-USED-WEIGHT-1M' in headers:
                exchange_weight_usage.labels(time_window='1m').set(
                    int(headers['X-MBX-USED-WEIGHT-1M'])
                )
            
            # Update order counts
            if 'X-MBX-ORDER-COUNT-1M' in headers:
                exchange_order_count.labels(time_window='1m').set(
                    int(headers['X-MBX-ORDER-COUNT-1M'])
                )
            
            if 'X-MBX-ORDER-COUNT-10S' in headers:
                exchange_order_count.labels(time_window='10s').set(
                    int(headers['X-MBX-ORDER-COUNT-10S'])
                )
                
        except Exception as e:
            self.logger.error("Failed to update exchange metrics", error=str(e))


# Global metrics collector instance
_metrics_collector: Optional[RateLimitMetricsCollector] = None


def get_metrics_collector() -> RateLimitMetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = RateLimitMetricsCollector()
    return _metrics_collector