# Genesis Distributed Tracing Best Practices

## Overview

This guide documents best practices for implementing and using distributed tracing in the Genesis trading system using OpenTelemetry and Jaeger.

## Quick Start

### Starting Jaeger Services

```bash
# Start Jaeger backend
docker-compose -f docker-compose.jaeger.yml up -d

# Verify services are running
docker-compose -f docker-compose.jaeger.yml ps

# Access Jaeger UI
open http://localhost:16686
```

### Environment Configuration

```bash
# .env configuration for tracing
OTLP_ENDPOINT=localhost:4317
OTLP_SECURE=false
TRACE_SAMPLING_RATE=0.01  # 1% sampling for production
JAEGER_UI_URL=http://localhost:16686
TRACE_RETENTION_DAYS=7
ENVIRONMENT=production
```

## Span Naming Conventions

### Format
- Use descriptive, hierarchical names: `service.component.operation`
- Examples:
  - `genesis.order.execute_market`
  - `genesis.risk.check_position_limits`
  - `genesis.market_data.process_ticker`

### Operation Types
- **HTTP**: `http_METHOD_/path` (e.g., `http_POST_/api/v3/order`)
- **Database**: `db.query.table_name` (e.g., `db.query.orders`)
- **Cache**: `cache.operation.key_pattern` (e.g., `cache.get.order_*`)
- **External API**: `api.exchange.endpoint` (e.g., `api.binance.place_order`)

## Attribute Standards

### Required Attributes
Every span should include:
- `service.name`: Service identifier
- `service.version`: Service version
- `correlation_id`: Request correlation ID
- `tier`: Current trading tier (sniper/hunter/strategist)

### Operation-Specific Attributes

#### Order Operations
```python
{
    "order.id": "uuid",
    "order.type": "market|limit",
    "order.side": "buy|sell",
    "order.symbol": "BTCUSDT",
    "order.quantity": "0.001",
    "order.price": "50000.00",  # For limit orders
    "order.status": "success|failed",
    "order.execution_time_ms": 25.5
}
```

#### Risk Checks
```python
{
    "risk.check_type": "position|leverage|drawdown",
    "risk.tier": "sniper|hunter|strategist",
    "risk.passed": True,
    "risk.limit": "10000",
    "risk.current_value": "8500"
}
```

#### Market Data
```python
{
    "market.symbol": "BTCUSDT",
    "market.type": "ticker|orderbook|trade",
    "market.timestamp": 1234567890,
    "processing.latency_ms": 2.5
}
```

## Sampling Configuration

### Adaptive Sampling Strategy
The system uses adaptive sampling based on operation characteristics:

```python
# Production sampling rates
BASE_SAMPLING_RATE = 0.01      # 1% for normal operations
ERROR_SAMPLING_RATE = 1.0      # 100% for errors
SLOW_SAMPLING_RATE = 1.0       # 100% for slow operations (>100ms)
CRITICAL_OPS_RATE = 0.1        # 10% for critical operations
```

### Override Sampling for Debugging
```python
# Force sampling for specific operations during debugging
tracer = get_opentelemetry_tracer(sampling_rate=1.0)  # 100% sampling
```

## Instrumentation Patterns

### Using Decorators

```python
from genesis.monitoring.tracing_init import setup_genesis_tracing

# Setup tracing
tracer, decorators = setup_genesis_tracing(service_name="genesis-trading")

# Use decorators for automatic instrumentation
@decorators["order_execution"]
async def execute_order(order):
    # Order execution logic
    pass

@decorators["track_performance"]("custom_operation")
async def custom_operation():
    # Custom operation logic
    pass
```

### Manual Span Creation

```python
from genesis.monitoring.opentelemetry_tracing import get_opentelemetry_tracer

tracer = get_opentelemetry_tracer()

# Create span with context manager
with tracer.create_span(
    "manual_operation",
    attributes={
        "custom.attribute": "value",
        "operation.type": "manual"
    }
) as span:
    try:
        # Operation logic
        result = await perform_operation()
        span.set_attribute("operation.result", "success")
    except Exception as e:
        span.record_exception(e)
        span.set_attribute("operation.result", "failed")
        raise
```

### Context Propagation

```python
# Inject context for cross-service calls
headers = {}
tracer.inject_context(headers)
await external_service.call(headers=headers)

# Extract context from incoming requests
context = tracer.extract_context(request.headers)
with tracer.tracer.start_as_current_span("handle_request", context=context):
    # Handle request with parent context
    pass
```

## Performance Optimization

### Reduce Span Overhead
1. **Batch span exports**: Use BatchSpanProcessor (configured by default)
2. **Limit attribute size**: Keep attributes under 1KB
3. **Avoid sensitive data**: Never include passwords, API keys, or PII
4. **Use sampling**: Don't trace everything in production

### Efficient Attribute Collection
```python
# Good: Selective attributes
span.set_attribute("order.id", order_id)
span.set_attribute("order.status", status)

# Bad: Entire object serialization
span.set_attribute("order", str(order))  # Avoid this!
```

## Troubleshooting Guide

### Common Issues

#### 1. Missing Traces
- **Check sampling rate**: Ensure it's not 0
- **Verify OTLP endpoint**: Test connection to Jaeger collector
- **Check span export**: Look for export errors in logs

```bash
# Test OTLP connectivity
curl -v http://localhost:14269/health
```

#### 2. High Memory Usage
- **Reduce span batch size**: Lower `max_export_batch_size`
- **Decrease retention**: Reduce trace retention period
- **Optimize attributes**: Remove unnecessary span attributes

#### 3. Slow Trace Queries
- **Add index**: Ensure Elasticsearch indices are optimized
- **Limit time range**: Query specific time windows
- **Use service/operation filters**: Narrow search scope

### Debug Mode
Enable debug logging for tracing:

```python
# Enable console export for debugging
tracer = get_opentelemetry_tracer(
    export_to_console=True,
    production_mode=False
)
```

## Alerting on Traces

### Alert Configuration
Configure alerts based on trace metrics:

```python
from genesis.monitoring.trace_alerting import TraceAlertManager

alert_manager = TraceAlertManager()

# Alert on high latency
alert_manager.add_rule(AlertRule(
    name="custom_high_latency",
    condition=lambda m: m.get("p99_latency_ms", 0) > 200,
    severity=AlertSeverity.HIGH,
    threshold_value=200.0
))
```

### Alert Response Procedures

#### Critical Latency Alert
1. Check Jaeger for slow operations
2. Identify bottleneck operations
3. Review recent deployments
4. Scale resources if needed

#### High Error Rate Alert
1. Check error traces in Jaeger
2. Identify error patterns
3. Review logs for root cause
4. Implement fixes or rollback

## Security Considerations

### Data Sanitization
```python
# Sanitize sensitive data before adding to spans
def sanitize_order_data(order):
    return {
        "order_id": order.id,
        "symbol": order.symbol,
        "side": order.side,
        # Don't include: api_key, secret, user_id
    }

span.set_attributes(sanitize_order_data(order))
```

### Access Control
- Restrict Jaeger UI access in production
- Use TLS for OTLP communication
- Rotate collector certificates regularly
- Audit trace data access

## Monitoring Trace System Health

### Key Metrics to Monitor
- **Traces per minute**: Should be > 0
- **Export failures**: Should be < 1%
- **Span drop rate**: Should be < 0.1%
- **Collector latency**: Should be < 10ms

### Health Check Endpoints
```bash
# Jaeger collector health
curl http://localhost:14269/health

# Jaeger query health
curl http://localhost:16687/health

# Elasticsearch health
curl http://localhost:9200/_cluster/health
```

## Integration with Other Tools

### Prometheus Metrics
Correlate traces with metrics using exemplars:

```python
# Add trace ID to Prometheus metrics
from prometheus_client import Counter

orders_counter = Counter(
    'orders_total',
    'Total orders processed',
    ['status', 'trace_id']
)

# In your code
trace_id = span.get_span_context().trace_id
orders_counter.labels(status='success', trace_id=trace_id).inc()
```

### Log Correlation
Include trace IDs in logs:

```python
import structlog

logger = structlog.get_logger()

# Log with trace context
logger.info(
    "Order executed",
    order_id=order.id,
    trace_id=span.get_span_context().trace_id,
    span_id=span.get_span_context().span_id
)
```

## Maintenance Procedures

### Daily Tasks
- Review active alerts
- Check trace collection rate
- Monitor storage usage

### Weekly Tasks
- Analyze performance trends
- Review top slow operations
- Optimize high-cardinality attributes

### Monthly Tasks
- Clean up old trace data
- Review and update alert rules
- Performance baseline updates

## Performance Baselines

### Expected Latencies
- Order execution: < 50ms p99
- Risk checks: < 10ms p99
- Market data processing: < 5ms p99
- Database queries: < 20ms p99

### Storage Requirements
- Average trace size: ~1KB
- Daily trace volume: ~1GB (at full load)
- 7-day retention: ~7GB
- Elasticsearch heap: 2GB minimum

## Useful Jaeger Queries

### Find Slow Operations
```
service="genesis-trading" AND duration>100ms
```

### Find Failed Orders
```
service="genesis-trading" AND operation="order_execution" AND error=true
```

### Trace Specific Order
```
service="genesis-trading" AND order.id="specific-order-uuid"
```

### High Latency Database Queries
```
operation="db.query.*" AND duration>50ms
```

## Emergency Procedures

### Disable Tracing
If tracing causes issues, disable it:

```bash
# Set environment variable
export OTLP_ENDPOINT=disabled

# Restart service
systemctl restart genesis-trading
```

### Clear Trace Data
```bash
# Delete old indices in Elasticsearch
curl -X DELETE "http://localhost:9200/genesis-traces-*"

# Restart Jaeger services
docker-compose -f docker-compose.jaeger.yml restart
```

## Support and Resources

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Genesis Internal Wiki](internal-wiki-link)
- Support Channel: #genesis-observability