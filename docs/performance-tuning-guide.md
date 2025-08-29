# Performance Tuning Guide for Genesis Trading Loop

## Overview

This guide provides comprehensive performance tuning recommendations for the Genesis Trading Loop to achieve optimal throughput and latency characteristics.

## Performance Targets

| Metric | Baseline | Optimized | Maximum |
|--------|----------|-----------|---------|
| Event Throughput | 100/sec | 1000/sec | 5000/sec |
| P50 Latency | 20ms | 8ms | 2ms |
| P99 Latency | 200ms | 45ms | 10ms |
| Memory Usage | 500MB | 200MB | 100MB |
| CPU Usage | 40% | 20% | 10% |

## 1. Event Bus Optimization

### Queue Configuration

```python
# Optimal queue sizes based on load testing
EVENT_BUS_CONFIG = {
    "max_queue_size": 10000,  # Prevent unbounded growth
    "batch_size": 100,        # Process events in batches
    "flush_interval": 0.1,     # 100ms flush interval
    "priority_lanes": 4        # Separate priority queues
}
```

### Priority Lane Tuning

Assign events to appropriate priority lanes:

- **CRITICAL (0-10ms)**: Stop losses, circuit breakers
- **HIGH (10-50ms)**: Order fills, trading signals  
- **NORMAL (50-200ms)**: Market data updates
- **LOW (200ms+)**: Analytics, monitoring

### Batch Processing

Enable batch processing for non-critical events:

```python
async def process_batch(events: List[Event]):
    # Process multiple events in single operation
    async with database.transaction():
        for event in events:
            await process_event(event)
```

## 2. Database Optimization

### Connection Pooling

```python
# PostgreSQL connection pool settings
DATABASE_CONFIG = {
    "pool_size": 20,           # Base connections
    "max_overflow": 10,        # Additional connections under load
    "pool_timeout": 30,        # Connection timeout
    "pool_recycle": 3600,      # Recycle connections hourly
    "pool_pre_ping": True      # Verify connections before use
}
```

### Query Optimization

1. **Use prepared statements** for frequently executed queries
2. **Create appropriate indexes**:
   ```sql
   CREATE INDEX idx_events_type_time ON events(event_type, created_at);
   CREATE INDEX idx_positions_symbol ON positions(symbol, status);
   CREATE INDEX idx_orders_status ON orders(status, created_at);
   ```

3. **Batch inserts** for event storage:
   ```python
   await session.execute(
       insert(Event.__table__),
       events_data  # List of event dictionaries
   )
   ```

### Archive Strategy

Move old events to archive tables:

```python
# Archive events older than 30 days
ARCHIVE_CONFIG = {
    "archive_after_days": 30,
    "compress_after_days": 7,
    "batch_size": 1000,
    "run_time": "02:00"  # Run at 2 AM
}
```

## 3. Memory Optimization

### Object Pooling

Reuse objects to reduce garbage collection:

```python
class OrderPool:
    def __init__(self, size=1000):
        self.pool = [Order() for _ in range(size)]
        self.available = list(self.pool)
    
    def acquire(self) -> Order:
        if self.available:
            return self.available.pop()
        return Order()  # Create new if pool exhausted
    
    def release(self, order: Order):
        order.reset()  # Clear order data
        self.available.append(order)
```

### Memory Profiling

Use memory profilers to identify leaks:

```bash
# Profile memory usage
python -m memory_profiler genesis/engine/trading_loop.py

# Generate memory report
mprof run python -m genesis
mprof plot
```

### Garbage Collection Tuning

```python
import gc

# Tune garbage collection for trading workload
gc.set_threshold(700, 10, 10)  # Adjust thresholds
gc.collect(1)  # Collect young generation more frequently
```

## 4. Async I/O Optimization

### Concurrent Execution

Maximize concurrent operations:

```python
# Process independent operations concurrently
results = await asyncio.gather(
    exchange.get_balance(),
    exchange.get_open_orders(),
    database.load_positions(),
    return_exceptions=True
)
```

### Connection Reuse

Maintain persistent connections:

```python
class ExchangeConnection:
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=100,  # Total connection limit
                    limit_per_host=30,  # Per-host limit
                    ttl_dns_cache=300  # DNS cache TTL
                )
            )
        return self.session
```

### Task Scheduling

Use appropriate task scheduling:

```python
# CPU-bound tasks
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(
    executor,  # ThreadPoolExecutor for CPU tasks
    cpu_intensive_function,
    data
)

# I/O-bound tasks - use native async
result = await io_operation()
```

## 5. Caching Strategy

### Multi-Level Cache

Implement tiered caching:

```python
class TieredCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory (microseconds)
        self.l2_cache = Redis()  # Redis (milliseconds)
        self.l3_cache = Database()  # Database (10s of ms)
    
    async def get(self, key):
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        # Check L3
        value = await self.l3_cache.get(key)
        if value:
            await self.l2_cache.set(key, value)
            self.l1_cache[key] = value
        
        return value
```

### Cache Configuration

```python
CACHE_CONFIG = {
    "market_data_ttl": 1,      # 1 second for prices
    "position_ttl": 5,         # 5 seconds for positions
    "account_ttl": 30,         # 30 seconds for account data
    "config_ttl": 300,         # 5 minutes for configuration
    "max_size": 10000,         # Maximum cache entries
    "eviction": "lru"          # Least recently used eviction
}
```

## 6. Network Optimization

### TCP Tuning

System-level TCP optimizations:

```bash
# Linux TCP tuning
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_keepalive_time = 60' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_keepalive_intvl = 10' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_keepalive_probes = 6' >> /etc/sysctl.conf
sysctl -p
```

### WebSocket Optimization

```python
WEBSOCKET_CONFIG = {
    "ping_interval": 20,       # Keepalive ping
    "ping_timeout": 10,        # Ping response timeout
    "close_timeout": 5,        # Graceful close timeout
    "max_size": 10**6,        # 1MB max message size
    "compression": None,       # Disable compression for low latency
    "read_limit": 2**16,      # 64KB read buffer
    "write_limit": 2**16      # 64KB write buffer
}
```

## 7. CPU Optimization

### Profile CPU Usage

```python
import cProfile
import pstats

# Profile specific function
profiler = cProfile.Profile()
profiler.enable()
await process_events()
profiler.disable()

# Generate report
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Optimize Hot Paths

Identify and optimize frequently executed code:

```python
# Before - Multiple dictionary lookups
def process_order(order):
    if order['type'] == 'MARKET':
        if order['side'] == 'BUY':
            price = market_data['ask']
        else:
            price = market_data['bid']
    # ...

# After - Single lookup with caching
def process_order(order):
    order_key = (order.type, order.side)
    price = price_cache.get(order_key)
    if not price:
        price = calculate_price(order)
        price_cache[order_key] = price
    # ...
```

### Use Efficient Data Structures

```python
# Use deque for queues (O(1) append/pop)
from collections import deque
event_queue = deque(maxlen=10000)

# Use set for membership testing (O(1) average)
active_symbols = set(['BTC/USDT', 'ETH/USDT'])

# Use bisect for sorted insertions (O(log n))
import bisect
bisect.insort(sorted_prices, new_price)
```

## 8. Monitoring and Alerting

### Key Metrics to Monitor

```python
MONITORING_METRICS = {
    "event_rate": {
        "warning": 500,   # Warn if below 500/sec
        "critical": 100   # Critical if below 100/sec
    },
    "latency_p99": {
        "warning": 100,   # Warn if above 100ms
        "critical": 500   # Critical if above 500ms
    },
    "memory_usage": {
        "warning": 1000,  # Warn if above 1GB
        "critical": 2000  # Critical if above 2GB
    },
    "error_rate": {
        "warning": 0.01,  # Warn if above 1%
        "critical": 0.05  # Critical if above 5%
    }
}
```

### Performance Regression Detection

Automated performance testing in CI/CD:

```yaml
# .github/workflows/performance.yml
performance-test:
  runs-on: ubuntu-latest
  steps:
    - name: Run performance tests
      run: |
        python -m pytest tests/load/ --benchmark-only
        python scripts/check_performance_regression.py
```

## 9. Configuration Recommendations

### Development Environment

```python
DEV_CONFIG = {
    "debug": True,
    "profile": True,
    "batch_size": 10,
    "flush_interval": 1.0,
    "cache_enabled": False
}
```

### Production Environment

```python
PROD_CONFIG = {
    "debug": False,
    "profile": False,
    "batch_size": 100,
    "flush_interval": 0.1,
    "cache_enabled": True,
    "connection_pool_size": 50,
    "worker_threads": 4
}
```

## 10. Troubleshooting Performance Issues

### Diagnostic Checklist

1. **Check system resources**
   ```bash
   top -p $(pgrep -f genesis)  # CPU usage
   free -h                      # Memory usage
   iotop -p $(pgrep -f genesis) # I/O usage
   netstat -tulpn              # Network connections
   ```

2. **Analyze logs for patterns**
   ```bash
   # Find slow queries
   grep "duration>" logs/trading.log | awk '{print $NF}' | sort -n | tail -20
   
   # Find errors
   grep ERROR logs/trading.log | head -20
   ```

3. **Check database performance**
   ```sql
   -- PostgreSQL slow queries
   SELECT query, calls, mean_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 10;
   ```

4. **Review event bus metrics**
   ```python
   metrics = event_bus.get_metrics()
   print(f"Queue depth: {metrics['queue_depth']}")
   print(f"Processing rate: {metrics['events_per_second']}")
   print(f"Dropped events: {metrics['dropped_events']}")
   ```

### Common Performance Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Memory Leak | Increasing memory over time | Profile with tracemalloc, fix circular references |
| Database Bottleneck | High query latency | Add indexes, optimize queries, increase pool size |
| Event Bus Overflow | Dropped events, high queue depth | Increase batch size, add workers, optimize handlers |
| Network Latency | Slow API responses | Use connection pooling, enable keepalive, check routing |
| CPU Saturation | 100% CPU usage | Profile hot paths, optimize algorithms, add workers |

## Performance Testing Commands

```bash
# Load test - 1000 events/second
python -m pytest tests/load/test_trading_loop_load.py::test_thousand_events_per_second

# Memory stability test
python -m pytest tests/load/test_trading_loop_load.py::test_sustained_load_memory

# Latency degradation test
python -m pytest tests/load/test_trading_loop_load.py::test_latency_under_load

# Full performance suite
python -m pytest tests/load/ -v --benchmark-json=performance_report.json
```

## Conclusion

Performance optimization is an iterative process. Start with profiling to identify bottlenecks, apply targeted optimizations, and validate improvements through load testing. Monitor production metrics continuously and be prepared to adjust configurations based on real-world usage patterns.