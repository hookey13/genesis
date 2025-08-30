# Performance Troubleshooting Guide

## Performance Baseline Metrics

### Normal Operating Ranges
| Metric | Baseline | Warning | Critical |
|--------|----------|---------|----------|
| Order Latency | <50ms | 50-200ms | >200ms |
| API Response Time | <100ms | 100-500ms | >500ms |
| CPU Usage | 20-40% | 40-70% | >70% |
| Memory Usage | 2-4GB | 4-6GB | >6GB |
| Database Query Time | <10ms | 10-50ms | >50ms |
| WebSocket Latency | <20ms | 20-100ms | >100ms |
| Event Processing | <5ms | 5-20ms | >20ms |
| Disk I/O Wait | <5% | 5-15% | >15% |
| Network RTT (Binance) | <30ms | 30-100ms | >100ms |
| Cache Hit Rate | >90% | 70-90% | <70% |

### Key Performance Indicators
- **Throughput**: 100-200 orders/second normal
- **Concurrent Connections**: 10-50 WebSocket connections
- **Queue Depth**: <100 messages normal, >1000 concerning
- **GC Pause Time**: <10ms acceptable, >50ms problematic

## Profiling Instructions

### CPU Profiling with cProfile

#### Start Profiling
```bash
# Profile the main application
python -m cProfile -o profile.stats genesis.__main__

# Profile specific module
python -m cProfile -o profile.stats -m genesis.engine.executor

# Profile with sorting
python -m cProfile -s cumulative genesis.__main__ > profile.txt
```

#### Analyze Profile Results
```python
# analyze_profile.py
import pstats
from pstats import SortKey

# Load profile
p = pstats.Stats('profile.stats')

# Sort by cumulative time
p.sort_stats(SortKey.CUMULATIVE)

# Print top 20 functions
p.print_stats(20)

# Find specific function
p.print_stats('execute_order')

# Print callers of a function
p.print_callers('calculate_position_size')

# Print callees
p.print_callees('process_market_data')
```

#### Visualize Profile
```bash
# Install visualization tools
pip install snakeviz gprof2dot

# Interactive visualization
snakeviz profile.stats

# Generate call graph
gprof2dot -f pstats profile.stats | dot -Tpng -o profile.png
```

### Memory Profiling with memory_profiler

#### Install and Setup
```bash
pip install memory_profiler pympler tracemalloc
```

#### Line-by-Line Memory Profiling
```python
# Add decorator to functions
from memory_profiler import profile

@profile
def process_order(order):
    # Function code here
    pass

# Run with memory profiling
python -m memory_profiler genesis/__main__.py
```

#### Memory Usage Over Time
```python
# monitor_memory.py
import tracemalloc
import asyncio
from datetime import datetime

tracemalloc.start()

async def monitor_memory():
    while True:
        current, peak = tracemalloc.get_traced_memory()
        print(f"{datetime.now()}: Current = {current / 10**6:.1f}MB, Peak = {peak / 10**6:.1f}MB")
        
        # Get top memory consumers
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("Top 5 memory consumers:")
        for stat in top_stats[:5]:
            print(stat)
        
        await asyncio.sleep(60)  # Check every minute

# Add to main application
asyncio.create_task(monitor_memory())
```

#### Heap Analysis
```python
# heap_analysis.py
from pympler import tracker, muppy, summary

# Create memory tracker
tr = tracker.SummaryTracker()

# ... application runs ...

# Print memory diff
tr.print_diff()

# Get all objects
all_objects = muppy.get_objects()

# Summarize by type
sum_obj = summary.summarize(all_objects)
summary.print_(sum_obj)

# Find specific objects
import gc
for obj in gc.get_objects():
    if isinstance(obj, Order):
        print(f"Order: {obj}, refs: {gc.get_referents(obj)}")
```

### Asyncio Profiling

#### Debug Mode
```python
# Enable asyncio debug mode
import asyncio
asyncio.set_debug(True)

# Or via environment
export PYTHONASYNCIODEBUG=1
```

#### Monitor Task Performance
```python
# async_profiler.py
import asyncio
import time
from collections import defaultdict

task_times = defaultdict(list)

def task_wrapper(coro):
    async def wrapped():
        start = time.perf_counter()
        try:
            result = await coro
            return result
        finally:
            duration = time.perf_counter() - start
            task_times[coro.__name__].append(duration)
            if duration > 0.1:  # Log slow tasks
                print(f"Slow task: {coro.__name__} took {duration:.3f}s")
    return wrapped()

# Wrap async tasks
task = asyncio.create_task(task_wrapper(process_order()))
```

#### Find Blocking Code
```python
# detect_blocking.py
import asyncio
import time
import threading

def detect_blocking():
    """Background thread to detect event loop blocking."""
    loop = asyncio.get_event_loop()
    last_check = time.time()
    
    def check():
        nonlocal last_check
        now = time.time()
        delay = now - last_check - 0.1  # Expected 100ms interval
        if delay > 0.05:  # Blocked for >50ms
            print(f"Event loop blocked for {delay*1000:.0f}ms")
        last_check = now
        loop.call_later(0.1, check)
    
    loop.call_soon(check)

# Start detector in background
threading.Thread(target=detect_blocking, daemon=True).start()
```

## Query Optimization Techniques

### Identify Slow Queries

#### PostgreSQL
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries >100ms
SELECT pg_reload_conf();

-- View slow queries
SELECT 
    query,
    calls,
    mean_exec_time,
    total_exec_time,
    min_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 50
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Current running queries
SELECT 
    pid,
    now() - query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
    AND query NOT LIKE '%pg_stat_activity%'
ORDER BY duration DESC;
```

#### SQLite
```python
# Enable query profiling
import sqlite3
import time

class ProfiledConnection(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        start = time.perf_counter()
        result = super().execute(sql, parameters)
        duration = time.perf_counter() - start
        if duration > 0.01:  # Log queries >10ms
            print(f"Slow query ({duration*1000:.1f}ms): {sql[:100]}")
        return result

# Use profiled connection
conn = sqlite3.connect('genesis.db', factory=ProfiledConnection)
```

### Query Optimization

#### Add Indexes
```sql
-- Find missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 100
    AND correlation < 0.1
ORDER BY n_distinct DESC;

-- Create indexes for common queries
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_positions_symbol ON positions(symbol);
```

#### Optimize Joins
```sql
-- Before: Nested loop join
SELECT o.*, t.*
FROM orders o
JOIN trades t ON o.id = t.order_id
WHERE o.created_at > NOW() - INTERVAL '1 hour';

-- After: Use index and hash join
CREATE INDEX idx_trades_order_id ON trades(order_id);

SELECT /*+ HashJoin(o t) */ o.*, t.*
FROM orders o
JOIN trades t ON o.id = t.order_id
WHERE o.created_at > NOW() - INTERVAL '1 hour';
```

## Latency Investigation Procedures

### Network Latency

#### Measure API Latency
```python
# measure_latency.py
import aiohttp
import asyncio
import time
from statistics import mean, stdev

async def measure_api_latency(url, count=100):
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        for _ in range(count):
            start = time.perf_counter()
            async with session.get(url) as response:
                await response.text()
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            await asyncio.sleep(0.1)
    
    print(f"Latency Statistics for {url}:")
    print(f"  Mean: {mean(latencies):.2f}ms")
    print(f"  StdDev: {stdev(latencies):.2f}ms")
    print(f"  Min: {min(latencies):.2f}ms")
    print(f"  Max: {max(latencies):.2f}ms")
    print(f"  P95: {sorted(latencies)[int(count*0.95)]:.2f}ms")
    print(f"  P99: {sorted(latencies)[int(count*0.99)]:.2f}ms")

# Run measurement
asyncio.run(measure_api_latency('https://api.binance.com/api/v3/ping'))
```

#### WebSocket Latency
```python
# ws_latency.py
import websockets
import asyncio
import time
import json

async def measure_ws_latency():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    
    async with websockets.connect(uri) as websocket:
        latencies = []
        
        # Send ping and measure pong
        for _ in range(100):
            start = time.perf_counter()
            await websocket.ping()
            pong_waiter = await websocket.ping()
            await pong_waiter
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            print(f"Ping-pong latency: {latency:.2f}ms")
            await asyncio.sleep(1)
        
        print(f"Average WebSocket latency: {mean(latencies):.2f}ms")
```

### Application Latency

#### Trace Request Flow
```python
# request_tracer.py
import time
import functools
from contextlib import contextmanager

class RequestTracer:
    def __init__(self):
        self.spans = []
    
    @contextmanager
    def span(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.spans.append({
                'name': name,
                'duration': duration * 1000,
                'timestamp': start
            })
    
    def print_trace(self):
        print("Request Trace:")
        for span in sorted(self.spans, key=lambda x: x['timestamp']):
            print(f"  {span['name']}: {span['duration']:.2f}ms")
        total = sum(s['duration'] for s in self.spans)
        print(f"  Total: {total:.2f}ms")

# Usage example
tracer = RequestTracer()

async def process_order(order):
    with tracer.span("validate_order"):
        await validate_order(order)
    
    with tracer.span("check_risk"):
        await check_risk_limits(order)
    
    with tracer.span("execute_order"):
        await execute_on_exchange(order)
    
    with tracer.span("update_database"):
        await save_to_database(order)
    
    tracer.print_trace()
```

## Resource Monitoring Commands

### System Resources
```bash
# CPU usage by process
top -p $(pgrep -f genesis) -b -n 1

# Memory usage details
ps aux | grep genesis | awk '{print $2, $4, $5, $6}'

# Disk I/O
iotop -p $(pgrep -f genesis)

# Network connections
netstat -tunp | grep genesis

# Open file descriptors
lsof -p $(pgrep -f genesis) | wc -l

# Thread count
ps -eLf | grep genesis | wc -l
```

### Python-Specific Monitoring
```python
# resource_monitor.py
import resource
import psutil
import os

def print_resource_usage():
    # Process info
    process = psutil.Process(os.getpid())
    
    print("Resource Usage:")
    print(f"  CPU %: {process.cpu_percent()}")
    print(f"  Memory RSS: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"  Memory VMS: {process.memory_info().vms / 1024 / 1024:.1f} MB")
    print(f"  Open Files: {len(process.open_files())}")
    print(f"  Connections: {len(process.connections())}")
    print(f"  Threads: {process.num_threads()}")
    
    # System limits
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"  File Descriptor Limit: {soft}/{hard}")
    
    # GC stats
    import gc
    for i, gen in enumerate(gc.get_stats()):
        print(f"  GC Gen {i}: collections={gen['collections']}, collected={gen['collected']}")
```

### Database Connection Pool
```python
# monitor_db_pool.py
from sqlalchemy.pool import QueuePool

def log_pool_status(dbapi_conn, connection_rec):
    pool = connection_rec.get_pool()
    print(f"Pool: size={pool.size()}, checked_in={pool.checkedin()}, "
          f"overflow={pool.overflow()}, total={pool.size() + pool.overflow()}")

# Attach to pool
engine.pool.connect().on_connect = log_pool_status
```

## Performance Optimization Checklist

### Immediate Actions (< 5 minutes)
- [ ] Check current resource usage (CPU, memory, disk)
- [ ] Review recent error logs for exceptions
- [ ] Check database connection pool status
- [ ] Verify network connectivity to exchange
- [ ] Check for blocking queries
- [ ] Review WebSocket connection status
- [ ] Clear application caches if needed

### Short-term Analysis (< 30 minutes)
- [ ] Run CPU profiler for 5 minutes
- [ ] Analyze slow query log
- [ ] Check for memory leaks with tracemalloc
- [ ] Review event loop blocking detection
- [ ] Analyze network latency patterns
- [ ] Check GC frequency and pause times
- [ ] Review cache hit rates

### Deep Investigation (< 2 hours)
- [ ] Generate flame graphs from profile data
- [ ] Perform heap dump analysis
- [ ] Run load testing to reproduce issue
- [ ] Analyze database query plans
- [ ] Review asyncio task performance
- [ ] Check for resource contention
- [ ] Analyze historical metrics trends

### Optimization Actions
- [ ] Add missing database indexes
- [ ] Optimize slow queries
- [ ] Implement caching for hot paths
- [ ] Batch database operations
- [ ] Use connection pooling effectively
- [ ] Optimize serialization/deserialization
- [ ] Reduce lock contention
- [ ] Implement circuit breakers for external calls

## Common Performance Issues and Solutions

### Issue: High CPU Usage
**Symptoms**: CPU >70%, slow response times
**Investigation**:
1. Profile with cProfile
2. Check for infinite loops
3. Review algorithmic complexity
**Solutions**:
- Optimize hot code paths
- Use more efficient algorithms
- Implement caching
- Use C extensions for critical paths

### Issue: Memory Leak
**Symptoms**: Gradually increasing memory, eventual OOM
**Investigation**:
1. Use tracemalloc to track allocations
2. Check for circular references
3. Review cache implementations
**Solutions**:
- Fix circular references
- Implement cache eviction
- Use weak references where appropriate
- Ensure proper cleanup in __del__

### Issue: Slow Database Queries
**Symptoms**: High query times, database CPU spike
**Investigation**:
1. Enable slow query log
2. Analyze query plans
3. Check for lock contention
**Solutions**:
- Add appropriate indexes
- Optimize query structure
- Implement query result caching
- Use read replicas for heavy reads

### Issue: Event Loop Blocking
**Symptoms**: Delayed async operations, WebSocket timeouts
**Investigation**:
1. Enable asyncio debug mode
2. Use blocking detector
3. Profile async functions
**Solutions**:
- Move blocking operations to thread pool
- Use async libraries for I/O
- Break up long computations
- Implement proper async/await patterns

### Issue: Network Latency
**Symptoms**: Slow API responses, order delays
**Investigation**:
1. Measure round-trip times
2. Check for packet loss
3. Review connection pooling
**Solutions**:
- Use connection pooling
- Implement request batching
- Add regional redundancy
- Optimize payload sizes