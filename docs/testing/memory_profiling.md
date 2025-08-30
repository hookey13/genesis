# Memory Profiling Documentation

## Overview

Memory profiling is a critical component of our production validation suite, designed to detect memory leaks, identify inefficient memory usage patterns, and ensure long-term system stability. The Memory Profiler provides continuous monitoring over extended periods (up to 7 days) to catch slow memory leaks that might not be apparent in shorter tests.

## Purpose

The Memory Profiler addresses several key concerns:
- Detects memory leaks before production deployment
- Identifies memory usage patterns and growth trends
- Validates garbage collection effectiveness
- Monitors memory allocation hotspots
- Ensures system stability over extended periods
- Prevents out-of-memory (OOM) crashes

## Architecture

### Core Components

#### 1. MemoryProfiler Class
The main profiling engine that:
- Tracks memory allocations using tracemalloc
- Collects periodic memory snapshots
- Analyzes memory growth patterns
- Detects potential leaks
- Generates detailed reports

#### 2. MemorySnapshot
Captures memory state at a point in time:
- Total memory usage
- Top memory consumers
- Allocation traceback
- Object counts by type
- GC statistics

#### 3. LeakDetector
Analyzes snapshots to identify leaks:
- Linear regression on memory growth
- Statistical anomaly detection
- Pattern matching for known leak signatures
- Threshold-based alerting

#### 4. MemoryMetrics
Tracks memory-related metrics:
- Current memory usage
- Peak memory usage
- Growth rate
- GC frequency and duration
- Allocation/deallocation patterns

## Configuration

### Basic Setup

```python
from tests.stress.memory_profiler import MemoryProfiler

# Initialize profiler
profiler = MemoryProfiler(
    snapshot_interval=60,  # Snapshot every minute
    trace_limit=100,      # Track top 100 allocations
    alert_threshold_mb=500  # Alert if memory exceeds 500MB
)

# Start profiling
await profiler.start()
```

### Advanced Configuration

```python
profiler = MemoryProfiler(
    snapshot_interval=30,
    trace_limit=200,
    alert_threshold_mb=1000,
    growth_threshold_mb_per_hour=10,  # Alert if growing >10MB/hour
    enable_tracemalloc=True,
    tracemalloc_frames=25,  # Traceback depth
    gc_threshold=(700, 10, 10),  # Custom GC thresholds
    output_dir="memory_profiles"
)
```

## Memory Leak Detection

### Detection Methodology

The profiler uses multiple techniques to detect leaks:

1. **Linear Growth Detection**
   - Tracks memory usage over time
   - Fits linear regression model
   - Identifies consistent upward trends

2. **Snapshot Comparison**
   - Compares memory snapshots
   - Identifies growing objects
   - Tracks allocation sites

3. **Statistical Analysis**
   - Calculates standard deviation
   - Detects anomalous growth
   - Uses moving averages

### Leak Patterns

Common leak patterns detected:

#### Unbounded Collections
```python
# Leak pattern: List grows without bounds
leaked_list = []
for message in message_stream:
    leaked_list.append(message)  # Never cleared
```

#### Circular References
```python
# Leak pattern: Circular reference prevents GC
class Node:
    def __init__(self):
        self.parent = None
        self.children = []
    
node1 = Node()
node2 = Node()
node1.children.append(node2)
node2.parent = node1  # Circular reference
```

#### Event Listener Accumulation
```python
# Leak pattern: Event listeners never removed
for client in clients:
    event_bus.subscribe(client.handler)  # Never unsubscribed
```

## Usage Examples

### Basic Memory Profiling

```python
import asyncio
from tests.stress.memory_profiler import MemoryProfiler

async def profile_application():
    # Initialize profiler
    profiler = MemoryProfiler()
    
    # Start profiling
    await profiler.start()
    
    # Run your application
    await run_trading_system()
    
    # Stop profiling
    await profiler.stop()
    
    # Analyze results
    report = profiler.generate_report()
    
    if report['memory_leak_detected']:
        print(f"Memory leak detected: {report['leak_rate_mb_per_hour']} MB/hour")
        print(f"Top allocators: {report['top_allocators']}")

asyncio.run(profile_application())
```

### 7-Day Continuous Monitoring

```python
async def long_term_monitoring():
    profiler = MemoryProfiler(
        snapshot_interval=300,  # Every 5 minutes
        alert_threshold_mb=2000,
        growth_threshold_mb_per_hour=5
    )
    
    # Set up alerting
    async def on_leak_detected(leak_info):
        await send_alert(f"Memory leak detected: {leak_info}")
    
    profiler.on_leak_detected = on_leak_detected
    
    # Run for 7 days
    await profiler.monitor_continuous(days=7)
    
    # Generate final report
    report = profiler.generate_report()
    await save_report(report, "7day_memory_profile.json")

asyncio.run(long_term_monitoring())
```

### Heap Dump on Threshold

```python
async def monitor_with_heap_dump():
    profiler = MemoryProfiler(
        alert_threshold_mb=1000,
        enable_heap_dump=True,
        heap_dump_dir="/tmp/heap_dumps"
    )
    
    await profiler.start()
    
    # Monitor until threshold breach
    while profiler.current_memory_mb < 1000:
        await asyncio.sleep(10)
    
    # Heap dump automatically created
    dump_file = profiler.latest_heap_dump
    print(f"Heap dump saved: {dump_file}")
    
    # Analyze dump
    analysis = await profiler.analyze_heap_dump(dump_file)
    print(f"Top memory consumers: {analysis['top_objects']}")

asyncio.run(monitor_with_heap_dump())
```

## Command-Line Usage

```bash
# Basic profiling
python -m tests.stress.memory_profiler --duration 3600

# 7-day monitoring
python -m tests.stress.memory_profiler \
    --duration 604800 \
    --snapshot-interval 300 \
    --alert-threshold 2000

# Profile specific process
python -m tests.stress.memory_profiler \
    --pid 12345 \
    --duration 7200 \
    --trace-limit 500

# Generate detailed report
python -m tests.stress.memory_profiler \
    --duration 3600 \
    --detailed-report \
    --output memory_report.html
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --duration | 3600 | Profiling duration in seconds |
| --snapshot-interval | 60 | Seconds between snapshots |
| --trace-limit | 100 | Number of top allocations to track |
| --alert-threshold | 1000 | Memory threshold in MB |
| --growth-threshold | 10 | Growth rate threshold (MB/hour) |
| --pid | current | Process ID to profile |
| --output | stdout | Output file for report |
| --detailed-report | false | Generate detailed HTML report |

## Analysis Tools

### Memory Snapshot Analysis

```python
# Analyze memory snapshot
snapshot = profiler.take_snapshot()

# Get top memory consumers
top_stats = snapshot.get_top_stats(limit=10)
for stat in top_stats:
    print(f"{stat.size / 1024 / 1024:.2f} MB: {stat.traceback}")

# Compare snapshots
snapshot1 = profiler.take_snapshot()
# ... run code ...
snapshot2 = profiler.take_snapshot()

diff = snapshot2.compare_to(snapshot1, 'lineno')
for stat in diff[:10]:
    print(f"Change: {stat.size_diff / 1024 / 1024:.2f} MB")
```

### Growth Pattern Analysis

```python
# Analyze memory growth patterns
growth_analysis = profiler.analyze_growth_pattern()

print(f"Growth rate: {growth_analysis['rate_mb_per_hour']:.2f} MB/hour")
print(f"R-squared: {growth_analysis['r_squared']:.4f}")
print(f"Projected 24h usage: {growth_analysis['projected_24h_mb']:.0f} MB")

if growth_analysis['is_leak']:
    print("WARNING: Memory leak detected!")
    print(f"Confidence: {growth_analysis['confidence']:.2%}")
```

### Object Tracking

```python
# Track specific object types
profiler.track_object_type(dict)
profiler.track_object_type(list)
profiler.track_object_type(OrderBook)

# Get object statistics
stats = profiler.get_object_stats()
for obj_type, count in stats.items():
    print(f"{obj_type.__name__}: {count} instances")
```

## Metrics and Reporting

### Real-Time Metrics

During profiling, the following metrics are collected:
- Current memory usage (RSS, VMS)
- Memory growth rate
- GC collection counts
- Object allocation rate
- Peak memory usage

### Memory Report Structure

```json
{
  "summary": {
    "duration_hours": 168,
    "start_memory_mb": 256,
    "end_memory_mb": 512,
    "peak_memory_mb": 768,
    "average_memory_mb": 384,
    "growth_rate_mb_per_hour": 1.5
  },
  "leak_detection": {
    "leak_detected": true,
    "confidence": 0.95,
    "leak_rate_mb_per_hour": 1.5,
    "projected_oom_hours": 240
  },
  "top_allocators": [
    {
      "file": "order_book.py",
      "line": 145,
      "function": "add_order",
      "size_mb": 128,
      "count": 1000000
    }
  ],
  "gc_statistics": {
    "gen0_collections": 5420,
    "gen1_collections": 492,
    "gen2_collections": 5,
    "total_gc_time_seconds": 45.2
  },
  "snapshots": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "memory_mb": 256,
      "top_types": {
        "dict": 45000,
        "list": 32000,
        "str": 128000
      }
    }
  ]
}
```

## Best Practices

### 1. Baseline Establishment
Always establish a baseline before optimization:
```python
# Run baseline profiling
baseline = await profiler.create_baseline(duration_hours=1)
print(f"Baseline memory: {baseline['average_mb']} MB")
```

### 2. Incremental Profiling
Profile incrementally during development:
```python
# Profile after each major change
async def profile_change(description):
    before = profiler.current_memory_mb
    await apply_change()
    after = profiler.current_memory_mb
    print(f"{description}: {after - before:+.2f} MB")
```

### 3. Production-Like Conditions
Profile under realistic conditions:
```python
# Simulate production load
await generate_production_load()
await profiler.monitor_continuous(hours=24)
```

### 4. Multiple Profiling Runs
Run multiple profiles to ensure consistency:
```python
results = []
for run in range(3):
    profiler = MemoryProfiler()
    await profiler.monitor_continuous(hours=1)
    results.append(profiler.generate_report())

# Compare results
validate_consistency(results)
```

### 5. Automated Leak Detection
Integrate leak detection into CI/CD:
```python
# CI/CD integration
async def test_memory_leak():
    profiler = MemoryProfiler(growth_threshold_mb_per_hour=1)
    await profiler.monitor_continuous(hours=1)
    
    report = profiler.generate_report()
    assert not report['leak_detection']['leak_detected'], \
        f"Memory leak detected: {report['leak_detection']}"
```

## Troubleshooting

### High Memory Usage

1. **Identify Top Consumers**
```python
top_stats = profiler.get_top_memory_consumers(10)
for stat in top_stats:
    print(f"{stat.size_mb:.2f} MB: {stat.location}")
```

2. **Check Object Counts**
```python
obj_counts = profiler.get_object_counts()
for type_name, count in sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{type_name}: {count:,}")
```

3. **Review Allocation Patterns**
```python
patterns = profiler.analyze_allocation_patterns()
for pattern in patterns['suspicious']:
    print(f"Suspicious pattern: {pattern}")
```

### Memory Leak Confirmation

1. **Extended Monitoring**
```python
# Monitor for extended period
await profiler.monitor_continuous(hours=24)
if profiler.confirm_memory_leak(confidence=0.95):
    print("Memory leak confirmed with 95% confidence")
```

2. **Snapshot Differencing**
```python
# Take snapshots at intervals
snapshots = []
for i in range(10):
    await asyncio.sleep(3600)  # 1 hour
    snapshots.append(profiler.take_snapshot())

# Analyze growth
growth = analyze_snapshot_growth(snapshots)
```

3. **Traceback Analysis**
```python
# Get allocation tracebacks
leaking_traces = profiler.get_leaking_tracebacks()
for trace in leaking_traces:
    print(f"Leak source: {trace}")
```

## Performance Optimization

### Memory-Efficient Patterns

1. **Object Pooling**
```python
# Reuse objects instead of creating new ones
order_pool = ObjectPool(Order, size=1000)
order = order_pool.acquire()
# Use order
order_pool.release(order)
```

2. **Weak References**
```python
# Use weak references for caches
import weakref
cache = weakref.WeakValueDictionary()
```

3. **Generator Functions**
```python
# Use generators for large datasets
def process_orders():
    for order in large_order_list:
        yield process(order)  # Don't hold all in memory
```

4. **Slots for Classes**
```python
# Use __slots__ to reduce memory overhead
class Order:
    __slots__ = ['id', 'price', 'quantity', 'side']
```

### Garbage Collection Tuning

```python
import gc

# Tune GC thresholds
gc.set_threshold(700, 10, 10)  # Default is (700, 10, 10)

# Force collection during idle periods
async def periodic_gc():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        if is_idle():
            gc.collect()
```

## Integration Examples

### GitHub Actions Integration

```yaml
name: Memory Leak Detection
on: [push, pull_request]

jobs:
  memory-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run memory profiling
        run: |
          python -m tests.stress.memory_profiler \
            --duration 3600 \
            --growth-threshold 5 \
            --output memory_report.json
      - name: Check for leaks
        run: |
          python -c "
          import json
          with open('memory_report.json') as f:
              report = json.load(f)
          assert not report['leak_detection']['leak_detected'], \
              'Memory leak detected'
          "
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: memory-report
          path: memory_report.json
```

### Docker Integration

```dockerfile
# Dockerfile with memory profiling
FROM python:3.11

# Install memory profiling tools
RUN pip install memory_profiler pympler tracemalloc

# Set memory limits
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRACE=1

# Run with memory profiling
CMD ["python", "-m", "tests.stress.memory_profiler", "--monitor"]
```

## Related Documentation

- [Load Generator](load_generator.md)
- [Continuous Operation](continuous_operation.md)
- [Database Stress Testing](database_stress.md)
- [Performance Baseline](../qa/performance_baseline.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*