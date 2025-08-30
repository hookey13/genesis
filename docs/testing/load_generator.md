# Load Generator Documentation

## Overview

The Load Generator is a comprehensive testing framework designed to simulate high-volume message traffic and validate system performance under stress conditions. It enables testing of system scalability, throughput limits, and performance degradation patterns by generating configurable loads that mimic real-world trading scenarios.

## Purpose

The Load Generator serves several critical functions:
- Validates system capacity to handle 100x normal message volume
- Identifies performance bottlenecks before production deployment  
- Tests graceful degradation under extreme load conditions
- Measures latency distribution and throughput metrics
- Simulates various market conditions (normal, volatile, flash crash)

## Architecture

The Load Generator consists of three main components:

### 1. LoadGenerator Class
The main orchestrator that manages load testing execution, coordinates message generation, and collects performance metrics.

### 2. WebSocketMessageGenerator
Generates realistic WebSocket messages that simulate exchange data feeds, including:
- Order book updates
- Trade executions
- Ticker updates
- Account balance changes

### 3. LoadProfile Enum
Defines predefined load patterns:
- NORMAL: Steady state trading conditions
- VOLATILE: High-frequency market movements
- SPIKE: Sudden traffic surges
- RAMP_UP: Gradual increase in load
- RAMP_DOWN: Gradual decrease in load

## Configuration

### Basic Configuration

```python
from tests.stress.load_generator import LoadGenerator, LoadProfile

# Initialize with target system
generator = LoadGenerator(
    target_system=your_message_processor,
    base_rate=100,  # Base messages per second
    report_interval=10  # Report metrics every 10 seconds
)
```

### Advanced Configuration

```python
# Configure with custom settings
generator = LoadGenerator(
    target_system=processor,
    base_rate=100,
    report_interval=10,
    max_concurrent=1000,  # Max concurrent messages
    timeout_seconds=5,  # Message processing timeout
    error_threshold=0.01  # 1% error rate threshold
)
```

### Load Profiles

| Profile | Description | Use Case |
|---------|------------|----------|
| NORMAL | Steady 1x rate | Baseline testing |
| VOLATILE | Variable 0.5x-2x rate | Market volatility simulation |
| SPIKE | Sudden 10x bursts | Flash crash scenarios |
| RAMP_UP | Gradual 0x-5x increase | Capacity testing |
| RAMP_DOWN | Gradual 5x-0x decrease | Recovery testing |

## Usage Examples

### Basic Load Test

```python
import asyncio
from tests.stress.load_generator import LoadGenerator, LoadProfile

async def run_basic_test():
    # Create message processor
    async def process_message(msg):
        # Your processing logic here
        await asyncio.sleep(0.001)  # Simulate processing
        return True
    
    # Initialize generator
    generator = LoadGenerator(target_system=process_message)
    
    # Run 60-second test at 10x normal load
    await generator.run_load_test(
        duration_seconds=60,
        multiplier=10,
        profile=LoadProfile.NORMAL
    )
    
    # Get results
    report = generator.get_report()
    print(f"Messages sent: {report['total_messages']}")
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Avg latency: {report['avg_latency_ms']:.2f}ms")

asyncio.run(run_basic_test())
```

### Stress Test with Ramp-Up

```python
async def run_stress_test():
    generator = LoadGenerator(
        target_system=process_message,
        base_rate=100,
        error_threshold=0.05  # Allow 5% errors
    )
    
    # Gradually increase load
    await generator.run_load_test(
        duration_seconds=300,  # 5 minutes
        multiplier=100,  # Target 100x load
        profile=LoadProfile.RAMP_UP
    )
    
    # Analyze breaking point
    metrics = generator.metrics
    if metrics.error_rate > 0.01:
        print(f"System degraded at {metrics.peak_throughput} msg/s")

asyncio.run(run_stress_test())
```

### Market Volatility Simulation

```python
async def simulate_volatility():
    generator = LoadGenerator(target_system=process_message)
    
    # Simulate volatile market conditions
    await generator.run_load_test(
        duration_seconds=600,  # 10 minutes
        multiplier=50,
        profile=LoadProfile.VOLATILE
    )
    
    # Check system stability
    report = generator.get_report()
    if report['p99_latency_ms'] > 100:
        print("Warning: High tail latency detected")

asyncio.run(simulate_volatility())
```

## Command-Line Usage

The load generator can be run from the command line:

```bash
# Basic test
python -m tests.stress.load_generator --duration 60 --multiplier 10

# Stress test with specific profile
python -m tests.stress.load_generator \
    --duration 300 \
    --multiplier 100 \
    --profile RAMP_UP \
    --base-rate 200

# Test with custom error threshold
python -m tests.stress.load_generator \
    --duration 120 \
    --multiplier 50 \
    --error-threshold 0.02 \
    --report-interval 5
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --duration | 60 | Test duration in seconds |
| --multiplier | 10 | Load multiplier (e.g., 10 = 10x normal) |
| --profile | NORMAL | Load profile to use |
| --base-rate | 100 | Base messages per second |
| --error-threshold | 0.01 | Maximum acceptable error rate |
| --report-interval | 10 | Seconds between metric reports |
| --output | stdout | Output file for results (JSON) |

## Metrics and Reporting

### Real-Time Metrics

During test execution, the following metrics are reported:
- Messages sent/processed per second
- Current error rate
- Average and P99 latency
- Memory usage
- Active concurrent messages

### Final Report

The final report includes:
- Total messages sent and processed
- Success/failure counts
- Latency percentiles (P50, P90, P95, P99)
- Throughput statistics
- Error analysis
- Resource utilization

### Example Report

```json
{
  "test_duration": 60,
  "total_messages": 60000,
  "messages_processed": 59940,
  "messages_failed": 60,
  "success_rate": 0.999,
  "avg_latency_ms": 12.5,
  "p50_latency_ms": 10.2,
  "p90_latency_ms": 18.7,
  "p95_latency_ms": 25.3,
  "p99_latency_ms": 45.8,
  "peak_throughput": 1250,
  "avg_throughput": 1000,
  "error_rate": 0.001,
  "memory_usage_mb": 256
}
```

## Best Practices

### 1. Gradual Load Increase
Always start with low multipliers and gradually increase to identify breaking points:
```python
for multiplier in [1, 10, 50, 100, 200]:
    await generator.run_load_test(60, multiplier, LoadProfile.NORMAL)
    if generator.metrics.error_rate > 0.01:
        break
```

### 2. Baseline Establishment
Run baseline tests at 1x load to establish normal performance:
```python
baseline = await generator.run_load_test(300, 1, LoadProfile.NORMAL)
```

### 3. Profile Selection
Choose appropriate profiles for different test scenarios:
- Use NORMAL for capacity testing
- Use VOLATILE for resilience testing
- Use SPIKE for failure recovery testing
- Use RAMP_UP for breaking point identification

### 4. Error Threshold Configuration
Set realistic error thresholds based on requirements:
- Production systems: 0.001 (0.1%)
- Development testing: 0.01 (1%)
- Stress testing: 0.05 (5%)

### 5. Resource Monitoring
Monitor system resources during tests:
```python
import psutil

async def monitor_resources():
    while generator.running:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        if cpu > 80 or memory > 80:
            logger.warning(f"High resource usage: CPU={cpu}%, MEM={memory}%")
        await asyncio.sleep(1)
```

## Troubleshooting

### High Error Rates
If experiencing high error rates:
1. Reduce multiplier value
2. Increase timeout_seconds
3. Check target system logs
4. Verify network capacity
5. Monitor resource utilization

### Memory Issues
For memory-related problems:
1. Reduce max_concurrent setting
2. Enable message batching
3. Implement backpressure handling
4. Use streaming processing

### Latency Spikes
To address latency spikes:
1. Check for GC pauses
2. Analyze message processing logic
3. Verify database query performance
4. Review network configuration

### AsyncIterator Import Error
If you encounter an import error for AsyncIterator:
```python
from typing import AsyncIterator  # Add this import
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Load Testing
on: [push, pull_request]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run load tests
        run: |
          python -m tests.stress.load_generator \
            --duration 120 \
            --multiplier 50 \
            --error-threshold 0.01
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: load-test-results
          path: load_test_report.json
```

## Performance Optimization Tips

1. **Use asyncio effectively**: Leverage asyncio.gather() for concurrent processing
2. **Implement connection pooling**: Reuse connections to reduce overhead
3. **Enable message batching**: Process messages in batches when possible
4. **Optimize serialization**: Use efficient serialization formats (msgpack, protobuf)
5. **Implement circuit breakers**: Prevent cascade failures under load

## Related Documentation

- [Continuous Operation Testing](continuous_operation.md)
- [Chaos Engineering](chaos_engineering.md)
- [Memory Profiling](memory_profiling.md)
- [Database Stress Testing](database_stress.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*