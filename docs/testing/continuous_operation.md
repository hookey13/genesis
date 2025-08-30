# Continuous Operation Testing Guide

## Overview

The continuous operation test validates system stability over extended periods (24+ hours) using paper trading mode. This test ensures the system can handle real-world conditions without degradation.

## Test Components

### 1. Stability Metrics
- **Uptime tracking**: Monitors total operational time
- **Error rates**: Tracks all types of errors per hour
- **Memory usage**: Monitors for memory leaks
- **CPU usage**: Ensures efficient resource utilization
- **Trading metrics**: Orders placed/filled, positions opened/closed

### 2. Mock Exchange Data
- Realistic price movements with configurable volatility
- Order book simulation with 20 depth levels
- Trade stream generation (1-10 trades/second)
- Market conditions simulation (trending, ranging, volatile)

### 3. Paper Trading Operations
- Order placement and cancellation
- Position management
- Market data fetching
- WebSocket stream handling
- Error recovery simulation

## Running the Test

### Quick Test (1 hour)
```bash
python -m tests.stress.continuous_operation --quick
```

### Full 24-Hour Test
```bash
python -m tests.stress.continuous_operation --duration 24
```

### Custom Duration
```bash
python -m tests.stress.continuous_operation --duration 48  # 48-hour test
```

## Success Criteria

The test is considered successful if:
1. **Zero critical errors** during the test period
2. **Memory usage stable** (no continuous growth)
3. **CPU usage < 80%** average
4. **Error rate < 1 per hour**
5. **Position consistency maintained** (no orphaned orders/positions)
6. **All validations pass**

## Monitoring During Test

The test provides real-time monitoring:
- Progress updates every minute
- Memory/CPU metrics every 30 seconds
- Anomaly alerts for high resource usage
- Trading operation statistics

## Output Reports

Reports are generated in `tests/stress/reports/` with:
- Complete metrics summary
- Error analysis
- Resource usage graphs
- Position/order consistency validation
- Pass/fail determination

### Report Format
```json
{
  "test_type": "continuous_operation",
  "duration_hours": 24,
  "consistency_valid": true,
  "metrics": {
    "uptime_hours": 24.0,
    "total_errors": 0,
    "error_rate_per_hour": 0.0,
    "average_memory_mb": 250.5,
    "peak_memory_mb": 380.2,
    "average_cpu_percent": 15.3,
    "positions_opened": 1250,
    "positions_closed": 1248,
    "orders_placed": 2500,
    "orders_filled": 2250
  },
  "success": true
}
```

## Troubleshooting

### High Memory Usage
- Check for unclosed database connections
- Review asyncio task cleanup
- Analyze object retention in positions/orders dicts

### High Error Rate
- Review error logs for patterns
- Check network connectivity
- Verify mock exchange data generation

### Consistency Failures
- Check for race conditions in order processing
- Verify position state transitions
- Review order lifecycle management

## Integration with CI/CD

### Pre-Production Validation
```yaml
# .github/workflows/pre-production.yml
- name: Run 24-hour stability test
  run: |
    python -m tests.stress.continuous_operation --duration 24
    if [ $? -ne 0 ]; then
      echo "Stability test failed"
      exit 1
    fi
```

### Nightly Regression
```yaml
# Run shorter 4-hour test nightly
- name: Nightly stability test
  run: python -m tests.stress.continuous_operation --duration 4
```

## Best Practices

1. **Run before major releases**: Always run full 24-hour test
2. **Monitor actively**: Check progress periodically during long tests
3. **Analyze trends**: Compare reports across multiple runs
4. **Incremental testing**: Start with 1-hour, then 4-hour, then 24-hour
5. **Resource allocation**: Ensure test environment has sufficient resources

## Related Tests

- Load Testing: `tests/stress/load_generator.py`
- Chaos Engineering: `tests/chaos/chaos_engine.py`
- Memory Profiling: `tests/stress/memory_profiler.py`
- Database Stress: `tests/stress/db_stress.py`