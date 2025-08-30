# Chaos Engineering Documentation

## Overview

Chaos Engineering is the discipline of experimenting on a distributed system to build confidence in the system's capability to withstand turbulent conditions in production. Our chaos engineering framework, the ChaosMonkey, systematically injects failures to validate system resilience and recovery mechanisms.

## Philosophy

The core principle of chaos engineering is to proactively discover weaknesses before they manifest in production. By introducing controlled failures, we can:
- Validate fault tolerance mechanisms
- Test recovery procedures
- Identify hidden dependencies
- Build confidence in system resilience
- Train teams for incident response

## Architecture

### Core Components

#### 1. ChaosMonkey Class
The main orchestrator that manages chaos injection, coordinates failure scenarios, and tracks recovery metrics.

#### 2. ChaosType Enum
Defines types of chaos that can be injected:
- PROCESS_KILL: Terminates processes
- PROCESS_RESTART: Restarts services
- NETWORK_DELAY: Adds network latency
- NETWORK_LOSS: Drops network packets
- NETWORK_PARTITION: Creates network splits
- CPU_STRESS: Causes high CPU usage
- MEMORY_STRESS: Causes memory pressure
- DISK_STRESS: Creates I/O bottlenecks
- DATABASE_SLOW: Slows database queries
- API_FAILURE: Simulates API failures

#### 3. ChaosEvent Dataclass
Records each chaos injection event with:
- Type of chaos
- Timestamp and duration
- Target component
- Parameters used
- Impact assessment
- Recovery status

#### 4. ChaosMetrics
Tracks metrics during chaos testing:
- Events injected
- Failures detected
- Recovery times
- Service availability
- Data consistency

## Configuration

### Basic Setup

```python
from tests.chaos.chaos_engine import ChaosMonkey, ChaosType

# Initialize with default settings
monkey = ChaosMonkey()

# Initialize with custom recovery validator
async def validate_system():
    # Check system health
    return all([
        await check_api_health(),
        await check_database_health(),
        await check_message_queue()
    ])

monkey = ChaosMonkey(recovery_validator=validate_system)
```

### Advanced Configuration

```python
# Configure with target system details
target_system = {
    "services": ["trading_engine", "risk_manager", "order_executor"],
    "databases": ["postgres_main", "redis_cache"],
    "network": {"nodes": ["node1", "node2", "node3"]},
    "processes": {"pids": [1234, 5678, 9012]}
}

monkey = ChaosMonkey(
    target_system=target_system,
    recovery_validator=validate_system
)
```

## Chaos Types

### Process Failures

#### Process Kill
Simulates sudden process termination:
```python
await monkey.inject_chaos(ChaosType.PROCESS_KILL)
```
**Impact**: Immediate service unavailability
**Recovery**: Process restart via supervisor
**Validation**: Service health checks

#### Process Restart
Simulates service restart:
```python
await monkey.inject_chaos(ChaosType.PROCESS_RESTART)
```
**Impact**: Temporary service interruption
**Recovery**: Automatic after restart
**Validation**: Connection reestablishment

### Network Failures

#### Network Delay
Adds latency to network communications:
```python
await monkey.inject_chaos(ChaosType.NETWORK_DELAY)
# Adds 100-1000ms random delay
```
**Impact**: Slow response times
**Recovery**: Automatic when removed
**Validation**: Latency measurements

#### Network Loss
Drops network packets randomly:
```python
await monkey.inject_chaos(ChaosType.NETWORK_LOSS)
# 5-30% packet loss rate
```
**Impact**: Connection errors, retries
**Recovery**: Automatic with retries
**Validation**: Packet delivery confirmation

#### Network Partition
Creates network splits (split-brain):
```python
await monkey.inject_chaos(ChaosType.NETWORK_PARTITION)
```
**Impact**: Cluster splits, inconsistency risk
**Recovery**: Partition healing
**Validation**: Data consistency checks

### Resource Stress

#### CPU Stress
Creates high CPU utilization:
```python
await monkey.inject_chaos(ChaosType.CPU_STRESS)
# 80% CPU usage on 2 cores
```
**Impact**: Performance degradation
**Recovery**: Load reduction
**Validation**: CPU metrics

#### Memory Stress
Causes memory pressure:
```python
await monkey.inject_chaos(ChaosType.MEMORY_STRESS)
# Allocates 100MB+
```
**Impact**: GC pressure, OOM risk
**Recovery**: Memory release
**Validation**: Memory metrics

#### Disk Stress
Creates I/O bottlenecks:
```python
await monkey.inject_chaos(ChaosType.DISK_STRESS)
```
**Impact**: Slow I/O operations
**Recovery**: I/O completion
**Validation**: Disk metrics

### Application Failures

#### Database Slowdown
Simulates slow database:
```python
await monkey.inject_chaos(ChaosType.DATABASE_SLOW)
# 10x query slowdown
```
**Impact**: Transaction delays
**Recovery**: Normal query speed
**Validation**: Query performance

#### API Failures
Simulates API errors:
```python
await monkey.inject_chaos(ChaosType.API_FAILURE)
# 50% failure rate
```
**Impact**: Request failures
**Recovery**: Retry mechanisms
**Validation**: API health checks

## Usage Examples

### Basic Chaos Test

```python
import asyncio
from tests.chaos.chaos_engine import ChaosMonkey, ChaosType

async def run_basic_chaos():
    # Create chaos monkey
    monkey = ChaosMonkey()
    
    # Run 1-hour chaos test
    await monkey.run_chaos_test(
        duration_minutes=60,
        chaos_probability=0.1,  # 10% chance per minute
        chaos_types=[
            ChaosType.NETWORK_DELAY,
            ChaosType.CPU_STRESS,
            ChaosType.API_FAILURE
        ]
    )
    
    # Check results
    report = monkey.metrics.to_dict()
    print(f"Total chaos events: {report['total_events']}")
    print(f"Successful recoveries: {report['successful_recoveries']}")
    print(f"Service availability: {report['service_availability_percent']}%")

asyncio.run(run_basic_chaos())
```

### Targeted Chaos Injection

```python
async def test_specific_failure():
    monkey = ChaosMonkey()
    
    # Test network partition recovery
    event = await monkey.inject_chaos(ChaosType.NETWORK_PARTITION)
    
    # Monitor recovery
    await asyncio.sleep(event.duration_seconds)
    
    if event.recovered:
        print(f"Recovered in {event.recovery_time_seconds}s")
    else:
        print("Recovery failed!")
    
    # Validate data consistency
    consistent = await monkey.validate_data_consistency()
    assert consistent, "Data inconsistency detected!"

asyncio.run(test_specific_failure())
```

### Game Day Simulation

```python
async def game_day_simulation():
    """Simulate production incident scenarios"""
    
    monkey = ChaosMonkey()
    scenarios = [
        # Scenario 1: Database failure
        [ChaosType.DATABASE_SLOW, ChaosType.API_FAILURE],
        
        # Scenario 2: Network issues
        [ChaosType.NETWORK_PARTITION, ChaosType.NETWORK_LOSS],
        
        # Scenario 3: Resource exhaustion
        [ChaosType.CPU_STRESS, ChaosType.MEMORY_STRESS]
    ]
    
    for scenario in scenarios:
        print(f"Running scenario: {scenario}")
        
        # Inject multiple failures
        for chaos_type in scenario:
            await monkey.inject_chaos(chaos_type)
        
        # Wait for recovery
        await asyncio.sleep(60)
        
        # Validate system state
        if await monkey.validate_data_consistency():
            print("✓ Scenario passed")
        else:
            print("✗ Scenario failed")

asyncio.run(game_day_simulation())
```

## Command-Line Usage

```bash
# Basic chaos test
python -m tests.chaos.chaos_engine --duration 60 --probability 0.1

# Test specific chaos types
python -m tests.chaos.chaos_engine \
    --duration 120 \
    --probability 0.2 \
    --types NETWORK_DELAY CPU_STRESS

# High-intensity chaos
python -m tests.chaos.chaos_engine \
    --duration 30 \
    --probability 0.5 \
    --types ALL
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --duration | 60 | Test duration in minutes |
| --probability | 0.1 | Chaos injection probability per minute |
| --types | ALL | Space-separated chaos types to inject |
| --output | chaos_report.json | Output file for results |
| --validate | true | Run data consistency validation |

## Recovery Procedures

### Automated Recovery

The system implements several automated recovery mechanisms:

1. **Process Recovery**
   - Supervisor monitors and restarts failed processes
   - Health checks validate service availability
   - Connection pools reconnect automatically

2. **Network Recovery**
   - Automatic retry with exponential backoff
   - Circuit breakers prevent cascade failures
   - Partition healing restores connectivity

3. **Resource Recovery**
   - Garbage collection frees memory
   - CPU throttling reduces load
   - I/O queues clear backlogs

### Manual Recovery

For severe failures, manual intervention may be required:

```python
# Emergency stop all chaos
await monkey.emergency_stop()

# Clear all network issues
await simulator.clear_all_issues()

# Force service restart
await restart_all_services()

# Validate and repair data
await validate_and_repair_data()
```

## Metrics and Monitoring

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MTTR | Mean Time To Recovery | <5 min |
| Availability | Service uptime percentage | >99.9% |
| Error Rate | Failed operations percentage | <0.1% |
| Recovery Rate | Successful recovery percentage | >95% |
| Data Consistency | Consistency check pass rate | 100% |

### Monitoring Dashboard

```python
# Real-time monitoring during chaos
async def monitor_chaos(monkey):
    while monkey.running:
        metrics = monkey.metrics
        print(f"Events: {len(metrics.events_injected)}")
        print(f"Failures: {metrics.failures_detected}")
        print(f"Recoveries: {metrics.successful_recoveries}")
        print(f"Availability: {metrics.calculate_availability()}%")
        await asyncio.sleep(10)
```

### Report Generation

```python
# Generate comprehensive report
await monkey.generate_report()

# Report includes:
# - All chaos events with timestamps
# - Recovery times for each event
# - Service availability calculation
# - Data consistency results
# - Success/failure determination
```

## Best Practices

### 1. Start Small
Begin with single-failure scenarios before combining multiple failures:
```python
# Start with one type
await test_single_failure(ChaosType.NETWORK_DELAY)

# Then combine
await test_multiple_failures([
    ChaosType.NETWORK_DELAY,
    ChaosType.CPU_STRESS
])
```

### 2. Monitor Everything
Comprehensive monitoring during chaos tests:
```python
monitors = [
    monitor_services(),
    monitor_resources(),
    monitor_network(),
    monitor_data_consistency()
]
await asyncio.gather(*monitors)
```

### 3. Validate Recovery
Always validate system recovery:
```python
# Check service health
assert await check_all_services_healthy()

# Validate data consistency
assert await validate_data_integrity()

# Verify no message loss
assert await check_message_delivery()
```

### 4. Document Findings
Record all findings and improvements:
- Failed scenarios and root causes
- Recovery time measurements
- Identified weaknesses
- Implemented fixes
- Lessons learned

### 5. Gradual Escalation
Increase chaos intensity gradually:
```python
for probability in [0.05, 0.1, 0.2, 0.5]:
    await monkey.run_chaos_test(
        duration_minutes=30,
        chaos_probability=probability
    )
    if monkey.metrics.failed_recoveries > 0:
        break
```

## Safety Considerations

### Production Safeguards

1. **Blast Radius Control**: Limit chaos to specific components
2. **Emergency Stop**: Ability to immediately halt all chaos
3. **Monitoring Alerts**: Real-time alerts for critical failures
4. **Rollback Plan**: Quick rollback procedures
5. **Data Backup**: Recent backups before chaos tests

### Pre-Test Checklist

- [ ] Recent backup completed
- [ ] Monitoring alerts configured
- [ ] Recovery procedures documented
- [ ] Team members notified
- [ ] Emergency contacts available
- [ ] Rollback plan ready

## Troubleshooting

### Common Issues

#### High Failure Rate
- Reduce chaos probability
- Increase recovery time allowance
- Check system resource limits
- Verify recovery mechanisms

#### Data Inconsistency
- Implement stronger consistency checks
- Add transaction boundaries
- Verify replication lag
- Check split-brain handling

#### Slow Recovery
- Optimize health check intervals
- Reduce recovery validation overhead
- Implement faster failover
- Check resource contention

## Integration Examples

### CI/CD Pipeline

```yaml
name: Chaos Testing
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  chaos-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run chaos test
        run: |
          python -m tests.chaos.chaos_engine \
            --duration 30 \
            --probability 0.1 \
            --validate true
      - name: Check results
        run: |
          python -c "
          import json
          with open('chaos_report.json') as f:
              report = json.load(f)
          assert report['success'], 'Chaos test failed'
          "
```

### Kubernetes Integration

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: chaos-monkey
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: chaos-monkey
            image: genesis/chaos-monkey:latest
            args:
              - --duration=60
              - --probability=0.1
              - --types=NETWORK_DELAY,CPU_STRESS
```

## Advanced Scenarios

### Multi-Region Failure

```python
async def test_region_failure():
    """Test multi-region failover"""
    regions = ["us-east", "us-west", "eu-central"]
    
    for region in regions:
        print(f"Failing region: {region}")
        await simulate_region_failure(region)
        
        # Verify failover
        active_region = await get_active_region()
        assert active_region != region
        
        # Restore region
        await restore_region(region)
```

### Cascade Failure Simulation

```python
async def test_cascade_failure():
    """Test cascade failure prevention"""
    
    # Start with single component failure
    await monkey.inject_chaos(ChaosType.PROCESS_KILL)
    
    # Monitor for cascade
    affected_services = await monitor_service_health()
    
    # Circuit breakers should prevent cascade
    assert len(affected_services) == 1
```

## Related Documentation

- [Load Generator](load_generator.md)
- [Network Simulation](network_simulation.md)
- [Memory Profiling](memory_profiling.md)
- [Disaster Recovery](../operations/disaster_recovery.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*