# DR Testing Guide

## Overview

The DR testing framework provides automated testing capabilities including scheduled drills, chaos engineering, and performance benchmarking. Regular testing ensures DR procedures work when needed.

## Testing Framework

### Components

1. **DR Test Runner** - Orchestrates test execution
2. **Chaos Engineering** - Injects controlled failures
3. **Performance Benchmarking** - Measures recovery metrics
4. **Automated Scheduling** - Monthly DR drills

## Running DR Drills

### Manual Drill Execution

```python
from genesis.dr import DRTestRunner

# Initialize test runner
test_runner = DRTestRunner(dr_orchestrator)

# Run basic drill
result = await test_runner.run_dr_drill(
    scenario_name="DATABASE_CORRUPTION",
    include_chaos=False,
    notification=True
)

print(f"Drill success: {result['success']}")
print(f"Recovery time: {result['duration_seconds']} seconds")
print(f"RTO met: {result['rto_met']}")
```

### Drill with Chaos Engineering

```python
# Run drill with chaos injection
result = await test_runner.run_dr_drill(
    scenario_name="PRIMARY_FAILURE",
    include_chaos=True,  # Randomly inject failures
    notification=True
)

# Specific chaos scenario
chaos_result = await test_runner.run_chaos_scenario("network_latency")
```

## Chaos Engineering Scenarios

### Available Scenarios

| Scenario | Description | Duration | Intensity |
|----------|-------------|----------|-----------|
| network_latency | Inject network delays | 60s | 0.5 |
| service_failure | Simulate service crashes | 30s | 0.7 |
| database_slowdown | Slow database queries | 45s | 0.6 |
| disk_pressure | Simulate disk space issues | 90s | 0.8 |
| api_errors | Inject API failures | 60s | 0.3 |

### Running Chaos Tests

```python
# Run specific chaos scenario
result = await test_runner.run_chaos_scenario("database_slowdown")

# Custom chaos configuration
from genesis.dr.dr_test_runner import ChaosScenario

custom_chaos = ChaosScenario(
    name="custom_failure",
    description="Custom failure injection",
    fault_injector=my_fault_injector,
    duration_seconds=120,
    intensity=0.9
)

result = await test_runner._run_chaos_scenario(custom_chaos)
```

## Scheduled Testing

### Monthly DR Drills

Automatically runs on the first Monday of each month at 2 AM:

```python
# Start scheduled drills
asyncio.create_task(test_runner.schedule_monthly_drill())

# Check next scheduled drill
next_drill = test_runner._get_next_drill_time()
print(f"Next drill: {next_drill}")
```

### Custom Schedule

```python
# Custom drill schedule
async def custom_schedule():
    while True:
        # Run drill every Sunday at 3 AM
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        
        if days_until_sunday == 0 and now.hour >= 3:
            days_until_sunday = 7
        
        next_run = now + timedelta(days=days_until_sunday)
        next_run = next_run.replace(hour=3, minute=0, second=0)
        
        wait_seconds = (next_run - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        
        # Run drill
        await test_runner.run_dr_drill(
            scenario_name="COMPLETE_DISASTER",
            include_chaos=True,
            notification=True
        )
```

## Test Scenarios

### Scenario 1: Database Corruption

Tests recovery from corrupted database:

```python
# Simulate database corruption
await test_runner.run_dr_drill(
    scenario_name="DATABASE_CORRUPTION",
    include_chaos=False,
    notification=True
)
```

Expected actions:
1. Stop trading
2. Create backup
3. Recover to last good state
4. Validate recovery
5. Resume trading

### Scenario 2: Primary Infrastructure Failure

Tests failover to backup infrastructure:

```python
# Simulate primary failure
await test_runner.run_dr_drill(
    scenario_name="PRIMARY_FAILURE",
    include_chaos=True,  # Add network issues
    notification=True
)
```

Expected actions:
1. Detect failure
2. Initiate failover
3. Verify backup services
4. Update DNS
5. Notify team

### Scenario 3: Complete Disaster

Tests full disaster recovery:

```python
# Simulate complete disaster
await test_runner.run_dr_drill(
    scenario_name="COMPLETE_DISASTER",
    include_chaos=True,
    notification=True
)
```

Expected actions:
1. Emergency close all positions
2. Activate DR site
3. Restore from backup
4. Verify all systems
5. Gradual resume

## Performance Benchmarking

### Collecting Metrics

```python
# Get performance metrics
metrics = test_runner.get_performance_metrics()

print(f"Total tests: {metrics['total_tests']}")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Avg recovery time: {metrics['average_recovery_time']:.1f} min")
print(f"RTO compliance: {metrics['rto_compliance']:.1%}")
```

### Test History

```python
# Get recent test results
history = test_runner.get_test_history(limit=10)

for test in history:
    print(f"{test['scenario']}: {test['success']} ({test['duration_seconds']}s)")
```

### Generate Report

```python
# Generate comprehensive test report
report = test_runner.generate_test_report()
print(report)

# Save to file
with open("dr_test_report.txt", "w") as f:
    f.write(report)
```

## Validation Testing

### Recovery Validation

After each drill, validate:

```python
# Validate recovery
validation = await test_runner._validate_recovery()

print(f"Valid: {validation['is_valid']}")
print(f"Passed: {validation['checks_passed']}")
print(f"Failed: {validation['checks_failed']}")
```

### Checklist Validation

- [ ] Backup available and recent
- [ ] Replication lag < 5 minutes
- [ ] Failover monitoring active
- [ ] All services healthy
- [ ] Database integrity verified
- [ ] Positions match exchange
- [ ] No data loss detected

## Test Playbooks

### Weekly Quick Test

Run every week to maintain readiness:

```python
async def weekly_quick_test():
    # Quick backup/restore test
    result = await test_runner.run_dr_drill(
        scenario_name="DATABASE_CORRUPTION",
        include_chaos=False,
        notification=False  # Silent test
    )
    
    if not result["success"]:
        # Alert on failure
        await send_alert("Weekly DR test failed")
```

### Monthly Full Test

Comprehensive monthly testing:

```python
async def monthly_full_test():
    scenarios = [
        "DATABASE_CORRUPTION",
        "PRIMARY_FAILURE",
        "DATA_LOSS",
        "COMPLETE_DISASTER"
    ]
    
    results = []
    for scenario in scenarios:
        result = await test_runner.run_dr_drill(
            scenario_name=scenario,
            include_chaos=True,
            notification=True
        )
        results.append(result)
    
    # Generate report
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    print(f"Monthly test success rate: {success_rate:.1%}")
```

### Quarterly Disaster Simulation

Full-scale disaster simulation:

```python
async def quarterly_disaster_simulation():
    # Notify team in advance
    await send_notification("Quarterly DR simulation starting in 1 hour")
    await asyncio.sleep(3600)
    
    # Run complete disaster scenario
    result = await test_runner.run_dr_drill(
        scenario_name="COMPLETE_DISASTER",
        include_chaos=True,
        notification=True
    )
    
    # Detailed analysis
    if result["success"]:
        print("Quarterly simulation PASSED")
        # Document lessons learned
    else:
        print("Quarterly simulation FAILED")
        # Schedule remediation
```

## Success Criteria

### RTO Targets

| Scenario | Target | Acceptable |
|----------|--------|------------|
| Database Corruption | 10 min | 15 min |
| Primary Failure | 15 min | 20 min |
| Network Partition | 5 min | 10 min |
| Data Loss | 15 min | 25 min |
| Complete Disaster | 30 min | 45 min |

### RPO Targets

- Maximum data loss: 5 minutes
- Backup frequency: 4 hours (full), 5 minutes (incremental)
- Replication lag: < 5 minutes

## Troubleshooting Test Failures

### Common Issues

#### Test Timeout
**Symptom**: Test exceeds time limit
**Solution**: 
- Check for blocking operations
- Verify network connectivity
- Review timeout settings

#### Validation Failures
**Symptom**: Post-test validation fails
**Solution**:
- Check backup availability
- Verify replication status
- Ensure services are healthy

#### Chaos Injection Errors
**Symptom**: Chaos scenarios fail to inject
**Solution**:
- Verify permissions
- Check resource availability
- Review chaos configuration

## Best Practices

1. **Test Regularly**
   - Weekly quick tests
   - Monthly comprehensive drills
   - Quarterly disaster simulations

2. **Document Results**
   - Record all test outcomes
   - Track performance trends
   - Document lessons learned

3. **Improve Continuously**
   - Address test failures immediately
   - Optimize recovery procedures
   - Update documentation

4. **Realistic Testing**
   - Include chaos engineering
   - Test during business hours
   - Simulate actual failures

## Metrics and KPIs

Track these metrics:

- **Test Success Rate**: Target > 95%
- **Average Recovery Time**: Target < RTO
- **RTO Compliance**: Target > 99%
- **Test Coverage**: All scenarios monthly
- **Time Since Last Test**: < 30 days

## Related Documentation

- [DR Runbook](dr-runbook.md)
- [Backup Procedures](backup-procedures.md)
- [Recovery Guide](recovery-guide.md)
- [Failover Procedures](failover-procedures.md)