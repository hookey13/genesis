# Failover Procedures

## Overview

The failover system provides automatic and manual failover capabilities between primary and backup infrastructure. It includes health monitoring, DNS management, and coordinated service switching.

## Components

### 1. Health Checker
- Monitors critical services (database, API, exchange connectivity)
- Configurable health check intervals and failure thresholds
- Supports multiple check types: HTTP, TCP, database, process

### 2. DNS Manager
- Manages DNS records for traffic routing
- Supports DigitalOcean, Cloudflare, and Route53
- Automatic DNS failover with propagation verification

### 3. Failover Coordinator
- Orchestrates failover operations
- Manages pre/post failover hooks
- Tracks failover state and history
- Implements cooldown periods to prevent flapping

## Automatic Failover

### Configuration

```python
from genesis.failover import FailoverCoordinator, HealthChecker, DNSManager

# Initialize components
health_checker = HealthChecker()
dns_manager = DNSManager(
    api_token=os.environ["DO_API_TOKEN"],
    provider="digitalocean"
)

coordinator = FailoverCoordinator(
    health_checker=health_checker,
    dns_manager=dns_manager,
    notification_channels=["email", "slack", "pagerduty"]
)

# Start monitoring
await coordinator.start_monitoring()
```

### Health Check Configuration

```python
from genesis.failover import HealthCheck

# Database health check
health_checker.add_check(
    HealthCheck(
        name="primary_database",
        check_type="database",
        target=".genesis/data/genesis.db",
        interval_seconds=10,
        timeout_seconds=5,
        failure_threshold=3  # Unhealthy after 3 failures
    )
)

# API health check
health_checker.add_check(
    HealthCheck(
        name="primary_api",
        check_type="http",
        target="https://api.genesis.primary/health",
        interval_seconds=10,
        timeout_seconds=5,
        failure_threshold=3
    )
)

# Exchange connectivity
health_checker.add_check(
    HealthCheck(
        name="exchange_connection",
        check_type="tcp",
        target="api.binance.com:443",
        interval_seconds=30,
        timeout_seconds=10,
        failure_threshold=5
    )
)
```

### Failover Triggers

Automatic failover is triggered when:
1. Critical service fails health checks
2. Failure threshold exceeded (default: 3 consecutive failures)
3. No failover in cooldown period (5 minutes)
4. Failover count under limit (5 per day)

## Manual Failover

### Execute Failover

```python
# Dry run first
result = await coordinator.execute_failover(
    reason="Planned maintenance",
    dry_run=True
)

if result["success"]:
    # Execute actual failover
    result = await coordinator.execute_failover(
        reason="Planned maintenance",
        dry_run=False
    )
    
    print(f"Failover completed in {result['duration_seconds']}s")
```

### Execute Failback

```python
# Verify primary is healthy
health_status = coordinator.health_checker.get_status()
if health_status["overall_health"]:
    # Execute failback
    result = await coordinator.execute_failback(dry_run=False)
    
    if result["success"]:
        print("Failback to primary completed")
```

## Failover Hooks

### Pre-Failover Hooks

```python
async def stop_trading():
    """Stop all trading before failover."""
    # Cancel open orders
    # Close positions if needed
    logger.info("Trading stopped for failover")

coordinator.register_pre_failover_hook(stop_trading)
```

### Post-Failover Hooks

```python
async def verify_services():
    """Verify services after failover."""
    # Check database connectivity
    # Verify API endpoints
    # Test exchange connection
    logger.info("Services verified after failover")

coordinator.register_post_failover_hook(verify_services)
```

## DNS Failover

### Manual DNS Update

```python
# Update DNS to backup IP
success = await dns_manager.failover_dns(
    domain="genesis.example.com",
    from_ip="1.2.3.4",  # Primary IP
    to_ip="5.6.7.8",    # Backup IP
    record_type="A"
)

# Verify propagation
propagation = await dns_manager.verify_dns_propagation(
    domain="genesis.example.com",
    expected_ip="5.6.7.8",
    nameservers=["8.8.8.8", "1.1.1.1", "9.9.9.9"]
)

for ns, propagated in propagation.items():
    print(f"{ns}: {'✓' if propagated else '✗'}")
```

## Monitoring and Alerts

### Health Status

```python
# Get current health status
status = health_checker.get_status()

print(f"Overall Health: {status['overall_health']}")
print(f"Healthy Checks: {status['healthy_checks']}/{status['total_checks']}")

for check_name, check_status in status["checks"].items():
    print(f"{check_name}: {'✓' if check_status['is_healthy'] else '✗'}")
    if check_status["last_error"]:
        print(f"  Error: {check_status['last_error']}")
```

### Failover Status

```python
# Get failover coordinator status
status = coordinator.get_status()

print(f"State: {status['state']}")
print(f"Failover Count: {status['failover_count']}")
print(f"Last Failover: {status['last_failover']}")

# Check if in failed over state
if status["state"] == "failed_over":
    print("WARNING: Running on backup infrastructure")
```

## Failover Checklist

### Pre-Failover
- [ ] Verify backup infrastructure is ready
- [ ] Check database replication lag
- [ ] Ensure backup configurations are current
- [ ] Notify team of impending failover
- [ ] Stop non-critical processes

### During Failover
- [ ] Monitor failover progress
- [ ] Watch for errors in logs
- [ ] Verify each step completes
- [ ] Check notification delivery

### Post-Failover
- [ ] Verify all services operational
- [ ] Check database connectivity
- [ ] Test API endpoints
- [ ] Verify exchange connection
- [ ] Monitor for issues
- [ ] Document failover details

### Failback Preparation
- [ ] Ensure primary is fully recovered
- [ ] Sync data from backup to primary
- [ ] Test primary services
- [ ] Schedule failback window
- [ ] Notify team

## Troubleshooting

### Common Issues

#### 1. Health Check Failures
**Symptom**: False positive health check failures
**Solution**: 
- Increase timeout values
- Adjust failure threshold
- Check network connectivity

#### 2. DNS Propagation Delays
**Symptom**: Traffic still going to old IP
**Solution**:
- Lower TTL before failover
- Use multiple DNS providers
- Implement client-side failover

#### 3. Failover Loops
**Symptom**: System keeps failing over
**Solution**:
- Check cooldown period
- Increase failure threshold
- Investigate root cause

#### 4. Partial Failover
**Symptom**: Some services on primary, others on backup
**Solution**:
- Execute full failover
- Verify all connection strings updated
- Check service dependencies

## Best Practices

1. **Regular Testing**
   - Monthly failover drills
   - Quarterly full DR tests
   - Document lessons learned

2. **Monitoring**
   - Alert on health check failures
   - Track failover metrics
   - Monitor replication lag

3. **Documentation**
   - Keep runbooks updated
   - Document configuration changes
   - Maintain contact lists

4. **Automation**
   - Automate routine checks
   - Script common procedures
   - Implement self-healing where possible

## Configuration Examples

### Production Configuration

```python
# Production health checks
PRODUCTION_CHECKS = [
    {
        "name": "primary_db",
        "type": "database",
        "target": "/var/genesis/data/genesis.db",
        "interval": 10,
        "timeout": 5,
        "threshold": 3
    },
    {
        "name": "api_health",
        "type": "http",
        "target": "https://api.genesis.io/health",
        "interval": 10,
        "timeout": 5,
        "threshold": 3
    },
    {
        "name": "binance_api",
        "type": "tcp",
        "target": "api.binance.com:443",
        "interval": 30,
        "timeout": 10,
        "threshold": 5
    },
    {
        "name": "redis_cache",
        "type": "tcp",
        "target": "localhost:6379",
        "interval": 10,
        "timeout": 3,
        "threshold": 3
    }
]

# Initialize with production config
for check_config in PRODUCTION_CHECKS:
    health_checker.add_check(
        HealthCheck(
            name=check_config["name"],
            check_type=check_config["type"],
            target=check_config["target"],
            interval_seconds=check_config["interval"],
            timeout_seconds=check_config["timeout"],
            failure_threshold=check_config["threshold"]
        )
    )
```

## Related Documentation
- [Backup Procedures](backup-procedures.md)
- [Recovery Guide](recovery-guide.md)
- [DR Runbook](dr-runbook.md)
- [Emergency Procedures](emergency-close.md)