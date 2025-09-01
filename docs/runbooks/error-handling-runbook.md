# Error Handling Runbook

## Overview

This runbook provides operational procedures for handling errors, failures, and degraded conditions in the GENESIS trading system. It covers monitoring, diagnosis, and recovery procedures for common failure scenarios.

## Table of Contents

1. [Error Code Reference](#error-code-reference)
2. [Circuit Breaker Management](#circuit-breaker-management)
3. [Dead Letter Queue Operations](#dead-letter-queue-operations)
4. [Error Budget Monitoring](#error-budget-monitoring)
5. [Feature Degradation](#feature-degradation)
6. [Recovery Procedures](#recovery-procedures)
7. [Troubleshooting Flowcharts](#troubleshooting-flowcharts)

## Error Code Reference

### GENESIS Error Code Format

All errors follow the format: `GENESIS-CSNN`
- `C`: Category code (1-6)
- `S`: Severity code (3,5,7,9)
- `NN`: Specific error identifier

### Category Codes

| Code | Category | Description |
|------|----------|-------------|
| 1 | EXCHANGE | Exchange API errors |
| 2 | NETWORK | Network connectivity issues |
| 3 | DATABASE | Database operation failures |
| 4 | VALIDATION | Input validation errors |
| 5 | BUSINESS | Business logic violations |
| 6 | SYSTEM | System-level errors |

### Severity Codes

| Code | Severity | Action Required |
|------|----------|-----------------|
| 9 | CRITICAL | Immediate intervention - money at risk |
| 7 | HIGH | Urgent attention - service degraded |
| 5 | MEDIUM | Investigation needed - recoverable |
| 3 | LOW | Informational - monitoring only |

### Common Error Codes

| Error Code | Description | Remediation |
|------------|-------------|-------------|
| GENESIS-1701 | Order Rejected | Check order parameters, verify balance |
| GENESIS-1502 | Order Not Found | Verify order ID, check order history |
| GENESIS-2501 | Connection Timeout | Check network, retry with backoff |
| GENESIS-2502 | Rate Limit Exceeded | Wait for rate limit window, reduce frequency |
| GENESIS-3501 | Database Locked | Retry transaction with jitter |
| GENESIS-5701 | Risk Limit Exceeded | Reduce position size, check tier limits |
| GENESIS-5702 | Tier Violation | Verify tier requirements, adjust parameters |
| GENESIS-5903 | Tilt Intervention | Review trading behavior, implement cooldown |

## Circuit Breaker Management

### Monitoring Circuit Breaker Status

```python
# Check circuit breaker status
from genesis.core.circuit_breaker import get_circuit_breaker_registry

registry = get_circuit_breaker_registry()
status = registry.get_all_status()

# Example output:
# {
#   "binance_api": {
#     "state": "closed",
#     "failure_count": 2,
#     "success_count": 98
#   }
# }
```

### Circuit Breaker States

| State | Description | Action |
|-------|-------------|--------|
| CLOSED | Normal operation | No action needed |
| OPEN | Service failing, requests blocked | Wait for recovery timeout |
| HALF_OPEN | Testing recovery | Monitor success rate |

### Manual Circuit Breaker Operations

#### Reset Circuit Breaker
When: Service has recovered but circuit is still open
```python
breaker = registry.get("binance_api")
await breaker.reset()
```

#### Trip Circuit Breaker
When: Preemptively protect against known issues
```python
breaker = registry.get("binance_api")
await breaker.trip()
```

## Dead Letter Queue Operations

### Monitoring DLQ

```python
from genesis.core.dead_letter_queue import DeadLetterQueue

dlq = DeadLetterQueue(name="main")
stats = dlq.get_statistics()

# Check for stuck items
pending_items = dlq.get_items(status=DLQItemStatus.PENDING)
failed_items = dlq.get_items(status=DLQItemStatus.FAILED)
```

### Manual DLQ Operations

#### Retry Failed Item
```python
# Retry specific item
await dlq.retry_item(item_id="abc-123")
```

#### Clear Failed Items
```python
# Clear permanently failed items
await dlq.clear(status=DLQItemStatus.FAILED)
```

#### Inspect DLQ Item
```python
item = dlq.get_item(item_id="abc-123")
print(f"Error: {item.error_message}")
print(f"Retries: {item.retry_count}/{item.max_retries}")
```

## Error Budget Monitoring

### Check Error Budget Status

```python
from genesis.monitoring.error_budget import ErrorBudget

budget = ErrorBudget()
status = budget.get_budget_status("order_execution")

if status.is_exhausted:
    print("ERROR BUDGET EXHAUSTED - Immediate action required")
    print(f"Success rate: {status.current_success_rate:.2%}")
    print(f"Target: {status.target:.2%}")
```

### SLO Targets

| SLO | Target | Measurement Window | Critical |
|-----|--------|-------------------|----------|
| Overall Availability | 99.9% | 30 days | Yes |
| Order Execution | 99.99% | 7 days | Yes |
| Data Accuracy | 99.9% | 1 day | No |
| Risk Compliance | 99.999% | 30 days | Yes |
| API Latency | 95% | 1 hour | No |

### Budget Exhaustion Response

1. **Immediate Actions**:
   - Switch to CRITICAL degradation level
   - Disable non-essential features
   - Alert operations team
   
2. **Investigation**:
   - Review error patterns in last hour
   - Check circuit breaker states
   - Verify external service health
   
3. **Recovery**:
   - Fix root cause
   - Reset error budget if appropriate
   - Document incident

## Feature Degradation

### Degradation Levels

| Level | Description | Features Disabled |
|-------|-------------|-------------------|
| NORMAL | All features enabled | None |
| MINOR | Non-critical features disabled | Analytics, UI enhancements |
| MAJOR | Most features disabled | Multi-pair trading, backtesting |
| CRITICAL | Essential features only | Everything except core trading |
| EMERGENCY | Minimal functionality | All except position closing |

### Managing Feature Flags

```python
from genesis.core.feature_flags import FeatureManager, DegradationLevel

manager = FeatureManager()

# Check current degradation level
current_level = manager._degradation_level

# Manually set degradation level
manager.set_degradation_level(DegradationLevel.MINOR)

# Check specific feature
if manager.is_enabled("multi_pair_trading"):
    print("Multi-pair trading is enabled")

# Get degraded features
degraded = manager.get_degraded_features()
```

### Auto-Degradation Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Error rate > 5% | 5 minutes | Move to MINOR |
| Error rate > 10% | 2 minutes | Move to MAJOR |
| Budget exhausted | Immediate | Move to CRITICAL |
| Critical error | Immediate | Feature-specific disable |

## Recovery Procedures

### Connection Timeout Recovery

**Symptoms**: `GENESIS-2501` errors, increased latency

**Procedure**:
1. Check network connectivity: `ping api.binance.com`
2. Verify firewall rules allow outbound HTTPS
3. Check circuit breaker status for `binance_api`
4. If circuit open, wait for recovery timeout
5. If persistent, check Binance API status page
6. Consider switching to backup connection

### Rate Limit Recovery

**Symptoms**: `GENESIS-2502` errors, HTTP 429 responses

**Procedure**:
1. Check current request rate in logs
2. Verify rate limiter configuration
3. Wait for rate limit window (typically 60s)
4. Reduce request frequency in configuration
5. Consider upgrading API tier if persistent

### Database Lock Recovery

**Symptoms**: `GENESIS-3501` errors, slow queries

**Procedure**:
1. Check active database connections
2. Look for long-running transactions
3. Add jitter to retry logic if not present
4. Consider database connection pool size
5. Review concurrent access patterns

### Order Rejection Recovery

**Symptoms**: `GENESIS-1701` errors, orders not executing

**Procedure**:
1. Verify account balance is sufficient
2. Check order parameters meet exchange requirements
3. Verify symbol is valid and tradeable
4. Check for exchange maintenance windows
5. Review order size against tier limits

### Tilt Intervention Recovery

**Symptoms**: `GENESIS-5903` errors, erratic trading behavior

**Procedure**:
1. System enters automatic cooldown period
2. Review recent trading metrics
3. Check for unusual patterns:
   - Rapid order cancellations
   - Increasing position sizes
   - Revenge trading indicators
4. Implement manual trading pause if needed
5. Review and adjust tilt detection thresholds

## Troubleshooting Flowcharts

### High Error Rate Troubleshooting

```
High Error Rate Detected
         |
         v
    Check Error Types
         |
    +---------+---------+
    |         |         |
Network   Exchange  Database
    |         |         |
    v         v         v
Check     Check     Check
Circuit   Balance   Locks
Breaker   & Limits
    |         |         |
    v         v         v
Reset if  Adjust    Add
Needed    Params    Jitter
```

### System Degradation Response

```
Error Budget Warning (80% consumed)
         |
         v
    Assess Impact
         |
    Critical SLO?
    /         \
   Yes         No
    |           |
    v           v
Immediate    Monitor
Action       Closely
    |           |
    v           |
Move to      Log for
MAJOR        Review
Degradation
```

### Complete System Recovery

```
System in CRITICAL/EMERGENCY Mode
         |
         v
    Root Cause Fixed?
    /            \
   No             Yes
    |              |
    v              v
Investigate    Reset Error
& Fix          Budgets
               |
               v
          Test Core
          Functions
               |
               v
          Gradually
          Re-enable
          Features
               |
               v
          Monitor
          24 Hours
```

## Operational Commands

### Health Check
```bash
# Check overall system health
python -m genesis.monitoring health

# Check specific component
python -m genesis.monitoring health --component error_handling
```

### Error Analysis
```bash
# Analyze recent errors
python -m genesis.monitoring errors --last 1h

# Get error distribution
python -m genesis.monitoring errors --distribution
```

### Manual Interventions
```bash
# Reset all circuit breakers
python -m genesis.core.admin reset-breakers

# Clear DLQ
python -m genesis.core.admin clear-dlq --status failed

# Set degradation level
python -m genesis.core.admin set-degradation --level minor
```

## Monitoring Dashboards

### Key Metrics to Monitor

1. **Error Rates**
   - Overall error rate (target < 0.1%)
   - Error rate by category
   - Error rate by severity

2. **Circuit Breakers**
   - Number of open circuits
   - Circuit trip frequency
   - Recovery success rate

3. **Error Budget**
   - Budget consumption rate
   - Time to exhaustion
   - SLO compliance

4. **Dead Letter Queue**
   - Queue depth
   - Retry success rate
   - Age of oldest item

5. **Feature Flags**
   - Degradation level
   - Disabled features count
   - Auto-degradation triggers

## Incident Response

### Severity Levels

| Level | Response Time | Escalation |
|-------|---------------|------------|
| P1 - Critical | < 5 minutes | Immediate page |
| P2 - High | < 30 minutes | Team notification |
| P3 - Medium | < 2 hours | Email alert |
| P4 - Low | Next business day | Log only |

### Incident Template

```markdown
## Incident Report

**Date**: [YYYY-MM-DD HH:MM UTC]
**Severity**: [P1/P2/P3/P4]
**Duration**: [XX minutes]
**Impact**: [Users affected, money at risk]

### Summary
[Brief description of incident]

### Timeline
- HH:MM - Initial detection
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Service restored

### Root Cause
[Detailed explanation]

### Resolution
[Steps taken to resolve]

### Prevention
[Changes to prevent recurrence]

### Lessons Learned
[Key takeaways]
```

## Best Practices

1. **Never ignore CRITICAL errors** - They indicate money at risk
2. **Monitor error budgets daily** - Proactive intervention prevents outages
3. **Test recovery procedures regularly** - Practice during low-volume periods
4. **Document all manual interventions** - Build knowledge base
5. **Review error patterns weekly** - Identify systemic issues
6. **Keep runbooks updated** - Document new failure modes
7. **Use correlation IDs** - Essential for debugging distributed failures
8. **Implement gradual rollouts** - Use feature flags for safe deployments

## Contact Information

- **On-Call Engineer**: Check PagerDuty
- **Team Lead**: [Contact via Slack]
- **Binance Support**: [API Support Portal]
- **Infrastructure Team**: [Internal escalation]

## Appendix

### Log Queries

```sql
-- Find errors by correlation ID
SELECT * FROM logs 
WHERE correlation_id = 'abc-123-def-456'
ORDER BY timestamp;

-- Get error distribution
SELECT error_code, COUNT(*) as count
FROM errors
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY error_code
ORDER BY count DESC;

-- Find related errors
SELECT * FROM errors
WHERE component = 'exchange_gateway'
AND severity IN ('HIGH', 'CRITICAL')
AND timestamp > NOW() - INTERVAL '30 minutes';
```

### Configuration Files

- Error handling config: `config/error_handling.yaml`
- Circuit breaker config: `config/circuit_breakers.yaml`
- Feature flags: `.genesis/feature_flags.json`
- SLO definitions: `config/slos.yaml`

---

*Last Updated: 2025-08-30*
*Version: 1.0*