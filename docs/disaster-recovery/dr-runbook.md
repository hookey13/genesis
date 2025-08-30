# Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for handling disaster recovery scenarios in Project GENESIS. Each scenario includes detection criteria, immediate actions, recovery procedures, and validation steps.

## Quick Reference

| Scenario | RTO | RPO | Automation | Page |
|----------|-----|-----|------------|------|
| Database Corruption | 15 min | 5 min | Full | [Link](#database-corruption) |
| Primary Failure | 15 min | 5 min | Full | [Link](#primary-infrastructure-failure) |
| Network Partition | 10 min | 0 min | Partial | [Link](#network-partition) |
| Data Loss | 20 min | 5 min | Full | [Link](#data-loss) |
| Security Breach | 30 min | 5 min | Partial | [Link](#security-breach) |
| Complete Disaster | 45 min | 5 min | Full | [Link](#complete-disaster) |

## DR Activation

### Automated DR Execution

```python
from genesis.dr import DROrchestrator
from genesis.dr import DRScenario

# Execute DR workflow
result = await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.DATABASE_CORRUPTION,
    dry_run=False,
    auto_execute=True  # Fully automated
)

print(f"Recovery completed: {result['success']}")
print(f"Recovery time: {result['recovery_time_minutes']} minutes")
print(f"RTO met: {result['rto_met']}")
print(f"RPO met: {result['rpo_met']}")
```

### Manual DR Execution

For manual control over each step:

```python
# Execute with prompts at each step
result = await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.PRIMARY_FAILURE,
    dry_run=False,
    auto_execute=False  # Manual confirmation required
)
```

## Scenarios

### Database Corruption

**Detection:**
- Database integrity check failures
- Unexpected query errors
- Data inconsistencies detected
- Application crashes when accessing data

**Immediate Actions:**
1. **STOP all trading immediately**
   ```bash
   systemctl stop genesis-trading
   ```

2. **Create immediate backup** (preserve evidence)
   ```bash
   cp .genesis/data/genesis.db .genesis/data/genesis.db.corrupted.$(date +%s)
   ```

3. **Notify team**
   ```bash
   ./scripts/send_alert.sh "DATABASE CORRUPTION DETECTED"
   ```

**Recovery Procedure:**

```python
# Automated recovery
await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.DATABASE_CORRUPTION,
    dry_run=False,
    auto_execute=True
)
```

Manual steps:
1. Identify last known good backup
2. Perform point-in-time recovery
3. Validate recovered data
4. Reconcile with exchange
5. Resume operations

**Validation:**
- Run database integrity check
- Verify position counts match
- Check account balances
- Confirm order states
- Test queries

### Primary Infrastructure Failure

**Detection:**
- Primary services unreachable
- Health checks failing
- Network timeout errors
- DNS resolution failures

**Immediate Actions:**
1. **Verify failure** (not network issue)
   ```bash
   ping primary.genesis.io
   curl https://api.genesis.primary/health
   ```

2. **Initiate failover**
   ```python
   await failover_coordinator.execute_failover(
       reason="Primary infrastructure failure",
       dry_run=False
   )
   ```

**Recovery Procedure:**

Automated:
```python
await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.PRIMARY_FAILURE,
    dry_run=False,
    auto_execute=True
)
```

Manual steps:
1. Confirm primary failure
2. Activate backup infrastructure
3. Update DNS records
4. Verify backup services
5. Monitor for issues

**Validation:**
- All services responding on backup
- DNS propagated to new IPs
- Database accessible
- Trading can resume
- Monitoring active

### Network Partition

**Detection:**
- Split brain detected
- Partial connectivity loss
- Inconsistent service states
- Replication lag increasing

**Immediate Actions:**
1. **Identify partition boundaries**
   ```bash
   ./scripts/network_diagnostic.sh
   ```

2. **Isolate affected services**
   ```bash
   iptables -A INPUT -s <affected_subnet> -j DROP
   ```

**Recovery Procedure:**

1. Determine authoritative side
2. Fence non-authoritative nodes
3. Reroute traffic to healthy nodes
4. Resolve partition (network team)
5. Resync when partition heals

**Validation:**
- No split brain
- All nodes see consistent state
- Replication caught up
- No duplicate operations

### Data Loss

**Detection:**
- Missing records discovered
- Accidental deletion reported
- Corruption affecting specific data
- Time gap in events

**Immediate Actions:**
1. **Stop operations**
   ```python
   await dr_orchestrator._stop_trading(dry_run=False)
   ```

2. **Assess damage**
   ```sql
   SELECT COUNT(*) FROM positions WHERE created_at > '2025-01-01';
   SELECT MAX(created_at) FROM events;
   ```

**Recovery Procedure:**

Automated:
```python
await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.DATA_LOSS,
    dry_run=False,
    auto_execute=True
)
```

Manual:
1. Identify data loss timeframe
2. Find appropriate backup
3. Restore to point before loss
4. Replay events if available
5. Reconcile with exchange
6. Document lost data

**Validation:**
- Data restored to expected state
- No orphaned references
- Positions match exchange
- Account balances correct

### Security Breach

**Detection:**
- Unauthorized access detected
- Suspicious transactions
- API keys compromised
- Unusual system behavior

**Immediate Actions:**
1. **EMERGENCY SHUTDOWN**
   ```python
   await emergency_closer.emergency_close_all(
       reason="Security breach",
       dry_run=False,
       force=True
   )
   ```

2. **Isolate systems**
   ```bash
   iptables -P INPUT DROP
   iptables -P OUTPUT DROP
   ```

3. **Rotate credentials**
   ```bash
   ./scripts/rotate_all_keys.sh
   ```

**Recovery Procedure:**

1. Assess breach scope
2. Preserve forensic evidence
3. Clean infected systems
4. Restore from clean backup
5. Implement additional security
6. Gradual service restoration
7. Security audit

**Validation:**
- No unauthorized access
- All credentials rotated
- Security patches applied
- Monitoring enhanced
- Incident documented

### Complete Disaster

**Detection:**
- Multiple system failures
- Data center offline
- Regional outage
- Catastrophic event

**Immediate Actions:**
1. **Activate disaster response team**
2. **Emergency position closure**
3. **Activate DR site**

**Recovery Procedure:**

Automated:
```python
await dr_orchestrator.execute_dr_workflow(
    scenario=DRScenario.COMPLETE_DISASTER,
    dry_run=False,
    auto_execute=True
)
```

Full procedure:
1. Emergency close all positions
2. Activate DR infrastructure
3. Restore from offsite backups
4. Rebuild primary systems
5. Verify all components
6. Gradual service restoration
7. Full system validation
8. Post-mortem analysis

**Validation:**
- All services operational
- Data integrity verified
- Performance acceptable
- Monitoring active
- Documentation updated

## Command Reference

### Status Commands

```bash
# Check DR readiness
python -m genesis.dr status

# View backup status
python -m genesis.dr backup-status

# Check replication lag
python -m genesis.dr replication-status

# Verify failover readiness
python -m genesis.dr failover-status
```

### Recovery Commands

```bash
# Execute DR workflow
python -m genesis.dr execute --scenario DATABASE_CORRUPTION

# Dry run
python -m genesis.dr execute --scenario PRIMARY_FAILURE --dry-run

# Manual recovery
python -m genesis.dr recover --timestamp "2025-01-01T12:00:00"

# Emergency close
python -m genesis.dr emergency-close --reason "Manual intervention"
```

### Testing Commands

```bash
# Run DR drill
python -m genesis.dr drill --scenario COMPLETE_DISASTER

# Validate recovery
python -m genesis.dr validate

# Calculate readiness
python -m genesis.dr readiness
```

## Contact Information

### Escalation Path

1. **On-Call Engineer**: Check PagerDuty
2. **Team Lead**: [Redacted]
3. **Infrastructure Team**: [Redacted]
4. **Security Team**: [Redacted]
5. **Executive**: [Redacted]

### External Contacts

- **DigitalOcean Support**: [Ticket System]
- **Binance Support**: [API Support]
- **DNS Provider**: [Support Portal]

## Appendix

### Pre-configured Scripts

Located in `/scripts/dr/`:
- `quick_recovery.sh` - Automated database recovery
- `failover.sh` - Execute failover
- `failback.sh` - Return to primary
- `emergency_close.sh` - Close all positions
- `validate_dr.sh` - Validate DR readiness

### Environment Variables

Required for DR operations:
```bash
export DR_S3_BUCKET=genesis-dr-backups
export DR_BACKUP_KEY=[Encrypted]
export DR_NOTIFICATION_WEBHOOK=[URL]
export DR_FAILOVER_DNS_TOKEN=[Token]
```

### Monitoring URLs

- Primary Dashboard: https://monitor.genesis.primary
- Backup Dashboard: https://monitor.genesis.backup
- DR Status: https://dr.genesis.io/status
- Metrics: https://metrics.genesis.io

### Documentation Updates

This runbook should be reviewed and updated:
- After each DR event
- Monthly during DR review
- When infrastructure changes
- After DR drills

Last Updated: 2025-01-01
Next Review: 2025-02-01