# Disaster Recovery Procedures

## Overview

This document outlines the disaster recovery procedures for Project GENESIS, including backup strategies, recovery procedures, and failover mechanisms.

## Backup Strategy

### Automated Backups
- **Frequency**: Every 4 hours
- **Retention**: 30 days local, 1 year in S3/Spaces
- **Types**:
  - Full backup: Daily at 02:00 UTC
  - Incremental: Every 4 hours
  - Transaction logs: Continuous

### Backup Components
1. **Database**: SQLite database files with WAL
2. **Configuration**: All config files and secrets
3. **State Files**: Tier state, positions, orders
4. **Logs**: Audit logs for forensic analysis

## Recovery Procedures

### 1. Data Corruption Recovery
```bash
# Stop trading engine
supervisorctl stop genesis

# Verify backup integrity
python -m tests.dr.disaster_recovery --verify-backup

# Restore from backup
python scripts/restore_backup.py --point-in-time "2024-01-15 14:00:00"

# Validate data integrity
python scripts/validate_data.py

# Restart services
supervisorctl start genesis
```

### 2. Complete System Failure
```bash
# On backup server
cd /opt/genesis

# Pull latest backup from S3
aws s3 sync s3://genesis-backups/latest ./backups/

# Restore system
./scripts/disaster_recovery.sh --full-restore

# Update DNS to point to backup server
./scripts/update_dns.sh --failover

# Start services
docker-compose up -d
```

### 3. Database Recovery
```bash
# For SQLite corruption
sqlite3 genesis.db ".recover" > recovered.sql
sqlite3 new_genesis.db < recovered.sql

# For PostgreSQL (future)
pg_restore -d genesis_prod backup_20240115.dump
```

## Failover Procedures

### Automatic Failover Triggers
- 3 failed trades in 10 minutes
- System crash with no recovery in 5 minutes
- Database corruption detected
- Network partition lasting > 30 seconds

### Manual Failover Steps
1. **Assess the situation**
   - Check system logs
   - Verify exchange connectivity
   - Review recent trades

2. **Initiate failover**
   ```bash
   ./scripts/failover.sh --target backup-singapore
   ```

3. **Verify failover**
   - Check new system health
   - Confirm order reconciliation
   - Validate position consistency

4. **Resume operations**
   - Re-enable trading
   - Monitor closely for 1 hour

## Recovery Time Objectives (RTO)

| Scenario | Target RTO | Maximum RTO |
|----------|------------|-------------|
| Service restart | < 1 minute | 5 minutes |
| Database recovery | < 5 minutes | 15 minutes |
| Complete failover | < 10 minutes | 30 minutes |
| Full disaster recovery | < 30 minutes | 2 hours |

## Recovery Point Objectives (RPO)

| Data Type | Target RPO | Maximum RPO |
|-----------|------------|-------------|
| Trades | 0 (real-time) | 1 minute |
| Positions | 0 (real-time) | 1 minute |
| Configuration | < 4 hours | 24 hours |
| Logs | < 1 hour | 4 hours |

## Testing Schedule

### DR Drills
- **Monthly**: Service restart and failover test
- **Quarterly**: Full disaster recovery drill
- **Annually**: Complete infrastructure rebuild

### Validation Tests
```bash
# Run DR validation suite
python -m tests.dr.disaster_recovery --full-drill

# Test specific scenarios
python -m tests.dr.disaster_recovery --scenario data-corruption
python -m tests.dr.disaster_recovery --scenario network-partition
python -m tests.dr.disaster_recovery --scenario complete-failure
```

## Runbook for Common Scenarios

### Scenario 1: Exchange API Degradation
1. Circuit breaker activates automatically
2. Switch to backup exchange connection
3. Queue orders for retry
4. Monitor and resume when stable

### Scenario 2: Memory Leak Detected
1. Graceful shutdown triggered at 85% memory
2. State persisted to disk
3. Process restarted by supervisor
4. State restored from disk
5. Trading resumes automatically

### Scenario 3: Disk Space Critical
1. Log rotation triggered
2. Old logs compressed and uploaded to S3
3. Local logs > 7 days deleted
4. Alert sent if still critical

## Contact Information

### Escalation Path
1. **Level 1**: Automated recovery (0-5 minutes)
2. **Level 2**: On-call engineer (5-15 minutes)
3. **Level 3**: Senior engineer (15-30 minutes)
4. **Level 4**: CTO/Founder (30+ minutes)

### Key Contacts
- **Primary On-Call**: Via PagerDuty
- **Backup On-Call**: Via PagerDuty
- **Exchange Support**: Binance API Support
- **Infrastructure**: DigitalOcean Support

## Post-Recovery Procedures

### Immediate Actions
1. Verify all positions are accurate
2. Reconcile orders with exchange
3. Check for any missed trades
4. Review audit logs for anomalies

### Within 24 Hours
1. Complete incident report
2. Update runbooks if needed
3. Schedule post-mortem meeting
4. Implement preventive measures

### Post-Mortem Template
```markdown
## Incident Post-Mortem

**Date**: [Date]
**Duration**: [Start] - [End]
**Impact**: [Description of impact]

### Timeline
- [Time]: Event description

### Root Cause
[Description of root cause]

### Resolution
[How it was resolved]

### Action Items
- [ ] Action item 1
- [ ] Action item 2

### Lessons Learned
- Lesson 1
- Lesson 2
```

## Automation Scripts

All DR procedures are automated in:
- `/scripts/disaster_recovery.sh` - Main DR orchestrator
- `/scripts/backup.sh` - Backup automation
- `/scripts/restore_backup.py` - Restoration tool
- `/scripts/failover.sh` - Failover automation
- `/tests/dr/disaster_recovery.py` - DR testing framework