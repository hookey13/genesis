# Operational Procedures

This document provides comprehensive operational procedures for Project GENESIS's automated maintenance and self-healing systems.

## Table of Contents

1. [Log Archival Procedures](#log-archival-procedures)
2. [Database Optimization](#database-optimization)
3. [Certificate Management](#certificate-management)
4. [Performance Baseline Management](#performance-baseline-management)
5. [Correlation Matrix Updates](#correlation-matrix-updates)
6. [Strategy A/B Testing](#strategy-ab-testing)
7. [Dependency Management](#dependency-management)
8. [Health Monitoring & Remediation](#health-monitoring--remediation)

## Log Archival Procedures

### Overview
The log archival system automatically rotates, compresses, and archives logs to long-term storage.

### Configuration
- **Local Retention**: 30 days
- **Archive Retention**: 1 year
- **Rotation Size**: 10MB per file
- **Compression**: gzip

### Manual Operations

#### Force Log Rotation
```bash
python -m genesis.operations.log_archiver --rotate-now
```

#### Retrieve Archived Logs
```bash
python -m genesis.operations.log_archiver --retrieve \
    --date-range "2024-01-01:2024-01-31" \
    --log-type "trading"
```

#### Cleanup Old Archives
```bash
python -m genesis.operations.log_archiver --cleanup \
    --older-than "365d"
```

### Monitoring
- Check archival status: `GET /api/operations/logs/status`
- Failed archival alerts sent to `#ops-alerts` channel
- Dashboard: Grafana → Operations → Log Management

## Database Optimization

### Overview
Automated database maintenance runs weekly during low-activity periods.

### Scheduled Maintenance
- **Time**: Sundays 03:00 UTC
- **Duration**: ~15-30 minutes
- **Operations**: VACUUM, ANALYZE, index optimization

### Manual Operations

#### Run Optimization Now
```bash
python scripts/optimize_db.py --force
```

#### Check Database Health
```bash
python -m genesis.operations.db_optimizer --health-check
```

#### Index Analysis
```bash
python -m genesis.operations.db_optimizer --analyze-indexes
```

### Pre-Maintenance Checklist
1. Verify backup completed successfully
2. Check active connections < 10
3. Ensure no critical trades in progress
4. Monitor disk space > 20% free

### Post-Maintenance Verification
1. Run query performance tests
2. Verify index usage statistics
3. Check connection pool health
4. Confirm backup integrity

## Certificate Management

### Overview
Automated Let's Encrypt certificate renewal with zero-downtime deployment.

### Certificate Lifecycle
- **Check Frequency**: Daily at 02:00 UTC
- **Renewal Threshold**: 30 days before expiry
- **Warning Alerts**: 30, 7, 1 days before expiry

### Manual Operations

#### Force Certificate Renewal
```bash
python -m genesis.operations.cert_manager --renew --domain "api.genesis.trading"
```

#### Verify Certificate
```bash
python -m genesis.operations.cert_manager --verify
```

#### Rollback to Previous Certificate
```bash
python -m genesis.operations.cert_manager --rollback
```

### Emergency Procedures

#### Certificate Expired
1. Switch to self-signed certificate (automatic)
2. Alert sent to on-call engineer
3. Manual renewal required within 4 hours
4. Update DNS if needed

#### Renewal Failure
1. Check Let's Encrypt service status
2. Verify DNS records
3. Check rate limits (5 per week per domain)
4. Use backup certificate if available

## Performance Baseline Management

### Overview
Weekly recalculation of performance baselines for anomaly detection.

### Baseline Metrics
- **p50, p95, p99** latencies per operation
- **Rolling Window**: 30 days
- **Update Frequency**: Weekly (Mondays 00:00 UTC)

### Manual Operations

#### Recalculate Baselines
```bash
python -m genesis.operations.performance_baseline --recalculate
```

#### View Current Baselines
```bash
python -m genesis.operations.performance_baseline --show
```

#### Export to Monitoring
```bash
python -m genesis.operations.performance_baseline --export-prometheus
```

### Performance Degradation Response

#### Alert Triggered (>2σ deviation)
1. Check recent deployments
2. Review infrastructure metrics
3. Analyze slow query logs
4. Consider rollback if degradation persists

## Correlation Matrix Updates

### Overview
Hot-reload correlation matrices without service interruption.

### Update Process
1. Calculate new correlations from 30-day window
2. Validate against historical patterns
3. A/B test with 10% traffic
4. Gradual rollout if successful

### Manual Operations

#### Force Correlation Update
```bash
python -m genesis.operations.correlation_updater --update
```

#### View Current Matrix
```bash
python -m genesis.operations.correlation_updater --show-matrix
```

#### Rollback Matrix
```bash
python -m genesis.operations.correlation_updater --rollback
```

### Validation Criteria
- Maximum drift: 15% from baseline
- Matrix must be positive semi-definite
- All correlations in range [-1, 1]
- Minimum 100 data points per pair

## Strategy A/B Testing

### Overview
Automated parameter optimization with statistical significance testing.

### Test Configuration
- **Minimum Trades**: 100 per group
- **Significance Level**: p < 0.05
- **Rollback Threshold**: -10% performance
- **Default Split**: 90/10 (control/variant)

### Manual Operations

#### Create A/B Test
```bash
python -m genesis.operations.strategy_optimizer create-test \
    --strategy "spread_capture" \
    --param "threshold:0.003" \
    --split "80/20"
```

#### View Active Tests
```bash
python -m genesis.operations.strategy_optimizer --list-tests
```

#### Force Test Conclusion
```bash
python -m genesis.operations.strategy_optimizer conclude-test \
    --test-id "TEST_123" \
    --action "promote|rollback"
```

### Analysis Procedures

#### Statistical Significance
1. Collect minimum 100 trades per group
2. Run two-sample t-test
3. Check p-value < 0.05
4. Verify effect size (Cohen's d > 0.2)

#### Decision Matrix
| Scenario | Action |
|----------|--------|
| Variant +10%, p<0.05 | Promote to production |
| Variant -10% | Immediate rollback |
| No significance after 500 trades | Conclude as no difference |
| Mixed results | Extend test period |

## Dependency Management

### Overview
Monthly automated dependency updates with security scanning.

### Update Schedule
- **Frequency**: First Monday of month
- **Time**: 04:00 UTC
- **Strategy**: Conservative (patch updates only)

### Manual Operations

#### Check for Updates
```bash
python scripts/update_deps.py --check
```

#### Run Security Scan
```bash
python scripts/update_deps.py --security-scan
```

#### Force Update
```bash
python scripts/update_deps.py --update --strategy moderate
```

### Update Process
1. Create backup of requirements
2. Run security scan
3. Update in test environment
4. Run full test suite
5. Deploy if tests pass
6. Rollback on failure

### Security Response

#### Critical Vulnerability Found
1. Immediate notification to security team
2. Patch within 24 hours
3. Emergency deployment if actively exploited
4. Post-mortem within 72 hours

## Health Monitoring & Remediation

### Overview
Multi-level health checks with automatic remediation.

### Check Levels
- **Shallow** (every 30s): API connectivity, basic metrics
- **Deep** (every 5m): Database integrity, queue depth
- **Diagnostic** (on-demand): Full system audit

### Remediation Matrix

| Issue | Detection | Automatic Action | Manual Fallback |
|-------|-----------|-----------------|-----------------|
| Connection Pool Exhaustion | Available < 10% | Reset connections | Restart service |
| Memory Leak | Usage > 85% | GC + cache clear | Controlled restart |
| Disk Space Low | Free < 10% | Log cleanup | Add storage |
| API Degradation | Latency > 2σ | Circuit breaker | Switch to backup |
| Database Lock | Deadlock detected | Kill query | Manual investigation |

### Manual Operations

#### Run Health Check
```bash
python -m genesis.operations.health_monitor --check --level deep
```

#### View Health Dashboard
```bash
python -m genesis.operations.health_monitor --dashboard
```

#### Disable Auto-Remediation
```bash
python -m genesis.operations.health_monitor --disable-remediation
```

### Escalation Procedures

#### Escalation Triggers
- 3+ failed remediation attempts
- Critical issues persisting > 15 minutes
- Multiple cascading failures
- Manual intervention flag

#### On-Call Response
1. Check health dashboard
2. Review recent remediations
3. Analyze error patterns
4. Execute manual remediation
5. Document resolution

## Emergency Procedures

### System-Wide Failure
1. Activate emergency mode: `python scripts/emergency.py --activate`
2. Close all positions: `python scripts/emergency_close.py`
3. Stop automated trading
4. Investigate root cause
5. Restore from backup if needed

### Rollback Procedures
1. Identify failed component
2. Retrieve last known good state
3. Execute rollback: `python scripts/rollback.py --component [name]`
4. Verify system health
5. Document incident

## Monitoring & Alerting

### Key Metrics
- Operation success rate > 99.5%
- Remediation success rate > 90%
- Update failure rate < 5%
- Certificate expiry > 7 days

### Alert Channels
- **Critical**: PagerDuty + SMS
- **High**: Slack #ops-critical
- **Medium**: Slack #ops-alerts
- **Low**: Email digest

### Dashboards
- **Operations Overview**: Overall system health
- **Remediation History**: Recent automated actions
- **Performance Trends**: Baseline deviations
- **Dependency Status**: Update history and vulnerabilities

## Best Practices

1. **Always verify backups** before maintenance operations
2. **Monitor during automated actions** for unexpected behavior
3. **Document all manual interventions** in operation log
4. **Test rollback procedures** quarterly
5. **Review remediation patterns** monthly for improvements
6. **Keep runbooks updated** with lessons learned
7. **Practice emergency procedures** in staging environment

## Contact Information

- **On-Call Engineer**: Check PagerDuty schedule
- **Escalation**: engineering-lead@genesis.trading
- **Security Issues**: security@genesis.trading
- **Non-urgent**: #ops-discussion on Slack