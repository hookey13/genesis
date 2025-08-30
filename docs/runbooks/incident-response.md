# Incident Response Runbook

## Quick Reference Decision Tree

```
START → Is the system accepting trades?
├─ NO → GO TO: Scenario #1 (Complete System Outage)
└─ YES → Are positions being executed?
    ├─ NO → GO TO: Scenario #2 (Order Execution Failure)
    └─ YES → Are market data feeds active?
        ├─ NO → GO TO: Scenario #3 (Market Data Loss)
        └─ YES → Check specific symptoms below
```

## Top 20 Failure Scenarios

### Scenario #1: Complete System Outage
**Detection Criteria:**
- No heartbeat from genesis process for >60 seconds
- All health checks failing
- No logs being written

**Recovery Procedure:**
1. Check process status: `supervisorctl status genesis`
2. Check system resources: `df -h && free -m && top -bn1`
3. Review last 100 log lines: `tail -n 100 /var/log/genesis/trading.log`
4. If crashed, restart: `supervisorctl restart genesis`
5. If won't start, check for port conflicts: `netstat -tulpn | grep :8080`
6. Verify database connectivity: `python scripts/verify_db.py`

**Validation:**
- Health endpoint returns 200: `curl http://localhost:8080/health`
- Logs show "System initialized successfully"
- Positions are being monitored

**Automation:** `scripts/recovery/system_restart.sh`

---

### Scenario #2: Order Execution Failure
**Detection Criteria:**
- Orders stuck in PENDING state >30 seconds
- Repeated "Order rejected" errors
- ExecutionError exceptions in logs

**Recovery Procedure:**
1. Check exchange connectivity: `python -m genesis.exchange.gateway --test`
2. Verify API credentials: `python scripts/verify_credentials.py`
3. Check rate limits: `grep "rate limit" /var/log/genesis/trading.log`
4. Clear order queue: `python scripts/clear_order_queue.py`
5. Reset circuit breaker: `python scripts/reset_circuit_breaker.py`
6. Restart executor: `supervisorctl restart genesis:executor`

**Validation:**
- Test order succeeds: `python scripts/test_order.py`
- No pending orders in queue
- Circuit breaker status: CLOSED

**Automation:** `scripts/recovery/fix_execution.sh`

---

### Scenario #3: Market Data Feed Loss
**Detection Criteria:**
- No price updates for >10 seconds
- WebSocket disconnection errors
- Stale market data warnings

**Recovery Procedure:**
1. Check WebSocket status: `python -m genesis.exchange.websocket_manager --status`
2. Test network connectivity: `ping api.binance.com`
3. Restart WebSocket connections: `python scripts/restart_websockets.py`
4. If persistent, switch to REST polling: `python scripts/enable_rest_fallback.py`
5. Clear market data cache: `redis-cli FLUSHDB` (if using Redis)

**Validation:**
- Live prices updating in UI
- WebSocket status: CONNECTED
- Price timestamps <2 seconds old

**Automation:** `scripts/recovery/restore_market_data.sh`

---

### Scenario #4: Database Connection Lost
**Detection Criteria:**
- "Connection refused" errors for database
- Transaction rollback failures
- Unable to persist state

**Recovery Procedure:**
1. Check database status: `systemctl status postgresql`
2. Verify connection string: `cat .env | grep DATABASE_URL`
3. Test connection: `psql -U genesis -h localhost -c "SELECT 1"`
4. If down, restart: `systemctl restart postgresql`
5. If corrupted, restore from backup: `scripts/restore_db.sh`
6. Run migrations: `alembic upgrade head`

**Validation:**
- Database queries succeed
- Recent trades visible in UI
- Audit trail being written

**Automation:** `scripts/recovery/database_recovery.sh`

---

### Scenario #5: Memory Leak / High Memory Usage
**Detection Criteria:**
- Memory usage >90% of available
- Gradual memory increase over time
- OOMKiller events in system logs

**Recovery Procedure:**
1. Identify memory usage: `ps aux | grep genesis | awk '{print $4}'`
2. Generate heap dump: `python scripts/heap_dump.py`
3. Graceful restart: `supervisorctl restart genesis`
4. If urgent, force restart: `kill -9 $(pgrep -f genesis) && supervisorctl start genesis`
5. Analyze heap dump: `python scripts/analyze_heap.py`

**Validation:**
- Memory usage <50% after restart
- No growing memory trend
- Performance metrics normal

**Automation:** `scripts/recovery/memory_cleanup.sh`

---

### Scenario #6: Tilt Detection Triggered
**Detection Criteria:**
- Tilt score >0.7
- Multiple risk limit violations
- Unusual trading patterns detected

**Recovery Procedure:**
1. Check tilt status: `python -m genesis.tilt.detector --status`
2. Review recent trades: `python scripts/recent_trades.py --last 50`
3. Apply cooling period: `python scripts/enable_cooling.py --minutes 30`
4. Reduce position limits: `python scripts/reduce_limits.py --factor 0.5`
5. Notify trader: Send alert via PagerDuty

**Validation:**
- Tilt score decreasing
- No new violations
- Trader acknowledged alert

**Automation:** `scripts/recovery/tilt_intervention.sh`

---

### Scenario #7: API Rate Limit Exceeded
**Detection Criteria:**
- HTTP 429 responses
- "Rate limit exceeded" in logs
- Exponential backoff triggered

**Recovery Procedure:**
1. Check current rate: `grep -c "API call" /var/log/genesis/trading.log | tail -1`
2. Enable rate limit protection: `python scripts/enable_rate_limiter.py`
3. Reduce request frequency: Edit config/trading_rules.yaml
4. Clear request queue: `python scripts/clear_request_queue.py`
5. Wait for limit reset (usually 1 minute)

**Validation:**
- API calls succeeding
- Rate below 1200/minute
- No 429 responses

**Automation:** `scripts/recovery/rate_limit_recovery.sh`

---

### Scenario #8: Position Size Violation
**Detection Criteria:**
- Position exceeds tier limits
- Risk engine rejecting orders
- Tier demotion warnings

**Recovery Procedure:**
1. Check current positions: `python scripts/show_positions.py`
2. Compare to limits: `python scripts/check_limits.py`
3. Close excess positions: `python scripts/reduce_positions.py`
4. Reset risk calculations: `python scripts/recalc_risk.py`
5. Verify tier status: `python -m genesis.engine.state_machine --status`

**Validation:**
- All positions within limits
- Risk engine accepting orders
- Tier status stable

**Automation:** `scripts/recovery/position_compliance.sh`

---

### Scenario #9: Certificate Expiration
**Detection Criteria:**
- SSL handshake failures
- Certificate expiry warnings
- API connection refused

**Recovery Procedure:**
1. Check cert expiry: `python -m genesis.operations.cert_manager --check`
2. Renew if needed: `python -m genesis.operations.cert_manager --renew`
3. Update cert store: `update-ca-certificates`
4. Restart connections: `supervisorctl restart genesis`
5. Verify connectivity: `python scripts/test_ssl.py`

**Validation:**
- SSL connections successful
- Certificates valid >30 days
- No handshake errors

**Automation:** `scripts/recovery/cert_renewal.sh`

---

### Scenario #10: Disk Space Exhaustion
**Detection Criteria:**
- Disk usage >95%
- Write failures in logs
- Database insert errors

**Recovery Procedure:**
1. Check disk usage: `df -h`
2. Clean old logs: `python -m genesis.operations.log_archiver --cleanup`
3. Archive old data: `python -m genesis.operations.data_retention --archive`
4. Clear temp files: `find /tmp -type f -mtime +7 -delete`
5. If critical, move to backup disk: `scripts/emergency_storage.sh`

**Validation:**
- Disk usage <80%
- Writes succeeding
- No space errors

**Automation:** `scripts/recovery/disk_cleanup.sh`

---

### Scenario #11: Network Partition
**Detection Criteria:**
- Split brain detected
- Conflicting state between nodes
- Network timeout errors

**Recovery Procedure:**
1. Identify partition: `python scripts/check_network.py`
2. Determine primary: `python scripts/elect_primary.py`
3. Fence secondary: `python scripts/fence_node.py --node secondary`
4. Reconcile state: `python scripts/reconcile_state.py`
5. Resume operations: `python scripts/resume_trading.py`

**Validation:**
- Single primary active
- State consistent
- No split brain warnings

**Automation:** `scripts/recovery/heal_partition.sh`

---

### Scenario #12: Correlation Matrix Corruption
**Detection Criteria:**
- Invalid correlation values (>1 or <-1)
- Matrix calculation failures
- Strategy errors due to bad correlations

**Recovery Procedure:**
1. Validate matrix: `python -m genesis.analytics.correlation --validate`
2. Rebuild from history: `python -m genesis.operations.correlation_updater --rebuild`
3. If failed, use defaults: `cp config/default_correlations.json .genesis/state/`
4. Restart strategies: `supervisorctl restart genesis:strategies`
5. Monitor for 5 minutes

**Validation:**
- Correlations in valid range
- Strategies executing normally
- No calculation errors

**Automation:** `scripts/recovery/fix_correlations.sh`

---

### Scenario #13: Order Book Desync
**Detection Criteria:**
- Local order book differs from exchange
- Impossible spreads detected
- Arbitrage calculations failing

**Recovery Procedure:**
1. Compare orderbooks: `python scripts/compare_orderbooks.py`
2. Force resync: `python scripts/resync_orderbook.py`
3. Clear local cache: `redis-cli DEL orderbook:*`
4. Restart market data: `supervisorctl restart genesis:market_data`
5. Validate prices: `python scripts/validate_prices.py`

**Validation:**
- Orderbooks match exchange
- Spreads realistic
- Arbitrage calculations working

**Automation:** `scripts/recovery/orderbook_sync.sh`

---

### Scenario #14: Authentication Failure
**Detection Criteria:**
- 401/403 responses from API
- "Invalid signature" errors
- Authentication token expired

**Recovery Procedure:**
1. Verify credentials: `python scripts/test_auth.py`
2. Check time sync: `ntpdate -q pool.ntp.org`
3. If drift >1s, sync: `ntpdate pool.ntp.org`
4. Regenerate signatures: `python scripts/regen_auth.py`
5. Update API keys if needed: Follow secure key rotation procedure

**Validation:**
- Authentication succeeding
- No 401/403 errors
- Time synchronized

**Automation:** `scripts/recovery/fix_auth.sh`

---

### Scenario #15: Strategy Malfunction
**Detection Criteria:**
- Strategy throwing exceptions
- No trades for active strategy
- Abnormal trade patterns

**Recovery Procedure:**
1. Check strategy status: `python -m genesis.strategies.loader --status`
2. Review strategy logs: `grep "strategy" /var/log/genesis/trading.log`
3. Disable problematic strategy: `python scripts/disable_strategy.py --name <strategy>`
4. Reset strategy state: `python scripts/reset_strategy.py --name <strategy>`
5. Re-enable with safeguards: `python scripts/enable_strategy.py --name <strategy> --safe-mode`

**Validation:**
- Strategy executing trades
- No exceptions in logs
- Trade patterns normal

**Automation:** `scripts/recovery/strategy_recovery.sh`

---

### Scenario #16: Backup Failure
**Detection Criteria:**
- Backup job failures
- Missing backup files
- Backup age >24 hours

**Recovery Procedure:**
1. Check backup status: `restic snapshots`
2. Test backup integrity: `restic check`
3. Run manual backup: `scripts/backup.sh`
4. Verify upload: `restic ls latest`
5. Fix scheduled job: `crontab -e` and verify backup schedule

**Validation:**
- Recent backup exists
- Integrity check passes
- Scheduled job running

**Automation:** `scripts/recovery/backup_recovery.sh`

---

### Scenario #17: Performance Degradation
**Detection Criteria:**
- Response times >1 second
- CPU usage >80% sustained
- Order latency increasing

**Recovery Procedure:**
1. Profile application: `python -m cProfile -o profile.out genesis.__main__`
2. Identify bottlenecks: `python scripts/analyze_profile.py profile.out`
3. Optimize database: `python -m genesis.operations.db_optimizer`
4. Clear caches: `python scripts/clear_caches.py`
5. Restart with optimization: `supervisorctl restart genesis`

**Validation:**
- Response times <500ms
- CPU usage <60%
- Order latency normal

**Automation:** `scripts/recovery/performance_tune.sh`

---

### Scenario #18: Audit Trail Gap
**Detection Criteria:**
- Missing audit entries
- Non-sequential audit IDs
- Audit write failures

**Recovery Procedure:**
1. Identify gaps: `python -m genesis.analytics.forensics --find-gaps`
2. Reconstruct from logs: `python scripts/rebuild_audit.py`
3. Verify completeness: `python scripts/verify_audit.py`
4. Enable write verification: `python scripts/enable_audit_verify.py`
5. Generate gap report: `python scripts/audit_gap_report.py`

**Validation:**
- No gaps in audit trail
- All events captured
- Write verification active

**Automation:** `scripts/recovery/audit_recovery.sh`

---

### Scenario #19: Configuration Drift
**Detection Criteria:**
- Config mismatch errors
- Unexpected behavior changes
- Version conflicts

**Recovery Procedure:**
1. Compare configs: `diff -r config/ config.backup/`
2. Validate current config: `python scripts/validate_config.py`
3. Restore known good: `cp -r config.backup/* config/`
4. Reload configuration: `supervisorctl signal HUP genesis`
5. Verify behavior: Run smoke tests

**Validation:**
- Config validation passes
- Expected behavior restored
- No version conflicts

**Automation:** `scripts/recovery/config_restore.sh`

---

### Scenario #20: Emergency Market Conditions
**Detection Criteria:**
- Extreme volatility (>10% in 1 minute)
- Exchange halt announced
- Liquidity disappearance

**Recovery Procedure:**
1. Activate emergency mode: `python scripts/emergency_mode.py`
2. Close all positions: `python scripts/emergency_close.py`
3. Cancel pending orders: `python scripts/cancel_all_orders.py`
4. Enable safe mode: `python scripts/enable_safe_mode.py`
5. Wait for conditions to normalize

**Validation:**
- No open positions
- No pending orders
- System in safe mode

**Automation:** `scripts/recovery/emergency_shutdown.sh`

---

## Escalation Matrix

| Severity | Response Time | Escalation Path | Alert Method |
|----------|---------------|-----------------|--------------|
| Critical | <5 minutes | On-call → Lead → Director | PagerDuty + Phone |
| High | <15 minutes | On-call → Lead | PagerDuty |
| Medium | <1 hour | On-call | Slack + Email |
| Low | <4 hours | On-call | Email |

## Quick Commands Reference

```bash
# System Status
supervisorctl status
python -m genesis.operations.health_monitor --full

# Emergency Actions
python scripts/emergency_close.py  # Close all positions
python scripts/emergency_mode.py   # Enter safe mode
python scripts/kill_switch.py      # Complete shutdown

# Diagnostics
python scripts/system_diagnostics.py
python scripts/performance_check.py
python scripts/connectivity_test.py

# Recovery
scripts/recovery/auto_recover.sh   # Attempt automatic recovery
scripts/recovery/manual_recover.sh # Step-by-step recovery
```

## Contact Information

See `genesis/operations/contact_manager.py` for current on-call rotation and emergency contacts.

## Post-Incident

After any incident:
1. Create post-mortem: `python -m genesis.operations.post_mortem --create`
2. Document timeline: Include all actions taken
3. Identify root cause: Use 5-whys analysis
4. Create action items: Prevent recurrence
5. Update runbook: Add new scenarios if needed