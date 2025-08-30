# DR Monitoring Integration

## Overview

The DR system integrates with Prometheus and Grafana to provide real-time monitoring, alerting, and visualization of disaster recovery metrics.

## Metrics Collection

### DR Metrics Collector

Initialize and start metrics collection:

```python
from genesis.monitoring.dr_metrics import DRMetricsCollector
from genesis.monitoring.metrics_collector import MetricsCollector

# Initialize collectors
metrics_collector = MetricsCollector()
dr_metrics = DRMetricsCollector(
    metrics_collector=metrics_collector,
    dr_orchestrator=dr_orchestrator,
    backup_manager=backup_manager,
    recovery_engine=recovery_engine,
    failover_coordinator=failover_coordinator
)

# Start collection loop
async def collect_metrics_loop():
    while True:
        await dr_metrics.collect_metrics()
        await asyncio.sleep(10)  # Collect every 10 seconds

asyncio.create_task(collect_metrics_loop())
```

## Available Metrics

### Backup Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_backup_total | Counter | Total backups created |
| genesis_backup_size_bytes | Gauge | Size of last backup |
| genesis_backup_duration_seconds | Histogram | Backup duration |
| genesis_backup_age_hours | Gauge | Age of last backup |

### Replication Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_replication_lag_seconds | Gauge | Replication lag |
| genesis_replication_queue_size | Gauge | Items in queue |

### Recovery Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_recovery_total | Counter | Total recoveries |
| genesis_recovery_duration_seconds | Histogram | Recovery duration |
| genesis_rpo_minutes | Gauge | Current RPO |
| genesis_rto_minutes | Gauge | Current RTO |

### Failover Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_failover_total | Counter | Total failovers |
| genesis_failover_duration_seconds | Histogram | Failover duration |
| genesis_health_check_failures | Counter | Health check failures |

### Readiness Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_dr_readiness_score | Gauge | Overall readiness (0-1) |
| genesis_dr_component_score | Gauge | Component scores |

### Test Metrics

| Metric | Type | Description |
|--------|------|-------------|
| genesis_dr_test_total | Counter | Total DR tests |
| genesis_dr_test_success_rate | Gauge | Test success rate |
| genesis_dr_test_last_run_timestamp | Gauge | Last test timestamp |

## Prometheus Configuration

### Scrape Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'genesis_dr'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
    metrics_path: '/metrics'
```

### Alert Rules

Create `dr_alerts.yml`:

```yaml
groups:
  - name: dr_alerts
    interval: 30s
    rules:
      - alert: BackupTooOld
        expr: genesis_backup_age_hours > 8
        for: 5m
        labels:
          severity: warning
          component: backup
        annotations:
          summary: "Backup is {{ $value }} hours old"
          description: "Last backup exceeds 8 hour threshold"
      
      - alert: HighReplicationLag
        expr: genesis_replication_lag_seconds > 300
        for: 5m
        labels:
          severity: critical
          component: replication
        annotations:
          summary: "Replication lag: {{ $value }}s"
          description: "Replication lag exceeds 5 minutes"
      
      - alert: LowDRReadiness
        expr: genesis_dr_readiness_score < 0.7
        for: 30m
        labels:
          severity: warning
          component: readiness
        annotations:
          summary: "DR readiness: {{ $value }}"
          description: "Readiness score below 70%"
      
      - alert: DRTestOverdue
        expr: (time() - genesis_dr_test_last_run_timestamp) > 2592000
        for: 1h
        labels:
          severity: warning
          component: testing
        annotations:
          summary: "DR test overdue"
          description: "No DR test in 30 days"
      
      - alert: RPOViolation
        expr: genesis_rpo_minutes > 5
        for: 10m
        labels:
          severity: critical
          component: recovery
        annotations:
          summary: "RPO violation: {{ $value }} minutes"
          description: "Data loss window exceeds 5 minutes"
```

## Grafana Dashboards

### Import Dashboard

```python
# Generate dashboard JSON
dashboard_config = dr_metrics.create_grafana_dashboard()

# Save to file
import json
with open("dr_dashboard.json", "w") as f:
    json.dump(dashboard_config, f, indent=2)
```

### Dashboard Panels

1. **DR Readiness Gauge**
   - Shows overall readiness score
   - Green > 80%, Yellow > 60%, Red < 60%

2. **Backup Status**
   - Backup age trend
   - Backup size history
   - Next backup countdown

3. **Replication Monitor**
   - Replication lag graph
   - Queue size
   - Sync status

4. **RTO/RPO Compliance**
   - Current vs target metrics
   - Historical compliance
   - Violation alerts

5. **Health Checks**
   - Service status table
   - Failure counts
   - Uptime percentages

6. **Test Results**
   - Success rate gauge
   - Test history
   - Time since last test

## Custom Queries

### Useful PromQL Queries

```promql
# Backup frequency (backups per day)
rate(genesis_backup_total[1d]) * 86400

# Average recovery time
rate(genesis_recovery_duration_seconds_sum[1h]) / rate(genesis_recovery_duration_seconds_count[1h])

# Replication backlog trend
deriv(genesis_replication_queue_size[5m])

# Health check success rate
1 - (rate(genesis_health_check_failures[5m]) / 60)

# DR test frequency
increase(genesis_dr_test_total[30d])

# Component readiness heatmap
genesis_dr_component_score

# Time since last backup
time() - genesis_backup_age_hours * 3600

# Failover frequency
rate(genesis_failover_total[7d]) * 604800
```

## AlertManager Integration

### Notification Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'dr_team'
  
  routes:
  - match:
      severity: critical
      component: dr
    receiver: 'dr_critical'
    continue: true

receivers:
- name: 'dr_team'
  email_configs:
  - to: 'dr-team@example.com'
    from: 'alerts@genesis.io'
    smarthost: 'smtp.example.com:587'
    
- name: 'dr_critical'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_KEY'
    description: 'DR Critical Alert'
  
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK'
    channel: '#dr-alerts'
    title: 'DR Critical Alert'
```

## SLA Tracking

### DR SLA Metrics

```python
# Track SLA compliance
sla_metrics = {
    "backup_sla": {
        "target": 4,  # hours
        "query": "genesis_backup_age_hours",
        "operator": "<"
    },
    "replication_sla": {
        "target": 300,  # seconds
        "query": "genesis_replication_lag_seconds",
        "operator": "<"
    },
    "readiness_sla": {
        "target": 0.8,
        "query": "genesis_dr_readiness_score",
        "operator": ">"
    },
    "test_frequency_sla": {
        "target": 30,  # days
        "query": "(time() - genesis_dr_test_last_run_timestamp) / 86400",
        "operator": "<"
    }
}

# Calculate compliance
async def calculate_sla_compliance():
    compliance = {}
    
    for sla_name, sla_config in sla_metrics.items():
        # Query Prometheus
        result = await query_prometheus(sla_config["query"])
        
        # Check compliance
        if sla_config["operator"] == "<":
            compliant = result < sla_config["target"]
        else:
            compliant = result > sla_config["target"]
        
        compliance[sla_name] = {
            "compliant": compliant,
            "current": result,
            "target": sla_config["target"]
        }
    
    return compliance
```

## Dashboard Access

### URLs

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **AlertManager**: http://localhost:9093

### Default Dashboards

1. **DR Overview**: `/d/dr-overview`
2. **Backup Status**: `/d/backup-status`
3. **Replication Monitor**: `/d/replication`
4. **Recovery Metrics**: `/d/recovery`
5. **Test Results**: `/d/dr-tests`

## Monitoring Best Practices

1. **Alert Fatigue Prevention**
   - Set appropriate thresholds
   - Use alert grouping
   - Implement quiet hours
   - Escalate gradually

2. **Dashboard Design**
   - Focus on actionable metrics
   - Use color coding consistently
   - Group related metrics
   - Include historical context

3. **Metric Collection**
   - Sample at appropriate intervals
   - Aggregate where possible
   - Use labels effectively
   - Avoid high cardinality

4. **Alert Response**
   - Document runbooks
   - Include context in alerts
   - Link to dashboards
   - Track MTTR

## Troubleshooting

### Common Issues

#### Metrics Not Appearing
- Check Prometheus targets page
- Verify metrics endpoint is accessible
- Check metric registration
- Review Prometheus logs

#### High Memory Usage
- Reduce retention period
- Downsample old data
- Optimize queries
- Check cardinality

#### Alert Storms
- Review alert thresholds
- Implement alert dependencies
- Use inhibition rules
- Check flapping detection

## Integration Testing

### Verify Metrics Collection

```python
# Test metrics collection
async def test_metrics_collection():
    # Trigger backup
    await backup_manager.create_full_backup()
    
    # Wait for metrics update
    await asyncio.sleep(15)
    
    # Query metrics
    metrics = await query_prometheus("genesis_backup_total")
    assert metrics > 0
    
    # Check all metrics registered
    all_metrics = await get_all_metrics()
    expected = [
        "genesis_backup_total",
        "genesis_replication_lag_seconds",
        "genesis_dr_readiness_score"
    ]
    
    for metric in expected:
        assert metric in all_metrics
```

## Related Documentation

- [Monitoring Platform](../monitoring/README.md)
- [Prometheus Setup](../monitoring/prometheus.md)
- [Grafana Dashboards](../monitoring/grafana.md)
- [Alert Configuration](../monitoring/alerts.md)