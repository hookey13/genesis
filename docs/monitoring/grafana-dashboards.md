# Grafana Dashboards Guide

## Overview

Grafana provides real-time visualization of GENESIS trading metrics, performance indicators, and system health through customized dashboards.

## Available Dashboards

### 1. P&L Dashboard (`genesis-pnl`)

Real-time profit and loss tracking with drawdown monitoring.

**Panels:**
- Current P&L (Gauge) - Total unrealized + realized P&L
- P&L Over Time (Time Series) - Historical P&L trend
- Current Drawdown (Gauge) - Percentage from peak
- Open Positions (Stat) - Current position count
- Order Status Distribution (Pie Chart) - Order success/failure rates
- Tilt Score (Gauge) - Behavioral monitoring

**Key Metrics:**
- `genesis_pnl_dollars` - Current P&L in USD
- `genesis_drawdown_percent` - Current drawdown percentage
- `genesis_position_count` - Number of open positions
- `genesis_tilt_score` - Psychological tilt indicator

### 2. Latency Dashboard (`genesis-latency`)

Performance monitoring for order execution and WebSocket connections.

**Panels:**
- Order Execution Latency (Time Series) - p50, p95, p99 percentiles
- WebSocket Latency (Time Series) - Network latency percentiles
- P99 Execution Time (Gauge) - 99th percentile execution time
- Rate Limit Usage (Gauge) - API rate limit consumption
- Latency Distribution (Donut Chart) - Latency bucket distribution

**Key Metrics:**
- `genesis_order_execution_time_seconds` - Order execution duration
- `genesis_websocket_latency_ms` - WebSocket message latency
- `genesis_rate_limit_usage_ratio` - Rate limit usage (0-1)

### 3. System Health Dashboard (`genesis-system`)

Infrastructure and application health monitoring.

**Panels:**
- Uptime (Stat) - Application uptime
- Connection Status (State Timeline) - Exchange connectivity
- Memory Usage (Time Series) - Process memory consumption
- CPU Usage (Time Series) - Process CPU utilization
- Error Rate (Time Series) - Application error rate
- Active Alerts (Table) - Current firing alerts

### 4. Trading Activity Dashboard (`genesis-trading`)

Detailed trading activity and order flow analysis.

**Panels:**
- Orders per Minute (Time Series) - Order placement rate
- Fill Rate (Gauge) - Order fill success percentage
- Trade Volume (Bar Chart) - Trading volume by pair
- Win Rate (Stat) - Profitable trade percentage
- Sharpe Ratio (Stat) - Risk-adjusted returns
- Correlation Matrix (Heatmap) - Strategy correlations

## Installation

### Docker Deployment

```bash
# Start Grafana with docker-compose
docker-compose -f docker/docker-compose.monitoring.yml up -d grafana

# Default credentials
# Username: admin
# Password: admin (change on first login)
```

### Manual Dashboard Import

1. Access Grafana at `http://localhost:3000`
2. Navigate to Dashboards → Import
3. Upload JSON files from `docker/grafana/dashboards/`
4. Select Prometheus datasource
5. Click Import

### Automated Provisioning

Dashboards are automatically provisioned via:
```yaml
# docker/grafana/provisioning/dashboards/default.yml
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: 'GENESIS'
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

## Dashboard Configuration

### Variables

Create template variables for dynamic filtering:

```yaml
tier:
  query: label_values(genesis_info, tier)
  type: query
  multi: false
  
exchange:
  query: label_values(genesis_connection_status, exchange)
  type: query
  multi: true
  
timeframe:
  type: interval
  options: [1m, 5m, 15m, 1h, 24h]
```

### Alerts

Configure alert rules in dashboards:

```json
{
  "alert": {
    "name": "High Drawdown Alert",
    "conditions": [
      {
        "evaluator": {
          "params": [10],
          "type": "gt"
        },
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "params": [],
          "type": "avg"
        },
        "type": "query"
      }
    ],
    "frequency": "60s",
    "handler": 1,
    "noDataState": "no_data",
    "notifications": [
      {"uid": "slack-channel"}
    ]
  }
}
```

## Query Examples

### Performance Queries

```promql
# Average execution time
rate(genesis_order_execution_time_seconds_sum[5m]) 
/ rate(genesis_order_execution_time_seconds_count[5m])

# Success rate
rate(genesis_orders_filled_total[5m]) 
/ rate(genesis_orders_total[5m]) * 100

# P99 latency
histogram_quantile(0.99, 
  rate(genesis_websocket_latency_ms_bucket[5m]))

# Rate limit headroom
100 - (genesis_rate_limit_usage_ratio * 100)
```

### Business Metrics

```promql
# Hourly P&L
increase(genesis_pnl_dollars[1h])

# Daily trade count
increase(genesis_trades_total[24h])

# Win rate (requires recording rule)
genesis_winning_trades / genesis_total_trades * 100

# Risk-adjusted returns
genesis_pnl_dollars / stddev_over_time(genesis_pnl_dollars[24h])
```

### System Health

```promql
# Uptime percentage
avg_over_time(up{job="genesis"}[24h]) * 100

# Memory growth rate
rate(genesis_memory_usage_bytes[1h])

# Connection stability
avg_over_time(genesis_connection_status[5m])

# Error rate
rate(genesis_errors_total[5m])
```

## Customization

### Color Schemes

Consistent color coding across dashboards:
- Green: Normal/Good (< 50% thresholds)
- Yellow: Warning (50-75% thresholds)
- Orange: Alert (75-90% thresholds)
- Red: Critical (> 90% thresholds)

### Panel Types

Recommended panel types by metric:
- **Gauges**: Current values (P&L, positions, scores)
- **Time Series**: Trends (latency, volume, rates)
- **Stats**: Key indicators (uptime, win rate)
- **Tables**: Detailed data (alerts, trades)
- **Heatmaps**: Correlations and distributions

### Refresh Rates

- Real-time dashboards: 5s
- Performance dashboards: 10s
- Historical analysis: 30s
- System monitoring: 15s

## Annotations

### Event Annotations

Mark significant events on dashboards:

```promql
# Tier progression events
genesis_tier_change

# Emergency interventions
genesis_emergency_close

# System restarts
changes(genesis_up[5m]) > 0

# Large trades
increase(genesis_pnl_dollars[1m]) > 1000
```

### Alert Annotations

Display alert state changes:

```json
{
  "datasource": "Prometheus",
  "enable": true,
  "expr": "ALERTS{alertstate=\"firing\"}",
  "iconColor": "red",
  "name": "Firing Alerts",
  "tagKeys": "alertname,severity"
}
```

## Mobile Access

### Responsive Design

Dashboards optimized for mobile viewing:
- Single column layout for phones
- Touch-friendly controls
- Simplified panels for small screens
- Critical metrics prioritized

### Grafana Mobile App

1. Install Grafana Cloud app
2. Add server: `https://your-server:3000`
3. Configure push notifications for alerts
4. Set up quick access to key dashboards

## Performance Optimization

### Query Optimization

1. **Use Recording Rules**: Pre-compute expensive queries
2. **Limit Time Ranges**: Default to reasonable ranges
3. **Reduce Resolution**: Use `step` parameter appropriately
4. **Cache Results**: Enable query caching in Grafana

### Dashboard Best Practices

1. **Limit Panel Count**: Max 20 panels per dashboard
2. **Use Variables**: Reduce duplicate queries
3. **Lazy Loading**: Enable for large dashboards
4. **Shared Queries**: Reuse queries across panels

## Backup and Recovery

### Dashboard Backup

```bash
# Export all dashboards
for dashboard in $(curl -s http://localhost:3000/api/search | jq -r '.[].uri'); do
  curl -s http://localhost:3000/api/dashboards/$dashboard \
    > backup/$(basename $dashboard).json
done

# Import dashboards
for file in backup/*.json; do
  curl -X POST http://localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d @$file
done
```

### Version Control

Store dashboard JSON in git:
```bash
docker/grafana/dashboards/
├── pnl-dashboard.json
├── latency-dashboard.json
├── system-dashboard.json
└── trading-dashboard.json
```

## Troubleshooting

### Common Issues

1. **No Data**
   - Verify Prometheus datasource configuration
   - Check metric names in Prometheus
   - Ensure time range includes data

2. **Slow Loading**
   - Reduce panel count
   - Optimize queries with recording rules
   - Increase Grafana memory allocation

3. **Authentication Issues**
   - Reset admin password via CLI
   - Check LDAP/OAuth configuration
   - Verify API key permissions

### Debug Mode

Enable debug logging:
```ini
[log]
level = debug

[log.console]
level = debug
format = json
```

## Integration with Alerts

### Notification Channels

Configure in Grafana UI:
1. Alerting → Notification channels
2. Add channel (Slack, Email, PagerDuty)
3. Test configuration
4. Attach to dashboard alerts

### Alert Rules

Create from panels:
1. Edit panel → Alert tab
2. Define conditions
3. Set evaluation frequency
4. Configure notifications
5. Add descriptive messages

## Next Steps

1. Configure ELK stack for log visualization (see `elk-setup.md`)
2. Set up distributed tracing (see `distributed-tracing.md`)
3. Create SLA tracking dashboard
4. Implement capacity planning metrics