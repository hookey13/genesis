# Strategy Configuration Schema Documentation

## Overview

This document describes the complete configuration schema for GENESIS trading strategies. All strategy configurations are defined in YAML or JSON format and support hot-reload, versioning, A/B testing, and environment-specific overrides.

## Configuration File Location

Strategy configuration files are located in:
```
config/strategies/
├── sniper_arbitrage.yaml      # Sniper tier strategies
├── hunter_mean_reversion.yaml # Hunter tier strategies
└── strategist_vwap.yaml       # Strategist tier strategies
```

## Schema Structure

### Root Level Structure

```yaml
strategy:       # Strategy metadata (required)
parameters:     # Trading parameters (required)
risk_limits:    # Risk management constraints (required)
execution:      # Order execution settings (required)
monitoring:     # Logging and alerting (required)
overrides:      # Environment-specific overrides (optional)
variants:       # A/B testing variants (optional)
```

## Field Definitions

### Strategy Section

**Required Fields:**

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `name` | string | Unique strategy identifier | Required, max 50 chars |
| `version` | string | Strategy version | Semantic versioning (e.g., "1.0.0") |
| `tier` | string | Trading tier | One of: "sniper", "hunter", "strategist" |
| `enabled` | boolean | Strategy activation status | true/false |

**Optional Fields:**

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `description` | string | Strategy description | Empty string |
| `tags` | array | Strategy tags for filtering | Empty array |

### Parameters Section

**Common Parameters:**

| Field | Type | Description | Range | Default |
|-------|------|-------------|-------|---------|
| `min_profit_pct` | decimal | Minimum profit percentage | 0.0 - 100.0 | Required |
| `max_position_pct` | decimal | Max position as % of capital | 0.0 - 100.0 | Required |
| `stop_loss_pct` | decimal | Stop loss percentage | 0.0 - 100.0 | 1.0 |
| `take_profit_pct` | decimal | Take profit percentage | 0.0 - 100.0 | 0.5 |
| `min_order_size` | decimal | Minimum order size | > 0 | 10.0 |
| `max_order_size` | decimal | Maximum order size | > min_order_size | 100.0 |

**Tier-Specific Limits:**

| Tier | max_order_size | max_positions | max_daily_loss_pct |
|------|---------------|---------------|-------------------|
| Sniper | 100 | 1 | 5% |
| Hunter | 500 | 5 | 10% |
| Strategist | 5000 | 20 | 20% |

### Risk Limits Section

| Field | Type | Description | Range | Default |
|-------|------|-------------|-------|---------|
| `max_positions` | integer | Max concurrent positions | 1 - 100 | Tier-dependent |
| `max_daily_loss_pct` | decimal | Max daily loss percentage | 0.0 - 100.0 | Tier-dependent |
| `max_correlation` | decimal | Max position correlation | 0.0 - 1.0 | 0.7 |
| `max_leverage` | decimal | Maximum leverage | 1.0 - 10.0 | 1.0 |
| `min_liquidity_ratio` | decimal | Minimum liquidity ratio | 0.0 - 1.0 | 0.1 |

### Execution Section

| Field | Type | Description | Options | Default |
|-------|------|-------------|---------|---------|
| `order_type` | string | Order type | "market", "limit", "stop" | "market" |
| `time_in_force` | string | Time in force | "GTC", "IOC", "FOK" | "IOC" |
| `retry_attempts` | integer | Order retry attempts | 0 - 10 | 3 |
| `retry_delay_ms` | integer | Delay between retries | 0 - 10000 | 100 |
| `slippage_tolerance_pct` | decimal | Max slippage tolerance | 0.0 - 10.0 | 0.1 |
| `enable_iceberg` | boolean | Enable iceberg orders | true/false | false |
| `iceberg_size_pct` | decimal | Iceberg chunk size | 10.0 - 100.0 | 20.0 |

### Monitoring Section

| Field | Type | Description | Options | Default |
|-------|------|-------------|---------|---------|
| `log_level` | string | Logging level | "DEBUG", "INFO", "WARNING", "ERROR" | "INFO" |
| `metrics_interval_seconds` | integer | Metrics collection interval | 10 - 3600 | 60 |
| `alert_on_loss` | boolean | Alert on position loss | true/false | true |
| `alert_on_error` | boolean | Alert on execution error | true/false | true |
| `alert_channels` | array | Alert notification channels | ["email", "slack", "webhook"] | ["email"] |
| `performance_tracking` | boolean | Track performance metrics | true/false | true |

## Environment Overrides

Environment-specific overrides allow different parameter values for dev, staging, and production environments:

```yaml
overrides:
  dev:
    parameters:
      min_profit_pct: 0.1  # Lower threshold for testing
    monitoring:
      log_level: "DEBUG"   # More verbose logging
  staging:
    parameters:
      min_profit_pct: 0.3
  prod:
    parameters:
      min_profit_pct: 0.5  # Higher threshold for production
    monitoring:
      alert_on_loss: true
```

### Override Hierarchy

1. Base configuration (from main sections)
2. Environment-specific overrides (if environment matches)
3. Command-line overrides (highest priority)

## A/B Testing Variants

Support for testing multiple parameter sets:

```yaml
variants:
  - name: "conservative"
    parameters:
      min_profit_pct: 0.5
      stop_loss_pct: 0.5
  - name: "aggressive"
    parameters:
      min_profit_pct: 0.2
      max_position_pct: 0.03
  - name: "balanced"
    parameters:
      min_profit_pct: 0.35
      stop_loss_pct: 0.75
```

### Variant Selection

- Random selection: Variants are randomly assigned to strategy instances
- Weighted selection: Optional weights can be specified
- Performance tracking: Metrics tracked per variant
- Automatic optimization: Best performing variant can be promoted

## Cross-Field Validation Rules

The following cross-field constraints are automatically validated:

1. **Order Size Consistency**: `min_order_size` must be less than `max_order_size`
2. **Stop Loss vs Take Profit**: `stop_loss_pct` should typically be less than `take_profit_pct`
3. **Tier Limits**: All parameters must respect tier-specific limits
4. **Correlation Limits**: `max_correlation` must be between 0.0 and 1.0
5. **Leverage Constraints**: `max_leverage` limited based on tier

## Example Configurations

### Minimal Configuration (Sniper Tier)

```yaml
strategy:
  name: "SimpleArbitrage"
  version: "1.0.0"
  tier: "sniper"
  enabled: true

parameters:
  min_profit_pct: 0.3
  max_position_pct: 0.02

risk_limits:
  max_positions: 1
  max_daily_loss_pct: 5.0

execution:
  order_type: "market"
  time_in_force: "IOC"

monitoring:
  log_level: "INFO"
```

### Full Configuration (Strategist Tier)

```yaml
strategy:
  name: "AdvancedVWAP"
  version: "2.1.0"
  tier: "strategist"
  enabled: true
  description: "VWAP execution with dynamic adjustments"
  tags: ["vwap", "institutional", "large-orders"]

parameters:
  min_profit_pct: 0.5
  max_position_pct: 0.1
  stop_loss_pct: 2.0
  take_profit_pct: 1.0
  min_order_size: 100.0
  max_order_size: 5000.0
  vwap_period_minutes: 30
  participation_rate_pct: 10.0
  urgency_factor: 0.7

risk_limits:
  max_positions: 20
  max_daily_loss_pct: 20.0
  max_correlation: 0.6
  max_leverage: 3.0
  min_liquidity_ratio: 0.2

execution:
  order_type: "limit"
  time_in_force: "GTC"
  retry_attempts: 5
  retry_delay_ms: 500
  slippage_tolerance_pct: 0.5
  enable_iceberg: true
  iceberg_size_pct: 25.0

monitoring:
  log_level: "INFO"
  metrics_interval_seconds: 30
  alert_on_loss: true
  alert_on_error: true
  alert_channels: ["email", "slack"]
  performance_tracking: true

overrides:
  dev:
    parameters:
      min_profit_pct: 0.2
      max_order_size: 1000.0
    monitoring:
      log_level: "DEBUG"
  prod:
    parameters:
      min_profit_pct: 0.7
    monitoring:
      alert_channels: ["email", "slack", "webhook"]

variants:
  - name: "conservative"
    parameters:
      participation_rate_pct: 5.0
      urgency_factor: 0.3
  - name: "aggressive"
    parameters:
      participation_rate_pct: 15.0
      urgency_factor: 0.9
```

## Hot-Reload Behavior

### File Watching

- Configuration files are monitored for changes
- Changes detected within 1 second (configurable polling interval)
- Automatic validation before applying changes
- Rollback to previous version on validation failure

### Reload Process

1. File change detected
2. New configuration loaded and parsed
3. Validation performed (schema + cross-field)
4. If valid: Configuration atomically updated
5. If invalid: Change rejected, previous config retained
6. Audit log entry created

### Performance Requirements

- Config load time: <100ms per file
- Hot-reload latency: <500ms
- File watch overhead: <1% CPU
- Memory usage: <10MB for all configs

## Version Management

### Version History

- Last 100 versions kept by default (configurable)
- Each version includes:
  - Timestamp
  - Configuration snapshot
  - Change source (file, API, rollback)
  - User/system identifier

### Rollback Capability

```python
# Rollback to previous version
await config_manager.rollback_config("StrategyName", version_index=-1)

# Rollback to specific version
await config_manager.rollback_config("StrategyName", version_index=5)
```

## Audit Logging

All configuration changes are logged with:

- **Timestamp**: UTC timestamp of change
- **Strategy**: Strategy name
- **Action**: Type of change (load, reload, rollback)
- **Changed Fields**: List of modified fields
- **Old Values**: Previous values (sensitive data redacted)
- **New Values**: New values (sensitive data redacted)
- **Source**: Origin of change (file, API, manual)
- **User/System**: Identifier of change initiator

### Sensitive Data Redaction

The following field patterns are automatically redacted in logs:
- `*key*`, `*secret*`, `*password*`, `*token*`
- API credentials
- Authentication tokens

## Best Practices

### Configuration Organization

1. **One strategy per file**: Keep configurations focused
2. **Meaningful names**: Use descriptive strategy names
3. **Version control**: Track all configuration changes in git
4. **Environment separation**: Use overrides for env-specific values
5. **Comment documentation**: Add YAML comments for complex parameters

### Parameter Tuning

1. **Start conservative**: Begin with safer parameter values
2. **Test in dev**: Validate changes in development first
3. **Use A/B testing**: Test parameter changes with variants
4. **Monitor metrics**: Track performance after changes
5. **Document rationale**: Record why parameters were chosen

### Security Considerations

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Restrict file permissions**: Limit config file access
3. **Validate all inputs**: Ensure validation is always enabled
4. **Audit all changes**: Review audit logs regularly
5. **Encrypt sensitive values**: Use encryption for API keys

## Migration Guide

### From Legacy Configuration

```python
# Legacy format
config = {
    "min_profit": 0.003,  # 0.3%
    "position_size": 100,
}

# New format
config = {
    "strategy": {
        "name": "LegacyStrategy",
        "version": "1.0.0",
        "tier": "sniper",
        "enabled": True,
    },
    "parameters": {
        "min_profit_pct": 0.3,
        "max_position_pct": 0.02,
    },
    # ... additional required sections
}
```

### Schema Evolution

Configuration schema versions are managed through:
1. Backward compatibility for minor versions
2. Migration scripts for major versions
3. Automatic conversion where possible
4. Clear deprecation warnings

## Troubleshooting

### Common Issues

**Configuration not loading:**
- Check YAML syntax
- Verify required fields present
- Check file permissions
- Review validation errors in logs

**Hot-reload not working:**
- Verify hot-reload enabled
- Check file watcher status
- Review file permissions
- Check polling interval

**Validation failures:**
- Review error messages
- Check cross-field constraints
- Verify tier limits
- Ensure proper data types

**Performance issues:**
- Reduce polling frequency
- Limit version history size
- Optimize configuration size
- Check CPU/memory usage

## API Reference

### ConfigManager Methods

```python
# Initialize manager
manager = StrategyConfigManager(
    config_path="config/strategies",
    environment=Environment.PROD,
    enable_hot_reload=True,
    enable_ab_testing=True,
)

# Load configurations
await manager.load_all_configs()
await manager.load_config_file("path/to/config.yaml")

# Get configuration
config = manager.get_strategy_config("StrategyName")
config = manager.get_config_variant("StrategyName", "variant_name")

# Manage versions
await manager.rollback_config("StrategyName", version_index=-1)
history = manager.get_version_history("StrategyName")

# Hot-reload control
await manager.start_watching()
await manager.stop_watching()

# Validation
result = await manager.validate_config(config_dict)
```

## Support

For configuration issues or questions:
1. Check this documentation
2. Review example configurations
3. Check audit logs for errors
4. Contact the trading team

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-09-04 | Initial documentation release |