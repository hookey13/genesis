# Paper Trading Validation Guide

## Overview

Paper trading mode allows you to validate the complete trading system with real market data but without risking actual capital. This mode is essential for:

- Testing new strategies before going live
- Validating system stability over extended periods
- Training and familiarization with the platform
- Compliance with tier progression requirements

## Quick Start

### 1. Enable Paper Trading Mode

```python
# In your configuration
PAPER_TRADING_MODE = True
PAPER_TRADING_SESSION_ID = "your-session-id"
```

### 2. Launch the System

```bash
# Start in paper trading mode
python -m genesis --paper-trading

# Or with specific configuration
python -m genesis --paper-trading --config config/paper_trading_config.py
```

### 3. Monitor Performance

The UI will display a yellow "PAPER TRADING MODE" indicator and all positions/P&L will be simulated.

## Features

### Realistic Order Execution
- Simulated market fills with configurable slippage (default 0.1%)
- Realistic latency simulation (50ms default)
- Order book depth simulation
- Partial fill simulation for limit orders

### P&L Tracking
- Real-time P&L calculation accurate to 2 decimal places
- Separate tracking of realized and unrealized P&L
- Session-level aggregation and metrics
- Win rate and average win/loss tracking

### UI Integration
- Live position display with paper trading indicator
- Real-time P&L updates
- Performance metrics dashboard
- Trade history and audit trail

### Continuous Operation Monitoring
- Heartbeat monitoring every 30 seconds
- Health checks every 60 seconds
- Automatic reconnection on disconnection
- Performance metrics logging

## Validation Requirements

To pass paper trading validation, the system must achieve:

### Acceptance Criteria

1. **AC1: Trade Execution**
   - Complete 10 successful round-trip trades
   - Trades must include both entry and exit

2. **AC2: P&L Accuracy**
   - P&L calculations accurate to 2 decimal places
   - Both realized and unrealized P&L tracked

3. **AC3: Continuous Operation**
   - 24-hour continuous operation
   - 99%+ uptime (less than 15 minutes downtime)
   - Automatic recovery from disconnections

4. **AC4: UI Responsiveness**
   - Live position updates
   - Real-time P&L display
   - No lag or freezing

5. **AC5: Autonomous Operation**
   - No manual intervention required
   - Automatic error recovery
   - Self-healing capabilities

## Configuration

### Basic Settings

```python
# genesis/config/paper_trading_config.py

class PaperTradingConfig:
    # Session settings
    session_duration_hours = 24
    min_trades_required = 10
    
    # Position sizing (SNIPER tier)
    max_position_size_usdt = Decimal("100")
    min_position_size_usdt = Decimal("10")
    
    # Risk settings
    stop_loss_percent = Decimal("2.0")
    slippage_percent = Decimal("0.1")
    
    # Initial balance
    initial_balance_usdt = Decimal("10000")
```

### Advanced Settings

```python
# Monitoring intervals
heartbeat_interval_seconds = 30
health_check_interval_seconds = 60
log_performance_interval_seconds = 60

# Reconnection settings
auto_reconnect = True
max_reconnect_attempts = 10

# UI refresh rate
ui_refresh_interval_ms = 1000
```

## Running Tests

### Unit Tests

```bash
# Run paper trading unit tests
pytest tests/unit/test_paper_trading.py -v
```

### Integration Tests

```bash
# Run full paper trading validation suite
pytest tests/integration/test_paper_trading_suite.py -v

# Run with specific duration (hours)
pytest tests/integration/test_paper_trading_suite.py::test_paper_trading_validation --duration=1
```

### Performance Tests

```bash
# Test P&L calculation accuracy
pytest tests/unit/test_paper_trading.py::TestPnLCalculation -v

# Test continuous operation
pytest tests/unit/test_paper_trading.py::TestContinuousOperation -v
```

## Test Reports

After running the paper trading validation, a comprehensive report is generated:

### Report Contents

1. **Trade Summary**
   - Total trades executed
   - Entry/exit details for each trade
   - Symbols traded

2. **P&L Analysis**
   - Total P&L
   - Realized vs Unrealized
   - Win rate and statistics
   - Accuracy verification

3. **System Performance**
   - Uptime percentage
   - Heartbeats sent/received
   - Health check results
   - Disconnection events

4. **UI Validation**
   - Position update events
   - P&L update frequency
   - Response times

### Sample Report Output

```json
{
  "test_duration": "1:00:00",
  "overall_success": true,
  "trades": {
    "total_trades": 10,
    "success": true
  },
  "pnl_accuracy": {
    "realized_pnl": "125.50",
    "unrealized_pnl": "15.25",
    "total_pnl": "140.75",
    "accuracy_valid": true,
    "required_decimals": 2
  },
  "continuous_operation": {
    "duration_hours": 1,
    "heartbeats_received": 120,
    "health_checks_passed": 60,
    "disconnections": 0,
    "uptime_percent": 100.0,
    "continuous_operation": true
  },
  "ui_updates": {
    "position_updates": 20,
    "pnl_updates": 10,
    "live_updates": true
  },
  "manual_interventions": {
    "count": 0,
    "no_intervention_required": true
  }
}
```

## Troubleshooting

### Common Issues

1. **Trades Not Executing**
   - Check risk engine limits
   - Verify position sizing configuration
   - Check event bus connectivity

2. **P&L Inaccuracy**
   - Verify decimal precision settings
   - Check slippage configuration
   - Review fee calculations

3. **Disconnections**
   - Check network connectivity
   - Review WebSocket settings
   - Verify heartbeat configuration

4. **UI Not Updating**
   - Check event subscriptions
   - Verify UI refresh interval
   - Review widget connections

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)
```

## Best Practices

1. **Start Small**
   - Begin with 1-hour test sessions
   - Gradually increase to 24-hour tests
   - Monitor closely during initial runs

2. **Realistic Configuration**
   - Use production-like settings
   - Test with actual market volatility
   - Include network latency simulation

3. **Comprehensive Testing**
   - Test all trading strategies
   - Include edge cases
   - Verify error recovery

4. **Documentation**
   - Document any issues encountered
   - Record configuration changes
   - Maintain test logs

## Migration to Live Trading

After successful paper trading validation:

1. **Review Results**
   - Analyze P&L performance
   - Check system stability metrics
   - Verify all acceptance criteria met

2. **Configuration Changes**
   ```python
   PAPER_TRADING_MODE = False
   # Use real exchange credentials
   EXCHANGE_API_KEY = "your-api-key"
   EXCHANGE_SECRET_KEY = "your-secret"
   ```

3. **Start with Minimum**
   - Begin with minimum position sizes
   - Monitor closely for first 24 hours
   - Gradually increase as confidence builds

## Support

For issues or questions about paper trading:

1. Check the test logs in `.ai/debug-log.md`
2. Review validation reports in `reports/paper_trading/`
3. Consult the development team

---

*Paper Trading Module v1.0 - Project GENESIS*