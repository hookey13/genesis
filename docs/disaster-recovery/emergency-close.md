# Emergency Position Closure Procedures

## Overview

The emergency position closure system provides rapid, automated closure of all trading positions during critical events. It includes intelligent position unwinding, correlation consideration, and dead man's switch activation.

## Emergency Closer System

### Components

1. **EmergencyCloser** - Main coordinator for emergency closures
2. **PositionUnwinder** - Intelligent position prioritization and unwinding
3. **DeadMansSwitch** - Automatic trigger if heartbeat stops

## Emergency Closure Execution

### Manual Emergency Closure

```python
from genesis.emergency import EmergencyCloser, PositionUnwinder

# Initialize components
position_unwinder = PositionUnwinder(correlation_threshold=0.7)
emergency_closer = EmergencyCloser(
    exchange_gateway=exchange_gateway,
    position_unwinder=position_unwinder,
    max_slippage_percent=Decimal("2.0"),
    notification_channels=["email", "slack", "pagerduty"]
)

# Execute emergency closure
result = await emergency_closer.emergency_close_all(
    reason="Market crash detected",
    dry_run=False,  # Set to True for simulation
    force=False     # Set to True to bypass safety checks
)

print(f"Positions closed: {result['positions_closed']}")
print(f"Total PnL: {result['total_realized_pnl']}")
print(f"Duration: {result['duration_seconds']} seconds")
```

### Dry Run Testing

Always test with dry run first:

```python
# Simulate closure without executing
result = await emergency_closer.emergency_close_all(
    reason="Test emergency closure",
    dry_run=True,
    force=False
)

# Review what would happen
for position in result["details"]:
    print(f"{position['symbol']}: {position['quantity']} @ {position['executed_price']}")
```

## Position Unwinding Strategy

### Prioritization Logic

Positions are prioritized based on:

1. **Risk Score** - Exposure × volatility × loss factor
2. **Correlation Groups** - Related positions grouped together
3. **Market Impact** - Large positions split into chunks

```python
# Get prioritized positions
positions = await exchange_gateway.get_positions()
prioritized = position_unwinder.prioritize_positions(
    positions=positions,
    risk_first=True  # Close highest risk first
)

# Review prioritization
for position in prioritized[:5]:
    print(f"{position['symbol']}: Risk={position['risk_score']:.2f}")
```

### Correlation Handling

Correlated positions are grouped to avoid market impact:

```python
# Known correlations
correlations = {
    ("BTC/USDT", "ETH/USDT"): 0.85,
    ("BTC/USDT", "SOL/USDT"): 0.80,
    ("ETH/USDT", "BNB/USDT"): 0.70
}

# Positions closed in correlation groups
groups = position_unwinder._group_correlated_positions(positions)
for group in groups:
    print(f"Group {group[0]['correlation_group']}: {len(group)} positions")
```

### Market Impact Mitigation

Large positions are automatically split:

```python
# Calculate unwinding schedule
schedule = position_unwinder.calculate_unwinding_schedule(
    positions=prioritized,
    max_orders_per_minute=60,
    market_impact_threshold=Decimal("0.1")
)

# Execute with delays
for position, delay_seconds in schedule:
    await asyncio.sleep(delay_seconds)
    await close_position(position)
```

## Dead Man's Switch

### Configuration

```python
from genesis.emergency import DeadMansSwitch

# Create dead man's switch
dead_mans_switch = DeadMansSwitch(
    timeout_seconds=300,  # 5 minutes without heartbeat
    check_interval_seconds=30,
    emergency_callback=lambda: emergency_closer.emergency_close_all(
        reason="Dead man's switch triggered",
        dry_run=False,
        force=True
    )
)

# Arm the switch
dead_mans_switch.arm()

# Regular heartbeat (must be called regularly)
while trading_active:
    dead_mans_switch.heartbeat()
    await asyncio.sleep(60)  # Heartbeat every minute
```

### Status Monitoring

```python
# Check dead man's switch status
status = dead_mans_switch.get_status()

print(f"Armed: {status['armed']}")
print(f"Time since heartbeat: {status['time_since_heartbeat']}s")
print(f"Will trigger in: {status['will_trigger_in']}s")

# Disarm when not needed
dead_mans_switch.disarm()
```

## Emergency Scenarios

### Scenario 1: Exchange Hack

```python
# Immediate closure with maximum urgency
result = await emergency_closer.emergency_close_all(
    reason="Exchange security breach detected",
    dry_run=False,
    force=True  # Bypass all safety checks
)
```

### Scenario 2: System Failure

```python
# Graceful closure with safety checks
result = await emergency_closer.emergency_close_all(
    reason="Critical system failure",
    dry_run=False,
    force=False  # Maintain safety checks
)
```

### Scenario 3: Regulatory Action

```python
# Controlled closure with documentation
result = await emergency_closer.emergency_close_all(
    reason="Regulatory compliance - immediate cessation required",
    dry_run=False,
    force=False
)

# Generate compliance report
await generate_closure_report(result)
```

## Closure Configuration

### Slippage Settings

```python
# Configure acceptable slippage
emergency_closer.max_slippage_percent = Decimal("5.0")  # Allow 5% slippage in emergency

# Position-specific slippage
volatile_pairs = ["DOGE/USDT", "SHIB/USDT"]
for position in positions:
    if position["symbol"] in volatile_pairs:
        position["max_slippage"] = Decimal("10.0")  # Higher for volatile
```

### Notification Channels

```python
# Configure notifications
emergency_closer.notification_channels = [
    "email",      # Email to team
    "slack",      # Slack alert
    "pagerduty",  # Page on-call
    "telegram",   # Telegram message
    "webhook"     # Custom webhook
]

# Custom notification handler
async def custom_notification(subject: str, message: str):
    # Send to monitoring system
    await monitoring.send_alert(subject, message)
    
    # Log to audit system
    await audit.log_emergency(subject, message)
```

## Hedge Positions

### Suggested Hedges

During unwinding, the system can suggest hedges:

```python
# Get hedge suggestions
hedges = position_unwinder.suggest_hedge_positions(positions)

for hedge in hedges:
    print(f"Suggested: {hedge['side']} {hedge['suggested_quantity']} {hedge['symbol']}")
    print(f"Reason: {hedge['reason']}")
```

## Audit Trail

### Emergency Closure Logging

All emergency closures are logged:

```python
# Audit trail automatically created
audit_entry = {
    "timestamp": "2025-01-01T12:00:00Z",
    "action": "emergency_closure",
    "reason": "Market crash",
    "positions_closed": 15,
    "total_pnl": -5000.00,
    "duration_seconds": 45,
    "details": [...]  # Full position details
}

# Query audit history
history = await get_emergency_closure_history()
for event in history:
    print(f"{event['timestamp']}: {event['reason']} - {event['positions_closed']} positions")
```

## Recovery After Emergency

### Post-Closure Checklist

- [ ] All positions confirmed closed
- [ ] Exchange reconciliation complete
- [ ] PnL calculated and recorded
- [ ] Notifications sent to all channels
- [ ] Audit trail created
- [ ] Root cause analysis initiated
- [ ] Recovery plan developed
- [ ] Team debriefing scheduled

### Gradual Resumption

```python
# After emergency resolved
async def resume_trading():
    # 1. Verify system stability
    if not await verify_system_health():
        return False
    
    # 2. Start with reduced limits
    await set_position_limits(max_positions=5, max_exposure=10000)
    
    # 3. Enable monitoring
    await enable_enhanced_monitoring()
    
    # 4. Gradual increase
    for day in range(7):
        await increase_limits(factor=1.5)
        await asyncio.sleep(86400)  # Daily increase
    
    return True
```

## Best Practices

1. **Regular Testing**
   - Weekly dry run tests
   - Monthly full simulation
   - Quarterly live test with minimal positions

2. **Clear Procedures**
   - Document trigger conditions
   - Define escalation path
   - Maintain contact lists

3. **Fast Execution**
   - Pre-authorize emergency actions
   - Minimize decision points
   - Automate where possible

4. **Communication**
   - Immediate team notification
   - Exchange communication if needed
   - Stakeholder updates

## Metrics and Monitoring

### Key Metrics

- **Closure Time**: Target < 60 seconds for all positions
- **Slippage**: Average < 2%, maximum < 5%
- **Success Rate**: 100% positions closed
- **Notification Delivery**: 100% within 1 minute

### Monitoring Dashboard

```python
# Real-time monitoring during emergency
async def monitor_emergency_closure():
    while emergency_closer.closure_in_progress:
        status = emergency_closer.get_closure_status()
        
        print(f"Progress: {len(status['closure_results'])} positions")
        print(f"Time elapsed: {(datetime.now() - status['last_closure_start']).seconds}s")
        
        await asyncio.sleep(1)
```

## Command Line Interface

### Emergency Commands

```bash
# Immediate emergency closure
python -m genesis.emergency close --reason "Emergency" --force

# Dry run
python -m genesis.emergency close --reason "Test" --dry-run

# Check status
python -m genesis.emergency status

# Arm dead man's switch
python -m genesis.emergency arm-dms --timeout 300

# Disarm dead man's switch
python -m genesis.emergency disarm-dms
```

## Related Documentation

- [DR Runbook](dr-runbook.md)
- [Failover Procedures](failover-procedures.md)
- [Recovery Guide](recovery-guide.md)
- [Risk Management](../risk-management.md)