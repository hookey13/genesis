# Recovery Guide

## Overview

The recovery system provides point-in-time database restoration capabilities using a combination of full backups, incremental WAL files, and event sourcing. The system can recover to any 5-minute window with complete state reconstruction.

## Recovery Components

### 1. RecoveryEngine
- Orchestrates the entire recovery process
- Downloads and stages backup files
- Applies incremental changes
- Validates recovered state
- Manages recovery metrics

### 2. EventReplayer
- Replays events from the event stream
- Reconstructs state transitions
- Handles all event types (positions, orders, balances, etc.)
- Maintains event ordering and consistency

### 3. StateReconstructor
- Validates database state consistency
- Reconstructs complete application state
- Calculates aggregate metrics
- Performs integrity checks

## Recovery Procedures

### Quick Recovery (Last Known Good State)

For immediate recovery to the most recent backup:

```python
from genesis.recovery import RecoveryEngine
from genesis.backup import BackupManager
from pathlib import Path

# Initialize components
backup_manager = BackupManager(...)
recovery_engine = RecoveryEngine(
    backup_manager=backup_manager,
    database_path=Path(".genesis/data/genesis.db"),
    recovery_staging_dir=Path(".genesis/recovery")
)

# Recover to 1 hour ago
from datetime import datetime, timedelta
target_time = datetime.utcnow() - timedelta(hours=1)

result = await recovery_engine.recover_to_timestamp(
    target_timestamp=target_time,
    validate=True,
    dry_run=False  # Set to True for testing
)

print(f"Recovery completed in {result['recovery_time_seconds']} seconds")
```

### Point-in-Time Recovery

For precise recovery to a specific timestamp:

```python
# Find available recovery points
backups = await backup_manager.list_backups()
for backup in backups[:5]:
    print(f"{backup.timestamp}: {backup.backup_type} - {backup.backup_id}")

# Choose target timestamp
target_time = datetime(2025, 1, 15, 14, 30, 0)  # Specific time

# Find required backups
full_backup, incremental_backups = await backup_manager.get_backup_for_timestamp(target_time)

print(f"Will use full backup from: {full_backup.timestamp}")
print(f"Will apply {len(incremental_backups)} incremental backups")

# Perform recovery
result = await recovery_engine.recover_to_timestamp(
    target_timestamp=target_time,
    validate=True,
    dry_run=True  # Test first
)

if result["success"]:
    print(f"Dry run successful. Database would be at: {result['recovered_db_path']}")
    
    # Perform actual recovery
    result = await recovery_engine.recover_to_timestamp(
        target_timestamp=target_time,
        validate=True,
        dry_run=False
    )
```

### Position Recovery

Recover trading positions from event sourcing:

```python
# Recover all positions
positions = await recovery_engine.recover_positions()

print(f"Total positions: {positions['total_positions']}")
print(f"Open positions: {positions['open_positions']}")

for pos_id, position in positions['positions'].items():
    print(f"{pos_id}: {position['symbol']} {position['side']} @ {position['entry_price']}")
```

### Order Reconciliation

Reconcile recovered state with exchange:

```python
# Get current orders from exchange
from genesis.exchange import Gateway
gateway = Gateway(...)
exchange_orders = await gateway.get_open_orders()

# Reconcile with recovered database
reconciliation = await recovery_engine.verify_order_reconciliation(exchange_orders)

if not reconciliation["is_reconciled"]:
    print("Discrepancies found:")
    print(f"Missing in DB: {len(reconciliation['missing_in_db'])}")
    print(f"Missing on exchange: {len(reconciliation['missing_on_exchange'])}")
    
    # Handle discrepancies
    for order in reconciliation["missing_on_exchange"]:
        print(f"Cancel local order: {order['client_order_id']}")
```

## Recovery Validation

### Pre-Recovery Validation

Before performing recovery:

```python
# Validate backup integrity
metadata = backups[0]  # Latest backup
is_valid = await backup_manager.verify_backup_integrity(metadata)

if not is_valid:
    print(f"WARNING: Backup {metadata.backup_id} failed integrity check")
    # Use next backup
```

### Post-Recovery Validation

After recovery completes:

```python
# Validate recovered state
from genesis.recovery import StateReconstructor
reconstructor = StateReconstructor()

import sqlite3
conn = sqlite3.connect(".genesis/data/genesis.db")
validation = reconstructor.validate_state(conn)
conn.close()

if validation["is_consistent"]:
    print("Recovery validation passed")
else:
    print(f"Validation errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
```

## Recovery Scenarios

### Scenario 1: Database Corruption

```bash
# Stop application
systemctl stop genesis-trading

# Backup corrupted database
cp .genesis/data/genesis.db .genesis/data/genesis.db.corrupted

# Run recovery
python -m genesis.recovery restore --timestamp "2025-01-15T12:00:00"

# Verify recovery
python -m genesis.recovery validate

# Restart application
systemctl start genesis-trading
```

### Scenario 2: Accidental Data Deletion

```python
# Find timestamp before deletion
target = datetime(2025, 1, 15, 10, 0, 0)  # Before incident

# Dry run to verify
result = await recovery_engine.recover_to_timestamp(
    target_timestamp=target,
    validate=True,
    dry_run=True
)

# Check recovered data
import sqlite3
conn = sqlite3.connect(result["recovered_db_path"])
cursor = conn.execute("SELECT COUNT(*) FROM positions")
print(f"Recovered positions: {cursor.fetchone()[0]}")
conn.close()

# Apply recovery if satisfied
result = await recovery_engine.recover_to_timestamp(
    target_timestamp=target,
    validate=True,
    dry_run=False
)
```

### Scenario 3: Rollback After Bad Deployment

```python
# Identify deployment time
deployment_time = datetime(2025, 1, 15, 9, 0, 0)

# Recover to pre-deployment state
target = deployment_time - timedelta(minutes=5)

result = await recovery_engine.recover_to_timestamp(
    target_timestamp=target,
    validate=True,
    dry_run=False
)

# Reconcile with exchange
await reconcile_with_exchange()
```

## Command Line Interface

### Basic Recovery Commands

```bash
# List available backups
python -m genesis.recovery list-backups --days 7

# Perform dry run recovery
python -m genesis.recovery restore --timestamp "2025-01-15T12:00:00" --dry-run

# Actual recovery
python -m genesis.recovery restore --timestamp "2025-01-15T12:00:00"

# Validate current database
python -m genesis.recovery validate

# Recover positions only
python -m genesis.recovery recover-positions

# Reconcile with exchange
python -m genesis.recovery reconcile-orders
```

### Advanced Recovery Options

```bash
# Recovery with specific backup
python -m genesis.recovery restore \
    --backup-id "full_abc123" \
    --validate \
    --reconcile

# Recovery without validation (faster but risky)
python -m genesis.recovery restore \
    --timestamp "2025-01-15T12:00:00" \
    --no-validate

# Recovery to staging location
python -m genesis.recovery restore \
    --timestamp "2025-01-15T12:00:00" \
    --staging-dir /tmp/recovery \
    --dry-run
```

## Recovery Metrics

Track these recovery KPIs:

- **Recovery Time Objective (RTO)**: Target < 15 minutes
- **Recovery Point Objective (RPO)**: Target < 5 minutes
- **Validation Success Rate**: Target 100%
- **Reconciliation Accuracy**: Target > 99.9%

## Troubleshooting Recovery

### Common Issues

#### 1. Recovery Takes Too Long
- **Cause**: Large database or slow network
- **Solution**: Use local backup cache, optimize network
- **Prevention**: Regular backup pruning, compression

#### 2. Validation Fails
- **Cause**: Data inconsistency or corruption
- **Solution**: Use earlier backup, check event stream
- **Monitoring**: Alert on validation failures

#### 3. Reconciliation Mismatches
- **Cause**: Orders executed during recovery
- **Solution**: Cancel unmatched orders, resync
- **Prevention**: Stop trading before recovery

#### 4. Missing Incremental Backups
- **Cause**: Backup failure or retention policy
- **Solution**: Use next available full backup
- **Prevention**: Monitor backup success rate

## Best Practices

1. **Test Recovery Regularly**
   - Monthly DR drills
   - Document recovery times
   - Validate against production data

2. **Monitor Recovery Readiness**
   - Check backup availability
   - Verify backup integrity
   - Track recovery metrics

3. **Maintain Recovery Documentation**
   - Update procedures after each drill
   - Document lessons learned
   - Train team on recovery process

4. **Prepare for Recovery**
   - Stop trading before recovery
   - Notify team of recovery window
   - Have rollback plan ready

## Recovery Checklist

- [ ] Identify target recovery timestamp
- [ ] Verify backup availability
- [ ] Stop application services
- [ ] Create pre-recovery backup
- [ ] Perform dry run recovery
- [ ] Validate recovered data
- [ ] Apply actual recovery
- [ ] Reconcile with exchange
- [ ] Restart application services
- [ ] Monitor for issues
- [ ] Document recovery details

## Related Documentation
- [Backup Procedures](backup-procedures.md)
- [DR Runbook](dr-runbook.md)
- [Emergency Procedures](emergency-close.md)