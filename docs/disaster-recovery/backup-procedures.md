# Backup Procedures

## Overview

Project GENESIS implements a comprehensive backup strategy with automated SQLite database backups to DigitalOcean Spaces (S3-compatible storage). The system provides both full and incremental backups with cross-region replication for geographic redundancy.

## Backup Schedule

### Automated Backups
- **Full Backups**: Every 4 hours
- **Incremental Backups**: Every 5 minutes (WAL files)
- **Retention Policy Application**: Daily at midnight

### Retention Policies
| Backup Type | Retention Period | Use Case |
|------------|------------------|----------|
| Hourly | 7 days | Recent recovery points |
| Daily | 30 days | Daily snapshots |
| Monthly | 365 days | Long-term compliance |
| Yearly | Indefinite | Annual archives |

## Backup Components

### 1. BackupManager
Primary component responsible for:
- Scheduling automated backups
- Creating full database snapshots
- Managing incremental WAL backups
- Applying retention policies
- Verifying backup integrity

### 2. S3Client
Handles storage operations:
- Upload backups to DigitalOcean Spaces
- Download backups for recovery
- Generate presigned URLs
- Calculate and verify checksums
- Apply server-side encryption (AES-256)

### 3. ReplicationManager
Manages cross-region redundancy:
- Async replication to secondary region
- Monitors replication lag
- Verifies consistency between regions
- Handles retry logic with exponential backoff

## Manual Backup Procedures

### Creating a Manual Full Backup
```python
from genesis.backup import BackupManager
from genesis.backup.s3_client import S3Client
from pathlib import Path

# Initialize S3 client
s3_client = S3Client(
    endpoint_url="https://sgp1.digitaloceanspaces.com",
    access_key=os.environ["DO_SPACES_KEY"],
    secret_key=os.environ["DO_SPACES_SECRET"],
    bucket_name="genesis-backups",
    region="sgp1"
)

# Create backup manager
backup_manager = BackupManager(
    database_path=Path(".genesis/data/genesis.db"),
    s3_client=s3_client,
    local_backup_dir=Path(".genesis/data/backups"),
    enable_scheduler=False  # Manual mode
)

# Create backup
metadata = await backup_manager.create_full_backup()
print(f"Backup created: {metadata.backup_id}")
print(f"S3 location: {metadata.destination_key}")
print(f"Size: {metadata.size_bytes / 1024 / 1024:.2f} MB")
```

### Listing Available Backups
```python
# List all full backups
full_backups = await backup_manager.list_backups(backup_type="full")
for backup in full_backups:
    print(f"{backup.timestamp}: {backup.backup_id} ({backup.size_bytes} bytes)")

# List backups within date range
from datetime import datetime, timedelta

start_date = datetime.utcnow() - timedelta(days=7)
recent_backups = await backup_manager.list_backups(
    start_date=start_date,
    end_date=datetime.utcnow()
)
```

### Verifying Backup Integrity
```python
# Get latest backup
backups = await backup_manager.list_backups(backup_type="full")
latest_backup = backups[0]

# Verify integrity
is_valid = await backup_manager.verify_backup_integrity(latest_backup)
if is_valid:
    print(f"Backup {latest_backup.backup_id} verified successfully")
else:
    print(f"WARNING: Backup {latest_backup.backup_id} failed verification")
```

## Monitoring Backup Health

### Check Backup Status
```python
status = backup_manager.get_backup_status()
print(f"Last full backup: {status['last_full_backup']}")
print(f"Last incremental: {status['last_incremental_backup']}")
print(f"Total backups: {status['backup_count']}")
print(f"Scheduler running: {status['scheduler_running']}")
```

### Monitor Replication Lag
```python
from genesis.backup import ReplicationManager

replication_manager = ReplicationManager(
    primary_client=s3_client_sgp,
    secondary_client=s3_client_nyc
)

status = replication_manager.get_replication_status()
print(f"Replication lag: {status['replication_lag_seconds']} seconds")
print(f"Queue size: {status['queue_size']}")
print(f"Active replications: {status['active_replications']}")
```

## Backup Configuration

### Environment Variables
```bash
# DigitalOcean Spaces credentials
export DO_SPACES_KEY="your_access_key"
export DO_SPACES_SECRET="your_secret_key"

# Primary region (Singapore)
export DO_SPACES_ENDPOINT_PRIMARY="https://sgp1.digitaloceanspaces.com"
export DO_SPACES_BUCKET_PRIMARY="genesis-backups-sgp1"

# Secondary region (New York)
export DO_SPACES_ENDPOINT_SECONDARY="https://nyc3.digitaloceanspaces.com"
export DO_SPACES_BUCKET_SECONDARY="genesis-backups-nyc3"
```

### Application Settings
```python
# config/settings.py
BACKUP_CONFIG = {
    "full_backup_interval_hours": 4,
    "incremental_interval_minutes": 5,
    "retention": {
        "hourly_days": 7,
        "daily_days": 30,
        "monthly_days": 365
    },
    "replication": {
        "max_concurrent": 5,
        "lag_threshold_seconds": 300
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Backup Fails with "Checksum Mismatch"
- **Cause**: File corruption during transfer
- **Solution**: Retry backup, check network stability
- **Prevention**: Ensure stable connection, use retry logic

#### 2. High Replication Lag
- **Cause**: Network issues or high backup volume
- **Solution**: Check network connectivity, increase workers
- **Monitoring**: Alert when lag > 5 minutes

#### 3. Storage Space Issues
- **Cause**: Retention policy not applied
- **Solution**: Manually run retention policy
```python
deleted = await backup_manager.apply_retention_policy()
print(f"Deleted backups: {deleted}")
```

#### 4. WAL File Growing Too Large
- **Cause**: Long-running transactions
- **Solution**: Checkpoint WAL manually
```bash
sqlite3 .genesis/data/genesis.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

## Best Practices

1. **Test Backups Regularly**
   - Verify backup integrity weekly
   - Perform test restores monthly
   - Document restoration times

2. **Monitor Backup Metrics**
   - Track backup sizes for anomalies
   - Monitor replication lag
   - Alert on failed backups

3. **Secure Backup Storage**
   - Use server-side encryption
   - Rotate access keys quarterly
   - Implement least privilege access

4. **Document Recovery Procedures**
   - Maintain up-to-date runbooks
   - Train team on recovery process
   - Practice disaster recovery drills

## Backup Metrics

Track these KPIs:
- **Backup Success Rate**: Target > 99.9%
- **Average Backup Duration**: < 30 seconds for full backup
- **Replication Lag**: < 5 minutes
- **Storage Utilization**: Monitor growth trends
- **Recovery Time**: Test monthly, target < 15 minutes

## Integration with Monitoring

Backup metrics are exported to Prometheus:
```python
# Prometheus metrics
backup_total = Counter('genesis_backup_total', 'Total backups created')
backup_duration = Histogram('genesis_backup_duration_seconds', 'Backup duration')
backup_size = Gauge('genesis_backup_size_bytes', 'Backup size in bytes')
replication_lag = Gauge('genesis_replication_lag_seconds', 'Replication lag')
```

## Emergency Procedures

### Force Immediate Backup
```bash
python -m genesis.backup force-backup --type full
```

### Pause Scheduled Backups
```python
backup_manager.stop()  # Stops scheduler
```

### Resume Scheduled Backups
```python
backup_manager.start()  # Restarts scheduler
```

## Related Documentation
- [Recovery Guide](recovery-guide.md)
- [DR Runbook](dr-runbook.md)
- [Replication Architecture](replication.md)