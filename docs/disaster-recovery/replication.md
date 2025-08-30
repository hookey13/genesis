# Cross-Region Replication Architecture

## Overview

The replication system provides geographic redundancy through asynchronous backup replication to secondary regions. This ensures data availability even in case of regional failures.

## Architecture

### Primary Region (Singapore - SGP1)
- Main backup storage
- Real-time backup creation
- Low latency to Binance servers

### Secondary Region (New York - NYC3)
- Replication target
- Geographic redundancy
- Disaster recovery site

## Replication Manager

### Configuration

```python
from genesis.backup import ReplicationManager, S3Client

# Initialize S3 clients for both regions
primary_client = S3Client(
    endpoint_url="https://sgp1.digitaloceanspaces.com",
    access_key=os.environ["DO_SPACES_KEY"],
    secret_key=os.environ["DO_SPACES_SECRET"],
    bucket_name="genesis-backups-sgp1",
    region="sgp1"
)

secondary_client = S3Client(
    endpoint_url="https://nyc3.digitaloceanspaces.com",
    access_key=os.environ["DO_SPACES_KEY"],
    secret_key=os.environ["DO_SPACES_SECRET"],
    bucket_name="genesis-backups-nyc3",
    region="nyc3"
)

# Create replication manager
replication_manager = ReplicationManager(
    primary_client=primary_client,
    secondary_client=secondary_client,
    max_concurrent_replications=5,
    replication_lag_threshold_seconds=300  # 5 minutes
)

# Start replication
await replication_manager.start()
```

## Replication Operations

### Automatic Replication

Backups are automatically replicated after creation:

```python
# Queue backup for replication
status = await replication_manager.replicate_backup(backup_metadata)

print(f"Replication status: {status.status}")
print(f"Size: {status.size_bytes / 1024 / 1024:.2f} MB")
```

### Manual Sync

Sync all backups between regions:

```python
# Full synchronization
results = await replication_manager.sync_all_backups()

print(f"Total backups: {results['total']}")
print(f"Replicated: {results['replicated']}")
print(f"Skipped (already exists): {results['skipped']}")
print(f"Failed: {results['failed']}")
```

### Verification

Verify replication consistency:

```python
# Check replication status
verification = await replication_manager.verify_replication()

if verification["is_consistent"]:
    print("Regions are synchronized")
else:
    print(f"Missing in secondary: {verification['missing_count']}")
    for key in verification["missing_backups"][:5]:
        print(f"  - {key}")
```

## Monitoring

### Replication Lag

Monitor real-time replication lag:

```python
# Get current status
status = replication_manager.get_replication_status()

print(f"Replication lag: {status['replication_lag_seconds']} seconds")
print(f"Queue size: {status['queue_size']}")
print(f"Active replications: {status['active_replications']}")
print(f"Total replicated: {status['total_replicated']}")
print(f"Failed replications: {status['total_failed']}")
```

### Performance Metrics

```python
# Get replication metrics
if status["average_duration_seconds"] > 0:
    throughput_mbps = (status["total_replicated"] * avg_size_mb) / status["average_duration_seconds"]
    print(f"Average duration: {status['average_duration_seconds']:.1f}s")
    print(f"Throughput: {throughput_mbps:.2f} MB/s")
```

## Retry Logic

### Exponential Backoff

Failed replications are retried with exponential backoff:

```python
# Retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds

# Automatic retry logic
for attempt in range(MAX_RETRIES):
    try:
        await replicate_backup(backup)
        break
    except Exception as e:
        wait_time = BACKOFF_BASE ** attempt
        logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s")
        await asyncio.sleep(wait_time)
```

## Queue Management

### Replication Queue

```python
# Check queue status
queue_size = replication_manager.replication_queue.qsize()
print(f"Items in queue: {queue_size}")

# Priority replication
high_priority_backup = BackupMetadata(...)
await replication_manager.replication_queue.put((high_priority_backup, status))
```

### Worker Configuration

```python
# Adjust worker count based on load
if queue_size > 100:
    replication_manager.max_concurrent_replications = 10
elif queue_size < 10:
    replication_manager.max_concurrent_replications = 3
```

## Bandwidth Optimization

### Compression

Reduce transfer size with compression:

```python
# Compress before replication
import gzip

async def compress_and_replicate(file_path: Path):
    compressed_path = file_path.with_suffix('.gz')
    
    with open(file_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            f_out.writelines(f_in)
    
    # Replicate compressed file
    await replicate_file(compressed_path)
```

### Incremental Replication

Only replicate changes:

```python
# Check if file needs replication
async def needs_replication(backup: BackupMetadata) -> bool:
    # Check if exists in secondary
    secondary_files = await secondary_client.list_backups()
    secondary_checksums = {f["metadata"]["checksum"] for f in secondary_files}
    
    return backup.checksum not in secondary_checksums
```

## Failover Scenarios

### Primary Region Failure

Switch to secondary region:

```python
# Failover to secondary
async def failover_to_secondary():
    # Update configuration
    config.backup_endpoint = "https://nyc3.digitaloceanspaces.com"
    config.backup_bucket = "genesis-backups-nyc3"
    
    # Verify secondary is up to date
    verification = await replication_manager.verify_replication()
    if not verification["is_consistent"]:
        logger.warning(f"Secondary missing {verification['missing_count']} backups")
    
    # Switch backup manager to secondary
    backup_manager.s3_client = secondary_client
    
    logger.info("Failover to secondary region complete")
```

### Failback to Primary

Return to primary when recovered:

```python
# Failback procedure
async def failback_to_primary():
    # Sync any new backups from secondary to primary
    await reverse_sync_backups()
    
    # Switch back to primary
    backup_manager.s3_client = primary_client
    
    # Resume normal replication
    await replication_manager.start()
    
    logger.info("Failback to primary region complete")
```

## Best Practices

### 1. Monitor Continuously

```python
# Alerting on high lag
if status["replication_lag_seconds"] > 300:
    await send_alert("High replication lag detected")
```

### 2. Test Regularly

```python
# Monthly replication test
async def test_replication():
    # Create test backup
    test_backup = await create_test_backup()
    
    # Replicate
    await replication_manager.replicate_backup(test_backup)
    
    # Verify in secondary
    exists = await verify_in_secondary(test_backup)
    
    # Clean up
    await cleanup_test_backup(test_backup)
    
    return exists
```

### 3. Handle Network Issues

```python
# Network resilience
async def replicate_with_retry(backup: BackupMetadata):
    for attempt in range(5):
        try:
            return await replication_manager.replicate_backup(backup)
        except NetworkError as e:
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Troubleshooting

### Common Issues

#### High Replication Lag
- **Cause**: Network congestion or large backup sizes
- **Solution**: Increase workers, optimize bandwidth, compress backups

#### Failed Replications
- **Cause**: Network errors, authentication issues
- **Solution**: Check credentials, verify network connectivity, review logs

#### Inconsistent Regions
- **Cause**: Replication failures, manual deletions
- **Solution**: Run full sync, investigate missing backups

## Metrics

### Key Performance Indicators

- **Replication Lag**: Target < 5 minutes
- **Success Rate**: Target > 99.9%
- **Queue Size**: Target < 10 items
- **Throughput**: Target > 10 MB/s

### Monitoring Queries

```sql
-- Prometheus queries
-- Average replication lag
avg(genesis_replication_lag_seconds)

-- Replication success rate
rate(genesis_replication_success[1h]) / rate(genesis_replication_total[1h])

-- Queue growth rate
deriv(genesis_replication_queue_size[5m])
```

## Configuration

### Environment Variables

```bash
# Primary region
export PRIMARY_REGION="sgp1"
export PRIMARY_ENDPOINT="https://sgp1.digitaloceanspaces.com"
export PRIMARY_BUCKET="genesis-backups-sgp1"

# Secondary region
export SECONDARY_REGION="nyc3"
export SECONDARY_ENDPOINT="https://nyc3.digitaloceanspaces.com"
export SECONDARY_BUCKET="genesis-backups-nyc3"

# Replication settings
export REPLICATION_MAX_WORKERS="5"
export REPLICATION_LAG_THRESHOLD="300"
export REPLICATION_RETRY_MAX="3"
```

## Related Documentation

- [Backup Procedures](backup-procedures.md)
- [Failover Procedures](failover-procedures.md)
- [DR Architecture](dr-runbook.md)