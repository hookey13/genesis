"""Cross-region backup replication manager."""

import asyncio
from datetime import datetime

import structlog
from pydantic import BaseModel

from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.utils.decorators import with_retry

logger = structlog.get_logger(__name__)


class ReplicationStatus(BaseModel):
    """Status of a replication operation."""

    source_key: str
    destination_key: str
    status: str  # pending, in_progress, completed, failed
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    size_bytes: int = 0


class ReplicationManager:
    """Manages cross-region backup replication."""

    def __init__(
        self,
        primary_client: S3Client,
        secondary_client: S3Client,
        max_concurrent_replications: int = 5,
        replication_lag_threshold_seconds: int = 300
    ):
        """Initialize replication manager.
        
        Args:
            primary_client: Primary region S3 client
            secondary_client: Secondary region S3 client
            max_concurrent_replications: Maximum concurrent replication operations
            replication_lag_threshold_seconds: Alert threshold for replication lag
        """
        self.primary_client = primary_client
        self.secondary_client = secondary_client
        self.max_concurrent_replications = max_concurrent_replications
        self.replication_lag_threshold_seconds = replication_lag_threshold_seconds

        # Track replication state
        self.replication_queue: asyncio.Queue = asyncio.Queue()
        self.active_replications: set[str] = set()
        self.replication_history: list[ReplicationStatus] = []
        self.replication_lag_seconds: float = 0

        # Start replication workers
        self.workers: list[asyncio.Task] = []
        self.running = False

    async def start(self) -> None:
        """Start replication workers."""
        if self.running:
            return

        self.running = True

        # Start worker tasks
        for i in range(min(self.max_concurrent_replications, 3)):
            worker = asyncio.create_task(self._replication_worker(i))
            self.workers.append(worker)

        # Start monitoring task
        monitor = asyncio.create_task(self._monitor_replication_health())
        self.workers.append(monitor)

        logger.info(
            "Replication manager started",
            workers=len(self.workers),
            primary_region=self.primary_client.region,
            secondary_region=self.secondary_client.region
        )

    async def stop(self) -> None:
        """Stop replication workers."""
        if not self.running:
            return

        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()

        logger.info("Replication manager stopped")

    async def replicate_backup(self, backup_metadata: BackupMetadata) -> ReplicationStatus:
        """Queue backup for replication.
        
        Args:
            backup_metadata: Backup to replicate
            
        Returns:
            Replication status
        """
        status = ReplicationStatus(
            source_key=backup_metadata.destination_key,
            destination_key=backup_metadata.destination_key.replace(
                self.primary_client.region,
                self.secondary_client.region
            ),
            status="pending",
            size_bytes=backup_metadata.size_bytes
        )

        # Add to queue
        await self.replication_queue.put((backup_metadata, status))

        logger.info(
            "Backup queued for replication",
            source_key=status.source_key,
            size_mb=status.size_bytes / 1024 / 1024
        )

        return status

    async def _replication_worker(self, worker_id: int) -> None:
        """Worker to process replication queue.
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Replication worker {worker_id} started")

        while self.running:
            try:
                # Get item from queue with timeout
                try:
                    backup_metadata, status = await asyncio.wait_for(
                        self.replication_queue.get(),
                        timeout=5.0
                    )
                except TimeoutError:
                    continue

                # Mark as active
                self.active_replications.add(status.source_key)
                status.status = "in_progress"
                status.started_at = datetime.utcnow()

                try:
                    # Perform replication
                    await self._replicate_single_backup(backup_metadata, status)

                    status.status = "completed"
                    status.completed_at = datetime.utcnow()

                    logger.info(
                        f"Worker {worker_id}: Replication completed",
                        source_key=status.source_key,
                        duration_seconds=(status.completed_at - status.started_at).total_seconds()
                    )

                except Exception as e:
                    status.status = "failed"
                    status.error_message = str(e)
                    status.retry_count += 1

                    logger.error(
                        f"Worker {worker_id}: Replication failed",
                        source_key=status.source_key,
                        error=str(e),
                        retry_count=status.retry_count
                    )

                    # Retry if under limit
                    if status.retry_count < 3:
                        await asyncio.sleep(2 ** status.retry_count)  # Exponential backoff
                        await self.replication_queue.put((backup_metadata, status))

                finally:
                    # Remove from active set
                    self.active_replications.discard(status.source_key)
                    self.replication_history.append(status)

            except Exception as e:
                logger.error(f"Worker {worker_id} error", error=str(e))
                await asyncio.sleep(1)

    @with_retry(max_attempts=3, backoff_factor=2)
    async def _replicate_single_backup(
        self,
        backup_metadata: BackupMetadata,
        status: ReplicationStatus
    ) -> None:
        """Replicate a single backup to secondary region.
        
        Args:
            backup_metadata: Backup metadata
            status: Replication status to update
        """
        # Download from primary
        temp_path = Path(f"/tmp/replication_{backup_metadata.backup_id}.tmp")

        try:
            # Download from primary region
            await self.primary_client.download_backup(
                key=backup_metadata.destination_key,
                destination_path=temp_path,
                verify_checksum=True
            )

            # Upload to secondary region
            secondary_key = await self.secondary_client.upload_backup(
                file_path=temp_path,
                key_prefix=backup_metadata.destination_key.rsplit("/", 1)[0] + "/",
                metadata=backup_metadata,
                encryption="AES256"
            )

            status.destination_key = secondary_key

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    async def sync_all_backups(self) -> dict[str, int]:
        """Sync all backups from primary to secondary region.
        
        Returns:
            Dictionary with sync statistics
        """
        logger.info("Starting full backup sync")

        stats = {
            "total": 0,
            "replicated": 0,
            "skipped": 0,
            "failed": 0
        }

        # List all backups in primary
        primary_backups = await self.primary_client.list_backups()
        stats["total"] = len(primary_backups)

        # List all backups in secondary
        secondary_backups = await self.secondary_client.list_backups()
        secondary_keys = {b["key"] for b in secondary_backups}

        # Queue missing backups for replication
        for backup in primary_backups:
            # Check if already replicated
            expected_key = backup["key"].replace(
                self.primary_client.region,
                self.secondary_client.region
            )

            if expected_key in secondary_keys:
                stats["skipped"] += 1
                continue

            # Create metadata from backup info
            metadata_dict = backup["metadata"]
            metadata = BackupMetadata(
                backup_id=metadata_dict.get("backup-id", "unknown"),
                timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                size_bytes=backup["size_bytes"],
                checksum=metadata_dict.get("checksum", ""),
                database_version=metadata_dict.get("database-version", "unknown"),
                backup_type=metadata_dict.get("backup-type", "full"),
                retention_policy=metadata_dict.get("retention-policy", "daily"),
                source_path="",
                destination_key=backup["key"]
            )

            # Queue for replication
            status = await self.replicate_backup(metadata)

            # Wait for completion
            while status.source_key in self.active_replications:
                await asyncio.sleep(0.1)

            if status.status == "completed":
                stats["replicated"] += 1
            else:
                stats["failed"] += 1

        logger.info("Backup sync completed", stats=stats)

        return stats

    async def _monitor_replication_health(self) -> None:
        """Monitor replication health and lag."""
        while self.running:
            try:
                # Calculate replication lag
                primary_latest = await self._get_latest_backup_time(self.primary_client)
                secondary_latest = await self._get_latest_backup_time(self.secondary_client)

                if primary_latest and secondary_latest:
                    self.replication_lag_seconds = (
                        primary_latest - secondary_latest
                    ).total_seconds()

                    # Alert if lag exceeds threshold
                    if self.replication_lag_seconds > self.replication_lag_threshold_seconds:
                        logger.warning(
                            "High replication lag detected",
                            lag_seconds=self.replication_lag_seconds,
                            threshold_seconds=self.replication_lag_threshold_seconds
                        )

                # Log health metrics
                logger.debug(
                    "Replication health",
                    queue_size=self.replication_queue.qsize(),
                    active_replications=len(self.active_replications),
                    lag_seconds=self.replication_lag_seconds
                )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(60)

    async def _get_latest_backup_time(self, client: S3Client) -> datetime | None:
        """Get timestamp of most recent backup.
        
        Args:
            client: S3 client to query
            
        Returns:
            Latest backup timestamp or None
        """
        backups = await client.list_backups(max_keys=1)

        if not backups:
            return None

        # Parse timestamp from metadata
        metadata = backups[0]["metadata"]
        timestamp_str = metadata.get("timestamp")

        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)

        return None

    async def verify_replication(self) -> dict[str, any]:
        """Verify replication consistency between regions.
        
        Returns:
            Verification results
        """
        logger.info("Verifying replication consistency")

        # List backups in both regions
        primary_backups = await self.primary_client.list_backups()
        secondary_backups = await self.secondary_client.list_backups()

        primary_keys = {b["key"] for b in primary_backups}
        secondary_keys = {b["key"] for b in secondary_backups}

        # Find missing backups
        missing_in_secondary = []
        for key in primary_keys:
            expected_key = key.replace(
                self.primary_client.region,
                self.secondary_client.region
            )
            if expected_key not in secondary_keys:
                missing_in_secondary.append(key)

        results = {
            "primary_count": len(primary_backups),
            "secondary_count": len(secondary_backups),
            "missing_count": len(missing_in_secondary),
            "missing_backups": missing_in_secondary[:10],  # First 10
            "replication_lag_seconds": self.replication_lag_seconds,
            "is_consistent": len(missing_in_secondary) == 0
        }

        if not results["is_consistent"]:
            logger.warning(
                "Replication inconsistency detected",
                missing_count=results["missing_count"]
            )
        else:
            logger.info("Replication verified - regions are consistent")

        return results

    def get_replication_status(self) -> dict[str, any]:
        """Get current replication status.
        
        Returns:
            Status dictionary
        """
        # Calculate statistics
        completed = [s for s in self.replication_history if s.status == "completed"]
        failed = [s for s in self.replication_history if s.status == "failed"]

        avg_duration = 0
        if completed:
            durations = [
                (s.completed_at - s.started_at).total_seconds()
                for s in completed
                if s.completed_at and s.started_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            "running": self.running,
            "queue_size": self.replication_queue.qsize(),
            "active_replications": len(self.active_replications),
            "total_replicated": len(completed),
            "total_failed": len(failed),
            "average_duration_seconds": avg_duration,
            "replication_lag_seconds": self.replication_lag_seconds,
            "primary_region": self.primary_client.region,
            "secondary_region": self.secondary_client.region
        }


from pathlib import Path  # Add this import at the top
