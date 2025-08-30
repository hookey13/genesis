"""Automated database backup management system."""

import asyncio
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.core.exceptions import BackupError
from genesis.utils.decorators import with_retry

logger = structlog.get_logger(__name__)


class BackupManager:
    """Manages automated database backups with S3 storage."""

    def __init__(
        self,
        database_path: Path,
        s3_client: S3Client,
        local_backup_dir: Path,
        backup_interval_hours: int = 4,
        incremental_interval_minutes: int = 5,
        enable_scheduler: bool = True
    ):
        """Initialize backup manager.
        
        Args:
            database_path: Path to SQLite database
            s3_client: S3 client for remote storage
            local_backup_dir: Local directory for staging backups
            backup_interval_hours: Hours between full backups
            incremental_interval_minutes: Minutes between incremental backups
            enable_scheduler: Whether to enable automatic scheduling
        """
        self.database_path = database_path
        self.s3_client = s3_client
        self.local_backup_dir = local_backup_dir
        self.backup_interval_hours = backup_interval_hours
        self.incremental_interval_minutes = incremental_interval_minutes

        # Create local backup directory
        self.local_backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scheduler
        self.scheduler: AsyncIOScheduler | None = None
        if enable_scheduler:
            self.scheduler = AsyncIOScheduler()
            self._setup_schedule()

        # Track backup state
        self.last_full_backup: datetime | None = None
        self.last_incremental_backup: datetime | None = None
        self.backup_history: list[BackupMetadata] = []

    def _setup_schedule(self) -> None:
        """Set up automated backup schedule."""
        if not self.scheduler:
            return

        # Schedule full backups
        self.scheduler.add_job(
            self.create_full_backup,
            IntervalTrigger(hours=self.backup_interval_hours),
            id="full_backup",
            name="Full Database Backup",
            replace_existing=True
        )

        # Schedule incremental backups
        self.scheduler.add_job(
            self.create_incremental_backup,
            IntervalTrigger(minutes=self.incremental_interval_minutes),
            id="incremental_backup",
            name="Incremental WAL Backup",
            replace_existing=True
        )

        # Schedule retention policy application
        self.scheduler.add_job(
            self.apply_retention_policy,
            IntervalTrigger(hours=24),
            id="retention_policy",
            name="Apply Retention Policy",
            replace_existing=True
        )

        logger.info(
            "Backup schedule configured",
            full_interval_hours=self.backup_interval_hours,
            incremental_interval_minutes=self.incremental_interval_minutes
        )

    def start(self) -> None:
        """Start automated backup scheduler."""
        if self.scheduler and not self.scheduler.running:
            self.scheduler.start()
            logger.info("Backup scheduler started")

    def stop(self) -> None:
        """Stop automated backup scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Backup scheduler stopped")

    @with_retry(max_attempts=3, backoff_factor=2)
    async def create_full_backup(self) -> BackupMetadata:
        """Create full database backup.
        
        Returns:
            Backup metadata
        """
        backup_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        logger.info("Starting full backup", backup_id=backup_id)

        try:
            # Create local backup file
            backup_filename = f"genesis_full_{timestamp.strftime('%Y%m%d_%H%M%S')}.db"
            local_backup_path = self.local_backup_dir / backup_filename

            # Perform SQLite backup with WAL checkpoint
            await self._backup_sqlite_database(
                source_path=self.database_path,
                destination_path=local_backup_path,
                checkpoint=True
            )

            # Calculate metadata
            file_stats = local_backup_path.stat()
            checksum = await self.s3_client._calculate_checksum(local_backup_path)

            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                size_bytes=file_stats.st_size,
                checksum=checksum,
                database_version=await self._get_database_version(),
                backup_type="full",
                retention_policy=self._determine_retention_policy(timestamp),
                source_path=str(self.database_path),
                destination_key=""
            )

            # Upload to S3
            s3_key = await self.s3_client.upload_backup(
                file_path=local_backup_path,
                key_prefix="backups/full/",
                metadata=metadata
            )

            metadata.destination_key = s3_key

            # Clean up local file after successful upload
            local_backup_path.unlink()

            # Update state
            self.last_full_backup = timestamp
            self.backup_history.append(metadata)

            logger.info(
                "Full backup completed",
                backup_id=backup_id,
                size_mb=metadata.size_bytes / 1024 / 1024,
                s3_key=s3_key
            )

            return metadata

        except Exception as e:
            logger.error("Full backup failed", error=str(e))
            raise BackupError(f"Full backup failed: {e}")

    @with_retry(max_attempts=3, backoff_factor=2)
    async def create_incremental_backup(self) -> BackupMetadata | None:
        """Create incremental backup using WAL files.
        
        Returns:
            Backup metadata if WAL exists, None otherwise
        """
        if not self.last_full_backup:
            logger.warning("No full backup exists, skipping incremental backup")
            return None

        backup_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Check for WAL file
        wal_path = Path(str(self.database_path) + "-wal")
        if not wal_path.exists() or wal_path.stat().st_size == 0:
            logger.debug("No WAL changes to backup")
            return None

        logger.info("Starting incremental backup", backup_id=backup_id)

        try:
            # Copy WAL file
            backup_filename = f"genesis_wal_{timestamp.strftime('%Y%m%d_%H%M%S')}.wal"
            local_backup_path = self.local_backup_dir / backup_filename

            shutil.copy2(wal_path, local_backup_path)

            # Calculate metadata
            file_stats = local_backup_path.stat()
            checksum = await self.s3_client._calculate_checksum(local_backup_path)

            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                size_bytes=file_stats.st_size,
                checksum=checksum,
                database_version=await self._get_database_version(),
                backup_type="incremental",
                retention_policy="hourly",  # Incremental backups use hourly retention
                source_path=str(wal_path),
                destination_key=""
            )

            # Upload to S3
            s3_key = await self.s3_client.upload_backup(
                file_path=local_backup_path,
                key_prefix="backups/incremental/",
                metadata=metadata
            )

            metadata.destination_key = s3_key

            # Clean up local file
            local_backup_path.unlink()

            # Update state
            self.last_incremental_backup = timestamp
            self.backup_history.append(metadata)

            logger.info(
                "Incremental backup completed",
                backup_id=backup_id,
                size_kb=metadata.size_bytes / 1024,
                s3_key=s3_key
            )

            return metadata

        except Exception as e:
            logger.error("Incremental backup failed", error=str(e))
            raise BackupError(f"Incremental backup failed: {e}")

    async def _backup_sqlite_database(
        self,
        source_path: Path,
        destination_path: Path,
        checkpoint: bool = False
    ) -> None:
        """Perform SQLite database backup.
        
        Args:
            source_path: Source database path
            destination_path: Destination backup path
            checkpoint: Whether to checkpoint WAL before backup
        """
        loop = asyncio.get_event_loop()

        def perform_backup():
            # Open source database
            source_conn = sqlite3.connect(str(source_path))

            try:
                # Checkpoint WAL if requested
                if checkpoint:
                    source_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    logger.debug("WAL checkpoint completed")

                # Open destination database
                dest_conn = sqlite3.connect(str(destination_path))

                try:
                    # Perform backup using SQLite backup API
                    with dest_conn:
                        source_conn.backup(dest_conn)

                    logger.debug("SQLite backup completed", destination=str(destination_path))

                finally:
                    dest_conn.close()

            finally:
                source_conn.close()

        await loop.run_in_executor(None, perform_backup)

    async def _get_database_version(self) -> str:
        """Get database schema version.
        
        Returns:
            Version string
        """
        loop = asyncio.get_event_loop()

        def get_version():
            conn = sqlite3.connect(str(self.database_path))
            try:
                cursor = conn.execute("PRAGMA user_version")
                version = cursor.fetchone()[0]
                return f"v{version}"
            finally:
                conn.close()

        return await loop.run_in_executor(None, get_version)

    def _determine_retention_policy(self, timestamp: datetime) -> str:
        """Determine retention policy based on timestamp.
        
        Args:
            timestamp: Backup timestamp
            
        Returns:
            Retention policy (hourly, daily, monthly, yearly)
        """
        # First backup of the month -> monthly
        if timestamp.day == 1 and timestamp.hour < self.backup_interval_hours:
            return "monthly"

        # First backup of the year -> yearly
        if timestamp.month == 1 and timestamp.day == 1 and timestamp.hour < self.backup_interval_hours:
            return "yearly"

        # First backup of the day -> daily
        if timestamp.hour < self.backup_interval_hours:
            return "daily"

        # Default to hourly
        return "hourly"

    async def apply_retention_policy(self) -> dict[str, int]:
        """Apply retention policy to existing backups.
        
        Returns:
            Counts of deleted backups by policy
        """
        logger.info("Applying retention policy")

        deleted_counts = await self.s3_client.apply_retention_policy(
            prefix="backups/",
            hourly_days=7,
            daily_days=30,
            monthly_days=365
        )

        return deleted_counts

    async def list_backups(
        self,
        backup_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[BackupMetadata]:
        """List available backups.
        
        Args:
            backup_type: Filter by backup type (full, incremental)
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of backup metadata
        """
        prefix = None
        if backup_type:
            prefix = f"backups/{backup_type}/"

        backups_data = await self.s3_client.list_backups(prefix=prefix)

        backups = []
        for backup_data in backups_data:
            metadata_dict = backup_data["metadata"]

            # Parse metadata
            metadata = BackupMetadata(
                backup_id=metadata_dict.get("backup-id", "unknown"),
                timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                size_bytes=backup_data["size_bytes"],
                checksum=metadata_dict.get("checksum", ""),
                database_version=metadata_dict.get("database-version", "unknown"),
                backup_type=metadata_dict.get("backup-type", "full"),
                retention_policy=metadata_dict.get("retention-policy", "daily"),
                source_path="",
                destination_key=backup_data["key"]
            )

            # Apply date filters
            if start_date and metadata.timestamp < start_date:
                continue
            if end_date and metadata.timestamp > end_date:
                continue

            backups.append(metadata)

        # Sort by timestamp descending
        backups.sort(key=lambda x: x.timestamp, reverse=True)

        return backups

    async def get_backup_for_timestamp(
        self,
        target_timestamp: datetime
    ) -> tuple[BackupMetadata | None, list[BackupMetadata]]:
        """Find backup files needed for point-in-time recovery.
        
        Args:
            target_timestamp: Target recovery timestamp
            
        Returns:
            Tuple of (full backup, list of incremental backups)
        """
        # Find the most recent full backup before target
        full_backups = await self.list_backups(backup_type="full", end_date=target_timestamp)

        if not full_backups:
            logger.warning("No full backup found before target timestamp")
            return None, []

        full_backup = full_backups[0]  # Most recent

        # Find incremental backups between full backup and target
        incremental_backups = await self.list_backups(
            backup_type="incremental",
            start_date=full_backup.timestamp,
            end_date=target_timestamp
        )

        logger.info(
            "Found backups for recovery",
            full_backup_time=full_backup.timestamp,
            incremental_count=len(incremental_backups)
        )

        return full_backup, incremental_backups

    async def verify_backup_integrity(self, metadata: BackupMetadata) -> bool:
        """Verify backup integrity by downloading and checking.
        
        Args:
            metadata: Backup metadata
            
        Returns:
            True if backup is valid
        """
        try:
            # Download to temporary location
            temp_path = self.local_backup_dir / f"verify_{metadata.backup_id}.tmp"

            downloaded_metadata = await self.s3_client.download_backup(
                key=metadata.destination_key,
                destination_path=temp_path,
                verify_checksum=True
            )

            # Clean up temporary file
            temp_path.unlink()

            logger.info(
                "Backup integrity verified",
                backup_id=metadata.backup_id,
                checksum=metadata.checksum
            )

            return True

        except Exception as e:
            logger.error(
                "Backup integrity check failed",
                backup_id=metadata.backup_id,
                error=str(e)
            )
            return False

    def get_backup_status(self) -> dict[str, any]:
        """Get current backup status.
        
        Returns:
            Status dictionary
        """
        return {
            "last_full_backup": self.last_full_backup.isoformat() if self.last_full_backup else None,
            "last_incremental_backup": self.last_incremental_backup.isoformat() if self.last_incremental_backup else None,
            "backup_count": len(self.backup_history),
            "scheduler_running": self.scheduler.running if self.scheduler else False,
            "next_full_backup": self._get_next_backup_time("full"),
            "next_incremental_backup": self._get_next_backup_time("incremental")
        }

    def _get_next_backup_time(self, backup_type: str) -> str | None:
        """Get next scheduled backup time.
        
        Args:
            backup_type: Type of backup (full or incremental)
            
        Returns:
            ISO format timestamp or None
        """
        if not self.scheduler or not self.scheduler.running:
            return None

        job_id = f"{backup_type}_backup"
        job = self.scheduler.get_job(job_id)

        if job and job.next_run_time:
            return job.next_run_time.isoformat()

        return None
