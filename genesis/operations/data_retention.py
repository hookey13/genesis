"""Data retention and archival system for compliance."""
import gzip
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class RetentionPeriod(Enum):
    """Standard retention periods for different data types."""

    TRADE_DATA = 365 * 5  # 5 years
    AUDIT_LOGS = 365 * 7  # 7 years
    CUSTOMER_DATA = 365 * 5  # 5 years after account closure
    COMPLIANCE_REPORTS = 365 * 7  # 7 years
    SYSTEM_LOGS = 90  # 90 days
    TEMP_FILES = 7  # 7 days
    BACKUPS = 365  # 1 year


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    data_type: str
    retention_days: int
    archive_enabled: bool = True
    compression_enabled: bool = True
    encryption_enabled: bool = False
    purge_enabled: bool = True

    def is_expired(self, file_date: datetime) -> bool:
        """Check if data has exceeded retention period."""
        age_days = (datetime.now() - file_date).days
        return age_days > self.retention_days


@dataclass
class ArchivalResult:
    """Result of archival operation."""

    files_processed: int
    files_archived: int
    files_purged: int
    total_size_archived: int
    total_size_purged: int
    errors: list[str]
    timestamp: datetime


class DataRetentionManager:
    """Manages data retention, archival, and purging."""

    def __init__(
        self,
        archive_dir: str = ".genesis/archive",
        policies: dict[str, RetentionPolicy] | None = None
    ):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Default retention policies
        self.policies = policies or {
            "trades": RetentionPolicy(
                data_type="trades",
                retention_days=RetentionPeriod.TRADE_DATA.value
            ),
            "audit": RetentionPolicy(
                data_type="audit",
                retention_days=RetentionPeriod.AUDIT_LOGS.value,
                purge_enabled=False  # Never purge audit logs
            ),
            "customer": RetentionPolicy(
                data_type="customer",
                retention_days=RetentionPeriod.CUSTOMER_DATA.value
            ),
            "compliance": RetentionPolicy(
                data_type="compliance",
                retention_days=RetentionPeriod.COMPLIANCE_REPORTS.value
            ),
            "system": RetentionPolicy(
                data_type="system",
                retention_days=RetentionPeriod.SYSTEM_LOGS.value
            ),
            "temp": RetentionPolicy(
                data_type="temp",
                retention_days=RetentionPeriod.TEMP_FILES.value,
                archive_enabled=False  # Don't archive temp files
            )
        }

    def apply_retention_policy(
        self,
        data_dir: str,
        data_type: str,
        dry_run: bool = False
    ) -> ArchivalResult:
        """Apply retention policy to a data directory."""
        if data_type not in self.policies:
            raise ValueError(f"Unknown data type: {data_type}")

        policy = self.policies[data_type]
        data_path = Path(data_dir)

        if not data_path.exists():
            self.logger.warning(
                "data_directory_not_found",
                directory=str(data_path)
            )
            return ArchivalResult(
                files_processed=0,
                files_archived=0,
                files_purged=0,
                total_size_archived=0,
                total_size_purged=0,
                errors=[f"Directory not found: {data_path}"],
                timestamp=datetime.now()
            )

        result = ArchivalResult(
            files_processed=0,
            files_archived=0,
            files_purged=0,
            total_size_archived=0,
            total_size_purged=0,
            errors=[],
            timestamp=datetime.now()
        )

        # Process files
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                result.files_processed += 1

                try:
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)

                    if policy.is_expired(file_date):
                        file_size = file_path.stat().st_size

                        if not dry_run:
                            # Archive if enabled
                            if policy.archive_enabled:
                                archived_path = self._archive_file(
                                    file_path,
                                    data_type,
                                    policy.compression_enabled
                                )
                                if archived_path:
                                    result.files_archived += 1
                                    result.total_size_archived += file_size

                                    self.logger.info(
                                        "file_archived",
                                        original=str(file_path),
                                        archive=str(archived_path),
                                        size=file_size
                                    )

                            # Purge if enabled
                            if policy.purge_enabled:
                                file_path.unlink()
                                result.files_purged += 1
                                result.total_size_purged += file_size

                                self.logger.info(
                                    "file_purged",
                                    path=str(file_path),
                                    age_days=(datetime.now() - file_date).days
                                )
                        else:
                            # Dry run - just log what would happen
                            self.logger.info(
                                "dry_run_would_process",
                                file=str(file_path),
                                action="archive_and_purge" if policy.purge_enabled else "archive_only",
                                age_days=(datetime.now() - file_date).days
                            )

                            if policy.archive_enabled:
                                result.files_archived += 1
                                result.total_size_archived += file_size
                            if policy.purge_enabled:
                                result.files_purged += 1
                                result.total_size_purged += file_size

                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e!s}"
                    result.errors.append(error_msg)
                    self.logger.error(
                        "file_processing_error",
                        file=str(file_path),
                        error=str(e)
                    )

        # Log summary
        self.logger.info(
            "retention_policy_applied",
            data_type=data_type,
            dry_run=dry_run,
            files_processed=result.files_processed,
            files_archived=result.files_archived,
            files_purged=result.files_purged,
            errors=len(result.errors)
        )

        # Create audit log entry
        self._create_audit_log(data_type, result, dry_run)

        return result

    def _archive_file(
        self,
        file_path: Path,
        data_type: str,
        compress: bool = True
    ) -> Path | None:
        """Archive a file to the archive directory."""
        try:
            # Create archive subdirectory structure
            archive_date = datetime.now()
            archive_subdir = self.archive_dir / data_type / f"{archive_date.year}/{archive_date.month:02d}"
            archive_subdir.mkdir(parents=True, exist_ok=True)

            # Generate archive filename
            archive_name = f"{file_path.stem}_{archive_date.strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"

            if compress:
                archive_name += ".gz"
                archive_path = archive_subdir / archive_name

                # Compress and archive
                with open(file_path, 'rb') as f_in:
                    with gzip.open(archive_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                archive_path = archive_subdir / archive_name
                shutil.copy2(file_path, archive_path)

            return archive_path

        except Exception as e:
            self.logger.error(
                "archival_failed",
                file=str(file_path),
                error=str(e)
            )
            return None

    def _create_audit_log(
        self,
        data_type: str,
        result: ArchivalResult,
        dry_run: bool
    ) -> None:
        """Create audit log entry for retention operation."""
        audit_entry = {
            "timestamp": result.timestamp.isoformat(),
            "operation": "data_retention",
            "data_type": data_type,
            "dry_run": dry_run,
            "files_processed": result.files_processed,
            "files_archived": result.files_archived,
            "files_purged": result.files_purged,
            "total_size_archived": result.total_size_archived,
            "total_size_purged": result.total_size_purged,
            "errors": result.errors
        }

        # Write to audit log
        audit_dir = Path(".genesis/logs/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)

        audit_file = audit_dir / f"retention_{result.timestamp.strftime('%Y%m%d')}.jsonl"

        with open(audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

    def restore_from_archive(
        self,
        archive_path: str,
        restore_dir: str
    ) -> bool:
        """Restore a file from archive."""
        try:
            archive_file = Path(archive_path)
            restore_path = Path(restore_dir)

            if not archive_file.exists():
                self.logger.error(
                    "archive_file_not_found",
                    path=str(archive_file)
                )
                return False

            restore_path.mkdir(parents=True, exist_ok=True)

            # Determine if file is compressed
            if archive_file.suffix == '.gz':
                # Decompress and restore
                restore_file = restore_path / archive_file.stem

                with gzip.open(archive_file, 'rb') as f_in:
                    with open(restore_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Simple copy
                restore_file = restore_path / archive_file.name
                shutil.copy2(archive_file, restore_file)

            self.logger.info(
                "file_restored",
                archive=str(archive_file),
                restored_to=str(restore_file)
            )

            return True

        except Exception as e:
            self.logger.error(
                "restore_failed",
                archive=str(archive_path),
                error=str(e)
            )
            return False

    def get_retention_status(
        self,
        data_dir: str,
        data_type: str
    ) -> dict[str, Any]:
        """Get retention status for a data directory."""
        if data_type not in self.policies:
            raise ValueError(f"Unknown data type: {data_type}")

        policy = self.policies[data_type]
        data_path = Path(data_dir)

        if not data_path.exists():
            return {
                "error": f"Directory not found: {data_path}"
            }

        total_files = 0
        expired_files = 0
        total_size = 0
        expired_size = 0
        oldest_file = None
        newest_file = None

        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size

                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)

                if policy.is_expired(file_date):
                    expired_files += 1
                    expired_size += file_size

                if oldest_file is None or file_date < oldest_file:
                    oldest_file = file_date

                if newest_file is None or file_date > newest_file:
                    newest_file = file_date

        return {
            "data_type": data_type,
            "retention_days": policy.retention_days,
            "total_files": total_files,
            "expired_files": expired_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expired_size_mb": round(expired_size / (1024 * 1024), 2),
            "oldest_file": oldest_file.isoformat() if oldest_file else None,
            "newest_file": newest_file.isoformat() if newest_file else None,
            "archive_enabled": policy.archive_enabled,
            "purge_enabled": policy.purge_enabled
        }

    def schedule_retention_tasks(self) -> list[dict[str, Any]]:
        """Get list of scheduled retention tasks."""
        tasks = []

        data_directories = {
            "trades": ".genesis/data/trades",
            "audit": ".genesis/logs/audit",
            "system": ".genesis/logs",
            "temp": "/tmp/genesis"
        }

        for data_type, directory in data_directories.items():
            if data_type in self.policies:
                status = self.get_retention_status(directory, data_type)

                if "error" not in status and status["expired_files"] > 0:
                    tasks.append({
                        "data_type": data_type,
                        "directory": directory,
                        "expired_files": status["expired_files"],
                        "expired_size_mb": status["expired_size_mb"],
                        "action": "archive_and_purge" if status["purge_enabled"] else "archive_only"
                    })

        return tasks
