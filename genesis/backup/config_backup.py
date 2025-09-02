"""Application configuration backup management."""

import hashlib
import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from deepdiff import DeepDiff

from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.core.exceptions import BackupError
from genesis.utils.decorators import with_retry

logger = structlog.get_logger(__name__)


class ConfigBackupManager:
    """Manages application configuration backups with versioning."""

    def __init__(
        self,
        config_dirs: list[Path] | None = None,
        s3_client: S3Client | None = None,
        backup_dir: Path | None = None
    ):
        """Initialize configuration backup manager.
        
        Args:
            config_dirs: List of configuration directories to backup
            s3_client: S3 client for remote storage
            backup_dir: Local backup directory
        """
        self.config_dirs = config_dirs or [
            Path("config"),
            Path(".env"),
            Path("alembic.ini"),
            Path("requirements")
        ]
        self.s3_client = s3_client or S3Client()
        self.backup_dir = backup_dir or Path("/tmp/config_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Version tracking
        self.version_history: list[dict[str, Any]] = []
        self.current_version: str | None = None

    @with_retry(max_attempts=3, backoff_factor=2)
    async def create_config_backup(self) -> BackupMetadata:
        """Create a versioned backup of all configuration files.
        
        Returns:
            Backup metadata
        """
        timestamp = datetime.utcnow()
        backup_id = f"config_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info("Creating configuration backup", backup_id=backup_id)

        try:
            # Create tarball of all config files
            tarball_path = self.backup_dir / f"{backup_id}.tar.gz"

            with tarfile.open(tarball_path, "w:gz") as tar:
                for config_path in self.config_dirs:
                    if config_path.exists():
                        # Add file or directory to archive
                        tar.add(config_path, arcname=config_path.name)
                        logger.debug(f"Added {config_path} to backup")

                # Add version manifest
                manifest = await self._create_version_manifest()
                manifest_path = self.backup_dir / "version_manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                tar.add(manifest_path, arcname="version_manifest.json")
                manifest_path.unlink()

            # Calculate metadata
            file_stats = tarball_path.stat()
            checksum = await self._calculate_checksum(tarball_path)

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                size_bytes=file_stats.st_size,
                checksum=checksum,
                database_version=manifest['version'],
                backup_type="configuration",
                retention_policy="daily",
                source_path="config",
                destination_key=""
            )

            # Upload to S3 with replication
            if self.s3_client:
                primary_key, replica_keys = await self.s3_client.upload_backup_with_replication(
                    file_path=tarball_path,
                    key_prefix="config/",
                    metadata=metadata
                )
                metadata.destination_key = primary_key
                metadata.replicated_regions = self.s3_client.replication_regions

            # Track version
            self.version_history.append({
                "version": manifest['version'],
                "timestamp": timestamp.isoformat(),
                "backup_id": backup_id,
                "checksum": checksum,
                "files": manifest['files']
            })
            self.current_version = manifest['version']

            # Clean up local file
            tarball_path.unlink()

            logger.info(
                "Configuration backup created",
                backup_id=backup_id,
                version=manifest['version'],
                file_count=len(manifest['files'])
            )

            return metadata

        except Exception as e:
            logger.error("Configuration backup failed", error=str(e))
            raise BackupError(f"Configuration backup failed: {e}")

    async def _create_version_manifest(self) -> dict[str, Any]:
        """Create a manifest of all configuration files with checksums.
        
        Returns:
            Version manifest dictionary
        """
        manifest = {
            "version": await self._calculate_version(),
            "timestamp": datetime.utcnow().isoformat(),
            "files": {}
        }

        for config_path in self.config_dirs:
            if config_path.exists():
                if config_path.is_file():
                    # Single file
                    checksum = await self._calculate_checksum(config_path)
                    manifest['files'][str(config_path)] = {
                        "checksum": checksum,
                        "size": config_path.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            config_path.stat().st_mtime
                        ).isoformat()
                    }
                elif config_path.is_dir():
                    # Directory - recursively add all files
                    for file_path in config_path.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            checksum = await self._calculate_checksum(file_path)
                            relative_path = file_path.relative_to(config_path.parent)
                            manifest['files'][str(relative_path)] = {
                                "checksum": checksum,
                                "size": file_path.stat().st_size,
                                "modified": datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat()
                            }

        return manifest

    async def _calculate_version(self) -> str:
        """Calculate version hash based on all config file contents.
        
        Returns:
            Version hash string
        """
        hasher = hashlib.sha256()

        for config_path in sorted(self.config_dirs):
            if config_path.exists():
                if config_path.is_file():
                    with open(config_path, 'rb') as f:
                        hasher.update(f.read())
                elif config_path.is_dir():
                    for file_path in sorted(config_path.rglob("*")):
                        if file_path.is_file():
                            with open(file_path, 'rb') as f:
                                hasher.update(f.read())

        return hasher.hexdigest()[:12]  # Use first 12 chars of hash

    async def compare_configs(
        self,
        backup_id1: str,
        backup_id2: str
    ) -> dict[str, Any]:
        """Compare two configuration backups and show differences.
        
        Args:
            backup_id1: First backup ID
            backup_id2: Second backup ID
            
        Returns:
            Dictionary containing differences
        """
        logger.info(f"Comparing configs {backup_id1} and {backup_id2}")

        try:
            # Download both backups
            temp_dir = Path(tempfile.mkdtemp())

            path1 = temp_dir / f"{backup_id1}.tar.gz"
            path2 = temp_dir / f"{backup_id2}.tar.gz"

            # Download from S3
            await self.s3_client.download_backup(
                key=f"config/{backup_id1}.tar.gz",
                destination_path=path1
            )
            await self.s3_client.download_backup(
                key=f"config/{backup_id2}.tar.gz",
                destination_path=path2
            )

            # Extract and load manifests
            manifest1 = await self._extract_manifest(path1)
            manifest2 = await self._extract_manifest(path2)

            # Compare using DeepDiff
            diff = DeepDiff(manifest1, manifest2, ignore_order=True)

            # Clean up
            path1.unlink()
            path2.unlink()
            temp_dir.rmdir()

            return {
                "backup_id1": backup_id1,
                "backup_id2": backup_id2,
                "version1": manifest1.get('version'),
                "version2": manifest2.get('version'),
                "differences": diff.to_dict() if diff else {},
                "files_added": list(diff.get('dictionary_item_added', [])),
                "files_removed": list(diff.get('dictionary_item_removed', [])),
                "files_changed": list(diff.get('values_changed', {}).keys())
            }

        except Exception as e:
            logger.error("Config comparison failed", error=str(e))
            raise BackupError(f"Config comparison failed: {e}")

    async def _extract_manifest(self, tarball_path: Path) -> dict[str, Any]:
        """Extract version manifest from a config backup.
        
        Args:
            tarball_path: Path to tarball
            
        Returns:
            Manifest dictionary
        """
        with tarfile.open(tarball_path, "r:gz") as tar:
            manifest_member = tar.getmember("version_manifest.json")
            manifest_file = tar.extractfile(manifest_member)
            if manifest_file:
                manifest = json.load(manifest_file)
                return manifest

        return {}

    async def restore_config(
        self,
        backup_id: str,
        target_dir: Path | None = None,
        dry_run: bool = False
    ) -> dict[str, Any]:
        """Restore configuration from a backup.
        
        Args:
            backup_id: Backup ID to restore
            target_dir: Target directory (defaults to original locations)
            dry_run: If True, only show what would be restored
            
        Returns:
            Restoration summary
        """
        logger.info(f"Restoring configuration from {backup_id}", dry_run=dry_run)

        try:
            # Download backup
            temp_path = self.backup_dir / f"restore_{backup_id}.tar.gz"

            metadata = await self.s3_client.download_backup(
                key=f"config/{backup_id}.tar.gz",
                destination_path=temp_path,
                verify_checksum=True
            )

            # Extract files
            restored_files = []

            with tarfile.open(temp_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name != "version_manifest.json":
                        if dry_run:
                            restored_files.append(member.name)
                        else:
                            if target_dir:
                                member.name = str(target_dir / member.name)
                            tar.extract(member)
                            restored_files.append(member.name)
                            logger.debug(f"Restored {member.name}")

            # Clean up
            temp_path.unlink()

            result = {
                "backup_id": backup_id,
                "timestamp": metadata.timestamp.isoformat(),
                "files_restored": len(restored_files),
                "files": restored_files,
                "dry_run": dry_run
            }

            if not dry_run:
                logger.info(
                    "Configuration restored successfully",
                    files_count=len(restored_files)
                )

            return result

        except Exception as e:
            logger.error("Configuration restore failed", error=str(e))
            raise BackupError(f"Configuration restore failed: {e}")

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of checksum
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    async def list_config_backups(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """List available configuration backups.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of backup information
        """
        backups = await self.s3_client.list_backups(prefix="config/")

        result = []
        for backup in backups:
            if backup.backup_type == "configuration":
                # Apply date filters
                if start_date and backup.timestamp < start_date:
                    continue
                if end_date and backup.timestamp > end_date:
                    continue

                result.append({
                    "backup_id": backup.backup_id,
                    "timestamp": backup.timestamp.isoformat(),
                    "version": backup.database_version,
                    "size_bytes": backup.size_bytes,
                    "checksum": backup.checksum
                })

        return sorted(result, key=lambda x: x['timestamp'], reverse=True)

    def get_version_history(self) -> list[dict[str, Any]]:
        """Get configuration version history.
        
        Returns:
            List of version history entries
        """
        return self.version_history
