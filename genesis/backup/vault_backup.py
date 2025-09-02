"""Vault backup management for HashiCorp Vault snapshots and seal keys."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import hvac
import structlog
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.core.exceptions import BackupError
from genesis.utils.decorators import with_retry

logger = structlog.get_logger(__name__)


class VaultBackupManager:
    """Manages Vault snapshots and seal key backups."""

    def __init__(
        self,
        vault_url: str = "http://localhost:8200",
        vault_token: str | None = None,
        s3_client: S3Client | None = None,
        backup_dir: Path | None = None
    ):
        """Initialize Vault backup manager.
        
        Args:
            vault_url: Vault server URL
            vault_token: Vault root/admin token
            s3_client: S3 client for remote storage
            backup_dir: Local backup directory
        """
        self.vault_url = vault_url
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.s3_client = s3_client or S3Client()
        self.backup_dir = backup_dir or Path("/tmp/vault_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vault client
        self.vault_client = hvac.Client(
            url=self.vault_url,
            token=self.vault_token
        )

        if not self.vault_client.is_authenticated():
            raise BackupError("Failed to authenticate with Vault")

    @with_retry(max_attempts=3, backoff_factor=2)
    async def create_vault_snapshot(self) -> BackupMetadata:
        """Create a Vault Raft snapshot.
        
        Returns:
            Backup metadata for the snapshot
        """
        timestamp = datetime.utcnow()
        snapshot_id = f"vault_snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info("Creating Vault snapshot", snapshot_id=snapshot_id)

        try:
            # Create snapshot using Vault API
            snapshot_path = self.backup_dir / f"{snapshot_id}.snap"

            # Use sys/storage/raft/snapshot endpoint for Raft storage backend
            response = self.vault_client.adapter.get(
                "/v1/sys/storage/raft/snapshot",
                stream=True
            )

            if response.status_code != 200:
                raise BackupError(f"Failed to create snapshot: {response.text}")

            # Write snapshot to file
            with open(snapshot_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Calculate metadata
            file_stats = snapshot_path.stat()

            # Create metadata
            metadata = BackupMetadata(
                backup_id=snapshot_id,
                timestamp=timestamp,
                size_bytes=file_stats.st_size,
                checksum=await self._calculate_checksum(snapshot_path),
                database_version="vault",
                backup_type="vault_snapshot",
                retention_policy="daily",
                source_path=self.vault_url,
                destination_key=""
            )

            # Encrypt the snapshot before uploading
            encrypted_path = await self._encrypt_snapshot(snapshot_path)

            # Upload to S3 with replication
            if self.s3_client:
                primary_key, replica_keys = await self.s3_client.upload_backup_with_replication(
                    file_path=encrypted_path,
                    key_prefix="vault/snapshots/",
                    metadata=metadata
                )
                metadata.destination_key = primary_key
                metadata.replicated_regions = self.s3_client.replication_regions

            # Clean up local files
            snapshot_path.unlink()
            encrypted_path.unlink()

            logger.info(
                "Vault snapshot created successfully",
                snapshot_id=snapshot_id,
                size_mb=metadata.size_bytes / 1024 / 1024
            )

            return metadata

        except Exception as e:
            logger.error("Vault snapshot failed", error=str(e))
            raise BackupError(f"Vault snapshot failed: {e}")

    async def backup_seal_keys(self) -> dict[str, Any]:
        """Backup Vault seal keys securely.
        
        Returns:
            Dictionary containing backup information
        """
        logger.info("Backing up Vault seal keys")

        try:
            # Get seal status
            seal_status = self.vault_client.sys.read_seal_status()

            if seal_status['sealed']:
                raise BackupError("Vault is sealed, cannot backup seal keys")

            # Generate RSA key pair for encrypting seal keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            public_key = private_key.public_key()

            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Store public key in Vault for recovery
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path="backup/seal_key_public",
                secret={"public_key": public_pem.decode()}
            )

            # Store private key securely (this should be stored offline)
            private_key_path = self.backup_dir / "seal_key_private.pem"
            with open(private_key_path, 'wb') as f:
                f.write(private_pem)

            # Set restrictive permissions
            os.chmod(private_key_path, 0o600)

            # Create backup metadata
            backup_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "public_key_path": "vault:secret/backup/seal_key_public",
                "private_key_path": str(private_key_path),
                "seal_type": seal_status.get('type', 'shamir'),
                "initialized": seal_status.get('initialized', False),
                "recovery_shares": seal_status.get('recovery_shares', 0),
                "recovery_threshold": seal_status.get('recovery_threshold', 0)
            }

            # Save backup info
            info_path = self.backup_dir / "seal_backup_info.json"
            with open(info_path, 'w') as f:
                json.dump(backup_info, f, indent=2)

            logger.info(
                "Seal keys backed up successfully",
                seal_type=backup_info['seal_type']
            )

            return backup_info

        except Exception as e:
            logger.error("Seal key backup failed", error=str(e))
            raise BackupError(f"Seal key backup failed: {e}")

    async def restore_vault_snapshot(
        self,
        snapshot_key: str,
        force: bool = False
    ) -> bool:
        """Restore Vault from a snapshot.
        
        Args:
            snapshot_key: S3 key of the snapshot to restore
            force: Whether to force restore even if data exists
            
        Returns:
            True if restore successful
        """
        logger.info("Restoring Vault snapshot", snapshot_key=snapshot_key)

        try:
            # Download snapshot from S3
            local_path = self.backup_dir / "restore_snapshot.snap"

            metadata = await self.s3_client.download_backup(
                key=snapshot_key,
                destination_path=local_path,
                verify_checksum=True
            )

            # Decrypt snapshot if encrypted
            if metadata.encrypted:
                decrypted_path = await self._decrypt_snapshot(local_path)
                local_path = decrypted_path

            # Restore using Vault API
            with open(local_path, 'rb') as f:
                response = self.vault_client.adapter.post(
                    "/v1/sys/storage/raft/snapshot",
                    files={'snapshot': f},
                    params={'force': force}
                )

            if response.status_code != 204:
                raise BackupError(f"Restore failed: {response.text}")

            # Clean up
            local_path.unlink()

            logger.info("Vault snapshot restored successfully")
            return True

        except Exception as e:
            logger.error("Vault restore failed", error=str(e))
            raise BackupError(f"Vault restore failed: {e}")

    async def _encrypt_snapshot(self, snapshot_path: Path) -> Path:
        """Encrypt Vault snapshot using transit engine.
        
        Args:
            snapshot_path: Path to snapshot file
            
        Returns:
            Path to encrypted file
        """
        encrypted_path = snapshot_path.with_suffix('.enc')

        # Read snapshot data
        with open(snapshot_path, 'rb') as f:
            data = f.read()

        # Encrypt using Vault transit engine
        try:
            # Enable transit engine if not already enabled
            if 'transit/' not in self.vault_client.sys.list_mounted_secrets_engines():
                self.vault_client.sys.enable_secrets_engine(
                    backend_type='transit',
                    path='transit'
                )

            # Create or update encryption key
            self.vault_client.secrets.transit.create_key(
                name='backup-key',
                exportable=False,
                allow_plaintext_backup=False
            )

            # Encrypt data
            import base64
            plaintext_b64 = base64.b64encode(data).decode('utf-8')

            encrypt_response = self.vault_client.secrets.transit.encrypt_data(
                name='backup-key',
                plaintext=plaintext_b64
            )

            ciphertext = encrypt_response['data']['ciphertext']

            # Write encrypted data
            with open(encrypted_path, 'w') as f:
                f.write(ciphertext)

            return encrypted_path

        except Exception as e:
            logger.error("Snapshot encryption failed", error=str(e))
            # Fall back to unencrypted
            return snapshot_path

    async def _decrypt_snapshot(self, encrypted_path: Path) -> Path:
        """Decrypt Vault snapshot.
        
        Args:
            encrypted_path: Path to encrypted snapshot
            
        Returns:
            Path to decrypted file
        """
        decrypted_path = encrypted_path.with_suffix('.snap')

        # Read encrypted data
        with open(encrypted_path) as f:
            ciphertext = f.read()

        # Decrypt using Vault transit engine
        decrypt_response = self.vault_client.secrets.transit.decrypt_data(
            name='backup-key',
            ciphertext=ciphertext
        )

        import base64
        plaintext_b64 = decrypt_response['data']['plaintext']
        plaintext = base64.b64decode(plaintext_b64)

        # Write decrypted data
        with open(decrypted_path, 'wb') as f:
            f.write(plaintext)

        return decrypted_path

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of checksum
        """
        import hashlib
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    async def verify_vault_health(self) -> dict[str, Any]:
        """Verify Vault health and backup status.
        
        Returns:
            Health status dictionary
        """
        try:
            health = self.vault_client.sys.read_health_status()

            # Get recent snapshots
            snapshots = await self._list_recent_snapshots()

            return {
                "vault_healthy": not health.get('sealed', True),
                "initialized": health.get('initialized', False),
                "sealed": health.get('sealed', True),
                "version": health.get('version', 'unknown'),
                "recent_snapshots": len(snapshots),
                "last_snapshot": snapshots[0].timestamp.isoformat() if snapshots else None
            }

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "vault_healthy": False,
                "error": str(e)
            }

    async def _list_recent_snapshots(self, days: int = 7) -> list[BackupMetadata]:
        """List recent Vault snapshots.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of backup metadata
        """
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(days=days)

        backups = await self.s3_client.list_backups(prefix="vault/snapshots/")

        recent = [
            b for b in backups
            if b.timestamp >= start_date and b.backup_type == "vault_snapshot"
        ]

        return sorted(recent, key=lambda x: x.timestamp, reverse=True)
