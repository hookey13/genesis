"""S3-compatible client for DigitalOcean Spaces."""

import asyncio
import hashlib
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError
from pydantic import BaseModel

from genesis.core.exceptions import BackupError

logger = structlog.get_logger(__name__)


class BackupMetadata(BaseModel):
    """Metadata for backup files."""

    backup_id: str
    timestamp: datetime
    size_bytes: int
    checksum: str
    database_version: str
    backup_type: str  # full, incremental, wal
    retention_policy: str  # hourly, daily, monthly, yearly
    source_path: str
    destination_key: str
    encryption_key_id: str | None = None
    compression_ratio: Decimal | None = None
    encrypted: bool = False
    compressed: bool = False
    database_type: str = "sqlite"  # sqlite or postgresql
    replicated_regions: list[str] = []

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str
        }


class S3Client:
    """S3-compatible client for backup storage with multi-region support."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str | None = None,
        region: str = "us-east-1",
        use_ssl: bool = True,
        replication_regions: list[str] | None = None
    ):
        """Initialize S3 client.
        
        Args:
            endpoint_url: S3-compatible endpoint URL
            access_key: Access key ID
            secret_key: Secret access key
            bucket_name: Bucket name for backups
            region: Primary region name
            use_ssl: Whether to use SSL/TLS
            replication_regions: List of regions for cross-region replication
        """
        # Try to get credentials from Vault if not provided
        if not access_key or not secret_key:
            try:
                from genesis.security.vault_manager import VaultManager
                vault = VaultManager()
                s3_creds = vault.get_s3_credentials()
                access_key = s3_creds.get('access_key', access_key)
                secret_key = s3_creds.get('secret_key', secret_key)
            except Exception:
                # Use environment variables as fallback
                import os
                access_key = access_key or os.getenv('AWS_ACCESS_KEY_ID')
                secret_key = secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')

        self.bucket_name = bucket_name or "genesis-backups-primary"
        self.region = region
        self.replication_regions = replication_regions or ['us-west-2']

        # Initialize primary boto3 client
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            use_ssl=use_ssl
        )

        # Initialize replication clients for other regions
        self.replication_clients = {}
        for rep_region in self.replication_regions:
            self.replication_clients[rep_region] = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=rep_region,
                use_ssl=use_ssl
            )

        self._ensure_bucket_exists()
        self._setup_cross_region_replication()

    def _setup_cross_region_replication(self) -> None:
        """Configure cross-region replication for backup buckets."""
        try:
            # Set up replication configuration
            replication_config = {
                'Role': 'arn:aws:iam::account-id:role/replication-role',
                'Rules': []
            }

            for idx, region in enumerate(self.replication_regions):
                bucket_name = f"{self.bucket_name}-{region}"

                # Ensure replica bucket exists
                try:
                    self.replication_clients[region].create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                    logger.info(f"Created replica bucket in {region}", bucket=bucket_name)
                except ClientError as e:
                    if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                        logger.warning(f"Could not create replica bucket: {e}")

                # Add replication rule
                replication_config['Rules'].append({
                    'ID': f'ReplicateToRegion{idx}',
                    'Priority': idx + 1,
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'backups/'},
                    'Destination': {
                        'Bucket': f'arn:aws:s3:::{bucket_name}',
                        'ReplicationTime': {
                            'Status': 'Enabled',
                            'Time': {'Minutes': 15}
                        },
                        'Metrics': {
                            'Status': 'Enabled',
                            'EventThreshold': {'Minutes': 15}
                        },
                        'StorageClass': 'STANDARD_IA'
                    },
                    'DeleteMarkerReplication': {'Status': 'Enabled'}
                })

            # Apply replication configuration to primary bucket
            if replication_config['Rules']:
                try:
                    self.client.put_bucket_replication(
                        Bucket=self.bucket_name,
                        ReplicationConfiguration=replication_config
                    )
                    logger.info("Cross-region replication configured", regions=self.replication_regions)
                except ClientError as e:
                    logger.warning(f"Could not set up replication: {e}")

        except Exception as e:
            logger.warning(f"Cross-region replication setup failed: {e}")

    def _ensure_bucket_exists(self) -> None:
        """Ensure the backup bucket exists."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info("Backup bucket verified", bucket=self.bucket_name)
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                # Create bucket if it doesn't exist
                self.client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region}
                )
                logger.info("Created backup bucket", bucket=self.bucket_name)
            else:
                raise BackupError(f"Failed to verify bucket: {e}")

    async def upload_backup_with_replication(
        self,
        file_path: Path,
        key_prefix: str,
        metadata: BackupMetadata,
        encryption: str | None = "AES256",
        parallel_upload: bool = True
    ) -> tuple[str, list[str]]:
        """Upload backup file to S3 with cross-region replication.
        
        Args:
            file_path: Local file path to upload
            key_prefix: S3 key prefix
            metadata: Backup metadata
            encryption: Server-side encryption method
            parallel_upload: Whether to upload to all regions in parallel
            
        Returns:
            Tuple of (primary key, list of replica keys)
        """
        # First upload to primary region
        primary_key = await self.upload_backup(file_path, key_prefix, metadata, encryption)

        replica_keys = []

        if parallel_upload and self.replication_regions:
            # Upload to replica regions in parallel
            upload_tasks = []
            for region in self.replication_regions:
                task = self._upload_to_region(
                    file_path, key_prefix, metadata, encryption, region
                )
                upload_tasks.append(task)

            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            for region, result in zip(self.replication_regions, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Failed to replicate to {region}", error=str(result))
                else:
                    replica_keys.append(result)
                    logger.info(f"Replicated backup to {region}", key=result)

        # Update metadata with replicated regions
        metadata.replicated_regions = self.replication_regions

        return primary_key, replica_keys

    async def _upload_to_region(
        self,
        file_path: Path,
        key_prefix: str,
        metadata: BackupMetadata,
        encryption: str,
        region: str
    ) -> str:
        """Upload backup to a specific region.
        
        Args:
            file_path: Local file path
            key_prefix: S3 key prefix
            metadata: Backup metadata
            encryption: Encryption method
            region: Target region
            
        Returns:
            S3 object key
        """
        client = self.replication_clients[region]
        bucket_name = f"{self.bucket_name}-{region}"

        # Generate S3 key
        timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
        key = f"{key_prefix}{timestamp_str}_{metadata.backup_id}.bak"

        # Upload with metadata
        extra_args = {
            "Metadata": {
                "backup-id": metadata.backup_id,
                "timestamp": metadata.timestamp.isoformat(),
                "checksum": metadata.checksum,
                "backup-type": metadata.backup_type,
                "database-version": metadata.database_version,
                "retention-policy": metadata.retention_policy,
                "primary-region": self.region
            }
        }

        if encryption:
            extra_args["ServerSideEncryption"] = encryption

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: client.upload_file(
                str(file_path),
                bucket_name,
                key,
                ExtraArgs=extra_args
            )
        )

        return key

    async def upload_backup(
        self,
        file_path: Path,
        key_prefix: str,
        metadata: BackupMetadata,
        encryption: str | None = "AES256"
    ) -> str:
        """Upload backup file to S3.
        
        Args:
            file_path: Local file path to upload
            key_prefix: S3 key prefix (e.g., "backups/sqlite/")
            metadata: Backup metadata
            encryption: Server-side encryption method
            
        Returns:
            S3 object key
        """
        if not file_path.exists():
            raise BackupError(f"Backup file not found: {file_path}")

        # Generate S3 key
        timestamp_str = metadata.timestamp.strftime("%Y%m%d_%H%M%S")
        key = f"{key_prefix}{timestamp_str}_{metadata.backup_id}.bak"

        # Calculate checksum
        checksum = await self._calculate_checksum(file_path)
        if checksum != metadata.checksum:
            raise BackupError("Checksum mismatch before upload")

        # Upload with metadata
        extra_args = {
            "Metadata": {
                "backup-id": metadata.backup_id,
                "timestamp": metadata.timestamp.isoformat(),
                "checksum": metadata.checksum,
                "backup-type": metadata.backup_type,
                "database-version": metadata.database_version,
                "retention-policy": metadata.retention_policy
            }
        }

        if encryption:
            extra_args["ServerSideEncryption"] = encryption

        try:
            # Run upload in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    key,
                    ExtraArgs=extra_args
                )
            )

            logger.info(
                "Backup uploaded successfully",
                key=key,
                size_mb=metadata.size_bytes / 1024 / 1024,
                checksum=metadata.checksum
            )

            return key

        except ClientError as e:
            logger.error("Failed to upload backup", error=str(e))
            raise BackupError(f"Upload failed: {e}")

    async def download_backup(
        self,
        key: str,
        destination_path: Path,
        verify_checksum: bool = True
    ) -> BackupMetadata:
        """Download backup from S3.
        
        Args:
            key: S3 object key
            destination_path: Local destination path
            verify_checksum: Whether to verify checksum after download
            
        Returns:
            Backup metadata
        """
        try:
            # Get object metadata first
            response = self.client.head_object(Bucket=self.bucket_name, Key=key)
            metadata_dict = response.get("Metadata", {})

            # Download file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.download_file(
                    self.bucket_name,
                    key,
                    str(destination_path)
                )
            )

            # Verify checksum if requested
            if verify_checksum and "checksum" in metadata_dict:
                calculated_checksum = await self._calculate_checksum(destination_path)
                if calculated_checksum != metadata_dict["checksum"]:
                    raise BackupError("Checksum verification failed after download")

            # Reconstruct metadata
            metadata = BackupMetadata(
                backup_id=metadata_dict.get("backup-id", "unknown"),
                timestamp=datetime.fromisoformat(metadata_dict.get("timestamp", datetime.now().isoformat())),
                size_bytes=response["ContentLength"],
                checksum=metadata_dict.get("checksum", ""),
                database_version=metadata_dict.get("database-version", "unknown"),
                backup_type=metadata_dict.get("backup-type", "full"),
                retention_policy=metadata_dict.get("retention-policy", "daily"),
                source_path=str(destination_path),
                destination_key=key
            )

            logger.info(
                "Backup downloaded successfully",
                key=key,
                size_mb=metadata.size_bytes / 1024 / 1024
            )

            return metadata

        except ClientError as e:
            logger.error("Failed to download backup", error=str(e))
            raise BackupError(f"Download failed: {e}")

    async def list_backups(
        self,
        prefix: str | None = None,
        max_keys: int = 1000
    ) -> list[dict[str, Any]]:
        """List backups in S3.
        
        Args:
            prefix: Key prefix to filter by
            max_keys: Maximum number of keys to return
            
        Returns:
            List of backup objects with metadata
        """
        try:
            params = {
                "Bucket": self.bucket_name,
                "MaxKeys": max_keys
            }

            if prefix:
                params["Prefix"] = prefix

            response = self.client.list_objects_v2(**params)

            backups = []
            for obj in response.get("Contents", []):
                # Get full metadata for each object
                head_response = self.client.head_object(
                    Bucket=self.bucket_name,
                    Key=obj["Key"]
                )

                backups.append({
                    "key": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "metadata": head_response.get("Metadata", {})
                })

            return backups

        except ClientError as e:
            logger.error("Failed to list backups", error=str(e))
            raise BackupError(f"List failed: {e}")

    async def delete_backup(self, key: str) -> None:
        """Delete backup from S3.
        
        Args:
            key: S3 object key to delete
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info("Backup deleted", key=key)
        except ClientError as e:
            logger.error("Failed to delete backup", error=str(e))
            raise BackupError(f"Delete failed: {e}")

    async def apply_retention_policy(
        self,
        prefix: str,
        hourly_days: int = 7,
        daily_days: int = 30,
        monthly_days: int = 365
    ) -> dict[str, int]:
        """Apply retention policy to backups.
        
        Args:
            prefix: Key prefix for backups
            hourly_days: Days to keep hourly backups
            daily_days: Days to keep daily backups
            monthly_days: Days to keep monthly backups
            
        Returns:
            Dictionary with counts of deleted backups by policy
        """
        now = datetime.utcnow()
        deleted_counts = {"hourly": 0, "daily": 0, "monthly": 0}

        backups = await self.list_backups(prefix=prefix)

        # Group backups by retention policy
        by_policy: dict[str, list[dict]] = {"hourly": [], "daily": [], "monthly": []}

        for backup in backups:
            policy = backup["metadata"].get("retention-policy", "daily")
            if policy in by_policy:
                by_policy[policy].append(backup)

        # Apply retention for each policy
        retention_days = {
            "hourly": hourly_days,
            "daily": daily_days,
            "monthly": monthly_days
        }

        for policy, days in retention_days.items():
            for backup in by_policy[policy]:
                age_days = (now - backup["last_modified"].replace(tzinfo=None)).days
                if age_days > days:
                    await self.delete_backup(backup["key"])
                    deleted_counts[policy] += 1

        if sum(deleted_counts.values()) > 0:
            logger.info("Retention policy applied", deleted=deleted_counts)

        return deleted_counts

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal checksum string
        """
        sha256_hash = hashlib.sha256()

        # Read in chunks to handle large files
        loop = asyncio.get_event_loop()

        def calculate():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        return await loop.run_in_executor(None, calculate)

    def get_backup_url(self, key: str, expiry_seconds: int = 3600) -> str:
        """Generate presigned URL for backup download.
        
        Args:
            key: S3 object key
            expiry_seconds: URL expiry time in seconds
            
        Returns:
            Presigned URL
        """
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiry_seconds
            )
            return url
        except ClientError as e:
            logger.error("Failed to generate presigned URL", error=str(e))
            raise BackupError(f"URL generation failed: {e}")
