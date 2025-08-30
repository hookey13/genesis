"""
Log Archival System for Project GENESIS.

Provides automated log rotation, compression, and archival to S3/DigitalOcean Spaces
with configurable retention policies and searchable metadata.
"""

import asyncio
import gzip
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from genesis.utils.logger import get_logger, LoggerType


class LogArchiveMetadata(BaseModel):
    """Metadata for archived log files."""
    
    archive_id: str = Field(description="Unique archive identifier")
    original_filename: str = Field(description="Original log filename")
    archive_timestamp: datetime = Field(description="When archived")
    start_timestamp: Optional[datetime] = Field(None, description="First log entry timestamp")
    end_timestamp: Optional[datetime] = Field(None, description="Last log entry timestamp")
    original_size: int = Field(description="Original file size in bytes")
    compressed_size: int = Field(description="Compressed size in bytes")
    compression_ratio: float = Field(description="Compression ratio achieved")
    checksum: str = Field(description="SHA256 checksum of original file")
    log_type: str = Field(description="Type of log (trading, audit, tilt, system)")
    entry_count: int = Field(0, description="Number of log entries")
    tier_level: Optional[str] = Field(None, description="Tier level when archived")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


class LogArchivalConfig(BaseModel):
    """Configuration for log archival system."""
    
    local_retention_days: int = Field(30, description="Days to keep logs locally")
    archive_retention_days: int = Field(365, description="Days to keep in archive")
    compression_level: int = Field(9, description="Gzip compression level (1-9)")
    batch_size: int = Field(10, description="Number of files to archive in batch")
    s3_bucket: str = Field(description="S3/Spaces bucket name")
    s3_prefix: str = Field("logs/", description="S3 key prefix for archives")
    archive_on_rotation: bool = Field(True, description="Archive when rotating")
    min_file_size_bytes: int = Field(1024, description="Minimum size to archive")
    metadata_index_enabled: bool = Field(True, description="Enable metadata indexing")


class LogArchiver:
    """
    Handles log archival to S3/DigitalOcean Spaces with metadata tracking.
    
    Features:
        - Automatic archival on rotation
        - Compression with configurable levels
        - Metadata extraction and indexing
        - Retention policy enforcement
        - Archive retrieval and search
    """
    
    def __init__(
        self,
        config: LogArchivalConfig,
        s3_client: Optional[Any] = None,
        log_dir: Path = Path(".genesis/logs"),
        archive_dir: Path = Path(".genesis/archives")
    ):
        """
        Initialize log archiver.
        
        Args:
            config: Archival configuration
            s3_client: Optional S3 client (for testing)
            log_dir: Directory containing logs
            archive_dir: Local archive staging directory
        """
        self.config = config
        self.log_dir = log_dir
        self.archive_dir = archive_dir
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
        
        # Create directories
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = s3_client or self._create_s3_client()
        
        # Metadata index
        self.metadata_index: Dict[str, LogArchiveMetadata] = {}
        self._load_metadata_index()
    
    def _create_s3_client(self) -> Any:
        """Create S3 client for DigitalOcean Spaces."""
        import os
        
        return boto3.client(
            's3',
            endpoint_url=os.getenv('SPACES_ENDPOINT', 'https://sgp1.digitaloceanspaces.com'),
            aws_access_key_id=os.getenv('SPACES_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SPACES_SECRET_KEY')
        )
    
    def _load_metadata_index(self) -> None:
        """Load metadata index from local cache."""
        index_file = self.archive_dir / "metadata_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    for archive_id, metadata in data.items():
                        self.metadata_index[archive_id] = LogArchiveMetadata(**metadata)
            except Exception as e:
                self.logger.error("failed_to_load_metadata_index", error=str(e))
    
    def _save_metadata_index(self) -> None:
        """Save metadata index to local cache."""
        index_file = self.archive_dir / "metadata_index.json"
        try:
            data = {
                aid: metadata.dict() for aid, metadata in self.metadata_index.items()
            }
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error("failed_to_save_metadata_index", error=str(e))
    
    async def archive_rotated_logs(self) -> List[str]:
        """
        Archive rotated log files (*.log.1, *.log.2, etc).
        
        Returns:
            List of archive IDs for archived files
        """
        archived = []
        
        # Find rotated log files
        rotated_files = []
        for log_file in self.log_dir.glob("*.log.*"):
            if log_file.suffix.isdigit() or log_file.suffix == '.gz':
                rotated_files.append(log_file)
        
        self.logger.info(
            "found_rotated_logs",
            count=len(rotated_files),
            files=[str(f.name) for f in rotated_files]
        )
        
        # Process in batches
        for i in range(0, len(rotated_files), self.config.batch_size):
            batch = rotated_files[i:i + self.config.batch_size]
            batch_archives = await self._archive_batch(batch)
            archived.extend(batch_archives)
        
        return archived
    
    async def _archive_batch(self, files: List[Path]) -> List[str]:
        """Archive a batch of files."""
        archives = []
        
        for file_path in files:
            try:
                # Skip if too small
                if file_path.stat().st_size < self.config.min_file_size_bytes:
                    continue
                
                archive_id = await self._archive_file(file_path)
                if archive_id:
                    archives.append(archive_id)
                    
            except Exception as e:
                self.logger.error(
                    "failed_to_archive_file",
                    file=str(file_path),
                    error=str(e)
                )
        
        return archives
    
    async def _archive_file(self, file_path: Path) -> Optional[str]:
        """
        Archive a single log file.
        
        Args:
            file_path: Path to log file
            
        Returns:
            Archive ID if successful, None otherwise
        """
        try:
            # Generate archive ID
            archive_id = self._generate_archive_id(file_path)
            
            # Extract metadata
            metadata = await self._extract_metadata(file_path)
            metadata.archive_id = archive_id
            
            # Compress file
            compressed_path = await self._compress_file(file_path, archive_id)
            metadata.compressed_size = compressed_path.stat().st_size
            metadata.compression_ratio = metadata.original_size / metadata.compressed_size
            
            # Upload to S3/Spaces
            s3_key = f"{self.config.s3_prefix}{archive_id}.tar.gz"
            await self._upload_to_s3(compressed_path, s3_key, metadata)
            
            # Update metadata index
            self.metadata_index[archive_id] = metadata
            self._save_metadata_index()
            
            # Clean up local files
            compressed_path.unlink()
            if self.config.archive_on_rotation:
                file_path.unlink()
            
            self.logger.info(
                "file_archived",
                archive_id=archive_id,
                original_file=str(file_path.name),
                compression_ratio=round(metadata.compression_ratio, 2),
                s3_key=s3_key
            )
            
            return archive_id
            
        except Exception as e:
            self.logger.error(
                "archive_failed",
                file=str(file_path),
                error=str(e)
            )
            return None
    
    def _generate_archive_id(self, file_path: Path) -> str:
        """Generate unique archive ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_path.stem}_{timestamp}_{file_hash}"
    
    async def _extract_metadata(self, file_path: Path) -> LogArchiveMetadata:
        """Extract metadata from log file."""
        metadata = LogArchiveMetadata(
            archive_id="",  # Set later
            original_filename=file_path.name,
            archive_timestamp=datetime.utcnow(),
            original_size=file_path.stat().st_size,
            compressed_size=0,  # Set after compression
            compression_ratio=0,  # Set after compression
            checksum=self._calculate_checksum(file_path),
            log_type=self._determine_log_type(file_path)
        )
        
        # Try to extract timestamps and count entries
        try:
            if file_path.suffix != '.gz':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    metadata.entry_count = len(lines)
                    
                    # Extract first and last timestamps
                    if lines:
                        metadata.start_timestamp = self._extract_timestamp(lines[0])
                        metadata.end_timestamp = self._extract_timestamp(lines[-1])
            else:
                # Handle gzipped files
                with gzip.open(file_path, 'rt') as f:
                    lines = f.readlines()
                    metadata.entry_count = len(lines)
                    if lines:
                        metadata.start_timestamp = self._extract_timestamp(lines[0])
                        metadata.end_timestamp = self._extract_timestamp(lines[-1])
                        
        except Exception as e:
            self.logger.warning(
                "metadata_extraction_partial",
                file=str(file_path),
                error=str(e)
            )
        
        return metadata
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _determine_log_type(self, file_path: Path) -> str:
        """Determine log type from filename."""
        name = file_path.stem.lower()
        if 'trading' in name:
            return 'trading'
        elif 'audit' in name:
            return 'audit'
        elif 'tilt' in name:
            return 'tilt'
        else:
            return 'system'
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        try:
            # Handle JSON format
            if line.startswith('{'):
                data = json.loads(line)
                if 'timestamp' in data:
                    return datetime.fromisoformat(data['timestamp'])
            # Handle ISO format
            import re
            iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
            match = re.search(iso_pattern, line)
            if match:
                return datetime.fromisoformat(match.group())
        except:
            pass
        return None
    
    async def _compress_file(self, file_path: Path, archive_id: str) -> Path:
        """Compress file with gzip and create tar archive."""
        archive_path = self.archive_dir / f"{archive_id}.tar.gz"
        
        with tarfile.open(archive_path, "w:gz", compresslevel=self.config.compression_level) as tar:
            tar.add(file_path, arcname=file_path.name)
            
            # Add metadata file
            metadata_content = json.dumps({
                "original_file": file_path.name,
                "archived_at": datetime.utcnow().isoformat(),
                "checksum": self._calculate_checksum(file_path)
            })
            
            import io
            metadata_file = io.BytesIO(metadata_content.encode())
            tarinfo = tarfile.TarInfo(name="metadata.json")
            tarinfo.size = len(metadata_content)
            metadata_file.seek(0)
            tar.addfile(tarinfo, metadata_file)
        
        return archive_path
    
    async def _upload_to_s3(
        self,
        file_path: Path,
        s3_key: str,
        metadata: LogArchiveMetadata
    ) -> None:
        """Upload archive to S3/Spaces."""
        try:
            # Prepare S3 metadata
            s3_metadata = {
                'original-filename': metadata.original_filename,
                'log-type': metadata.log_type,
                'entry-count': str(metadata.entry_count),
                'checksum': metadata.checksum
            }
            
            # Upload with metadata
            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=s3_key,
                    Body=f,
                    Metadata=s3_metadata,
                    StorageClass='STANDARD_IA'  # Infrequent access for cost savings
                )
            
            self.logger.info(
                "uploaded_to_s3",
                key=s3_key,
                size=file_path.stat().st_size
            )
            
        except ClientError as e:
            self.logger.error(
                "s3_upload_failed",
                key=s3_key,
                error=str(e)
            )
            raise
    
    async def retrieve_archive(
        self,
        archive_id: str,
        destination: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Retrieve archived log from S3/Spaces.
        
        Args:
            archive_id: Archive identifier
            destination: Local destination path
            
        Returns:
            Path to retrieved file if successful
        """
        if archive_id not in self.metadata_index:
            self.logger.error("archive_not_found", archive_id=archive_id)
            return None
        
        metadata = self.metadata_index[archive_id]
        s3_key = f"{self.config.s3_prefix}{archive_id}.tar.gz"
        
        if destination is None:
            destination = self.archive_dir / f"retrieved_{archive_id}.tar.gz"
        
        try:
            # Download from S3
            self.s3_client.download_file(
                self.config.s3_bucket,
                s3_key,
                str(destination)
            )
            
            self.logger.info(
                "archive_retrieved",
                archive_id=archive_id,
                destination=str(destination)
            )
            
            return destination
            
        except ClientError as e:
            self.logger.error(
                "retrieval_failed",
                archive_id=archive_id,
                error=str(e)
            )
            return None
    
    async def search_archives(
        self,
        log_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_entries: Optional[int] = None
    ) -> List[LogArchiveMetadata]:
        """
        Search archived logs by criteria.
        
        Args:
            log_type: Type of log to search
            start_date: Start date range
            end_date: End date range
            min_entries: Minimum number of entries
            
        Returns:
            List of matching archive metadata
        """
        results = []
        
        for metadata in self.metadata_index.values():
            # Apply filters
            if log_type and metadata.log_type != log_type:
                continue
            
            if start_date and metadata.end_timestamp:
                if metadata.end_timestamp < start_date:
                    continue
            
            if end_date and metadata.start_timestamp:
                if metadata.start_timestamp > end_date:
                    continue
            
            if min_entries and metadata.entry_count < min_entries:
                continue
            
            results.append(metadata)
        
        # Sort by archive timestamp
        results.sort(key=lambda x: x.archive_timestamp, reverse=True)
        
        return results
    
    async def enforce_retention_policy(self) -> Tuple[int, int]:
        """
        Enforce retention policies for local and archived logs.
        
        Returns:
            Tuple of (local_deleted, archives_deleted)
        """
        local_deleted = 0
        archives_deleted = 0
        
        # Clean local logs
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.local_retention_days)
        
        for log_file in self.log_dir.glob("*.log*"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff_date:
                    # Archive before deleting if not already archived
                    if not any(log_file.name in m.original_filename 
                              for m in self.metadata_index.values()):
                        await self._archive_file(log_file)
                    
                    log_file.unlink()
                    local_deleted += 1
                    
            except Exception as e:
                self.logger.error(
                    "retention_enforcement_failed",
                    file=str(log_file),
                    error=str(e)
                )
        
        # Clean archived logs
        archive_cutoff = datetime.utcnow() - timedelta(days=self.config.archive_retention_days)
        
        for archive_id, metadata in list(self.metadata_index.items()):
            if metadata.archive_timestamp < archive_cutoff:
                try:
                    # Delete from S3
                    s3_key = f"{self.config.s3_prefix}{archive_id}.tar.gz"
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=s3_key
                    )
                    
                    # Remove from index
                    del self.metadata_index[archive_id]
                    archives_deleted += 1
                    
                except Exception as e:
                    self.logger.error(
                        "archive_deletion_failed",
                        archive_id=archive_id,
                        error=str(e)
                    )
        
        # Save updated index
        self._save_metadata_index()
        
        self.logger.info(
            "retention_policy_enforced",
            local_deleted=local_deleted,
            archives_deleted=archives_deleted
        )
        
        return local_deleted, archives_deleted
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for logs and archives."""
        stats = {
            "local": {
                "count": 0,
                "total_size": 0,
                "by_type": {}
            },
            "archived": {
                "count": len(self.metadata_index),
                "total_original_size": 0,
                "total_compressed_size": 0,
                "average_compression_ratio": 0,
                "by_type": {}
            }
        }
        
        # Local stats
        for log_file in self.log_dir.glob("*.log*"):
            stats["local"]["count"] += 1
            stats["local"]["total_size"] += log_file.stat().st_size
            
            log_type = self._determine_log_type(log_file)
            if log_type not in stats["local"]["by_type"]:
                stats["local"]["by_type"][log_type] = {"count": 0, "size": 0}
            stats["local"]["by_type"][log_type]["count"] += 1
            stats["local"]["by_type"][log_type]["size"] += log_file.stat().st_size
        
        # Archive stats
        compression_ratios = []
        for metadata in self.metadata_index.values():
            stats["archived"]["total_original_size"] += metadata.original_size
            stats["archived"]["total_compressed_size"] += metadata.compressed_size
            compression_ratios.append(metadata.compression_ratio)
            
            if metadata.log_type not in stats["archived"]["by_type"]:
                stats["archived"]["by_type"][metadata.log_type] = {
                    "count": 0,
                    "original_size": 0,
                    "compressed_size": 0
                }
            stats["archived"]["by_type"][metadata.log_type]["count"] += 1
            stats["archived"]["by_type"][metadata.log_type]["original_size"] += metadata.original_size
            stats["archived"]["by_type"][metadata.log_type]["compressed_size"] += metadata.compressed_size
        
        if compression_ratios:
            stats["archived"]["average_compression_ratio"] = sum(compression_ratios) / len(compression_ratios)
        
        return stats


async def setup_log_archival() -> LogArchiver:
    """Setup and configure log archival system."""
    config = LogArchivalConfig(
        local_retention_days=30,
        archive_retention_days=365,
        compression_level=9,
        batch_size=10,
        s3_bucket=os.getenv('BACKUP_BUCKET', 'genesis-backups'),
        s3_prefix="logs/",
        archive_on_rotation=True,
        min_file_size_bytes=1024,
        metadata_index_enabled=True
    )
    
    archiver = LogArchiver(config)
    
    # Setup scheduled archival
    async def scheduled_archival():
        while True:
            try:
                # Archive rotated logs every hour
                await asyncio.sleep(3600)
                await archiver.archive_rotated_logs()
                
                # Enforce retention daily
                if datetime.utcnow().hour == 2:  # Run at 2 AM
                    await archiver.enforce_retention_policy()
                    
            except Exception as e:
                logger = get_logger(__name__, LoggerType.SYSTEM)
                logger.error("scheduled_archival_failed", error=str(e))
    
    # Start background task
    asyncio.create_task(scheduled_archival())
    
    return archiver