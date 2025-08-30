"""Unit tests for log archival system."""

import asyncio
import gzip
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from genesis.operations.log_archiver import (
    LogArchivalConfig,
    LogArchiveMetadata,
    LogArchiver
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        log_dir = tmpdir / "logs"
        archive_dir = tmpdir / "archives"
        log_dir.mkdir()
        archive_dir.mkdir()
        yield log_dir, archive_dir


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    client = MagicMock()
    client.put_object = MagicMock()
    client.download_file = MagicMock()
    client.delete_object = MagicMock()
    return client


@pytest.fixture
def archival_config():
    """Create test archival configuration."""
    return LogArchivalConfig(
        local_retention_days=7,
        archive_retention_days=30,
        compression_level=6,
        batch_size=5,
        s3_bucket="test-bucket",
        s3_prefix="test/logs/",
        archive_on_rotation=True,
        min_file_size_bytes=100,
        metadata_index_enabled=True
    )


@pytest.fixture
def log_archiver(archival_config, mock_s3_client, temp_dirs):
    """Create log archiver instance."""
    log_dir, archive_dir = temp_dirs
    return LogArchiver(
        config=archival_config,
        s3_client=mock_s3_client,
        log_dir=log_dir,
        archive_dir=archive_dir
    )


def create_test_log_file(
    log_dir: Path,
    filename: str,
    num_lines: int = 100,
    with_timestamps: bool = True
) -> Path:
    """Create a test log file."""
    log_file = log_dir / filename
    
    with open(log_file, 'w') as f:
        for i in range(num_lines):
            if with_timestamps:
                timestamp = (datetime.utcnow() - timedelta(hours=num_lines-i)).isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "level": "INFO",
                    "message": f"Test log entry {i}",
                    "data": {"index": i}
                }
                f.write(json.dumps(log_entry) + "\n")
            else:
                f.write(f"Test log line {i}\n")
    
    return log_file


class TestLogArchiver:
    """Test cases for LogArchiver."""
    
    @pytest.mark.asyncio
    async def test_archive_rotated_logs(self, log_archiver, temp_dirs):
        """Test archiving rotated log files."""
        log_dir, _ = temp_dirs
        
        # Create rotated log files
        create_test_log_file(log_dir, "trading.log.1", 50)
        create_test_log_file(log_dir, "audit.log.2", 75)
        create_test_log_file(log_dir, "system.log.3", 100)
        
        # Archive rotated logs
        archive_ids = await log_archiver.archive_rotated_logs()
        
        # Verify archives created
        assert len(archive_ids) == 3
        assert log_archiver.s3_client.put_object.call_count == 3
        
        # Verify metadata index updated
        assert len(log_archiver.metadata_index) == 3
        
        # Verify files deleted after archival
        assert not (log_dir / "trading.log.1").exists()
        assert not (log_dir / "audit.log.2").exists()
        assert not (log_dir / "system.log.3").exists()
    
    @pytest.mark.asyncio
    async def test_archive_file_with_metadata(self, log_archiver, temp_dirs):
        """Test archiving single file with metadata extraction."""
        log_dir, _ = temp_dirs
        
        # Create test log file
        log_file = create_test_log_file(log_dir, "trading.log", 100, with_timestamps=True)
        
        # Archive file
        archive_id = await log_archiver._archive_file(log_file)
        
        assert archive_id is not None
        assert archive_id in log_archiver.metadata_index
        
        # Verify metadata
        metadata = log_archiver.metadata_index[archive_id]
        assert metadata.original_filename == "trading.log"
        assert metadata.log_type == "trading"
        assert metadata.entry_count == 100
        assert metadata.start_timestamp is not None
        assert metadata.end_timestamp is not None
        assert metadata.compression_ratio > 1.0
        
        # Verify S3 upload
        log_archiver.s3_client.put_object.assert_called_once()
        call_args = log_archiver.s3_client.put_object.call_args
        assert call_args[1]['Bucket'] == "test-bucket"
        assert call_args[1]['Key'].startswith("test/logs/")
    
    @pytest.mark.asyncio
    async def test_compression(self, log_archiver, temp_dirs):
        """Test file compression."""
        log_dir, archive_dir = temp_dirs
        
        # Create large test file
        log_file = create_test_log_file(log_dir, "large.log", 1000)
        original_size = log_file.stat().st_size
        
        # Compress file
        archive_id = log_archiver._generate_archive_id(log_file)
        compressed_path = await log_archiver._compress_file(log_file, archive_id)
        
        assert compressed_path.exists()
        compressed_size = compressed_path.stat().st_size
        
        # Verify compression achieved
        assert compressed_size < original_size
        compression_ratio = original_size / compressed_size
        assert compression_ratio > 1.5  # Expect decent compression for text
    
    def test_checksum_calculation(self, log_archiver, temp_dirs):
        """Test SHA256 checksum calculation."""
        log_dir, _ = temp_dirs
        
        # Create test file
        log_file = create_test_log_file(log_dir, "test.log", 10)
        
        # Calculate checksum twice
        checksum1 = log_archiver._calculate_checksum(log_file)
        checksum2 = log_archiver._calculate_checksum(log_file)
        
        # Verify consistency
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex length
    
    def test_log_type_determination(self, log_archiver):
        """Test log type determination from filename."""
        assert log_archiver._determine_log_type(Path("trading.log")) == "trading"
        assert log_archiver._determine_log_type(Path("audit.log.1")) == "audit"
        assert log_archiver._determine_log_type(Path("tilt_events.log")) == "tilt"
        assert log_archiver._determine_log_type(Path("application.log")) == "system"
    
    def test_timestamp_extraction(self, log_archiver):
        """Test timestamp extraction from log lines."""
        # JSON format
        json_line = '{"timestamp": "2024-01-15T10:30:45", "message": "test"}'
        timestamp = log_archiver._extract_timestamp(json_line)
        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15
        
        # ISO format in text
        text_line = "2024-01-15T10:30:45 INFO Test message"
        timestamp = log_archiver._extract_timestamp(text_line)
        assert timestamp is not None
        
        # No timestamp
        no_timestamp = "This line has no timestamp"
        timestamp = log_archiver._extract_timestamp(no_timestamp)
        assert timestamp is None
    
    @pytest.mark.asyncio
    async def test_retrieve_archive(self, log_archiver, temp_dirs):
        """Test archive retrieval from S3."""
        log_dir, archive_dir = temp_dirs
        
        # Create and archive a file
        log_file = create_test_log_file(log_dir, "test.log", 50)
        archive_id = await log_archiver._archive_file(log_file)
        
        # Mock S3 download
        log_archiver.s3_client.download_file = MagicMock()
        
        # Retrieve archive
        retrieved_path = await log_archiver.retrieve_archive(archive_id)
        
        assert retrieved_path is not None
        log_archiver.s3_client.download_file.assert_called_once()
        
        # Test retrieval of non-existent archive
        result = await log_archiver.retrieve_archive("non_existent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_archives(self, log_archiver):
        """Test archive search functionality."""
        # Add test metadata
        now = datetime.utcnow()
        
        metadata1 = LogArchiveMetadata(
            archive_id="archive1",
            original_filename="trading.log.1",
            archive_timestamp=now - timedelta(days=5),
            start_timestamp=now - timedelta(days=10),
            end_timestamp=now - timedelta(days=9),
            original_size=1000,
            compressed_size=300,
            compression_ratio=3.33,
            checksum="abc123",
            log_type="trading",
            entry_count=500
        )
        
        metadata2 = LogArchiveMetadata(
            archive_id="archive2",
            original_filename="audit.log.1",
            archive_timestamp=now - timedelta(days=2),
            start_timestamp=now - timedelta(days=3),
            end_timestamp=now - timedelta(days=2),
            original_size=2000,
            compressed_size=400,
            compression_ratio=5.0,
            checksum="def456",
            log_type="audit",
            entry_count=1000
        )
        
        log_archiver.metadata_index["archive1"] = metadata1
        log_archiver.metadata_index["archive2"] = metadata2
        
        # Search by log type
        results = await log_archiver.search_archives(log_type="trading")
        assert len(results) == 1
        assert results[0].archive_id == "archive1"
        
        # Search by date range
        results = await log_archiver.search_archives(
            start_date=now - timedelta(days=4),
            end_date=now
        )
        assert len(results) == 1
        assert results[0].archive_id == "archive2"
        
        # Search by entry count
        results = await log_archiver.search_archives(min_entries=800)
        assert len(results) == 1
        assert results[0].archive_id == "archive2"
    
    @pytest.mark.asyncio
    async def test_retention_policy_enforcement(self, log_archiver, temp_dirs):
        """Test retention policy enforcement."""
        log_dir, _ = temp_dirs
        
        # Create old log files
        old_file = create_test_log_file(log_dir, "old.log", 10)
        recent_file = create_test_log_file(log_dir, "recent.log", 10)
        
        # Modify file times
        old_time = (datetime.utcnow() - timedelta(days=10)).timestamp()
        recent_time = (datetime.utcnow() - timedelta(days=1)).timestamp()
        
        import os
        os.utime(old_file, (old_time, old_time))
        os.utime(recent_file, (recent_time, recent_time))
        
        # Add old archive to index
        old_metadata = LogArchiveMetadata(
            archive_id="old_archive",
            original_filename="ancient.log",
            archive_timestamp=datetime.utcnow() - timedelta(days=40),
            original_size=1000,
            compressed_size=200,
            compression_ratio=5.0,
            checksum="xyz789",
            log_type="system",
            entry_count=100
        )
        log_archiver.metadata_index["old_archive"] = old_metadata
        
        # Enforce retention
        local_deleted, archives_deleted = await log_archiver.enforce_retention_policy()
        
        # Verify old local file deleted
        assert local_deleted == 1
        assert not old_file.exists()
        assert recent_file.exists()
        
        # Verify old archive deleted
        assert archives_deleted == 1
        assert "old_archive" not in log_archiver.metadata_index
        log_archiver.s3_client.delete_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_storage_stats(self, log_archiver, temp_dirs):
        """Test storage statistics calculation."""
        log_dir, _ = temp_dirs
        
        # Create test files
        create_test_log_file(log_dir, "trading.log", 100)
        create_test_log_file(log_dir, "audit.log", 200)
        
        # Add test metadata
        metadata = LogArchiveMetadata(
            archive_id="test_archive",
            original_filename="old_trading.log",
            archive_timestamp=datetime.utcnow(),
            original_size=10000,
            compressed_size=2000,
            compression_ratio=5.0,
            checksum="abc123",
            log_type="trading",
            entry_count=1000
        )
        log_archiver.metadata_index["test_archive"] = metadata
        
        # Get stats
        stats = await log_archiver.get_storage_stats()
        
        # Verify local stats
        assert stats["local"]["count"] == 2
        assert stats["local"]["total_size"] > 0
        assert "trading" in stats["local"]["by_type"]
        assert "audit" in stats["local"]["by_type"]
        
        # Verify archive stats
        assert stats["archived"]["count"] == 1
        assert stats["archived"]["total_original_size"] == 10000
        assert stats["archived"]["total_compressed_size"] == 2000
        assert stats["archived"]["average_compression_ratio"] == 5.0
    
    @pytest.mark.asyncio
    async def test_s3_error_handling(self, log_archiver, temp_dirs):
        """Test S3 error handling."""
        log_dir, _ = temp_dirs
        
        # Create test file
        log_file = create_test_log_file(log_dir, "test.log", 10)
        
        # Simulate S3 error
        log_archiver.s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket"}},
            "PutObject"
        )
        
        # Attempt archive
        archive_id = await log_archiver._archive_file(log_file)
        
        # Should return None on error
        assert archive_id is None
        
        # File should still exist
        assert log_file.exists()
    
    def test_metadata_index_persistence(self, log_archiver, temp_dirs):
        """Test metadata index save/load."""
        _, archive_dir = temp_dirs
        
        # Add test metadata
        metadata = LogArchiveMetadata(
            archive_id="test_id",
            original_filename="test.log",
            archive_timestamp=datetime.utcnow(),
            original_size=1000,
            compressed_size=200,
            compression_ratio=5.0,
            checksum="abc123",
            log_type="trading",
            entry_count=100
        )
        log_archiver.metadata_index["test_id"] = metadata
        
        # Save index
        log_archiver._save_metadata_index()
        
        # Verify file created
        index_file = archive_dir / "metadata_index.json"
        assert index_file.exists()
        
        # Create new archiver and load index
        new_archiver = LogArchiver(
            config=log_archiver.config,
            s3_client=log_archiver.s3_client,
            log_dir=log_archiver.log_dir,
            archive_dir=archive_dir
        )
        
        # Verify loaded correctly
        assert "test_id" in new_archiver.metadata_index
        loaded_metadata = new_archiver.metadata_index["test_id"]
        assert loaded_metadata.original_filename == "test.log"
        assert loaded_metadata.entry_count == 100


@pytest.mark.asyncio
async def test_scheduled_archival():
    """Test scheduled archival setup."""
    with patch('genesis.operations.log_archiver.LogArchiver') as MockArchiver:
        mock_archiver = AsyncMock()
        MockArchiver.return_value = mock_archiver
        
        from genesis.operations.log_archiver import setup_log_archival
        
        # Setup archival
        archiver = await setup_log_archival()
        
        # Verify archiver created
        assert archiver is not None
        
        # Let background task start
        await asyncio.sleep(0.1)