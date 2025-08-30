"""Unit tests for backup management system."""

import asyncio
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from freezegun import freeze_time

from genesis.backup.backup_manager import BackupManager
from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.core.exceptions import BackupError


class TestBackupManager:
    """Test suite for BackupManager."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary SQLite database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO test (data) VALUES ('test_data')")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        return db_path
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        client = Mock(spec=S3Client)
        client.upload_backup = AsyncMock(return_value="backups/full/test.bak")
        client.download_backup = AsyncMock()
        client.list_backups = AsyncMock(return_value=[])
        client.apply_retention_policy = AsyncMock(return_value={"hourly": 0, "daily": 0, "monthly": 0})
        client._calculate_checksum = AsyncMock(return_value="abc123")
        return client
    
    @pytest.fixture
    def backup_manager(self, temp_db_path, mock_s3_client, tmp_path):
        """Create BackupManager instance."""
        backup_dir = tmp_path / "backups"
        return BackupManager(
            database_path=temp_db_path,
            s3_client=mock_s3_client,
            local_backup_dir=backup_dir,
            backup_interval_hours=4,
            incremental_interval_minutes=5,
            enable_scheduler=False
        )
    
    @pytest.mark.asyncio
    async def test_create_full_backup(self, backup_manager, mock_s3_client):
        """Test creating full database backup."""
        # Execute backup
        metadata = await backup_manager.create_full_backup()
        
        # Verify metadata
        assert metadata.backup_type == "full"
        assert metadata.database_version == "v1"
        assert metadata.size_bytes > 0
        assert metadata.checksum == "abc123"
        assert metadata.destination_key == "backups/full/test.bak"
        
        # Verify S3 upload was called
        mock_s3_client.upload_backup.assert_called_once()
        
        # Verify state updated
        assert backup_manager.last_full_backup is not None
        assert len(backup_manager.backup_history) == 1
    
    @pytest.mark.asyncio
    async def test_create_incremental_backup_without_full(self, backup_manager):
        """Test incremental backup fails without full backup."""
        result = await backup_manager.create_incremental_backup()
        
        assert result is None
        assert backup_manager.last_incremental_backup is None
    
    @pytest.mark.asyncio
    async def test_create_incremental_backup_with_wal(self, backup_manager, temp_db_path, mock_s3_client):
        """Test creating incremental WAL backup."""
        # Create full backup first
        await backup_manager.create_full_backup()
        
        # Create WAL file
        wal_path = Path(str(temp_db_path) + "-wal")
        wal_path.write_bytes(b"WAL_DATA")
        
        # Execute incremental backup
        metadata = await backup_manager.create_incremental_backup()
        
        # Verify metadata
        assert metadata is not None
        assert metadata.backup_type == "incremental"
        assert metadata.retention_policy == "hourly"
        
        # Verify S3 upload was called twice (full + incremental)
        assert mock_s3_client.upload_backup.call_count == 2
    
    @pytest.mark.asyncio
    async def test_backup_sqlite_database(self, backup_manager, temp_db_path, tmp_path):
        """Test SQLite backup mechanism."""
        dest_path = tmp_path / "backup.db"
        
        await backup_manager._backup_sqlite_database(
            source_path=temp_db_path,
            destination_path=dest_path,
            checkpoint=True
        )
        
        # Verify backup exists
        assert dest_path.exists()
        
        # Verify backup content
        conn = sqlite3.connect(str(dest_path))
        cursor = conn.execute("SELECT data FROM test")
        data = cursor.fetchone()[0]
        conn.close()
        
        assert data == "test_data"
    
    @pytest.mark.asyncio
    async def test_get_database_version(self, backup_manager):
        """Test getting database schema version."""
        version = await backup_manager._get_database_version()
        assert version == "v1"
    
    def test_determine_retention_policy(self, backup_manager):
        """Test retention policy determination."""
        # Yearly backup
        timestamp = datetime(2025, 1, 1, 2, 0, 0)
        assert backup_manager._determine_retention_policy(timestamp) == "yearly"
        
        # Monthly backup
        timestamp = datetime(2025, 3, 1, 2, 0, 0)
        assert backup_manager._determine_retention_policy(timestamp) == "monthly"
        
        # Daily backup
        timestamp = datetime(2025, 3, 15, 2, 0, 0)
        assert backup_manager._determine_retention_policy(timestamp) == "daily"
        
        # Hourly backup
        timestamp = datetime(2025, 3, 15, 10, 0, 0)
        assert backup_manager._determine_retention_policy(timestamp) == "hourly"
    
    @pytest.mark.asyncio
    async def test_apply_retention_policy(self, backup_manager, mock_s3_client):
        """Test applying retention policy."""
        deleted_counts = await backup_manager.apply_retention_policy()
        
        assert deleted_counts == {"hourly": 0, "daily": 0, "monthly": 0}
        mock_s3_client.apply_retention_policy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_backups(self, backup_manager, mock_s3_client):
        """Test listing available backups."""
        # Mock S3 response
        mock_s3_client.list_backups.return_value = [
            {
                "key": "backups/full/20250101_120000_test.bak",
                "size_bytes": 1024,
                "last_modified": datetime(2025, 1, 1, 12, 0, 0),
                "metadata": {
                    "backup-id": "test-id",
                    "timestamp": "2025-01-01T12:00:00",
                    "checksum": "abc123",
                    "database-version": "v1",
                    "backup-type": "full",
                    "retention-policy": "daily"
                }
            }
        ]
        
        backups = await backup_manager.list_backups(backup_type="full")
        
        assert len(backups) == 1
        assert backups[0].backup_type == "full"
        assert backups[0].database_version == "v1"
    
    @pytest.mark.asyncio
    async def test_get_backup_for_timestamp(self, backup_manager, mock_s3_client):
        """Test finding backups for point-in-time recovery."""
        # Mock backup history
        full_backup = BackupMetadata(
            backup_id="full-1",
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            size_bytes=1024,
            checksum="abc",
            database_version="v1",
            backup_type="full",
            retention_policy="daily",
            source_path="",
            destination_key="backups/full/test.bak"
        )
        
        incremental_backup = BackupMetadata(
            backup_id="inc-1",
            timestamp=datetime(2025, 1, 1, 10, 5, 0),
            size_bytes=512,
            checksum="def",
            database_version="v1",
            backup_type="incremental",
            retention_policy="hourly",
            source_path="",
            destination_key="backups/incremental/test.wal"
        )
        
        mock_s3_client.list_backups.side_effect = [
            # Full backups
            [{
                "key": full_backup.destination_key,
                "size_bytes": full_backup.size_bytes,
                "metadata": {
                    "backup-id": full_backup.backup_id,
                    "timestamp": full_backup.timestamp.isoformat(),
                    "backup-type": "full"
                }
            }],
            # Incremental backups
            [{
                "key": incremental_backup.destination_key,
                "size_bytes": incremental_backup.size_bytes,
                "metadata": {
                    "backup-id": incremental_backup.backup_id,
                    "timestamp": incremental_backup.timestamp.isoformat(),
                    "backup-type": "incremental"
                }
            }]
        ]
        
        target_time = datetime(2025, 1, 1, 10, 10, 0)
        full, incrementals = await backup_manager.get_backup_for_timestamp(target_time)
        
        assert full is not None
        assert len(incrementals) == 1
    
    @pytest.mark.asyncio
    async def test_verify_backup_integrity(self, backup_manager, mock_s3_client, tmp_path):
        """Test backup integrity verification."""
        metadata = BackupMetadata(
            backup_id="test-id",
            timestamp=datetime.utcnow(),
            size_bytes=1024,
            checksum="abc123",
            database_version="v1",
            backup_type="full",
            retention_policy="daily",
            source_path="",
            destination_key="backups/full/test.bak"
        )
        
        # Mock successful download
        mock_s3_client.download_backup.return_value = metadata
        
        # Create temp file for verification
        temp_file = tmp_path / f"verify_{metadata.backup_id}.tmp"
        temp_file.write_text("test")
        
        with patch.object(backup_manager.local_backup_dir, "iterdir", return_value=[]):
            result = await backup_manager.verify_backup_integrity(metadata)
        
        assert result is True
        mock_s3_client.download_backup.assert_called_once()
    
    def test_get_backup_status(self, backup_manager):
        """Test getting backup status."""
        backup_manager.last_full_backup = datetime(2025, 1, 1, 12, 0, 0)
        backup_manager.last_incremental_backup = datetime(2025, 1, 1, 12, 5, 0)
        backup_manager.backup_history = [Mock(), Mock()]
        
        status = backup_manager.get_backup_status()
        
        assert status["last_full_backup"] == "2025-01-01T12:00:00"
        assert status["last_incremental_backup"] == "2025-01-01T12:05:00"
        assert status["backup_count"] == 2
        assert status["scheduler_running"] is False
    
    def test_scheduler_setup(self, temp_db_path, mock_s3_client, tmp_path):
        """Test scheduler configuration."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(
            database_path=temp_db_path,
            s3_client=mock_s3_client,
            local_backup_dir=backup_dir,
            backup_interval_hours=4,
            incremental_interval_minutes=5,
            enable_scheduler=True
        )
        
        # Verify jobs are scheduled
        assert manager.scheduler is not None
        jobs = manager.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]
        
        assert "full_backup" in job_ids
        assert "incremental_backup" in job_ids
        assert "retention_policy" in job_ids
    
    @pytest.mark.asyncio
    async def test_backup_error_handling(self, backup_manager, mock_s3_client):
        """Test error handling during backup."""
        # Simulate S3 upload failure
        mock_s3_client.upload_backup.side_effect = Exception("S3 error")
        
        with pytest.raises(BackupError) as exc_info:
            await backup_manager.create_full_backup()
        
        assert "Full backup failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_backup_with_retry(self, backup_manager, mock_s3_client):
        """Test backup retry mechanism."""
        # Fail first two attempts, succeed on third
        mock_s3_client.upload_backup.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            "backups/full/test.bak"
        ]
        
        # Should succeed after retries
        metadata = await backup_manager.create_full_backup()
        
        assert metadata.destination_key == "backups/full/test.bak"
        assert mock_s3_client.upload_backup.call_count == 3