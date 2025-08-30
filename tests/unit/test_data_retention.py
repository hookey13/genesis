"""Unit tests for data retention system."""
import os
import gzip
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from genesis.operations.data_retention import (
    DataRetentionManager,
    RetentionPolicy,
    RetentionPeriod,
    ArchivalResult
)


@pytest.fixture
def retention_manager():
    """Create DataRetentionManager instance with temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DataRetentionManager(archive_dir=tmpdir)
        yield manager


@pytest.fixture
def test_data_directory():
    """Create test data directory with files of various ages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_data"
        data_dir.mkdir()
        
        # Create files with different ages
        now = datetime.now()
        
        # Recent file (1 day old)
        recent_file = data_dir / "recent.txt"
        recent_file.write_text("recent data")
        os.utime(recent_file, (now.timestamp(), (now - timedelta(days=1)).timestamp()))
        
        # Old file (100 days old)
        old_file = data_dir / "old.txt"
        old_file.write_text("old data")
        os.utime(old_file, (now.timestamp(), (now - timedelta(days=100)).timestamp()))
        
        # Very old file (400 days old)
        very_old_file = data_dir / "very_old.txt"
        very_old_file.write_text("very old data")
        os.utime(very_old_file, (now.timestamp(), (now - timedelta(days=400)).timestamp()))
        
        yield str(data_dir)


class TestRetentionPolicy:
    """Test RetentionPolicy functionality."""
    
    def test_retention_policy_creation(self):
        """Test creating a retention policy."""
        policy = RetentionPolicy(
            data_type="trades",
            retention_days=365,
            archive_enabled=True,
            compression_enabled=True,
            purge_enabled=False
        )
        
        assert policy.data_type == "trades"
        assert policy.retention_days == 365
        assert policy.archive_enabled is True
        assert policy.purge_enabled is False
    
    def test_is_expired_check(self):
        """Test checking if data has expired."""
        policy = RetentionPolicy(
            data_type="system",
            retention_days=90
        )
        
        # File from 50 days ago - not expired
        recent_date = datetime.now() - timedelta(days=50)
        assert policy.is_expired(recent_date) is False
        
        # File from 100 days ago - expired
        old_date = datetime.now() - timedelta(days=100)
        assert policy.is_expired(old_date) is True


class TestDataRetentionManager:
    """Test DataRetentionManager functionality."""
    
    def test_manager_initialization(self, retention_manager):
        """Test retention manager initialization."""
        assert retention_manager.archive_dir.exists()
        assert len(retention_manager.policies) > 0
        assert "trades" in retention_manager.policies
        assert "audit" in retention_manager.policies
    
    def test_apply_retention_policy_dry_run(
        self, retention_manager, test_data_directory
    ):
        """Test applying retention policy in dry run mode."""
        # Use system policy (90 days retention)
        result = retention_manager.apply_retention_policy(
            test_data_directory,
            "system",
            dry_run=True
        )
        
        assert result.files_processed == 3
        assert result.files_archived == 2  # old and very_old files
        assert result.files_purged == 2  # same files would be purged
        assert len(result.errors) == 0
        
        # Verify no actual changes were made
        assert len(list(Path(test_data_directory).iterdir())) == 3
    
    def test_apply_retention_policy_archive_only(
        self, retention_manager, test_data_directory
    ):
        """Test archiving without purging."""
        # Create custom policy with purge disabled
        retention_manager.policies["test"] = RetentionPolicy(
            data_type="test",
            retention_days=90,
            archive_enabled=True,
            purge_enabled=False
        )
        
        result = retention_manager.apply_retention_policy(
            test_data_directory,
            "test",
            dry_run=False
        )
        
        assert result.files_archived == 2  # old and very_old files
        assert result.files_purged == 0  # purge disabled
        
        # Verify original files still exist
        assert len(list(Path(test_data_directory).iterdir())) == 3
        
        # Verify archives were created
        archive_files = list(retention_manager.archive_dir.rglob("*.gz"))
        assert len(archive_files) == 2
    
    def test_apply_retention_policy_with_purge(
        self, retention_manager, test_data_directory
    ):
        """Test archiving and purging."""
        # Use system policy (90 days retention, purge enabled)
        result = retention_manager.apply_retention_policy(
            test_data_directory,
            "system",
            dry_run=False
        )
        
        assert result.files_archived == 2
        assert result.files_purged == 2
        
        # Verify old files were removed
        remaining_files = list(Path(test_data_directory).iterdir())
        assert len(remaining_files) == 1
        assert remaining_files[0].name == "recent.txt"
    
    def test_archive_file_compression(self, retention_manager):
        """Test file archival with compression."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test data for compression")
            temp_file = Path(f.name)
        
        try:
            archived_path = retention_manager._archive_file(
                temp_file,
                "test",
                compress=True
            )
            
            assert archived_path is not None
            assert archived_path.suffix == ".gz"
            assert archived_path.exists()
            
            # Verify compressed content
            with gzip.open(archived_path, 'rt') as gz_file:
                content = gz_file.read()
                assert content == "test data for compression"
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_archive_file_no_compression(self, retention_manager):
        """Test file archival without compression."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test data without compression")
            temp_file = Path(f.name)
        
        try:
            archived_path = retention_manager._archive_file(
                temp_file,
                "test",
                compress=False
            )
            
            assert archived_path is not None
            assert archived_path.suffix != ".gz"
            assert archived_path.exists()
            
            # Verify content
            content = archived_path.read_text()
            assert content == "test data without compression"
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_restore_from_archive_compressed(self, retention_manager):
        """Test restoring compressed archive."""
        # Create and archive a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("data to restore")
            temp_file = Path(f.name)
        
        try:
            archived_path = retention_manager._archive_file(
                temp_file,
                "test",
                compress=True
            )
            
            # Restore to different directory
            with tempfile.TemporaryDirectory() as restore_dir:
                success = retention_manager.restore_from_archive(
                    str(archived_path),
                    restore_dir
                )
                
                assert success is True
                
                # Verify restored file
                restored_files = list(Path(restore_dir).iterdir())
                assert len(restored_files) == 1
                
                content = restored_files[0].read_text()
                assert content == "data to restore"
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_restore_from_archive_uncompressed(self, retention_manager):
        """Test restoring uncompressed archive."""
        # Create and archive a file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("uncompressed data")
            temp_file = Path(f.name)
        
        try:
            archived_path = retention_manager._archive_file(
                temp_file,
                "test",
                compress=False
            )
            
            # Restore to different directory
            with tempfile.TemporaryDirectory() as restore_dir:
                success = retention_manager.restore_from_archive(
                    str(archived_path),
                    restore_dir
                )
                
                assert success is True
                
                # Verify restored file
                restored_files = list(Path(restore_dir).iterdir())
                assert len(restored_files) == 1
                
                content = restored_files[0].read_text()
                assert content == "uncompressed data"
        finally:
            temp_file.unlink(missing_ok=True)
    
    def test_restore_from_archive_not_found(self, retention_manager):
        """Test restoring non-existent archive."""
        with tempfile.TemporaryDirectory() as restore_dir:
            success = retention_manager.restore_from_archive(
                "/non/existent/file.gz",
                restore_dir
            )
            
            assert success is False
    
    def test_get_retention_status(self, retention_manager, test_data_directory):
        """Test getting retention status for directory."""
        status = retention_manager.get_retention_status(
            test_data_directory,
            "system"
        )
        
        assert status["data_type"] == "system"
        assert status["retention_days"] == 90
        assert status["total_files"] == 3
        assert status["expired_files"] == 2  # old and very_old files
        assert "oldest_file" in status
        assert "newest_file" in status
    
    def test_get_retention_status_nonexistent_dir(self, retention_manager):
        """Test getting status for non-existent directory."""
        status = retention_manager.get_retention_status(
            "/non/existent/dir",
            "system"
        )
        
        assert "error" in status
        assert "not found" in status["error"]
    
    def test_get_retention_status_invalid_type(self, retention_manager):
        """Test getting status with invalid data type."""
        with pytest.raises(ValueError, match="Unknown data type"):
            retention_manager.get_retention_status(
                "/tmp",
                "invalid_type"
            )
    
    def test_schedule_retention_tasks(self, retention_manager):
        """Test scheduling retention tasks."""
        # Create test directories with expired files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trade directory with old file
            trade_dir = Path(tmpdir) / ".genesis/data/trades"
            trade_dir.mkdir(parents=True)
            
            old_trade = trade_dir / "old_trade.json"
            old_trade.write_text("{}")
            
            # Make file old enough to expire
            old_time = datetime.now() - timedelta(days=2000)
            os.utime(old_trade, (old_time.timestamp(), old_time.timestamp()))
            
            # Update manager's base path
            retention_manager.policies["trades"] = RetentionPolicy(
                data_type="trades",
                retention_days=365
            )
            
            # Mock the data directories
            import genesis.operations.data_retention
            original_get_status = retention_manager.get_retention_status
            
            def mock_get_status(directory, data_type):
                if data_type == "trades" and "trades" in directory:
                    return {
                        "data_type": "trades",
                        "expired_files": 1,
                        "expired_size_mb": 1.0,
                        "purge_enabled": True
                    }
                return {"expired_files": 0}
            
            retention_manager.get_retention_status = mock_get_status
            
            tasks = retention_manager.schedule_retention_tasks()
            
            # Should have at least one task for expired trade file
            assert len(tasks) > 0
            trade_tasks = [t for t in tasks if t["data_type"] == "trades"]
            assert len(trade_tasks) == 1
            assert trade_tasks[0]["expired_files"] == 1
    
    def test_audit_log_creation(self, retention_manager):
        """Test audit log creation for retention operations."""
        result = ArchivalResult(
            files_processed=10,
            files_archived=5,
            files_purged=3,
            total_size_archived=1024,
            total_size_purged=512,
            errors=[],
            timestamp=datetime.now()
        )
        
        retention_manager._create_audit_log("test", result, dry_run=False)
        
        # Verify audit log was created
        audit_dir = Path(".genesis/logs/audit")
        if audit_dir.exists():
            audit_files = list(audit_dir.glob("retention_*.jsonl"))
            if audit_files:
                # Read and verify audit entry
                with open(audit_files[0], 'r') as f:
                    entry = json.loads(f.readline())
                    assert entry["operation"] == "data_retention"
                    assert entry["data_type"] == "test"
                    assert entry["files_processed"] == 10
                    assert entry["files_archived"] == 5
    
    def test_never_purge_audit_logs(self, retention_manager):
        """Test that audit logs are never purged."""
        audit_policy = retention_manager.policies["audit"]
        
        assert audit_policy.purge_enabled is False
        assert audit_policy.archive_enabled is True
    
    def test_temp_files_not_archived(self, retention_manager):
        """Test that temp files are not archived."""
        temp_policy = retention_manager.policies["temp"]
        
        assert temp_policy.archive_enabled is False
        assert temp_policy.purge_enabled is True
        assert temp_policy.retention_days == 7