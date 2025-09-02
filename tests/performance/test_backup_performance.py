"""Performance tests for backup system."""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.backup.backup_manager import BackupManager
from genesis.backup.recovery_manager import RecoveryManager
from genesis.backup.s3_client import BackupMetadata, S3Client


class TestBackupPerformance:
    """Performance tests for backup operations."""
    
    @pytest.mark.asyncio
    async def test_backup_completion_time(self, tmp_path):
        """Verify backup completes within 5 minutes."""
        # Setup
        db_path = tmp_path / "large_test.db"
        
        # Create a large test database (100 MB)
        with open(db_path, "wb") as f:
            f.write(b"0" * (100 * 1024 * 1024))  # 100 MB
        
        mock_s3_client = MagicMock(spec=S3Client)
        mock_s3_client.upload_backup = AsyncMock(return_value="s3://bucket/key")
        mock_s3_client._calculate_checksum = AsyncMock(return_value="test_checksum")
        
        manager = BackupManager(
            database_path=db_path,
            s3_client=mock_s3_client,
            local_backup_dir=tmp_path,
            enable_scheduler=False,
            database_type="sqlite"
        )
        
        # Disable encryption/compression for performance test
        manager.encryption_enabled = False
        manager.compression_enabled = False
        
        # Execute and measure time
        start_time = time.time()
        metadata = await manager.create_full_backup()
        end_time = time.time()
        
        completion_time = end_time - start_time
        
        # Verify
        assert metadata is not None
        assert completion_time < 300  # Should complete in less than 5 minutes
        print(f"Backup completed in {completion_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_recovery_time_objective(self, tmp_path):
        """Verify recovery achieves <15 minute RTO."""
        # Setup
        mock_s3_client = MagicMock(spec=S3Client)
        mock_s3_client.list_backups = AsyncMock(return_value=[
            BackupMetadata(
                backup_id="test_backup",
                timestamp=datetime.utcnow() - timedelta(hours=1),
                size_bytes=100 * 1024 * 1024,  # 100 MB
                checksum="test_checksum",
                database_version="v1",
                backup_type="full",
                retention_policy="daily",
                source_path="test.db",
                destination_key="s3://bucket/test_backup"
            )
        ])
        
        # Mock download with simulated delay
        async def mock_download(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate download time
            return MagicMock()
        
        mock_s3_client.download_backup = mock_download
        
        recovery_manager = RecoveryManager(s3_client=mock_s3_client)
        
        # Mock component recovery methods
        async def mock_recover(*args, **kwargs):
            await asyncio.sleep(1)  # Simulate recovery time
            return {"status": "success"}
        
        recovery_manager._recover_database = mock_recover
        recovery_manager._recover_vault = mock_recover
        recovery_manager._recover_config = mock_recover
        recovery_manager._recover_state = mock_recover
        
        # Execute and measure time
        start_time = time.time()
        report = await recovery_manager.perform_full_recovery(
            target_timestamp=datetime.utcnow() - timedelta(hours=2),
            dry_run=False
        )
        end_time = time.time()
        
        recovery_time = end_time - start_time
        
        # Verify
        assert report["status"] == "success"
        assert recovery_time < 900  # Should complete in less than 15 minutes
        assert report["rto_achieved"] is True
        print(f"Recovery completed in {recovery_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, tmp_path):
        """Test performance with concurrent backup operations."""
        # Setup multiple backup managers
        managers = []
        for i in range(5):
            db_path = tmp_path / f"test_{i}.db"
            db_path.write_bytes(b"test data" * 1000)
            
            mock_s3_client = MagicMock(spec=S3Client)
            mock_s3_client.upload_backup = AsyncMock(return_value=f"s3://bucket/key_{i}")
            mock_s3_client._calculate_checksum = AsyncMock(return_value=f"checksum_{i}")
            
            manager = BackupManager(
                database_path=db_path,
                s3_client=mock_s3_client,
                local_backup_dir=tmp_path / f"backup_{i}",
                enable_scheduler=False
            )
            manager.encryption_enabled = False
            manager.compression_enabled = False
            managers.append(manager)
        
        # Execute concurrent backups
        start_time = time.time()
        tasks = [manager.create_full_backup() for manager in managers]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify
        assert len(results) == 5
        assert all(r is not None for r in results)
        assert total_time < 60  # Should handle 5 concurrent backups in less than 1 minute
        print(f"5 concurrent backups completed in {total_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_backup_compression_ratio(self, tmp_path):
        """Test compression effectiveness on backup files."""
        # Create test data with repetitive content (highly compressible)
        test_file = tmp_path / "test_data.txt"
        test_content = "AAAAAAAAAA" * 10000  # 100KB of repetitive data
        test_file.write_text(test_content)
        original_size = test_file.stat().st_size
        
        manager = BackupManager(
            local_backup_dir=tmp_path,
            enable_scheduler=False
        )
        
        # Test compression
        with patch("genesis.security.vault_manager.VaultManager"):
            compressed_file = await manager._encrypt_and_compress_backup(
                test_file,
                encrypt=False,
                compress=True
            )
        
        compressed_size = compressed_file.stat().st_size
        compression_ratio = original_size / compressed_size
        
        # Verify
        assert compression_ratio > 10  # Should achieve at least 10:1 compression
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"Original: {original_size} bytes, Compressed: {compressed_size} bytes")
    
    @pytest.mark.asyncio
    async def test_incremental_backup_performance(self, tmp_path):
        """Test incremental backup performance."""
        # Setup
        wal_path = tmp_path / "test.db-wal"
        wal_path.write_bytes(b"WAL data" * 1000)  # Small WAL file
        
        mock_s3_client = MagicMock(spec=S3Client)
        mock_s3_client.upload_backup = AsyncMock(return_value="s3://bucket/wal")
        mock_s3_client._calculate_checksum = AsyncMock(return_value="wal_checksum")
        
        manager = BackupManager(
            database_path=tmp_path / "test.db",
            s3_client=mock_s3_client,
            local_backup_dir=tmp_path,
            enable_scheduler=False
        )
        manager.last_full_backup = datetime.utcnow()
        
        # Execute multiple incremental backups
        start_time = time.time()
        for _ in range(10):
            await manager.create_incremental_backup()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        # Verify
        assert avg_time < 1  # Each incremental backup should take less than 1 second
        print(f"10 incremental backups completed in {total_time:.2f} seconds")
        print(f"Average time per incremental backup: {avg_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_backup_memory_usage(self, tmp_path):
        """Test memory usage during backup operations."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large test file
        db_path = tmp_path / "large_test.db"
        with open(db_path, "wb") as f:
            # Write in chunks to avoid loading all in memory
            chunk = b"0" * (10 * 1024 * 1024)  # 10 MB chunks
            for _ in range(10):  # 100 MB total
                f.write(chunk)
        
        mock_s3_client = MagicMock(spec=S3Client)
        mock_s3_client.upload_backup = AsyncMock(return_value="s3://bucket/key")
        
        manager = BackupManager(
            database_path=db_path,
            s3_client=mock_s3_client,
            local_backup_dir=tmp_path,
            enable_scheduler=False
        )
        
        # Perform backup
        await manager.create_full_backup()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify
        assert memory_increase < 200  # Should not use more than 200 MB additional memory
        print(f"Memory usage: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB")
        print(f"Memory increase during backup: {memory_increase:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_parallel_upload_performance(self, tmp_path):
        """Test performance of parallel uploads to multiple regions."""
        # Setup
        test_file = tmp_path / "test_backup.db"
        test_file.write_bytes(b"test data" * 10000)
        
        # Create S3 client with multiple regions
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        
        mock_clients = {}
        for region in regions:
            mock_client = MagicMock()
            mock_client.upload_file = MagicMock()
            mock_clients[region] = mock_client
        
        s3_client = S3Client(
            bucket_name="test-bucket",
            replication_regions=regions[1:]
        )
        s3_client.replication_clients = mock_clients
        
        # Mock the upload method
        async def mock_upload_to_region(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate upload time
            return f"s3://bucket/{args[0]}"
        
        s3_client._upload_to_region = mock_upload_to_region
        s3_client.upload_backup = AsyncMock(return_value="s3://primary/key")
        
        # Create metadata
        metadata = BackupMetadata(
            backup_id="test",
            timestamp=datetime.utcnow(),
            size_bytes=test_file.stat().st_size,
            checksum="test_checksum",
            database_version="v1",
            backup_type="full",
            retention_policy="daily",
            source_path=str(test_file),
            destination_key=""
        )
        
        # Execute parallel upload
        start_time = time.time()
        primary_key, replica_keys = await s3_client.upload_backup_with_replication(
            file_path=test_file,
            key_prefix="backups/",
            metadata=metadata,
            parallel_upload=True
        )
        end_time = time.time()
        
        upload_time = end_time - start_time
        
        # Verify
        assert primary_key is not None
        assert len(replica_keys) == len(regions) - 1
        # Parallel upload should be faster than sequential (0.5s * 4 = 2s)
        assert upload_time < 1.5
        print(f"Parallel upload to {len(regions)} regions completed in {upload_time:.2f} seconds")


@pytest.mark.benchmark
class TestBackupBenchmarks:
    """Benchmark tests for backup operations."""
    
    @pytest.mark.asyncio
    async def test_checksum_calculation_speed(self, tmp_path, benchmark):
        """Benchmark checksum calculation speed."""
        # Create test file
        test_file = tmp_path / "test.dat"
        test_file.write_bytes(b"0" * (50 * 1024 * 1024))  # 50 MB
        
        manager = BackupManager(local_backup_dir=tmp_path, enable_scheduler=False)
        
        # Benchmark checksum calculation
        async def calculate():
            return await manager._calculate_sha256_checksum(test_file)
        
        result = benchmark(asyncio.run, calculate())
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_encryption_speed(self, tmp_path, benchmark):
        """Benchmark encryption speed."""
        # Create test file
        test_file = tmp_path / "test.dat"
        test_file.write_bytes(b"0" * (10 * 1024 * 1024))  # 10 MB
        
        manager = BackupManager(local_backup_dir=tmp_path, enable_scheduler=False)
        
        # Mock vault for encryption key
        with patch("genesis.security.vault_manager.VaultManager") as mock_vault:
            mock_vault.return_value.get_encryption_key = AsyncMock(
                return_value=b"0" * 32
            )
            
            # Benchmark encryption
            async def encrypt():
                return await manager._encrypt_and_compress_backup(
                    test_file,
                    encrypt=True,
                    compress=False
                )
            
            result = benchmark(asyncio.run, encrypt())
            assert result is not None