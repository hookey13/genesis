"""Integration tests for the automated backup system."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from freezegun import freeze_time

from genesis.backup.backup_manager import BackupManager
from genesis.backup.config_backup import ConfigBackupManager
from genesis.backup.recovery_manager import RecoveryManager
from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.backup.state_backup import StateBackupManager, TradingStateSnapshot
from genesis.backup.vault_backup import VaultBackupManager
from genesis.core.exceptions import BackupError
from genesis.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def temp_backup_dir():
    """Create temporary backup directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    client = MagicMock(spec=S3Client)
    client.upload_backup = AsyncMock(return_value="s3://bucket/key")
    client.upload_backup_with_replication = AsyncMock(
        return_value=("s3://bucket/key", ["s3://replica1/key", "s3://replica2/key"])
    )
    client.download_backup = AsyncMock()
    client.list_backups = AsyncMock(return_value=[])
    client.apply_retention_policy = AsyncMock(return_value={"hourly": 5, "daily": 2})
    return client


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector."""
    collector = MagicMock(spec=MetricsCollector)
    collector.record_backup_metric = AsyncMock()
    return collector


@pytest.fixture
def postgres_config():
    """PostgreSQL configuration for testing."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "genesis_test",
        "user": "genesis",
        "password": "test_password"
    }


class TestBackupManager:
    """Test backup manager functionality."""
    
    @pytest.mark.asyncio
    async def test_create_full_backup_sqlite(self, temp_backup_dir, mock_s3_client):
        """Test creating full SQLite backup."""
        # Setup
        db_path = temp_backup_dir / "test.db"
        db_path.touch()
        
        manager = BackupManager(
            database_path=db_path,
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            backup_interval_minutes=15,
            enable_scheduler=False,
            database_type="sqlite"
        )
        
        # Execute
        metadata = await manager.create_full_backup()
        
        # Verify
        assert metadata.backup_type == "full"
        assert metadata.database_type == "sqlite"
        assert metadata.encrypted is True
        assert metadata.compressed is True
        mock_s3_client.upload_backup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_full_backup_postgresql(self, temp_backup_dir, mock_s3_client, postgres_config):
        """Test creating full PostgreSQL backup."""
        # Setup
        manager = BackupManager(
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            backup_interval_minutes=15,
            enable_scheduler=False,
            database_type="postgresql",
            postgres_config=postgres_config
        )
        
        # Mock pg_dump command
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = MagicMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            # Execute
            metadata = await manager.create_full_backup()
            
            # Verify
            assert metadata.backup_type == "full"
            assert metadata.database_type == "postgresql"
            mock_subprocess.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_backup_encryption_and_compression(self, temp_backup_dir):
        """Test backup encryption and compression."""
        # Setup
        test_file = temp_backup_dir / "test_data.txt"
        test_file.write_text("Test data for encryption and compression")
        
        manager = BackupManager(
            local_backup_dir=temp_backup_dir,
            enable_scheduler=False
        )
        
        # Execute
        with patch("genesis.security.vault_manager.VaultManager") as mock_vault:
            mock_vault.return_value.get_encryption_key = AsyncMock(
                return_value=b"0" * 32  # 32 bytes for AES-256
            )
            
            processed_file = await manager._encrypt_and_compress_backup(
                test_file,
                encrypt=True,
                compress=True
            )
            
            # Verify
            assert processed_file.suffix == ".gz"
            assert processed_file.exists()
            assert not test_file.exists()  # Original should be deleted
    
    @pytest.mark.asyncio
    async def test_automated_backup_scheduling(self, temp_backup_dir, mock_s3_client):
        """Test automated backup scheduling."""
        # Setup
        db_path = temp_backup_dir / "test.db"
        db_path.touch()
        
        manager = BackupManager(
            database_path=db_path,
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            backup_interval_minutes=15,
            enable_scheduler=True
        )
        
        # Verify scheduler is configured
        assert manager.scheduler is not None
        assert len(manager.scheduler.get_jobs()) == 3  # full, incremental, retention
        
        # Check job intervals
        full_job = manager.scheduler.get_job("full_backup")
        assert full_job is not None
    
    @pytest.mark.asyncio
    async def test_backup_integrity_verification(self, temp_backup_dir, mock_s3_client):
        """Test backup integrity verification."""
        # Setup
        test_file = temp_backup_dir / "test_backup.db"
        test_file.write_bytes(b"Test backup data")
        
        metadata = BackupMetadata(
            backup_id="test_123",
            timestamp=datetime.utcnow(),
            size_bytes=len(test_file.read_bytes()),
            checksum="abc123",
            database_version="v1",
            backup_type="full",
            retention_policy="daily",
            source_path=str(test_file),
            destination_key="s3://bucket/test_123.db"
        )
        
        manager = BackupManager(
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            enable_scheduler=False
        )
        
        # Mock download with correct checksum
        mock_s3_client.download_backup.return_value = metadata
        mock_s3_client._calculate_checksum = AsyncMock(return_value="abc123")
        
        # Execute
        is_valid = await manager.verify_backup_integrity(metadata)
        
        # Verify
        assert is_valid is True
        mock_s3_client.download_backup.assert_called_once()


class TestVaultBackupManager:
    """Test Vault backup functionality."""
    
    @pytest.mark.asyncio
    async def test_create_vault_snapshot(self, temp_backup_dir, mock_s3_client):
        """Test creating Vault snapshot."""
        # Setup
        with patch("hvac.Client") as mock_hvac:
            mock_client = MagicMock()
            mock_client.is_authenticated.return_value = True
            mock_client.adapter.get.return_value = MagicMock(
                status_code=200,
                iter_content=lambda chunk_size: [b"snapshot_data"]
            )
            mock_hvac.return_value = mock_client
            
            manager = VaultBackupManager(
                s3_client=mock_s3_client,
                backup_dir=temp_backup_dir
            )
            
            # Execute
            metadata = await manager.create_vault_snapshot()
            
            # Verify
            assert metadata.backup_type == "vault_snapshot"
            assert metadata.database_version == "vault"
            mock_s3_client.upload_backup_with_replication.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_backup_seal_keys(self, temp_backup_dir):
        """Test Vault seal key backup."""
        # Setup
        with patch("hvac.Client") as mock_hvac:
            mock_client = MagicMock()
            mock_client.is_authenticated.return_value = True
            mock_client.sys.read_seal_status.return_value = {
                "sealed": False,
                "initialized": True,
                "type": "shamir",
                "recovery_shares": 5,
                "recovery_threshold": 3
            }
            mock_client.secrets.kv.v2.create_or_update_secret = MagicMock()
            mock_hvac.return_value = mock_client
            
            manager = VaultBackupManager(backup_dir=temp_backup_dir)
            
            # Execute
            backup_info = await manager.backup_seal_keys()
            
            # Verify
            assert backup_info["seal_type"] == "shamir"
            assert backup_info["recovery_shares"] == 5
            assert backup_info["recovery_threshold"] == 3
            assert "private_key_path" in backup_info
            
            # Check private key file was created
            private_key_path = Path(backup_info["private_key_path"])
            assert private_key_path.exists()


class TestConfigBackupManager:
    """Test configuration backup functionality."""
    
    @pytest.mark.asyncio
    async def test_create_config_backup(self, temp_backup_dir, mock_s3_client):
        """Test creating configuration backup."""
        # Setup config files
        config_dir = temp_backup_dir / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "settings.yaml"
        config_file.write_text("database:\n  host: localhost\n  port: 5432")
        
        env_file = temp_backup_dir / ".env"
        env_file.write_text("API_KEY=secret123\nDATABASE_URL=postgresql://localhost")
        
        manager = ConfigBackupManager(
            config_dirs=[config_dir, env_file],
            s3_client=mock_s3_client,
            backup_dir=temp_backup_dir
        )
        
        # Execute
        metadata = await manager.create_config_backup()
        
        # Verify
        assert metadata.backup_type == "configuration"
        assert len(manager.version_history) == 1
        mock_s3_client.upload_backup_with_replication.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compare_configs(self, temp_backup_dir, mock_s3_client):
        """Test configuration comparison."""
        # Setup
        manager = ConfigBackupManager(
            s3_client=mock_s3_client,
            backup_dir=temp_backup_dir
        )
        
        # Create mock config backups
        config1 = {"version": "v1", "files": {"config.yaml": {"checksum": "abc123"}}}
        config2 = {"version": "v2", "files": {"config.yaml": {"checksum": "def456"}}}
        
        with patch.object(manager, "_extract_manifest") as mock_extract:
            mock_extract.side_effect = [config1, config2]
            mock_s3_client.download_backup = AsyncMock()
            
            # Execute
            diff = await manager.compare_configs("backup1", "backup2")
            
            # Verify
            assert diff["version1"] == "v1"
            assert diff["version2"] == "v2"
            assert "differences" in diff


class TestStateBackupManager:
    """Test trading state backup functionality."""
    
    @pytest.mark.asyncio
    async def test_create_state_snapshot(self, temp_backup_dir, mock_s3_client):
        """Test creating trading state snapshot."""
        # Setup
        trading_state = {
            "tier": "sniper",
            "positions": [
                {"symbol": "BTC/USDT", "size": 0.5, "entry_price": 50000}
            ],
            "open_orders": [
                {"symbol": "ETH/USDT", "side": "BUY", "amount": 1.0}
            ],
            "balances": {"USDT": Decimal("10000"), "BTC": Decimal("0.5")},
            "risk_metrics": {"var": 0.05, "sharpe": 1.2},
            "tilt_status": {"score": 0.3, "level": "low"},
            "session_id": "session_123",
            "pnl_total": 1500.50,
            "pnl_today": 250.25
        }
        
        manager = StateBackupManager(
            s3_client=mock_s3_client,
            backup_dir=temp_backup_dir
        )
        
        # Execute
        metadata = await manager.create_state_snapshot(trading_state)
        
        # Verify
        assert metadata.backup_type == "trading_state"
        assert manager.last_snapshot is not None
        assert len(manager.last_snapshot.positions) == 1
        assert len(manager.last_snapshot.open_orders) == 1
        mock_s3_client.upload_backup_with_replication.assert_called()
    
    @pytest.mark.asyncio
    async def test_recover_state(self, temp_backup_dir, mock_s3_client):
        """Test recovering trading state from snapshot."""
        # Setup
        snapshot = TradingStateSnapshot(
            timestamp=datetime.utcnow(),
            tier="sniper",
            positions=[],
            open_orders=[],
            balances={},
            risk_metrics={},
            tilt_status={},
            session_id="test_session",
            pnl_total=Decimal("1000"),
            pnl_today=Decimal("100")
        )
        
        # Save snapshot as pickle
        import pickle
        snapshot_file = temp_backup_dir / "test_snapshot.pkl"
        with open(snapshot_file, "wb") as f:
            pickle.dump(snapshot, f)
        
        mock_s3_client.download_backup = AsyncMock()
        
        manager = StateBackupManager(
            s3_client=mock_s3_client,
            backup_dir=temp_backup_dir
        )
        
        with patch.object(Path, "unlink"):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = snapshot_file.read_bytes()
                
                # Execute
                with patch("pickle.load", return_value=snapshot):
                    recovered = await manager.recover_state("test_snapshot", use_pickle=True)
                
                # Verify
                assert recovered.session_id == "test_session"
                assert recovered.pnl_total == Decimal("1000")
                assert manager.recovery_point == snapshot.timestamp
    
    @pytest.mark.asyncio
    async def test_reconcile_positions(self, temp_backup_dir):
        """Test position reconciliation."""
        # Setup
        snapshot = TradingStateSnapshot(
            timestamp=datetime.utcnow(),
            tier="sniper",
            positions=[
                {"symbol": "BTC/USDT", "size": "0.5"},
                {"symbol": "ETH/USDT", "size": "2.0"}
            ],
            open_orders=[],
            balances={},
            risk_metrics={},
            tilt_status={},
            session_id="test",
            pnl_total=Decimal("0"),
            pnl_today=Decimal("0")
        )
        
        current_positions = [
            {"symbol": "BTC/USDT", "size": "0.6"},  # Size mismatch
            {"symbol": "SOL/USDT", "size": "10.0"}  # Extra position
        ]
        
        manager = StateBackupManager(backup_dir=temp_backup_dir)
        
        # Execute
        reconciliation = await manager.reconcile_positions(snapshot, current_positions)
        
        # Verify
        assert len(reconciliation["missing_in_current"]) == 1  # ETH/USDT
        assert len(reconciliation["extra_in_current"]) == 1  # SOL/USDT
        assert len(reconciliation["size_mismatches"]) == 1  # BTC/USDT
        assert reconciliation["needs_adjustment"] is True


class TestRecoveryManager:
    """Test recovery orchestration."""
    
    @pytest.mark.asyncio
    async def test_full_recovery(self, temp_backup_dir, mock_s3_client, mock_metrics_collector):
        """Test complete system recovery."""
        # Setup
        recovery_manager = RecoveryManager(
            s3_client=mock_s3_client,
            metrics_collector=mock_metrics_collector
        )
        
        # Mock backup listings
        mock_s3_client.list_backups.return_value = [
            BackupMetadata(
                backup_id="db_backup",
                timestamp=datetime.utcnow() - timedelta(hours=1),
                size_bytes=1024,
                checksum="abc123",
                database_version="v1",
                backup_type="full",
                retention_policy="daily",
                source_path="test.db",
                destination_key="s3://bucket/db_backup"
            )
        ]
        
        with patch.object(recovery_manager, "_recover_database") as mock_db:
            with patch.object(recovery_manager, "_recover_vault") as mock_vault:
                with patch.object(recovery_manager, "_recover_config") as mock_config:
                    with patch.object(recovery_manager, "_recover_state") as mock_state:
                        mock_db.return_value = {"component": "database", "status": "success"}
                        mock_vault.return_value = {"component": "vault", "status": "success"}
                        mock_config.return_value = {"component": "config", "status": "success"}
                        mock_state.return_value = {"component": "state", "status": "success"}
                        
                        # Execute
                        target_time = datetime.utcnow() - timedelta(hours=2)
                        report = await recovery_manager.perform_full_recovery(
                            target_timestamp=target_time,
                            dry_run=True
                        )
                        
                        # Verify
                        assert report["status"] == "dry_run"
                        assert len(report["results"]) == 4
                        assert report["rto_achieved"] is True  # Should complete quickly in dry run
                        mock_metrics_collector.record_backup_metric.assert_called()
    
    @pytest.mark.asyncio
    async def test_recovery_metrics(self, temp_backup_dir, mock_s3_client):
        """Test recovery metrics measurement."""
        # Setup
        recovery_manager = RecoveryManager(s3_client=mock_s3_client)
        
        # Mock backup listings with recent backups
        mock_s3_client.list_backups.return_value = [
            BackupMetadata(
                backup_id=f"backup_{i}",
                timestamp=datetime.utcnow() - timedelta(minutes=i*5),
                size_bytes=1024,
                checksum=f"checksum_{i}",
                database_version="v1",
                backup_type="full",
                retention_policy="hourly",
                source_path="test",
                destination_key=f"s3://bucket/backup_{i}"
            )
            for i in range(3)
        ]
        
        with patch.object(recovery_manager, "test_recovery_procedure") as mock_test:
            mock_test.return_value = {
                "all_recoverable": True,
                "estimated_recovery_time": 300,  # 5 minutes
                "meets_rto": True
            }
            
            # Execute
            metrics = await recovery_manager.measure_recovery_metrics()
            
            # Verify
            assert "rpo_metrics" in metrics
            assert metrics["rto_meets_target"] is True
            assert metrics["all_components_recoverable"] is True


@pytest.mark.asyncio
class TestBackupIntegration:
    """End-to-end integration tests."""
    
    async def test_backup_and_recovery_cycle(self, temp_backup_dir, mock_s3_client):
        """Test complete backup and recovery cycle."""
        # Create test database
        db_path = temp_backup_dir / "test.db"
        db_path.write_text("test database content")
        
        # Setup managers
        backup_manager = BackupManager(
            database_path=db_path,
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            enable_scheduler=False
        )
        
        recovery_manager = RecoveryManager(
            backup_manager=backup_manager,
            s3_client=mock_s3_client
        )
        
        # Create backup
        backup_metadata = await backup_manager.create_full_backup()
        assert backup_metadata is not None
        
        # Simulate recovery
        with patch.object(recovery_manager, "_recover_database") as mock_recover:
            mock_recover.return_value = {"status": "success"}
            
            report = await recovery_manager.perform_full_recovery(
                target_timestamp=datetime.utcnow(),
                components=["database"],
                dry_run=True
            )
            
            assert report["status"] == "dry_run"
            assert "database" in report["results"]
    
    @freeze_time("2025-01-01 12:00:00")
    async def test_retention_policy_application(self, temp_backup_dir, mock_s3_client):
        """Test backup retention policy."""
        # Setup
        manager = BackupManager(
            s3_client=mock_s3_client,
            local_backup_dir=temp_backup_dir,
            enable_scheduler=False
        )
        
        # Execute retention policy
        deleted = await manager.apply_retention_policy()
        
        # Verify
        mock_s3_client.apply_retention_policy.assert_called_once_with(
            prefix="backups/",
            hourly_days=7,
            daily_days=30,
            monthly_days=365
        )
    
    async def test_cross_region_replication(self, temp_backup_dir):
        """Test cross-region backup replication."""
        # Setup S3 client with replication
        s3_client = S3Client(
            endpoint_url="https://s3.amazonaws.com",
            access_key="test_key",
            secret_key="test_secret",
            bucket_name="test-bucket",
            replication_regions=["us-west-2", "eu-west-1"]
        )
        
        # Mock boto3 clients
        with patch("boto3.client") as mock_boto:
            mock_client = MagicMock()
            mock_client.create_bucket = MagicMock()
            mock_client.put_bucket_replication = MagicMock()
            mock_boto.return_value = mock_client
            
            # Verify replication setup
            assert len(s3_client.replication_regions) == 2
            assert len(s3_client.replication_clients) == 2
    
    async def test_backup_monitoring_integration(self, temp_backup_dir, mock_metrics_collector):
        """Test backup monitoring and alerting."""
        # Setup
        db_path = temp_backup_dir / "test.db"
        db_path.touch()
        
        with patch("genesis.backup.s3_client.S3Client") as mock_s3:
            mock_s3.return_value.upload_backup = AsyncMock(return_value="s3://bucket/key")
            
            manager = BackupManager(
                database_path=db_path,
                local_backup_dir=temp_backup_dir,
                enable_scheduler=False
            )
            
            # Create backup and record metrics
            metadata = await manager.create_full_backup()
            
            # Record backup metrics
            await mock_metrics_collector.record_backup_metric(
                "backup_success",
                metadata.size_bytes / 1024 / 1024  # Size in MB
            )
            
            await mock_metrics_collector.record_backup_metric(
                "backup_completion_time",
                5.0  # 5 seconds
            )
            
            # Verify metrics were recorded
            assert mock_metrics_collector.record_backup_metric.call_count == 2