"""Integration tests for recovery engine."""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from genesis.backup.backup_manager import BackupManager
from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.recovery.recovery_engine import RecoveryEngine


class TestRecoveryEngine:
    """Integration tests for recovery engine."""
    
    @pytest.fixture
    def test_database(self, tmp_path):
        """Create test database with sample data."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        
        # Create tables
        conn.executescript("""
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                event_data TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            
            CREATE TABLE positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                quantity REAL,
                status TEXT,
                opened_at TEXT,
                strategy_id TEXT,
                unrealized_pnl REAL,
                realized_pnl REAL,
                current_price REAL,
                exit_price REAL,
                closed_at TEXT
            );
            
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                client_order_id TEXT UNIQUE,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                quantity REAL,
                price REAL,
                status TEXT,
                created_at TEXT,
                filled_quantity REAL,
                executed_price REAL,
                filled_at TEXT,
                cancelled_at TEXT,
                position_id TEXT
            );
            
            CREATE TABLE balances (
                asset TEXT PRIMARY KEY,
                free_balance REAL,
                locked_balance REAL,
                total_balance REAL,
                updated_at TEXT
            );
            
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );
            
            CREATE TABLE tier_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_tier TEXT,
                new_tier TEXT,
                reason TEXT,
                changed_at TEXT
            );
            
            CREATE TABLE risk_limits (
                limit_type TEXT PRIMARY KEY,
                limit_value REAL,
                tier TEXT,
                updated_at TEXT
            );
            
            CREATE TABLE checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                state_hash TEXT,
                created_at TEXT
            );
            
            PRAGMA user_version = 1;
        """)
        
        # Insert sample data
        now = datetime.utcnow()
        
        # Add events
        events = [
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "BALANCE_UPDATED",
                "aggregate_id": "balance_1",
                "event_data": json.dumps({
                    "asset": "USDT",
                    "free_balance": 10000,
                    "locked_balance": 0,
                    "total_balance": 10000
                }),
                "sequence_number": 1,
                "created_at": (now - timedelta(hours=2)).isoformat()
            },
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "POSITION_OPENED",
                "aggregate_id": "pos_1",
                "event_data": json.dumps({
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "entry_price": 50000,
                    "quantity": 0.1,
                    "strategy_id": "strategy_1"
                }),
                "sequence_number": 2,
                "created_at": (now - timedelta(hours=1)).isoformat()
            },
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "CHECKPOINT",
                "aggregate_id": "checkpoint_1",
                "event_data": json.dumps({
                    "state_hash": "abc123"
                }),
                "sequence_number": 3,
                "created_at": (now - timedelta(minutes=30)).isoformat()
            }
        ]
        
        for event in events:
            conn.execute("""
                INSERT INTO events (event_id, event_type, aggregate_id, 
                                  event_data, sequence_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event["event_id"],
                event["event_type"],
                event["aggregate_id"],
                event["event_data"],
                event["sequence_number"],
                event["created_at"]
            ))
        
        # Add initial balance
        conn.execute("""
            INSERT INTO balances (asset, free_balance, locked_balance, total_balance, updated_at)
            VALUES ('USDT', 10000, 0, 10000, ?)
        """, (now.isoformat(),))
        
        # Add system state
        conn.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES ('current_tier', 'sniper', ?)
        """, (now.isoformat(),))
        
        conn.commit()
        conn.close()
        
        return db_path
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        client = Mock(spec=S3Client)
        client.download_backup = AsyncMock()
        client._calculate_checksum = AsyncMock(return_value="test_checksum")
        return client
    
    @pytest.fixture
    def mock_backup_manager(self, mock_s3_client):
        """Create mock backup manager."""
        manager = Mock(spec=BackupManager)
        manager.s3_client = mock_s3_client
        
        # Mock backup metadata
        full_backup = BackupMetadata(
            backup_id="full_1",
            timestamp=datetime.utcnow() - timedelta(hours=3),
            size_bytes=1024,
            checksum="abc123",
            database_version="v1",
            backup_type="full",
            retention_policy="daily",
            source_path="",
            destination_key="backups/full/test.db"
        )
        
        manager.get_backup_for_timestamp = AsyncMock(return_value=(full_backup, []))
        
        return manager
    
    @pytest.fixture
    def recovery_engine(self, mock_backup_manager, test_database, tmp_path):
        """Create recovery engine instance."""
        staging_dir = tmp_path / "recovery"
        return RecoveryEngine(
            backup_manager=mock_backup_manager,
            database_path=test_database,
            recovery_staging_dir=staging_dir
        )
    
    @pytest.mark.asyncio
    async def test_recover_to_timestamp(self, recovery_engine, test_database, tmp_path):
        """Test point-in-time recovery."""
        target_time = datetime.utcnow() - timedelta(minutes=15)
        
        # Mock staging recovery
        staging_db = tmp_path / "recovery" / f"recovery_test.db"
        shutil.copy2(test_database, staging_db)
        
        with patch.object(recovery_engine, "_stage_recovery", return_value=staging_db):
            with patch.object(recovery_engine, "_apply_recovery"):
                result = await recovery_engine.recover_to_timestamp(
                    target_timestamp=target_time,
                    validate=False,
                    dry_run=False
                )
        
        assert result["success"] is True
        assert result["target_timestamp"] == target_time.isoformat()
        assert result["full_backup_used"] == "full_1"
        assert result["recovery_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_recover_positions(self, recovery_engine, test_database):
        """Test position recovery from events."""
        # Add position events to database
        conn = sqlite3.connect(str(test_database))
        
        events = [
            ("POSITION_OPENED", "pos_1", {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 50000,
                "quantity": 0.1
            }),
            ("POSITION_UPDATED", "pos_1", {
                "current_price": 51000,
                "unrealized_pnl": 100
            })
        ]
        
        for i, (event_type, aggregate_id, event_data) in enumerate(events, start=10):
            conn.execute("""
                INSERT INTO events (event_id, event_type, aggregate_id, 
                                  event_data, sequence_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                event_type,
                aggregate_id,
                json.dumps(event_data),
                i,
                datetime.utcnow().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        # Recover positions
        result = await recovery_engine.recover_positions()
        
        assert result["total_positions"] >= 1
        assert "pos_1" in result["positions"]
        assert result["positions"]["pos_1"]["status"] == "open"
        assert result["positions"]["pos_1"]["unrealized_pnl"] == 100
    
    @pytest.mark.asyncio
    async def test_verify_order_reconciliation(self, recovery_engine, test_database):
        """Test order reconciliation with exchange."""
        # Add orders to database
        conn = sqlite3.connect(str(test_database))
        conn.execute("""
            INSERT INTO orders (order_id, client_order_id, status, symbol, 
                              side, quantity, price)
            VALUES ('order_1', 'client_1', 'new', 'BTC/USDT', 'buy', 0.1, 50000)
        """)
        conn.commit()
        conn.close()
        
        # Mock exchange orders
        exchange_orders = [
            {
                "orderId": "order_1",
                "clientOrderId": "client_1",
                "status": "NEW",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "origQty": "0.1",
                "price": "50000"
            },
            {
                "orderId": "order_2",
                "clientOrderId": "client_2",
                "status": "NEW",
                "symbol": "ETHUSDT",
                "side": "BUY",
                "origQty": "1",
                "price": "3000"
            }
        ]
        
        result = await recovery_engine.verify_order_reconciliation(exchange_orders)
        
        assert result["is_reconciled"] is False
        assert len(result["missing_in_db"]) == 1
        assert result["missing_in_db"][0]["clientOrderId"] == "client_2"
        assert result["db_order_count"] == 1
        assert result["exchange_order_count"] == 2
    
    @pytest.mark.asyncio
    async def test_validate_recovery(self, recovery_engine, test_database):
        """Test recovery validation."""
        # Test validation
        validation = await recovery_engine._validate_recovery(
            test_database,
            datetime.utcnow()
        )
        
        assert validation["is_valid"] is True
        assert len(validation["errors"]) == 0
        assert "database_path" in validation
    
    @pytest.mark.asyncio
    async def test_dry_run_recovery(self, recovery_engine, test_database, tmp_path):
        """Test dry run recovery."""
        target_time = datetime.utcnow() - timedelta(minutes=15)
        
        # Mock staging
        staging_db = tmp_path / "recovery" / "test_recovery.db"
        shutil.copy2(test_database, staging_db)
        
        with patch.object(recovery_engine, "_stage_recovery", return_value=staging_db):
            result = await recovery_engine.recover_to_timestamp(
                target_timestamp=target_time,
                validate=False,
                dry_run=True
            )
        
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["recovered_db_path"] is not None
        
        # Verify original database unchanged
        assert test_database.exists()
    
    @pytest.mark.asyncio
    async def test_recovery_with_incrementals(self, recovery_engine, mock_backup_manager):
        """Test recovery with incremental backups."""
        # Add incremental backups
        incremental = BackupMetadata(
            backup_id="inc_1",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            size_bytes=512,
            checksum="def456",
            database_version="v1",
            backup_type="incremental",
            retention_policy="hourly",
            source_path="",
            destination_key="backups/incremental/test.wal"
        )
        
        full_backup = mock_backup_manager.get_backup_for_timestamp.return_value[0]
        mock_backup_manager.get_backup_for_timestamp.return_value = (
            full_backup,
            [incremental]
        )
        
        target_time = datetime.utcnow() - timedelta(minutes=30)
        
        with patch.object(recovery_engine, "_stage_recovery") as mock_stage:
            with patch.object(recovery_engine, "_apply_recovery"):
                await recovery_engine.recover_to_timestamp(
                    target_timestamp=target_time,
                    validate=False
                )
        
        # Verify incremental was included
        mock_stage.assert_called_once()
        call_args = mock_stage.call_args[0]
        assert len(call_args[1]) == 1  # One incremental backup
        assert call_args[1][0].backup_id == "inc_1"
    
    @pytest.mark.asyncio
    async def test_recovery_status(self, recovery_engine):
        """Test getting recovery status."""
        status = recovery_engine.get_recovery_status()
        
        assert status["recovery_in_progress"] is False
        assert status["last_recovery_timestamp"] is None
        assert "staging_directory" in status
    
    @pytest.mark.asyncio
    async def test_concurrent_recovery_prevented(self, recovery_engine):
        """Test concurrent recovery is prevented."""
        recovery_engine.recovery_in_progress = True
        
        with pytest.raises(Exception) as exc_info:
            await recovery_engine.recover_to_timestamp(datetime.utcnow())
        
        assert "Recovery already in progress" in str(exc_info.value)


import shutil  # Add this import at the top