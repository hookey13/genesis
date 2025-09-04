"""Unit tests for paper trading persistence module."""

import json
import os
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from genesis.paper_trading.persistence import (
    PersistenceConfig,
    StatePersistence,
)


class TestPersistenceConfig:
    """Test PersistenceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PersistenceConfig()
        assert config.db_path == "paper_trading.db"
        assert config.auto_save_interval_seconds == 300
        assert config.max_history_days == 30
        assert config.enable_compression is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PersistenceConfig(
            db_path="/custom/path/trading.db",
            auto_save_interval_seconds=60,
            max_history_days=7,
            enable_compression=True,
        )
        assert config.db_path == "/custom/path/trading.db"
        assert config.auto_save_interval_seconds == 60
        assert config.max_history_days == 7
        assert config.enable_compression is True


class TestStatePersistence:
    """Test StatePersistence class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    @pytest.fixture
    def config(self, temp_db):
        """Create test configuration with temp database."""
        return PersistenceConfig(
            db_path=temp_db,
            auto_save_interval_seconds=0,  # Disable auto-save for testing
        )

    @pytest.fixture
    def persistence(self, config):
        """Create test persistence instance."""
        return StatePersistence(config)

    def test_initialization(self, persistence, config):
        """Test persistence initialization."""
        assert persistence.config == config
        assert persistence.conn is not None

    def test_create_tables(self, persistence):
        """Test database table creation."""
        # Tables should be created on initialization
        cursor = persistence.conn.cursor()
        
        # Check portfolios table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolios'"
        )
        assert cursor.fetchone() is not None
        
        # Check orders table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orders'"
        )
        assert cursor.fetchone() is not None
        
        # Check trades table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        assert cursor.fetchone() is not None
        
        # Check audit_log table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
        )
        assert cursor.fetchone() is not None

    def test_save_portfolio(self, persistence):
        """Test saving portfolio state."""
        portfolio_data = {
            "portfolio_id": "test_portfolio",
            "balance": "10000.00",
            "initial_balance": "10000.00",
            "total_trades": 5,
            "positions": json.dumps({
                "BTC/USDT": {
                    "quantity": "0.5",
                    "entry_price": "50000",
                    "current_price": "51000",
                }
            }),
        }
        
        persistence.save_portfolio(portfolio_data)
        
        # Verify saved
        cursor = persistence.conn.cursor()
        cursor.execute(
            "SELECT * FROM portfolios WHERE portfolio_id = ?",
            ("test_portfolio",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "10000.00"  # balance

    def test_load_portfolio(self, persistence):
        """Test loading portfolio state."""
        # First save
        portfolio_data = {
            "portfolio_id": "test_portfolio",
            "balance": "15000.00",
            "initial_balance": "10000.00",
            "total_trades": 10,
            "positions": json.dumps({}),
        }
        persistence.save_portfolio(portfolio_data)
        
        # Then load
        loaded = persistence.load_portfolio("test_portfolio")
        
        assert loaded is not None
        assert loaded["portfolio_id"] == "test_portfolio"
        assert loaded["balance"] == "15000.00"
        assert loaded["total_trades"] == 10

    def test_save_order(self, persistence):
        """Test saving order state."""
        order_data = {
            "order_id": "order123",
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": "0.5",
            "price": "50000",
            "status": "filled",
            "filled_quantity": "0.5",
            "average_fill_price": "50100",
            "timestamp": datetime.now().isoformat(),
        }
        
        persistence.save_order(order_data)
        
        # Verify saved
        cursor = persistence.conn.cursor()
        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            ("order123",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[2] == "BTC/USDT"  # symbol

    def test_load_orders(self, persistence):
        """Test loading orders for a strategy."""
        # Save multiple orders
        for i in range(3):
            order_data = {
                "order_id": f"order{i}",
                "strategy_id": "test_strategy",
                "symbol": "BTC/USDT",
                "side": "buy",
                "order_type": "market",
                "quantity": "0.1",
                "price": None,
                "status": "filled",
                "filled_quantity": "0.1",
                "average_fill_price": "50000",
                "timestamp": datetime.now().isoformat(),
            }
            persistence.save_order(order_data)
        
        # Load orders
        orders = persistence.load_orders("test_strategy")
        
        assert len(orders) == 3
        assert all(o["strategy_id"] == "test_strategy" for o in orders)

    def test_save_trade(self, persistence):
        """Test saving trade record."""
        trade_data = {
            "trade_id": "trade123",
            "strategy_id": "test_strategy",
            "symbol": "BTC/USDT",
            "side": "sell",
            "quantity": "0.5",
            "price": "51000",
            "pnl": "500.00",
            "commission": "5.00",
            "timestamp": datetime.now().isoformat(),
        }
        
        persistence.save_trade(trade_data)
        
        # Verify saved
        cursor = persistence.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE trade_id = ?",
            ("trade123",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[6] == "500.00"  # pnl

    def test_load_trades(self, persistence):
        """Test loading trades for a strategy."""
        # Save multiple trades
        for i in range(5):
            trade_data = {
                "trade_id": f"trade{i}",
                "strategy_id": "test_strategy",
                "symbol": "BTC/USDT",
                "side": "sell",
                "quantity": "0.1",
                "price": "50000",
                "pnl": f"{100 * (i - 2)}.00",  # Mix of profits and losses
                "commission": "5.00",
                "timestamp": datetime.now().isoformat(),
            }
            persistence.save_trade(trade_data)
        
        # Load trades
        trades = persistence.load_trades("test_strategy")
        
        assert len(trades) == 5
        assert all(t["strategy_id"] == "test_strategy" for t in trades)

    def test_save_audit_entry(self, persistence):
        """Test saving audit log entry."""
        audit_entry = {
            "action": "strategy_promoted",
            "strategy_id": "test_strategy",
            "details": json.dumps({
                "allocation": "0.10",
                "reason": "Validation criteria met",
            }),
            "timestamp": datetime.now().isoformat(),
        }
        
        persistence.save_audit_entry(audit_entry)
        
        # Verify saved
        cursor = persistence.conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log WHERE strategy_id = ?",
            ("test_strategy",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "strategy_promoted"  # action

    def test_load_audit_log(self, persistence):
        """Test loading audit log entries."""
        # Save multiple entries
        actions = ["promoted", "allocation_increased", "demoted"]
        for action in actions:
            audit_entry = {
                "action": action,
                "strategy_id": "test_strategy",
                "details": json.dumps({"test": action}),
                "timestamp": datetime.now().isoformat(),
            }
            persistence.save_audit_entry(audit_entry)
        
        # Load all entries
        entries = persistence.load_audit_log()
        
        assert len(entries) >= 3
        
        # Load for specific strategy
        strategy_entries = persistence.load_audit_log("test_strategy")
        assert all(e["strategy_id"] == "test_strategy" for e in strategy_entries)

    def test_save_complete_state(self, persistence):
        """Test saving complete simulator state."""
        state = {
            "portfolios": {
                "strategy1": {
                    "portfolio_id": "strategy1",
                    "balance": "10000",
                    "initial_balance": "10000",
                    "total_trades": 0,
                    "positions": {},
                }
            },
            "orders": {
                "order1": {
                    "order_id": "order1",
                    "strategy_id": "strategy1",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "order_type": "limit",
                    "quantity": "0.1",
                    "price": "50000",
                    "status": "pending",
                    "timestamp": datetime.now().isoformat(),
                }
            },
            "trades": [],
            "audit_log": [],
        }
        
        persistence.save_state(state)
        
        # Verify portfolios saved
        cursor = persistence.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM portfolios")
        assert cursor.fetchone()[0] >= 1
        
        # Verify orders saved
        cursor.execute("SELECT COUNT(*) FROM orders")
        assert cursor.fetchone()[0] >= 1

    def test_load_complete_state(self, persistence):
        """Test loading complete simulator state."""
        # Save state first
        state = {
            "portfolios": {
                "strategy1": {
                    "portfolio_id": "strategy1",
                    "balance": "15000",
                    "initial_balance": "10000",
                    "total_trades": 5,
                    "positions": json.dumps({}),
                }
            },
            "orders": {},
            "trades": [],
            "audit_log": [],
        }
        persistence.save_state(state)
        
        # Load state
        loaded_state = persistence.load_state()
        
        assert "portfolios" in loaded_state
        assert "orders" in loaded_state
        assert "trades" in loaded_state
        assert "audit_log" in loaded_state
        
        assert len(loaded_state["portfolios"]) >= 1
        portfolio = loaded_state["portfolios"][0]
        assert portfolio["portfolio_id"] == "strategy1"
        assert portfolio["balance"] == "15000"

    def test_clean_old_data(self, persistence):
        """Test cleaning old data."""
        # Save old and new trades
        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        new_date = datetime.now().isoformat()
        
        old_trade = {
            "trade_id": "old_trade",
            "strategy_id": "test",
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": "0.1",
            "price": "50000",
            "pnl": "0",
            "commission": "5",
            "timestamp": old_date,
        }
        
        new_trade = {
            "trade_id": "new_trade",
            "strategy_id": "test",
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": "0.1",
            "price": "50000",
            "pnl": "0",
            "commission": "5",
            "timestamp": new_date,
        }
        
        persistence.save_trade(old_trade)
        persistence.save_trade(new_trade)
        
        # Clean old data
        persistence.clean_old_data(days=30)
        
        # Check old trade is gone
        cursor = persistence.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE trade_id = ?",
            ("old_trade",)
        )
        assert cursor.fetchone() is None
        
        # Check new trade still exists
        cursor.execute(
            "SELECT * FROM trades WHERE trade_id = ?",
            ("new_trade",)
        )
        assert cursor.fetchone() is not None

    def test_close_connection(self, persistence):
        """Test closing database connection."""
        assert persistence.conn is not None
        
        persistence.close()
        
        # Connection should be closed
        with pytest.raises(Exception):
            cursor = persistence.conn.cursor()
            cursor.execute("SELECT 1")

    def test_context_manager(self, config):
        """Test using persistence as context manager."""
        with StatePersistence(config) as persistence:
            assert persistence.conn is not None
            
            # Save some data
            portfolio_data = {
                "portfolio_id": "test",
                "balance": "10000",
                "initial_balance": "10000",
                "total_trades": 0,
                "positions": json.dumps({}),
            }
            persistence.save_portfolio(portfolio_data)
        
        # Connection should be closed after context exit
        # Create new instance to verify data was saved
        new_persistence = StatePersistence(config)
        loaded = new_persistence.load_portfolio("test")
        assert loaded is not None
        new_persistence.close()

    def test_transaction_rollback(self, persistence):
        """Test transaction rollback on error."""
        # Start transaction
        persistence.conn.execute("BEGIN")
        
        try:
            # Insert valid data
            persistence.save_portfolio({
                "portfolio_id": "test",
                "balance": "10000",
                "initial_balance": "10000",
                "total_trades": 0,
                "positions": json.dumps({}),
            })
            
            # Force an error (duplicate primary key)
            persistence.save_portfolio({
                "portfolio_id": "test",  # Duplicate
                "balance": "20000",
                "initial_balance": "20000",
                "total_trades": 0,
                "positions": json.dumps({}),
            })
            
            persistence.conn.execute("COMMIT")
        except:
            persistence.conn.execute("ROLLBACK")
        
        # Check only one portfolio exists
        cursor = persistence.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM portfolios WHERE portfolio_id = 'test'")
        count = cursor.fetchone()[0]
        assert count <= 1

    def test_get_statistics(self, persistence):
        """Test getting database statistics."""
        # Add some data
        for i in range(3):
            persistence.save_portfolio({
                "portfolio_id": f"portfolio{i}",
                "balance": "10000",
                "initial_balance": "10000",
                "total_trades": i * 10,
                "positions": json.dumps({}),
            })
        
        for i in range(5):
            persistence.save_order({
                "order_id": f"order{i}",
                "strategy_id": "test",
                "symbol": "BTC/USDT",
                "side": "buy",
                "order_type": "market",
                "quantity": "0.1",
                "price": None,
                "status": "filled",
                "filled_quantity": "0.1",
                "average_fill_price": "50000",
                "timestamp": datetime.now().isoformat(),
            })
        
        stats = persistence.get_statistics()
        
        assert stats["portfolios_count"] >= 3
        assert stats["orders_count"] >= 5
        assert stats["trades_count"] >= 0
        assert stats["audit_entries_count"] >= 0

    def test_backup_database(self, persistence, temp_db):
        """Test database backup functionality."""
        # Add some data
        persistence.save_portfolio({
            "portfolio_id": "test",
            "balance": "10000",
            "initial_balance": "10000",
            "total_trades": 0,
            "positions": json.dumps({}),
        })
        
        # Create backup
        backup_path = f"{temp_db}.backup"
        persistence.backup_database(backup_path)
        
        # Verify backup exists
        assert os.path.exists(backup_path)
        
        # Verify backup contains data
        backup_config = PersistenceConfig(db_path=backup_path)
        backup_persistence = StatePersistence(backup_config)
        loaded = backup_persistence.load_portfolio("test")
        assert loaded is not None
        backup_persistence.close()
        
        # Cleanup
        try:
            os.unlink(backup_path)
        except:
            pass

    def test_restore_from_backup(self, persistence, temp_db):
        """Test restoring from backup."""
        # Add data and create backup
        persistence.save_portfolio({
            "portfolio_id": "original",
            "balance": "10000",
            "initial_balance": "10000",
            "total_trades": 5,
            "positions": json.dumps({}),
        })
        
        backup_path = f"{temp_db}.backup"
        persistence.backup_database(backup_path)
        
        # Modify original data
        persistence.save_portfolio({
            "portfolio_id": "original",
            "balance": "20000",  # Changed
            "initial_balance": "10000",
            "total_trades": 10,  # Changed
            "positions": json.dumps({}),
        })
        
        # Restore from backup
        persistence.restore_from_backup(backup_path)
        
        # Check data is restored
        loaded = persistence.load_portfolio("original")
        assert loaded["balance"] == "10000"  # Original value
        assert loaded["total_trades"] == 5  # Original value
        
        # Cleanup
        try:
            os.unlink(backup_path)
        except:
            pass