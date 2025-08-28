"""Unit tests for TWAP execution tables migration."""

import os
import tempfile
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config


class TestMigration008:
    """Test TWAP execution tables migration."""

    @pytest.fixture
    def alembic_config(self):
        """Create Alembic config for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        config = Config()
        config.set_main_option("script_location", "alembic")
        config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        yield config

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def db_session(self, alembic_config):
        """Create database session for testing."""
        # Run migrations up to 008
        command.upgrade(alembic_config, "008")

        # Create session
        engine = create_engine(alembic_config.get_main_option("sqlalchemy.url"))
        Session = sessionmaker(bind=engine)
        session = Session()

        yield session

        session.close()
        engine.dispose()

    def test_twap_executions_table_exists(self, db_session):
        """Test that twap_executions table is created."""
        result = db_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='twap_executions'"
            )
        )
        assert result.fetchone() is not None

    def test_twap_slice_history_table_exists(self, db_session):
        """Test that twap_slice_history table is created."""
        result = db_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='twap_slice_history'"
            )
        )
        assert result.fetchone() is not None

    def test_twap_executions_columns(self, db_session):
        """Test twap_executions table has all required columns."""
        result = db_session.execute(text("PRAGMA table_info(twap_executions)"))
        columns = {row[1] for row in result}

        expected_columns = {
            "execution_id",
            "symbol",
            "side",
            "total_quantity",
            "duration_minutes",
            "slice_count",
            "executed_quantity",
            "remaining_quantity",
            "arrival_price",
            "average_price",
            "twap_price",
            "implementation_shortfall",
            "participation_rate",
            "status",
            "early_completion",
            "early_completion_reason",
            "started_at",
            "completed_at",
            "paused_at",
            "resumed_at",
            "created_at",
            "updated_at",
        }

        assert expected_columns.issubset(columns)

    def test_twap_slice_history_columns(self, db_session):
        """Test twap_slice_history table has all required columns."""
        result = db_session.execute(text("PRAGMA table_info(twap_slice_history)"))
        columns = {row[1] for row in result}

        expected_columns = {
            "slice_id",
            "execution_id",
            "slice_number",
            "target_quantity",
            "executed_quantity",
            "execution_price",
            "market_price",
            "slippage_bps",
            "volume_at_execution",
            "participation_rate",
            "market_impact_bps",
            "time_delay_ms",
            "order_id",
            "client_order_id",
            "status",
            "error_message",
            "executed_at",
            "created_at",
        }

        assert expected_columns.issubset(columns)

    def test_twap_executions_indexes(self, db_session):
        """Test that proper indexes exist on twap_executions."""
        result = db_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='twap_executions'"
            )
        )
        indexes = {row[0] for row in result}

        expected_indexes = {
            "idx_twap_executions_symbol",
            "idx_twap_executions_status",
            "idx_twap_executions_started_at",
            "idx_twap_executions_symbol_status",
        }

        assert expected_indexes.issubset(indexes)

    def test_twap_slice_history_indexes(self, db_session):
        """Test that proper indexes exist on twap_slice_history."""
        result = db_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='twap_slice_history'"
            )
        )
        indexes = {row[0] for row in result}

        expected_indexes = {
            "idx_twap_slice_execution_id",
            "idx_twap_slice_executed_at",
            "idx_twap_slice_execution_slice",
        }

        assert expected_indexes.issubset(indexes)

    def test_can_insert_twap_execution(self, db_session):
        """Test inserting a TWAP execution record."""
        execution_id = str(uuid4())

        db_session.execute(
            text(
                """
            INSERT INTO twap_executions (
                execution_id, symbol, side, total_quantity, duration_minutes,
                slice_count, executed_quantity, remaining_quantity, arrival_price,
                status, started_at
            ) VALUES (
                :execution_id, :symbol, :side, :total_quantity, :duration_minutes,
                :slice_count, :executed_quantity, :remaining_quantity, :arrival_price,
                :status, :started_at
            )
        """
            ),
            {
                "execution_id": execution_id,
                "symbol": "BTC/USDT",
                "side": "BUY",
                "total_quantity": "1.5",
                "duration_minutes": 15,
                "slice_count": 5,
                "executed_quantity": "0",
                "remaining_quantity": "1.5",
                "arrival_price": "50000",
                "status": "ACTIVE",
                "started_at": datetime.utcnow(),
            },
        )

        db_session.commit()

        result = db_session.execute(
            text("SELECT * FROM twap_executions WHERE execution_id = :id"),
            {"id": execution_id},
        )

        row = result.fetchone()
        assert row is not None
        assert row[1] == "BTC/USDT"  # symbol
        assert row[2] == "BUY"  # side

    def test_can_insert_twap_slice(self, db_session):
        """Test inserting a TWAP slice record."""
        execution_id = str(uuid4())
        slice_id = str(uuid4())

        # First insert execution
        db_session.execute(
            text(
                """
            INSERT INTO twap_executions (
                execution_id, symbol, side, total_quantity, duration_minutes,
                slice_count, executed_quantity, remaining_quantity, arrival_price,
                status, started_at
            ) VALUES (
                :execution_id, 'BTC/USDT', 'BUY', '1.5', 15, 5, '0', '1.5', '50000',
                'ACTIVE', :started_at
            )
        """
            ),
            {"execution_id": execution_id, "started_at": datetime.utcnow()},
        )

        # Then insert slice
        db_session.execute(
            text(
                """
            INSERT INTO twap_slice_history (
                slice_id, execution_id, slice_number, target_quantity,
                executed_quantity, execution_price, market_price,
                status, executed_at
            ) VALUES (
                :slice_id, :execution_id, :slice_number, :target_quantity,
                :executed_quantity, :execution_price, :market_price,
                :status, :executed_at
            )
        """
            ),
            {
                "slice_id": slice_id,
                "execution_id": execution_id,
                "slice_number": 1,
                "target_quantity": "0.3",
                "executed_quantity": "0.3",
                "execution_price": "50100",
                "market_price": "50000",
                "status": "EXECUTED",
                "executed_at": datetime.utcnow(),
            },
        )

        db_session.commit()

        result = db_session.execute(
            text("SELECT * FROM twap_slice_history WHERE slice_id = :id"),
            {"id": slice_id},
        )

        row = result.fetchone()
        assert row is not None
        assert row[2] == 1  # slice_number
        assert row[14] == "EXECUTED"  # status

    def test_foreign_key_constraint(self, db_session):
        """Test that foreign key constraint works."""
        slice_id = str(uuid4())
        invalid_execution_id = str(uuid4())

        # Try to insert slice with non-existent execution_id
        # In SQLite, foreign keys need to be explicitly enabled
        db_session.execute(text("PRAGMA foreign_keys = ON"))

        with pytest.raises(Exception):
            db_session.execute(
                text(
                    """
                INSERT INTO twap_slice_history (
                    slice_id, execution_id, slice_number, target_quantity,
                    executed_quantity, execution_price, market_price,
                    status, executed_at
                ) VALUES (
                    :slice_id, :execution_id, 1, '0.3', '0.3', '50100', '50000',
                    'EXECUTED', :executed_at
                )
            """
                ),
                {
                    "slice_id": slice_id,
                    "execution_id": invalid_execution_id,
                    "executed_at": datetime.utcnow(),
                },
            )
            db_session.commit()

    def test_downgrade(self, alembic_config):
        """Test that downgrade removes tables properly."""
        # First upgrade
        command.upgrade(alembic_config, "008")

        # Then downgrade
        command.downgrade(alembic_config, "007")

        # Check tables don't exist
        engine = create_engine(alembic_config.get_main_option("sqlalchemy.url"))
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('twap_executions', 'twap_slice_history')"
                )
            )
            assert result.fetchone() is None
