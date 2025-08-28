"""Unit tests for migration 005 - tier transition tables."""

import os
import sys
import tempfile
import uuid

import pytest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config


class TestMigration005:
    """Test tier transition tables migration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.fixture
    def alembic_config(self, temp_db):
        """Create Alembic configuration for testing."""
        config = Config()
        config.set_main_option("script_location", "alembic")
        config.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db}")
        return config

    @pytest.fixture
    def db_engine(self, temp_db, alembic_config):
        """Create database engine and run migrations."""
        # Run migrations up to 004 first (assuming it exists)
        try:
            command.upgrade(alembic_config, "004")
        except:
            # If migration 004 doesn't exist, create base tables
            engine = create_engine(f"sqlite:///{temp_db}")
            with engine.connect() as conn:
                # Create minimal accounts table for foreign key constraints
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS accounts (
                        account_id TEXT PRIMARY KEY,
                        balance_usdt DECIMAL(20, 8),
                        current_tier TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                    )
                )
                conn.commit()

        # Run migration 005
        command.upgrade(alembic_config, "005")

        engine = create_engine(f"sqlite:///{temp_db}")
        return engine

    def test_tier_transitions_table_created(self, db_engine):
        """Test that tier_transitions table is created with correct schema."""
        inspector = inspect(db_engine)

        # Check table exists
        assert "tier_transitions" in inspector.get_table_names()

        # Check columns
        columns = {
            col["name"]: col for col in inspector.get_columns("tier_transitions")
        }

        required_columns = [
            "transition_id",
            "account_id",
            "from_tier",
            "to_tier",
            "readiness_score",
            "checklist_completed",
            "funeral_completed",
            "paper_trading_completed",
            "adjustment_period_start",
            "adjustment_period_end",
            "transition_status",
            "created_at",
            "updated_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Column {col_name} not found"

        # Check primary key
        pk = inspector.get_pk_constraint("tier_transitions")
        assert pk["constrained_columns"] == ["transition_id"]

        # Check foreign keys
        fks = inspector.get_foreign_keys("tier_transitions")
        assert any(fk["referred_table"] == "accounts" for fk in fks)

    def test_paper_trading_sessions_table_created(self, db_engine):
        """Test that paper_trading_sessions table is created correctly."""
        inspector = inspect(db_engine)

        # Check table exists
        assert "paper_trading_sessions" in inspector.get_table_names()

        # Check columns
        columns = {
            col["name"]: col for col in inspector.get_columns("paper_trading_sessions")
        }

        required_columns = [
            "session_id",
            "account_id",
            "transition_id",
            "strategy_name",
            "required_duration_hours",
            "actual_duration_hours",
            "success_rate",
            "total_trades",
            "profitable_trades",
            "started_at",
            "completed_at",
            "status",
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Column {col_name} not found"

        # Check foreign keys
        fks = inspector.get_foreign_keys("paper_trading_sessions")
        assert any(fk["referred_table"] == "accounts" for fk in fks)
        assert any(fk["referred_table"] == "tier_transitions" for fk in fks)

    def test_transition_checklists_table_created(self, db_engine):
        """Test that transition_checklists table is created correctly."""
        inspector = inspect(db_engine)

        # Check table exists
        assert "transition_checklists" in inspector.get_table_names()

        # Check columns
        columns = {
            col["name"]: col for col in inspector.get_columns("transition_checklists")
        }

        required_columns = [
            "checklist_id",
            "transition_id",
            "item_name",
            "item_description",
            "item_response",
            "is_required",
            "completed_at",
            "created_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Column {col_name} not found"

    def test_habit_funeral_records_table_created(self, db_engine):
        """Test that habit_funeral_records table is created correctly."""
        inspector = inspect(db_engine)

        # Check table exists
        assert "habit_funeral_records" in inspector.get_table_names()

        # Check columns
        columns = {
            col["name"]: col for col in inspector.get_columns("habit_funeral_records")
        }

        required_columns = [
            "funeral_id",
            "transition_id",
            "old_habits",
            "commitments",
            "ceremony_timestamp",
            "certificate_generated",
            "created_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Column {col_name} not found"

    def test_adjustment_periods_table_created(self, db_engine):
        """Test that adjustment_periods table is created correctly."""
        inspector = inspect(db_engine)

        # Check table exists
        assert "adjustment_periods" in inspector.get_table_names()

        # Check columns
        columns = {
            col["name"]: col for col in inspector.get_columns("adjustment_periods")
        }

        required_columns = [
            "period_id",
            "transition_id",
            "account_id",
            "original_position_limit",
            "reduced_position_limit",
            "monitoring_sensitivity_multiplier",
            "start_time",
            "end_time",
            "current_phase",
            "interventions_triggered",
            "created_at",
            "updated_at",
        ]

        for col_name in required_columns:
            assert col_name in columns, f"Column {col_name} not found"

    def test_indices_created(self, db_engine):
        """Test that performance indices are created."""
        inspector = inspect(db_engine)

        # Check indices
        transitions_indices = inspector.get_indexes("tier_transitions")
        paper_trading_indices = inspector.get_indexes("paper_trading_sessions")
        checklist_indices = inspector.get_indexes("transition_checklists")
        adjustment_indices = inspector.get_indexes("adjustment_periods")

        # Verify key indices exist
        assert any(
            "account_id" in idx.get("column_names", []) for idx in transitions_indices
        )
        assert any(
            "account_id" in idx.get("column_names", []) for idx in paper_trading_indices
        )
        assert any(
            "transition_id" in idx.get("column_names", []) for idx in checklist_indices
        )
        assert any(
            "account_id" in idx.get("column_names", []) for idx in adjustment_indices
        )

    def test_data_insertion(self, db_engine):
        """Test that data can be inserted correctly with constraints."""
        Session = sessionmaker(bind=db_engine)
        session = Session()

        try:
            # Insert test account
            account_id = str(uuid.uuid4())
            session.execute(
                text(
                    """
                INSERT INTO accounts (account_id, balance_usdt, current_tier)
                VALUES (:id, :balance, :tier)
            """
                ),
                {"id": account_id, "balance": 1900.0, "tier": "SNIPER"},
            )

            # Insert tier transition
            transition_id = str(uuid.uuid4())
            session.execute(
                text(
                    """
                INSERT INTO tier_transitions (
                    transition_id, account_id, from_tier, to_tier,
                    transition_status, readiness_score
                )
                VALUES (:tid, :aid, :from_tier, :to_tier, :status, :score)
            """
                ),
                {
                    "tid": transition_id,
                    "aid": account_id,
                    "from_tier": "SNIPER",
                    "to_tier": "HUNTER",
                    "status": "APPROACHING",
                    "score": 75,
                },
            )

            # Insert paper trading session
            session_id = str(uuid.uuid4())
            session.execute(
                text(
                    """
                INSERT INTO paper_trading_sessions (
                    session_id, account_id, transition_id, strategy_name,
                    required_duration_hours, status
                )
                VALUES (:sid, :aid, :tid, :strategy, :hours, :status)
            """
                ),
                {
                    "sid": session_id,
                    "aid": account_id,
                    "tid": transition_id,
                    "strategy": "iceberg_orders",
                    "hours": 24,
                    "status": "ACTIVE",
                },
            )

            # Insert checklist item
            checklist_id = str(uuid.uuid4())
            session.execute(
                text(
                    """
                INSERT INTO transition_checklists (
                    checklist_id, transition_id, item_name, is_required
                )
                VALUES (:cid, :tid, :name, :required)
            """
                ),
                {
                    "cid": checklist_id,
                    "tid": transition_id,
                    "name": "Review risk management rules",
                    "required": True,
                },
            )

            session.commit()

            # Verify data was inserted
            result = session.execute(
                text("SELECT COUNT(*) FROM tier_transitions WHERE account_id = :aid"),
                {"aid": account_id},
            ).scalar()
            assert result == 1

            result = session.execute(
                text(
                    "SELECT COUNT(*) FROM paper_trading_sessions WHERE account_id = :aid"
                ),
                {"aid": account_id},
            ).scalar()
            assert result == 1

        finally:
            session.close()

    def test_transition_status_constraint(self, db_engine):
        """Test that transition_status check constraint works."""
        Session = sessionmaker(bind=db_engine)
        session = Session()

        try:
            # Insert test account
            account_id = str(uuid.uuid4())
            session.execute(
                text(
                    """
                INSERT INTO accounts (account_id, balance_usdt, current_tier)
                VALUES (:id, :balance, :tier)
            """
                ),
                {"id": account_id, "balance": 2000.0, "tier": "SNIPER"},
            )

            # Try to insert with invalid status - should fail
            with pytest.raises(Exception):
                session.execute(
                    text(
                        """
                    INSERT INTO tier_transitions (
                        transition_id, account_id, from_tier, to_tier,
                        transition_status
                    )
                    VALUES (:tid, :aid, :from_tier, :to_tier, :status)
                """
                    ),
                    {
                        "tid": str(uuid.uuid4()),
                        "aid": account_id,
                        "from_tier": "SNIPER",
                        "to_tier": "HUNTER",
                        "status": "INVALID_STATUS",
                    },
                )
                session.commit()

        finally:
            session.rollback()
            session.close()

    def test_foreign_key_constraints(self, db_engine):
        """Test that foreign key constraints are enforced."""
        Session = sessionmaker(bind=db_engine)
        session = Session()

        try:
            # Try to insert transition with non-existent account - should fail
            with pytest.raises(Exception):
                session.execute(
                    text(
                        """
                    INSERT INTO tier_transitions (
                        transition_id, account_id, from_tier, to_tier,
                        transition_status
                    )
                    VALUES (:tid, :aid, :from_tier, :to_tier, :status)
                """
                    ),
                    {
                        "tid": str(uuid.uuid4()),
                        "aid": "non-existent-account",
                        "from_tier": "SNIPER",
                        "to_tier": "HUNTER",
                        "status": "APPROACHING",
                    },
                )
                session.commit()

        finally:
            session.rollback()
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
