"""Unit tests for migration 009: Add tier transition tables."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import json
from sqlalchemy import create_engine, MetaData, Table, select, insert
from sqlalchemy.exc import IntegrityError
from alembic import command
from alembic.config import Config
import tempfile
import os
from pathlib import Path


class TestMigration009:
    """Test suite for tier transition tables migration."""
    
    @pytest.fixture
    def alembic_config(self):
        """Create temporary database and alembic config for testing."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Configure alembic
        config = Config()
        config.set_main_option("script_location", "alembic")
        config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        
        yield config, db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.fixture
    def migrated_db(self, alembic_config):
        """Apply migrations up to 009 and return engine."""
        config, db_path = alembic_config
        
        # Run migrations up to 009
        command.upgrade(config, "009")
        
        # Create engine for testing
        engine = create_engine(f"sqlite:///{db_path}")
        return engine
    
    def test_tier_transitions_table_created(self, migrated_db):
        """Test that tier_transitions table is created with correct schema."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check table exists
        assert 'tier_transitions' in metadata.tables
        
        # Check columns
        table = metadata.tables['tier_transitions']
        columns = {col.name for col in table.columns}
        expected_columns = {
            'id', 'account_id', 'timestamp', 'from_tier', 'to_tier',
            'reason', 'gates_passed', 'transition_type', 'grace_period_hours',
            'created_at'
        }
        assert expected_columns.issubset(columns)
        
        # Check indexes
        index_names = {idx.name for idx in table.indexes}
        assert 'idx_tier_transitions_account' in index_names
        assert 'idx_tier_transitions_timestamp' in index_names
    
    def test_tier_gate_progress_table_created(self, migrated_db):
        """Test that tier_gate_progress table is created with correct schema."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check table exists
        assert 'tier_gate_progress' in metadata.tables
        
        # Check columns
        table = metadata.tables['tier_gate_progress']
        columns = {col.name for col in table.columns}
        expected_columns = {
            'id', 'account_id', 'target_tier', 'gate_name', 'required_value',
            'current_value', 'is_met', 'last_checked', 'completion_date',
            'created_at', 'updated_at'
        }
        assert expected_columns.issubset(columns)
        
        # Check unique constraint
        constraint_names = {const.name for const in table.constraints}
        assert 'uq_gate_progress' in constraint_names
    
    def test_tier_feature_unlocks_table_created(self, migrated_db):
        """Test that tier_feature_unlocks table is created with correct schema."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check table exists
        assert 'tier_feature_unlocks' in metadata.tables
        
        # Check columns
        table = metadata.tables['tier_feature_unlocks']
        columns = {col.name for col in table.columns}
        expected_columns = {
            'id', 'tier', 'feature_name', 'feature_description',
            'tutorial_content', 'min_balance_required',
            'additional_requirements', 'enabled', 'created_at', 'updated_at'
        }
        assert expected_columns.issubset(columns)
        
        # Check unique constraint
        constraint_names = {const.name for const in table.constraints}
        assert 'uq_tier_feature' in constraint_names
    
    def test_transition_ceremonies_table_created(self, migrated_db):
        """Test that transition_ceremonies table is created with correct schema."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check table exists
        assert 'transition_ceremonies' in metadata.tables
        
        # Check columns
        table = metadata.tables['transition_ceremonies']
        columns = {col.name for col in table.columns}
        expected_columns = {
            'id', 'transition_id', 'ceremony_started', 'ceremony_completed',
            'checklist_items', 'completed_items', 'tutorial_views', 'created_at'
        }
        assert expected_columns.issubset(columns)
    
    def test_valley_of_death_events_table_created(self, migrated_db):
        """Test that valley_of_death_events table is created with correct schema."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check table exists
        assert 'valley_of_death_events' in metadata.tables
        
        # Check columns
        table = metadata.tables['valley_of_death_events']
        columns = {col.name for col in table.columns}
        expected_columns = {
            'id', 'account_id', 'transition_id', 'event_type', 'severity',
            'metric_value', 'threshold_value', 'action_taken', 'timestamp',
            'created_at'
        }
        assert expected_columns.issubset(columns)
        
        # Check indexes
        index_names = {idx.name for idx in table.indexes}
        assert 'idx_valley_events_account' in index_names
        assert 'idx_valley_events_timestamp' in index_names
    
    def test_prevent_manual_tier_update_trigger(self, migrated_db):
        """Test that the trigger preventing manual tier updates works."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # First create an account for testing
        accounts_table = metadata.tables.get('accounts')
        if accounts_table is not None:
            with migrated_db.begin() as conn:
                # Insert test account
                account_id = 'test-account-123'
                conn.execute(
                    insert(accounts_table).values(
                        id=account_id,
                        tier='SNIPER',
                        balance=Decimal('1000'),
                        created_at=datetime.now()
                    )
                )
                
                # Try to update tier directly (should fail due to trigger)
                with pytest.raises(IntegrityError) as exc_info:
                    conn.execute(
                        accounts_table.update()
                        .where(accounts_table.c.id == account_id)
                        .values(tier='HUNTER')
                    )
                assert 'Direct tier modifications not allowed' in str(exc_info.value)
    
    def test_foreign_key_relationships(self, migrated_db):
        """Test that foreign key relationships are properly established."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check foreign keys in tier_transitions
        transitions_table = metadata.tables['tier_transitions']
        fk_columns = {fk.parent.name for fk in transitions_table.foreign_keys}
        assert 'account_id' in fk_columns
        
        # Check foreign keys in valley_of_death_events
        valley_table = metadata.tables['valley_of_death_events']
        fk_columns = {fk.parent.name for fk in valley_table.foreign_keys}
        assert 'account_id' in fk_columns
        assert 'transition_id' in fk_columns
    
    def test_json_columns_work(self, migrated_db):
        """Test that JSON columns can store and retrieve data correctly."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Test tier_transitions gates_passed JSON column
        transitions_table = metadata.tables['tier_transitions']
        test_gates = ['min_balance', 'win_rate', 'behavioral_baseline']
        
        with migrated_db.begin() as conn:
            # Need to create account first if accounts table exists
            accounts_table = metadata.tables.get('accounts')
            account_id = 'json-test-account'
            
            if accounts_table is not None:
                conn.execute(
                    insert(accounts_table).values(
                        id=account_id,
                        tier='HUNTER',
                        balance=Decimal('2000'),
                        created_at=datetime.now()
                    )
                )
            
            # Insert transition with JSON data
            result = conn.execute(
                insert(transitions_table).values(
                    account_id=account_id,
                    timestamp=datetime.now(),
                    from_tier='SNIPER',
                    to_tier='HUNTER',
                    reason='All gates passed',
                    gates_passed=json.dumps(test_gates),
                    transition_type='PROGRESSION',
                    created_at=datetime.now()
                )
            )
            
            # Retrieve and verify JSON data
            row = conn.execute(
                select(transitions_table.c.gates_passed)
                .where(transitions_table.c.account_id == account_id)
            ).first()
            
            retrieved_gates = json.loads(row[0])
            assert retrieved_gates == test_gates
    
    def test_downgrade_removes_tables(self, alembic_config):
        """Test that downgrade properly removes all created tables."""
        config, db_path = alembic_config
        
        # First upgrade to 009
        command.upgrade(config, "009")
        
        # Then downgrade
        command.downgrade(config, "008")
        
        # Check tables are removed
        engine = create_engine(f"sqlite:///{db_path}")
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        # All new tables should be gone
        assert 'tier_transitions' not in metadata.tables
        assert 'tier_gate_progress' not in metadata.tables
        assert 'tier_feature_unlocks' not in metadata.tables
        assert 'transition_ceremonies' not in metadata.tables
        assert 'valley_of_death_events' not in metadata.tables
    
    def test_nullable_constraints(self, migrated_db):
        """Test that nullable constraints are properly set."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Check tier_transitions nullable fields
        transitions_table = metadata.tables['tier_transitions']
        assert transitions_table.c.gates_passed.nullable == True
        assert transitions_table.c.grace_period_hours.nullable == True
        assert transitions_table.c.account_id.nullable == False
        assert transitions_table.c.from_tier.nullable == False
        
        # Check tier_gate_progress nullable fields
        progress_table = metadata.tables['tier_gate_progress']
        assert progress_table.c.current_value.nullable == True
        assert progress_table.c.completion_date.nullable == True
        assert progress_table.c.gate_name.nullable == False
    
    def test_decimal_columns(self, migrated_db):
        """Test that decimal columns work correctly for financial data."""
        metadata = MetaData()
        metadata.reflect(bind=migrated_db)
        
        # Test valley_of_death_events metric columns
        valley_table = metadata.tables['valley_of_death_events']
        
        with migrated_db.begin() as conn:
            # Insert test data with decimal values
            conn.execute(
                insert(valley_table).values(
                    account_id='decimal-test',
                    event_type='RAPID_LOSS',
                    severity='WARNING',
                    metric_value=Decimal('0.1523'),
                    threshold_value=Decimal('0.1500'),
                    timestamp=datetime.now(),
                    created_at=datetime.now()
                )
            )
            
            # Retrieve and verify decimal precision
            row = conn.execute(
                select(valley_table.c.metric_value, valley_table.c.threshold_value)
                .where(valley_table.c.account_id == 'decimal-test')
            ).first()
            
            assert abs(row[0] - Decimal('0.1523')) < Decimal('0.0001')
            assert abs(row[1] - Decimal('0.1500')) < Decimal('0.0001')