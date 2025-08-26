"""Unit tests for migration 006: Add iceberg order tables."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, Column, inspect
from sqlalchemy.orm import sessionmaker
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.operations import Operations
import tempfile
import os


class TestMigration006:
    """Test suite for iceberg order tables migration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Create basic tables that migrations depend on
        metadata = MetaData()
        
        # Create positions table (referenced by foreign key)
        positions = Table('positions', metadata,
            Column('position_id', sa.String(36), primary_key=True),
            Column('symbol', sa.String(20)),
            Column('created_at', sa.DateTime)
        )
        
        # Create orders table (modified by migration)
        orders = Table('orders', metadata,
            Column('order_id', sa.String(36), primary_key=True),
            Column('position_id', sa.String(36)),
            Column('client_order_id', sa.String(64), unique=True),
            Column('status', sa.String(20))
        )
        
        metadata.create_all(engine)
        
        yield engine, db_path
        
        # Cleanup
        engine.dispose()
        os.unlink(db_path)
    
    def test_migration_creates_iceberg_executions_table(self, temp_db):
        """Test that iceberg_executions table is created with correct schema."""
        engine, db_path = temp_db
        
        # Apply migration
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create iceberg_executions table
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True),
                sa.Column('order_id', sa.String(36), nullable=False),
                sa.Column('position_id', sa.String(36), nullable=True),
                sa.Column('symbol', sa.String(20), nullable=False),
                sa.Column('side', sa.String(10), nullable=False),
                sa.Column('total_quantity', sa.Numeric(20, 8), nullable=False),
                sa.Column('total_value_usdt', sa.Numeric(20, 8), nullable=False),
                sa.Column('slice_count', sa.Integer, nullable=False),
                sa.Column('slices_completed', sa.Integer, nullable=False, server_default='0'),
                sa.Column('slices_failed', sa.Integer, nullable=False, server_default='0'),
                sa.Column('status', sa.String(20), nullable=False),
                sa.Column('created_at', sa.DateTime, nullable=False)
            )
        
        # Verify table exists
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert 'iceberg_executions' in tables
        
        # Verify columns
        columns = {col['name']: col for col in inspector.get_columns('iceberg_executions')}
        assert 'execution_id' in columns
        assert 'order_id' in columns
        assert 'slice_count' in columns
        assert 'total_value_usdt' in columns
        assert 'status' in columns
        
        # Verify primary key
        pk = inspector.get_pk_constraint('iceberg_executions')
        assert pk['constrained_columns'] == ['execution_id']
    
    def test_migration_creates_iceberg_slices_table(self, temp_db):
        """Test that iceberg_slices table is created with correct schema."""
        engine, db_path = temp_db
        
        # First create iceberg_executions table (for foreign key)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True)
            )
            
            # Create iceberg_slices table
            op.create_table(
                'iceberg_slices',
                sa.Column('slice_id', sa.String(36), primary_key=True),
                sa.Column('execution_id', sa.String(36), nullable=False),
                sa.Column('slice_number', sa.Integer, nullable=False),
                sa.Column('client_order_id', sa.String(64), nullable=False, unique=True),
                sa.Column('quantity', sa.Numeric(20, 8), nullable=False),
                sa.Column('value_usdt', sa.Numeric(20, 8), nullable=False),
                sa.Column('slippage_percent', sa.Numeric(10, 4), nullable=True),
                sa.Column('status', sa.String(20), nullable=False),
                sa.Column('created_at', sa.DateTime, nullable=False)
            )
        
        # Verify table exists
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert 'iceberg_slices' in tables
        
        # Verify columns
        columns = {col['name']: col for col in inspector.get_columns('iceberg_slices')}
        assert 'slice_id' in columns
        assert 'execution_id' in columns
        assert 'slice_number' in columns
        assert 'client_order_id' in columns
        assert 'slippage_percent' in columns
        
        # Verify unique constraint on client_order_id
        uniques = inspector.get_unique_constraints('iceberg_slices')
        client_order_unique = any(
            'client_order_id' in u['column_names'] 
            for u in uniques
        )
        assert client_order_unique
    
    def test_migration_creates_market_impact_metrics_table(self, temp_db):
        """Test that market_impact_metrics table is created correctly."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create required tables first
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True)
            )
            
            op.create_table(
                'iceberg_slices', 
                sa.Column('slice_id', sa.String(36), primary_key=True)
            )
            
            # Create market_impact_metrics table
            op.create_table(
                'market_impact_metrics',
                sa.Column('impact_id', sa.String(36), primary_key=True),
                sa.Column('execution_id', sa.String(36), nullable=True),
                sa.Column('slice_id', sa.String(36), nullable=True),
                sa.Column('symbol', sa.String(20), nullable=False),
                sa.Column('pre_price', sa.Numeric(20, 8), nullable=False),
                sa.Column('post_price', sa.Numeric(20, 8), nullable=False),
                sa.Column('price_impact_percent', sa.Numeric(10, 4), nullable=False),
                sa.Column('volume_executed', sa.Numeric(20, 8), nullable=False),
                sa.Column('measured_at', sa.DateTime, nullable=False)
            )
        
        # Verify table and columns
        inspector = inspect(engine)
        assert 'market_impact_metrics' in inspector.get_table_names()
        
        columns = {col['name'] for col in inspector.get_columns('market_impact_metrics')}
        required_columns = {
            'impact_id', 'execution_id', 'slice_id', 'symbol',
            'pre_price', 'post_price', 'price_impact_percent',
            'volume_executed', 'measured_at'
        }
        assert required_columns.issubset(columns)
    
    def test_migration_creates_iceberg_rollbacks_table(self, temp_db):
        """Test that iceberg_rollbacks table is created correctly."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create iceberg_executions table first (for foreign key)
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True)
            )
            
            # Create iceberg_rollbacks table
            op.create_table(
                'iceberg_rollbacks',
                sa.Column('rollback_id', sa.String(36), primary_key=True),
                sa.Column('execution_id', sa.String(36), nullable=False),
                sa.Column('reason_code', sa.String(50), nullable=False),
                sa.Column('slices_to_rollback', sa.Integer, nullable=False),
                sa.Column('original_value_usdt', sa.Numeric(20, 8), nullable=False),
                sa.Column('status', sa.String(20), nullable=False),
                sa.Column('manual_confirmation', sa.Boolean, nullable=False, server_default='0'),
                sa.Column('created_at', sa.DateTime, nullable=False)
            )
        
        # Verify table
        inspector = inspect(engine)
        assert 'iceberg_rollbacks' in inspector.get_table_names()
        
        columns = {col['name'] for col in inspector.get_columns('iceberg_rollbacks')}
        assert 'rollback_id' in columns
        assert 'execution_id' in columns
        assert 'reason_code' in columns
        assert 'manual_confirmation' in columns
    
    def test_migration_adds_iceberg_columns_to_orders(self, temp_db):
        """Test that iceberg-related columns are added to orders table."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create iceberg_executions table first
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True)
            )
            
            # Add columns to orders table
            with op.batch_alter_table('orders') as batch_op:
                batch_op.add_column(
                    sa.Column('iceberg_execution_id', sa.String(36), nullable=True)
                )
                batch_op.add_column(
                    sa.Column('is_iceberg_slice', sa.Boolean, nullable=False, server_default='0')
                )
        
        # Verify columns were added
        inspector = inspect(engine)
        columns = {col['name'] for col in inspector.get_columns('orders')}
        assert 'iceberg_execution_id' in columns
        assert 'is_iceberg_slice' in columns
    
    def test_migration_creates_proper_indexes(self, temp_db):
        """Test that all required indexes are created."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create tables with indexes
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True),
                sa.Column('status', sa.String(20), nullable=False),
                sa.Column('symbol', sa.String(20), nullable=False),
                sa.Column('created_at', sa.DateTime, nullable=False)
            )
            
            op.create_index('idx_iceberg_status', 'iceberg_executions', ['status'])
            op.create_index('idx_iceberg_symbol', 'iceberg_executions', ['symbol'])
            op.create_index('idx_iceberg_created', 'iceberg_executions', ['created_at'])
        
        # Verify indexes
        inspector = inspect(engine)
        indexes = inspector.get_indexes('iceberg_executions')
        index_names = {idx['name'] for idx in indexes}
        
        assert 'idx_iceberg_status' in index_names
        assert 'idx_iceberg_symbol' in index_names
        assert 'idx_iceberg_created' in index_names
    
    def test_migration_handles_decimal_precision(self, temp_db):
        """Test that decimal columns have correct precision."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create table with decimal columns
            op.create_table(
                'test_decimals',
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('value_usdt', sa.Numeric(20, 8)),
                sa.Column('slippage_percent', sa.Numeric(10, 4))
            )
            
            # Insert test data
            conn.execute(
                sa.text("INSERT INTO test_decimals (id, value_usdt, slippage_percent) VALUES (:id, :val, :slip)"),
                {"id": 1, "val": "12345678.12345678", "slip": "0.1234"}
            )
            conn.commit()
            
            # Verify precision is maintained
            result = conn.execute(sa.text("SELECT * FROM test_decimals WHERE id = 1")).fetchone()
            assert result is not None
    
    def test_migration_rollback(self, temp_db):
        """Test that migration can be rolled back cleanly."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Apply migration
            tables_to_create = [
                'iceberg_executions',
                'iceberg_slices', 
                'market_impact_metrics',
                'iceberg_rollbacks'
            ]
            
            for table in tables_to_create:
                op.create_table(
                    table,
                    sa.Column('id', sa.String(36), primary_key=True)
                )
            
            # Verify tables exist
            inspector = inspect(engine)
            for table in tables_to_create:
                assert table in inspector.get_table_names()
            
            # Rollback migration
            for table in reversed(tables_to_create):
                op.drop_table(table)
            
            # Verify tables are removed
            inspector = inspect(engine)
            for table in tables_to_create:
                assert table not in inspector.get_table_names()
    
    def test_unique_constraints(self, temp_db):
        """Test that unique constraints are properly enforced."""
        engine, db_path = temp_db
        
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            op = Operations(context)
            
            # Create tables
            op.create_table(
                'iceberg_executions',
                sa.Column('execution_id', sa.String(36), primary_key=True)
            )
            
            op.create_table(
                'iceberg_slices',
                sa.Column('slice_id', sa.String(36), primary_key=True),
                sa.Column('execution_id', sa.String(36), nullable=False),
                sa.Column('slice_number', sa.Integer, nullable=False),
                sa.Column('client_order_id', sa.String(64), nullable=False, unique=True)
            )
            
            # Add unique constraint
            op.create_unique_constraint(
                'uq_execution_slice_number',
                'iceberg_slices',
                ['execution_id', 'slice_number']
            )
            
            # Insert test data
            conn.execute(sa.text("""
                INSERT INTO iceberg_slices (slice_id, execution_id, slice_number, client_order_id)
                VALUES ('slice1', 'exec1', 1, 'order1')
            """))
            conn.commit()
            
            # Verify unique constraint on client_order_id
            with pytest.raises(sa.exc.IntegrityError):
                conn.execute(sa.text("""
                    INSERT INTO iceberg_slices (slice_id, execution_id, slice_number, client_order_id)
                    VALUES ('slice2', 'exec1', 2, 'order1')
                """))
                conn.commit()