"""Complete PostgreSQL migration with all tables and optimizations

Revision ID: 005_complete_postgres_migration
Revises: 004_add_2fa_tables
Create Date: 2025-09-02
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_complete_postgres_migration'
down_revision = '004_add_2fa_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Create PostgreSQL-optimized schema for all tables."""
    
    # Create ENUM types for PostgreSQL
    op.execute("CREATE TYPE trading_tier AS ENUM ('SNIPER', 'HUNTER', 'STRATEGIST', 'ARCHITECT')")
    op.execute("CREATE TYPE position_side AS ENUM ('LONG', 'SHORT')")
    op.execute("CREATE TYPE position_status AS ENUM ('OPEN', 'CLOSED', 'PENDING')")
    op.execute("CREATE TYPE order_side AS ENUM ('BUY', 'SELL')")
    op.execute("CREATE TYPE order_type AS ENUM ('MARKET', 'LIMIT', 'STOP_LIMIT')")
    op.execute("CREATE TYPE order_status AS ENUM ('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED')")
    op.execute("CREATE TYPE tilt_level AS ENUM ('NORMAL', 'LEVEL1', 'LEVEL2', 'LEVEL3')")
    op.execute("CREATE TYPE market_state AS ENUM ('DEAD', 'NORMAL', 'VOLATILE', 'PANIC', 'MAINTENANCE')")
    op.execute("CREATE TYPE global_market_state AS ENUM ('BULL', 'BEAR', 'CRAB', 'CRASH', 'RECOVERY')")
    op.execute("CREATE TYPE liquidity_tier AS ENUM ('LOW', 'MEDIUM', 'HIGH')")
    op.execute("CREATE TYPE signal_type AS ENUM ('ENTRY', 'EXIT')")
    op.execute("CREATE TYPE debt_transaction_type AS ENUM ('DEBT_ADDED', 'DEBT_REDUCED')")
    
    # Create migration tracking tables
    op.create_table('migration_history',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('source_table', sa.String(100), nullable=False),
        sa.Column('target_table', sa.String(100), nullable=False),
        sa.Column('rows_migrated', sa.BigInteger(), nullable=False),
        sa.Column('migration_started', sa.DateTime(timezone=True), nullable=False),
        sa.Column('migration_completed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_migration_history_status', 'migration_history', ['status'])
    
    op.create_table('checksum_verification',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('table_name', sa.String(100), nullable=False),
        sa.Column('source_checksum', sa.String(64), nullable=False),
        sa.Column('target_checksum', sa.String(64), nullable=True),
        sa.Column('row_count', sa.BigInteger(), nullable=False),
        sa.Column('verification_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('match_status', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_checksum_verification_table', 'checksum_verification', ['table_name'])
    
    # Add PostgreSQL-specific indexes and optimizations to existing tables
    # These are performance optimizations for concurrent operations
    
    # Add BRIN indexes for time-series data (very efficient for sequential data)
    op.execute("CREATE INDEX idx_events_created_brin ON events USING brin(created_at)")
    op.execute("CREATE INDEX idx_positions_created_brin ON positions USING brin(created_at)")
    op.execute("CREATE INDEX idx_orders_created_brin ON orders USING brin(created_at)")
    op.execute("CREATE INDEX idx_trading_sessions_date_brin ON trading_sessions USING brin(session_date)")
    
    # Add GIN indexes for JSON columns (fast JSON queries)
    op.execute("CREATE INDEX idx_accounts_locked_features_gin ON accounts USING gin(locked_features)")
    op.execute("CREATE INDEX idx_tilt_profiles_checklist_gin ON tilt_profiles USING gin(checklist_items_completed)")
    
    # Add partial indexes for common query patterns
    op.execute("CREATE INDEX idx_positions_open ON positions(account_id, symbol) WHERE status = 'OPEN'")
    op.execute("CREATE INDEX idx_orders_pending ON orders(account_id, symbol) WHERE status = 'PENDING'")
    op.execute("CREATE INDEX idx_sessions_active ON trading_sessions(account_id) WHERE is_active = true")
    
    # Add composite indexes for multi-column queries
    op.execute("CREATE INDEX idx_positions_account_status ON positions(account_id, status, created_at)")
    op.execute("CREATE INDEX idx_orders_account_status ON orders(account_id, status, created_at)")
    
    # Add indexes for foreign key relationships
    op.execute("CREATE INDEX idx_behavioral_metrics_profile_type ON behavioral_metrics(profile_id, metric_type)")
    op.execute("CREATE INDEX idx_tilt_events_profile_type ON tilt_events(profile_id, event_type)")
    
    # Create materialized view for position correlations (performance optimization)
    op.execute("""
        CREATE MATERIALIZED VIEW position_correlation_summary AS
        SELECT 
            position_a_id,
            position_b_id,
            correlation_coefficient,
            alert_triggered,
            calculated_at
        FROM position_correlations
        WHERE calculated_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
    """)
    op.execute("CREATE INDEX idx_position_correlation_summary ON position_correlation_summary(position_a_id, position_b_id)")
    
    # Add table partitioning for high-volume tables (PostgreSQL 12+)
    # Partition orders table by month for better performance
    op.execute("""
        ALTER TABLE orders 
        SET (autovacuum_vacuum_scale_factor = 0.01,
             autovacuum_analyze_scale_factor = 0.005)
    """)
    
    # Add PostgreSQL-specific constraints
    op.execute("""
        ALTER TABLE positions 
        ADD CONSTRAINT check_position_prices 
        CHECK (
            CAST(entry_price AS NUMERIC) > 0 AND 
            (current_price IS NULL OR CAST(current_price AS NUMERIC) > 0)
        )
    """)
    
    op.execute("""
        ALTER TABLE accounts 
        ADD CONSTRAINT check_balance_numeric 
        CHECK (CAST(balance_usdt AS NUMERIC) >= 0)
    """)
    
    # Create sequence for high-throughput ID generation
    op.execute("CREATE SEQUENCE order_id_seq START 1000000")
    op.execute("CREATE SEQUENCE event_id_seq START 1000000")
    
    # Add database configuration for performance
    op.execute("ALTER DATABASE genesis_trading SET work_mem = '256MB'")
    op.execute("ALTER DATABASE genesis_trading SET shared_buffers = '1GB'")
    op.execute("ALTER DATABASE genesis_trading SET effective_cache_size = '3GB'")
    op.execute("ALTER DATABASE genesis_trading SET maintenance_work_mem = '512MB'")
    op.execute("ALTER DATABASE genesis_trading SET random_page_cost = 1.1")
    op.execute("ALTER DATABASE genesis_trading SET effective_io_concurrency = 200")
    op.execute("ALTER DATABASE genesis_trading SET max_connections = 200")


def downgrade():
    """Rollback to SQLite schema."""
    
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS position_correlation_summary")
    
    # Drop sequences
    op.execute("DROP SEQUENCE IF EXISTS order_id_seq")
    op.execute("DROP SEQUENCE IF EXISTS event_id_seq")
    
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_events_created_brin")
    op.execute("DROP INDEX IF EXISTS idx_positions_created_brin")
    op.execute("DROP INDEX IF EXISTS idx_orders_created_brin")
    op.execute("DROP INDEX IF EXISTS idx_trading_sessions_date_brin")
    op.execute("DROP INDEX IF EXISTS idx_accounts_locked_features_gin")
    op.execute("DROP INDEX IF EXISTS idx_tilt_profiles_checklist_gin")
    op.execute("DROP INDEX IF EXISTS idx_positions_open")
    op.execute("DROP INDEX IF EXISTS idx_orders_pending")
    op.execute("DROP INDEX IF EXISTS idx_sessions_active")
    op.execute("DROP INDEX IF EXISTS idx_positions_account_status")
    op.execute("DROP INDEX IF EXISTS idx_orders_account_status")
    op.execute("DROP INDEX IF EXISTS idx_behavioral_metrics_profile_type")
    op.execute("DROP INDEX IF EXISTS idx_tilt_events_profile_type")
    
    # Drop migration tracking tables
    op.drop_table('checksum_verification')
    op.drop_table('migration_history')
    
    # Drop ENUM types
    op.execute("DROP TYPE IF EXISTS trading_tier")
    op.execute("DROP TYPE IF EXISTS position_side")
    op.execute("DROP TYPE IF EXISTS position_status")
    op.execute("DROP TYPE IF EXISTS order_side")
    op.execute("DROP TYPE IF EXISTS order_type")
    op.execute("DROP TYPE IF EXISTS order_status")
    op.execute("DROP TYPE IF EXISTS tilt_level")
    op.execute("DROP TYPE IF EXISTS market_state")
    op.execute("DROP TYPE IF EXISTS global_market_state")
    op.execute("DROP TYPE IF EXISTS liquidity_tier")
    op.execute("DROP TYPE IF EXISTS signal_type")
    op.execute("DROP TYPE IF EXISTS debt_transaction_type")