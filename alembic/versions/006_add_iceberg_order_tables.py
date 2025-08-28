"""Add iceberg order tracking tables

Revision ID: 006_iceberg_orders
Revises: 005_add_tier_transition_tables
Create Date: 2025-08-26

"""
from decimal import Decimal
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create iceberg order tracking and market impact tables."""
    
    # Create iceberg_executions table for tracking sliced orders
    op.create_table(
        'iceberg_executions',
        sa.Column('execution_id', sa.String(36), primary_key=True),
        sa.Column('order_id', sa.String(36), nullable=False),
        sa.Column('position_id', sa.String(36), sa.ForeignKey('positions.position_id'), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),  # BUY or SELL
        sa.Column('total_quantity', sa.Numeric(20, 8), nullable=False),
        sa.Column('total_value_usdt', sa.Numeric(20, 8), nullable=False),
        sa.Column('slice_count', sa.Integer, nullable=False),
        sa.Column('slices_completed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('slices_failed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('avg_slice_size_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('slice_size_variation', sa.Numeric(5, 2), nullable=False, server_default='20.0'),  # ±20% default
        sa.Column('min_delay_seconds', sa.Numeric(5, 2), nullable=False, server_default='1.0'),
        sa.Column('max_delay_seconds', sa.Numeric(5, 2), nullable=False, server_default='5.0'),
        sa.Column('cumulative_slippage', sa.Numeric(10, 4), nullable=True),
        sa.Column('max_slice_slippage', sa.Numeric(10, 4), nullable=True),
        sa.Column('abort_threshold', sa.Numeric(10, 4), nullable=False, server_default='0.5'),  # 0.5% default
        sa.Column('status', sa.String(20), nullable=False),  # PENDING, EXECUTING, COMPLETED, ABORTED, FAILED
        sa.Column('abort_reason', sa.String(255), nullable=True),
        sa.Column('rollback_requested', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('rollback_completed', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('rollback_cost_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('execution_time_seconds', sa.Numeric(10, 2), nullable=True),
        sa.Index('idx_iceberg_status', 'status'),
        sa.Index('idx_iceberg_symbol', 'symbol'),
        sa.Index('idx_iceberg_created', 'created_at')
    )
    
    # Create iceberg_slices table for individual slice tracking
    op.create_table(
        'iceberg_slices',
        sa.Column('slice_id', sa.String(36), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('iceberg_executions.execution_id'), nullable=False),
        sa.Column('slice_number', sa.Integer, nullable=False),
        sa.Column('client_order_id', sa.String(64), nullable=False, unique=True),  # Idempotency key
        sa.Column('exchange_order_id', sa.String(64), nullable=True),
        sa.Column('quantity', sa.Numeric(20, 8), nullable=False),
        sa.Column('price_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('value_usdt', sa.Numeric(20, 8), nullable=False),
        sa.Column('expected_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('actual_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('slippage_percent', sa.Numeric(10, 4), nullable=True),
        sa.Column('delay_seconds', sa.Numeric(5, 2), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),  # PENDING, SUBMITTED, FILLED, PARTIAL, CANCELLED, FAILED
        sa.Column('fill_percent', sa.Numeric(5, 2), nullable=True),
        sa.Column('fees_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('error_message', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('submitted_at', sa.DateTime, nullable=True),
        sa.Column('filled_at', sa.DateTime, nullable=True),
        sa.Column('latency_ms', sa.Integer, nullable=True),
        sa.Index('idx_slice_execution', 'execution_id'),
        sa.Index('idx_slice_status', 'status'),
        sa.UniqueConstraint('execution_id', 'slice_number', name='uq_execution_slice_number')
    )
    
    # Create market_impact_metrics table for impact monitoring
    op.create_table(
        'market_impact_metrics',
        sa.Column('impact_id', sa.String(36), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('iceberg_executions.execution_id'), nullable=True),
        sa.Column('slice_id', sa.String(36), sa.ForeignKey('iceberg_slices.slice_id'), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('pre_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('post_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('price_impact_percent', sa.Numeric(10, 4), nullable=False),
        sa.Column('volume_executed', sa.Numeric(20, 8), nullable=False),
        sa.Column('order_book_depth_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('bid_ask_spread', sa.Numeric(20, 8), nullable=True),
        sa.Column('liquidity_consumed_percent', sa.Numeric(10, 4), nullable=True),
        sa.Column('market_depth_1pct', sa.Numeric(20, 8), nullable=True),  # Volume to move price 1%
        sa.Column('market_depth_2pct', sa.Numeric(20, 8), nullable=True),  # Volume to move price 2%
        sa.Column('cumulative_impact', sa.Numeric(10, 4), nullable=True),  # For tracking across slices
        sa.Column('measured_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Index('idx_impact_execution', 'execution_id'),
        sa.Index('idx_impact_symbol', 'symbol'),
        sa.Index('idx_impact_measured', 'measured_at')
    )
    
    # Create iceberg_rollbacks table for rollback tracking
    op.create_table(
        'iceberg_rollbacks',
        sa.Column('rollback_id', sa.String(36), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('iceberg_executions.execution_id'), nullable=False),
        sa.Column('reason_code', sa.String(50), nullable=False),  # SLIPPAGE, USER_REQUEST, ERROR, TILT
        sa.Column('reason_description', sa.String(500), nullable=True),
        sa.Column('slices_to_rollback', sa.Integer, nullable=False),
        sa.Column('slices_rolled_back', sa.Integer, nullable=False, server_default='0'),
        sa.Column('original_value_usdt', sa.Numeric(20, 8), nullable=False),
        sa.Column('rollback_value_usdt', sa.Numeric(20, 8), nullable=True),
        sa.Column('cost_usdt', sa.Numeric(20, 8), nullable=True),  # Fees + slippage
        sa.Column('status', sa.String(20), nullable=False),  # PENDING, EXECUTING, COMPLETED, FAILED
        sa.Column('manual_confirmation', sa.Boolean, nullable=False, server_default='0'),
        sa.Column('confirmed_by', sa.String(100), nullable=True),
        sa.Column('confirmed_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Index('idx_rollback_execution', 'execution_id'),
        sa.Index('idx_rollback_status', 'status')
    )
    
    # Add iceberg-related columns to existing orders table if not present
    with op.batch_alter_table('orders') as batch_op:
        # Check if columns exist before adding (for idempotency)
        inspector = sa.inspect(op.get_bind())
        existing_columns = [col['name'] for col in inspector.get_columns('orders')]
        
        if 'iceberg_execution_id' not in existing_columns:
            batch_op.add_column(sa.Column('iceberg_execution_id', sa.String(36), nullable=True))
        
        if 'is_iceberg_slice' not in existing_columns:
            batch_op.add_column(sa.Column('is_iceberg_slice', sa.Boolean, nullable=False, server_default='0'))


def downgrade() -> None:
    """Remove iceberg order tracking tables."""
    
    # Remove iceberg-related columns from orders table
    with op.batch_alter_table('orders') as batch_op:
        batch_op.drop_column('iceberg_execution_id')
        batch_op.drop_column('is_iceberg_slice')
    
    # Drop tables in reverse order of creation (respect foreign keys)
    op.drop_table('iceberg_rollbacks')
    op.drop_table('market_impact_metrics')
    op.drop_table('iceberg_slices')
    op.drop_table('iceberg_executions')