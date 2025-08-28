"""Add smart routing fields

Revision ID: 010_add_smart_routing_fields
Revises: 009_add_tier_transition_tables
Create Date: 2025-08-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '010'
down_revision: Union[str, None] = '009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add smart routing fields to orders and create execution quality table."""
    
    # Add new columns to orders table
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.add_column(sa.Column('routing_method', sa.String(50), nullable=True))
        batch_op.add_column(sa.Column('maker_fee_paid', sa.Numeric(precision=20, scale=8), nullable=True))
        batch_op.add_column(sa.Column('taker_fee_paid', sa.Numeric(precision=20, scale=8), nullable=True))
        batch_op.add_column(sa.Column('execution_score', sa.Float(), nullable=True))
    
    # Create execution_quality table
    op.create_table('execution_quality',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('order_id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('routing_method', sa.String(50), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('slippage_bps', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('total_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('maker_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('taker_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('time_to_fill_ms', sa.Integer(), nullable=False),
        sa.Column('fill_rate', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('price_improvement_bps', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('execution_score', sa.Float(), nullable=False),
        sa.Column('market_conditions', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for performance
    op.create_index('idx_execution_quality_order_id', 'execution_quality', ['order_id'])
    op.create_index('idx_execution_quality_symbol', 'execution_quality', ['symbol'])
    op.create_index('idx_execution_quality_timestamp', 'execution_quality', ['timestamp'])
    op.create_index('idx_execution_quality_score', 'execution_quality', ['execution_score'])
    
    # Create execution_stats table for aggregated metrics
    op.create_table('execution_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('period', sa.String(10), nullable=False),  # '1h', '24h', '7d'
        sa.Column('symbol', sa.String(20), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('total_orders', sa.Integer(), nullable=False),
        sa.Column('avg_slippage_bps', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('total_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('avg_maker_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('avg_taker_fees', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('avg_time_to_fill_ms', sa.Integer(), nullable=False),
        sa.Column('avg_fill_rate', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('price_improvement_rate', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('avg_execution_score', sa.Float(), nullable=False),
        sa.Column('best_execution_score', sa.Float(), nullable=False),
        sa.Column('worst_execution_score', sa.Float(), nullable=False),
        sa.Column('rejection_rate', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('orders_by_type', sa.Text(), nullable=True),  # JSON
        sa.Column('orders_by_routing', sa.Text(), nullable=True),  # JSON
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for stats table
    op.create_index('idx_execution_stats_period', 'execution_stats', ['period'])
    op.create_index('idx_execution_stats_symbol', 'execution_stats', ['symbol'])
    op.create_index('idx_execution_stats_timestamp', 'execution_stats', ['timestamp'])


def downgrade() -> None:
    """Remove smart routing fields and tables."""
    
    # Drop execution_stats table
    op.drop_index('idx_execution_stats_timestamp', 'execution_stats')
    op.drop_index('idx_execution_stats_symbol', 'execution_stats')
    op.drop_index('idx_execution_stats_period', 'execution_stats')
    op.drop_table('execution_stats')
    
    # Drop execution_quality table
    op.drop_index('idx_execution_quality_score', 'execution_quality')
    op.drop_index('idx_execution_quality_timestamp', 'execution_quality')
    op.drop_index('idx_execution_quality_symbol', 'execution_quality')
    op.drop_index('idx_execution_quality_order_id', 'execution_quality')
    op.drop_table('execution_quality')
    
    # Remove columns from orders table
    with op.batch_alter_table('orders', schema=None) as batch_op:
        batch_op.drop_column('execution_score')
        batch_op.drop_column('taker_fee_paid')
        batch_op.drop_column('maker_fee_paid')
        batch_op.drop_column('routing_method')