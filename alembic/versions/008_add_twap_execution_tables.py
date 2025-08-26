"""Add TWAP execution tracking tables

Revision ID: 008
Revises: 007
Create Date: 2025-08-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Index


# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create twap_executions table for overall TWAP tracking
    op.create_table(
        'twap_executions',
        sa.Column('execution_id', sa.String(36), primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('side', sa.String(4), nullable=False),  # BUY or SELL
        sa.Column('total_quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('duration_minutes', sa.Integer, nullable=False),
        sa.Column('slice_count', sa.Integer, nullable=False),
        sa.Column('executed_quantity', sa.Numeric(precision=18, scale=8), nullable=False, default=0),
        sa.Column('remaining_quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('arrival_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('average_price', sa.Numeric(precision=18, scale=8)),
        sa.Column('twap_price', sa.Numeric(precision=18, scale=8)),
        sa.Column('implementation_shortfall', sa.Numeric(precision=10, scale=4)),  # in percent
        sa.Column('participation_rate', sa.Numeric(precision=5, scale=2)),  # average %
        sa.Column('status', sa.String(20), nullable=False),  # ACTIVE, PAUSED, COMPLETED, CANCELLED
        sa.Column('early_completion', sa.Boolean, default=False),
        sa.Column('early_completion_reason', sa.String(100)),
        sa.Column('started_at', sa.DateTime, nullable=False),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('paused_at', sa.DateTime),
        sa.Column('resumed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create twap_slice_history table for individual slice tracking
    op.create_table(
        'twap_slice_history',
        sa.Column('slice_id', sa.String(36), primary_key=True),
        sa.Column('execution_id', sa.String(36), sa.ForeignKey('twap_executions.execution_id'), nullable=False),
        sa.Column('slice_number', sa.Integer, nullable=False),
        sa.Column('target_quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('executed_quantity', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('execution_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('market_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('slippage_bps', sa.Numeric(precision=10, scale=2)),  # basis points
        sa.Column('volume_at_execution', sa.Numeric(precision=18, scale=8)),
        sa.Column('participation_rate', sa.Numeric(precision=5, scale=2)),  # % of volume
        sa.Column('market_impact_bps', sa.Numeric(precision=10, scale=2)),  # estimated impact
        sa.Column('time_delay_ms', sa.Integer),  # actual vs planned timing
        sa.Column('order_id', sa.String(36)),  # reference to orders table
        sa.Column('client_order_id', sa.String(36)),  # for idempotency
        sa.Column('status', sa.String(20), nullable=False),  # EXECUTED, FAILED, SKIPPED
        sa.Column('error_message', sa.Text),
        sa.Column('executed_at', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP'))
    )
    
    # Create indexes for performance queries
    op.create_index('idx_twap_executions_symbol', 'twap_executions', ['symbol'])
    op.create_index('idx_twap_executions_status', 'twap_executions', ['status'])
    op.create_index('idx_twap_executions_started_at', 'twap_executions', ['started_at'])
    op.create_index('idx_twap_executions_symbol_status', 'twap_executions', ['symbol', 'status'])
    
    op.create_index('idx_twap_slice_execution_id', 'twap_slice_history', ['execution_id'])
    op.create_index('idx_twap_slice_executed_at', 'twap_slice_history', ['executed_at'])
    op.create_index('idx_twap_slice_execution_slice', 'twap_slice_history', ['execution_id', 'slice_number'])


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_twap_slice_execution_slice', 'twap_slice_history')
    op.drop_index('idx_twap_slice_executed_at', 'twap_slice_history')
    op.drop_index('idx_twap_slice_execution_id', 'twap_slice_history')
    
    op.drop_index('idx_twap_executions_symbol_status', 'twap_executions')
    op.drop_index('idx_twap_executions_started_at', 'twap_executions')
    op.drop_index('idx_twap_executions_status', 'twap_executions')
    op.drop_index('idx_twap_executions_symbol', 'twap_executions')
    
    # Drop tables
    op.drop_table('twap_slice_history')
    op.drop_table('twap_executions')