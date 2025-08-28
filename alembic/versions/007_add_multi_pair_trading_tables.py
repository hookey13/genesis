"""Add multi-pair trading tables

Revision ID: 007
Revises: 006
Create Date: 2025-08-26

"""
from decimal import Decimal
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create portfolio_limits table for per-pair and global limits
    op.create_table(
        'portfolio_limits',
        sa.Column('limit_id', sa.String(36), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=True),  # NULL for global limits
        sa.Column('max_position_size', sa.Numeric(20, 8), nullable=False),
        sa.Column('max_dollar_value', sa.Numeric(20, 8), nullable=False),
        sa.Column('max_open_positions', sa.Integer, nullable=True),  # Only for global
        sa.Column('limit_type', sa.String(10), nullable=False),  # 'PAIR' or 'GLOBAL'
        sa.Column('tier', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
        sa.Index('idx_portfolio_limits_account', 'account_id'),
        sa.Index('idx_portfolio_limits_symbol', 'symbol'),
        sa.CheckConstraint("limit_type IN ('PAIR', 'GLOBAL')", name='limit_type_check'),
        sa.CheckConstraint("tier IN ('SNIPER', 'HUNTER', 'STRATEGIST', 'ARCHITECT')", name='tier_check')
    )

    # Create signal_queue table for managing competing trade signals
    op.create_table(
        'signal_queue',
        sa.Column('signal_id', sa.String(36), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('strategy_name', sa.String(50), nullable=False),
        sa.Column('signal_type', sa.String(10), nullable=False),  # BUY, SELL, CLOSE
        sa.Column('confidence_score', sa.Numeric(5, 4), nullable=False),
        sa.Column('priority', sa.Integer, nullable=False),
        sa.Column('size_recommendation', sa.Numeric(20, 8), nullable=True),
        sa.Column('price_target', sa.Numeric(20, 8), nullable=True),
        sa.Column('stop_loss', sa.Numeric(20, 8), nullable=True),
        sa.Column('expiry_time', sa.DateTime, nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING'),
        sa.Column('conflict_resolution', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime, nullable=True),
        sa.Index('idx_signal_queue_status', 'status'),
        sa.Index('idx_signal_queue_priority', 'priority'),
        sa.Index('idx_signal_queue_symbol', 'symbol'),
        sa.CheckConstraint("signal_type IN ('BUY', 'SELL', 'CLOSE')", name='signal_type_check'),
        sa.CheckConstraint("status IN ('PENDING', 'PROCESSING', 'EXECUTED', 'REJECTED', 'EXPIRED', 'CONFLICTED')", name='status_check'),
        sa.CheckConstraint("confidence_score BETWEEN 0 AND 1", name='confidence_check'),
        sa.CheckConstraint("priority BETWEEN 0 AND 100", name='priority_check')
    )

    # Create pair_performance table for performance attribution tracking
    op.create_table(
        'pair_performance',
        sa.Column('performance_id', sa.String(36), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('period_start', sa.DateTime, nullable=False),
        sa.Column('period_end', sa.DateTime, nullable=False),
        sa.Column('total_trades', sa.Integer, nullable=False, server_default='0'),
        sa.Column('winning_trades', sa.Integer, nullable=False, server_default='0'),
        sa.Column('losing_trades', sa.Integer, nullable=False, server_default='0'),
        sa.Column('total_pnl_dollars', sa.Numeric(20, 8), nullable=False, server_default='0'),
        sa.Column('average_win_dollars', sa.Numeric(20, 8), nullable=True),
        sa.Column('average_loss_dollars', sa.Numeric(20, 8), nullable=True),
        sa.Column('win_rate', sa.Numeric(5, 4), nullable=True),
        sa.Column('profit_factor', sa.Numeric(10, 4), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(10, 4), nullable=True),
        sa.Column('max_drawdown_dollars', sa.Numeric(20, 8), nullable=True),
        sa.Column('volume_traded_base', sa.Numeric(20, 8), nullable=False, server_default='0'),
        sa.Column('volume_traded_quote', sa.Numeric(20, 8), nullable=False, server_default='0'),
        sa.Column('fees_paid_dollars', sa.Numeric(20, 8), nullable=False, server_default='0'),
        sa.Column('attribution_weight', sa.Numeric(5, 4), nullable=True),  # Weight in portfolio
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now()),
        sa.Index('idx_pair_performance_account', 'account_id'),
        sa.Index('idx_pair_performance_symbol', 'symbol'),
        sa.Index('idx_pair_performance_period', 'period_start', 'period_end'),
        sa.UniqueConstraint('account_id', 'symbol', 'period_start', 'period_end', name='unique_performance_period')
    )

    # Create position_correlations table
    op.create_table('position_correlations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('symbol_1', sa.String(20), nullable=False),
        sa.Column('symbol_2', sa.String(20), nullable=False),
        sa.Column('correlation_coefficient', sa.Numeric(5, 4), nullable=False),
        sa.Column('risk_adjustment_factor', sa.Numeric(5, 4), nullable=False, server_default='1.0'),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Index('idx_correlations_symbols', 'symbol_1', 'symbol_2'),
        sa.Index('idx_correlations_account', 'account_id'),
        sa.Index('idx_correlations_period', 'period_start', 'period_end')
    )


def downgrade() -> None:
    # Drop position_correlations table
    op.drop_table('position_correlations')
    
    # Drop tables in reverse order
    op.drop_table('pair_performance')
    op.drop_table('signal_queue')
    op.drop_table('portfolio_limits')