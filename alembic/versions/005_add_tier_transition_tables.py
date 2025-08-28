"""Add tier transition tables

Revision ID: 005
Revises: 004
Create Date: 2025-08-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text


# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create tier transition tables for valley of death protection."""
    
    # Create tier_transitions table
    op.create_table(
        'tier_transitions',
        sa.Column('transition_id', sa.String(36), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('from_tier', sa.String(20), nullable=False),
        sa.Column('to_tier', sa.String(20), nullable=False),
        sa.Column('readiness_score', sa.Integer()),
        sa.Column('checklist_completed', sa.Boolean(), default=False),
        sa.Column('funeral_completed', sa.Boolean(), default=False),
        sa.Column('paper_trading_completed', sa.Boolean(), default=False),
        sa.Column('adjustment_period_start', sa.DateTime()),
        sa.Column('adjustment_period_end', sa.DateTime()),
        sa.Column('transition_status', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
    )
    
    # Skip foreign key constraint as accounts table doesn't exist yet
    
    # SQLite doesn't support ALTER TABLE ADD CONSTRAINT, skip this for SQLite
    # The constraint would be handled during table creation in production
    
    # Create paper_trading_sessions table
    op.create_table(
        'paper_trading_sessions',
        sa.Column('session_id', sa.String(36), primary_key=True),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('transition_id', sa.String(36)),
        sa.Column('strategy_name', sa.String(100), nullable=False),
        sa.Column('required_duration_hours', sa.Integer(), nullable=False),
        sa.Column('actual_duration_hours', sa.Numeric(10, 2)),
        sa.Column('success_rate', sa.Numeric(5, 2)),
        sa.Column('total_trades', sa.Integer(), default=0),
        sa.Column('profitable_trades', sa.Integer(), default=0),
        sa.Column('started_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('status', sa.String(20), nullable=False, default='ACTIVE'),
    )
    
    # Add foreign key constraints
    with op.batch_alter_table('paper_trading_sessions') as batch_op:
        batch_op.create_foreign_key(
            'fk_paper_trading_account_id',
            'accounts',
            ['account_id'],
            ['account_id']
        )
        batch_op.create_foreign_key(
            'fk_paper_trading_transition_id',
            'tier_transitions',
            ['transition_id'],
            ['transition_id']
        )
    
    # Create transition_checklists table
    op.create_table(
        'transition_checklists',
        sa.Column('checklist_id', sa.String(36), primary_key=True),
        sa.Column('transition_id', sa.String(36), nullable=False),
        sa.Column('item_name', sa.String(200), nullable=False),
        sa.Column('item_description', sa.Text()),
        sa.Column('item_response', sa.Text()),
        sa.Column('is_required', sa.Boolean(), default=True),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
    )
    
    # Add foreign key constraint
    with op.batch_alter_table('transition_checklists') as batch_op:
        batch_op.create_foreign_key(
            'fk_checklist_transition_id',
            'tier_transitions',
            ['transition_id'],
            ['transition_id']
        )
    
    # Create habit_funeral_records table
    op.create_table(
        'habit_funeral_records',
        sa.Column('funeral_id', sa.String(36), primary_key=True),
        sa.Column('transition_id', sa.String(36), nullable=False),
        sa.Column('old_habits', sa.JSON(), nullable=False),
        sa.Column('commitments', sa.JSON(), nullable=False),
        sa.Column('ceremony_timestamp', sa.DateTime(), nullable=False),
        sa.Column('certificate_generated', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
    )
    
    # Add foreign key constraint
    with op.batch_alter_table('habit_funeral_records') as batch_op:
        batch_op.create_foreign_key(
            'fk_funeral_transition_id',
            'tier_transitions',
            ['transition_id'],
            ['transition_id']
        )
    
    # Create adjustment_periods table
    op.create_table(
        'adjustment_periods',
        sa.Column('period_id', sa.String(36), primary_key=True),
        sa.Column('transition_id', sa.String(36), nullable=False),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('original_position_limit', sa.Numeric(20, 8), nullable=False),
        sa.Column('reduced_position_limit', sa.Numeric(20, 8), nullable=False),
        sa.Column('monitoring_sensitivity_multiplier', sa.Numeric(3, 1), nullable=False, default=2.0),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=False),
        sa.Column('current_phase', sa.String(20), nullable=False, default='INITIAL'),
        sa.Column('interventions_triggered', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
    )
    
    # Add foreign key constraints
    with op.batch_alter_table('adjustment_periods') as batch_op:
        batch_op.create_foreign_key(
            'fk_adjustment_transition_id',
            'tier_transitions',
            ['transition_id'],
            ['transition_id']
        )
        batch_op.create_foreign_key(
            'fk_adjustment_account_id',
            'accounts',
            ['account_id'],
            ['account_id']
        )
    
    # Create indices for performance
    op.create_index('idx_tier_transitions_account_status', 'tier_transitions', ['account_id', 'transition_status'])
    op.create_index('idx_paper_trading_account_status', 'paper_trading_sessions', ['account_id', 'status'])
    op.create_index('idx_checklists_transition_completed', 'transition_checklists', ['transition_id', 'completed_at'])
    op.create_index('idx_adjustment_account_active', 'adjustment_periods', ['account_id', 'end_time'])


def downgrade() -> None:
    """Drop tier transition tables."""
    
    # Drop indices
    op.drop_index('idx_adjustment_account_active')
    op.drop_index('idx_checklists_transition_completed')
    op.drop_index('idx_paper_trading_account_status')
    op.drop_index('idx_tier_transitions_account_status')
    
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('adjustment_periods')
    op.drop_table('habit_funeral_records')
    op.drop_table('transition_checklists')
    op.drop_table('paper_trading_sessions')
    op.drop_table('tier_transitions')