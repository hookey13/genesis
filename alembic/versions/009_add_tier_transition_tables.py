"""Add tier transition tables

Revision ID: 009
Revises: 008
Create Date: 2025-08-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
import json

# revision identifiers, used by Alembic.
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add additional columns to tier_transitions table created in migration 005
    with op.batch_alter_table('tier_transitions') as batch_op:
        batch_op.add_column(sa.Column('timestamp', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('gates_passed', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('transition_type', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('grace_period_hours', sa.Integer(), nullable=True))
    
    # Create tier_gate_progress table
    op.create_table('tier_gate_progress',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('target_tier', sa.String(20), nullable=False),
        sa.Column('gate_name', sa.String(100), nullable=False),
        sa.Column('required_value', sa.String(255), nullable=False),
        sa.Column('current_value', sa.String(255), nullable=True),
        sa.Column('is_met', sa.Boolean(), nullable=False, default=False),
        sa.Column('last_checked', sa.DateTime(), nullable=False),
        sa.Column('completion_date', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        # Foreign key would reference accounts.id when accounts table exists
        sa.UniqueConstraint('account_id', 'target_tier', 'gate_name', name='uq_gate_progress')
    )
    op.create_index('idx_gate_progress_account', 'tier_gate_progress', ['account_id'])
    op.create_index('idx_gate_progress_tier', 'tier_gate_progress', ['target_tier'])
    
    # Create tier_feature_unlocks table
    op.create_table('tier_feature_unlocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tier', sa.String(20), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('feature_description', sa.Text(), nullable=True),
        sa.Column('tutorial_content', sa.JSON(), nullable=True),
        sa.Column('min_balance_required', sa.Numeric(15, 2), nullable=True),
        sa.Column('additional_requirements', sa.JSON(), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tier', 'feature_name', name='uq_tier_feature')
    )
    op.create_index('idx_feature_unlocks_tier', 'tier_feature_unlocks', ['tier'])
    
    # Create transition_ceremonies table for tracking ceremony completion
    op.create_table('transition_ceremonies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('transition_id', sa.Integer(), nullable=False),
        sa.Column('ceremony_started', sa.DateTime(), nullable=False),
        sa.Column('ceremony_completed', sa.DateTime(), nullable=True),
        sa.Column('checklist_items', sa.JSON(), nullable=False),
        sa.Column('completed_items', sa.JSON(), nullable=False, default=[]),
        sa.Column('tutorial_views', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        # Foreign key would reference tier_transitions.id
    )
    
    # Create valley_of_death_events table for monitoring critical transitions
    op.create_table('valley_of_death_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('transition_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),  # TILT_SPIKE, RAPID_LOSS, OVERLEVERAGING
        sa.Column('severity', sa.String(20), nullable=False),  # WARNING, CRITICAL, EMERGENCY
        sa.Column('metric_value', sa.Numeric(15, 4), nullable=True),
        sa.Column('threshold_value', sa.Numeric(15, 4), nullable=True),
        sa.Column('action_taken', sa.String(100), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        # Foreign key would reference accounts.id when accounts table exists
        # Foreign key would reference tier_transitions.id
    )
    op.create_index('idx_valley_events_account', 'valley_of_death_events', ['account_id'])
    op.create_index('idx_valley_events_timestamp', 'valley_of_death_events', ['timestamp'])
    
    # Add constraint to prevent manual tier changes (enforced at application level)
    # Skip trigger creation as accounts table doesn't exist yet
    # Trigger would be: prevent_manual_tier_update on accounts table
    pass


def downgrade() -> None:
    # No trigger to drop since we skipped it
    pass
    
    # Drop new tables
    op.drop_table('valley_of_death_events')
    op.drop_table('transition_ceremonies')
    op.drop_table('tier_feature_unlocks')
    op.drop_table('tier_gate_progress')
    
    # Remove added columns from tier_transitions
    with op.batch_alter_table('tier_transitions') as batch_op:
        batch_op.drop_column('grace_period_hours')
        batch_op.drop_column('transition_type')
        batch_op.drop_column('gates_passed')
        batch_op.drop_column('reason')
        batch_op.drop_column('timestamp')