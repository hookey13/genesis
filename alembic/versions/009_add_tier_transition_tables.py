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
    # Create tier_transitions table
    op.create_table('tier_transitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('from_tier', sa.String(20), nullable=False),
        sa.Column('to_tier', sa.String(20), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('gates_passed', sa.JSON(), nullable=True),
        sa.Column('transition_type', sa.String(20), nullable=False),  # PROGRESSION or DEMOTION
        sa.Column('grace_period_hours', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ),
    )
    op.create_index('idx_tier_transitions_account', 'tier_transitions', ['account_id'])
    op.create_index('idx_tier_transitions_timestamp', 'tier_transitions', ['timestamp'])
    
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
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ),
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
        sa.ForeignKeyConstraint(['transition_id'], ['tier_transitions.id'], ),
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
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ),
        sa.ForeignKeyConstraint(['transition_id'], ['tier_transitions.id'], ),
    )
    op.create_index('idx_valley_events_account', 'valley_of_death_events', ['account_id'])
    op.create_index('idx_valley_events_timestamp', 'valley_of_death_events', ['timestamp'])
    
    # Add constraint to prevent manual tier changes (enforced at application level)
    # This is a documentation constraint - actual enforcement happens in state machine
    op.execute("""
        CREATE TRIGGER prevent_manual_tier_update
        BEFORE UPDATE ON accounts
        FOR EACH ROW
        WHEN NEW.tier != OLD.tier
        BEGIN
            SELECT CASE
                WHEN NOT EXISTS (
                    SELECT 1 FROM tier_transitions
                    WHERE account_id = NEW.id
                    AND to_tier = NEW.tier
                    AND datetime(created_at) >= datetime('now', '-1 second')
                )
                THEN RAISE(ABORT, 'Direct tier modifications not allowed. Use state machine.')
            END;
        END;
    """)


def downgrade() -> None:
    # Drop trigger first
    op.execute("DROP TRIGGER IF EXISTS prevent_manual_tier_update")
    
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('valley_of_death_events')
    op.drop_table('transition_ceremonies')
    op.drop_table('tier_feature_unlocks')
    op.drop_table('tier_gate_progress')
    op.drop_table('tier_transitions')