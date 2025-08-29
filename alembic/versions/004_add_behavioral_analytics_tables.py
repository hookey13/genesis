"""Add behavioral analytics tables

Revision ID: 004
Revises: 003
Create Date: 2025-08-25

"""


from alembic import op

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    """Create behavioral analytics tables if they don't exist."""

    # Check if tables already exist (they were created in models_db.py)
    # These SQL commands will only create if not exists

    # behavioral_metrics table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS behavioral_metrics (
            metric_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES tilt_profiles(profile_id),
            session_id TEXT REFERENCES trading_sessions(session_id),
            metric_type TEXT NOT NULL,
            metric_value TEXT NOT NULL,
            metadata TEXT,
            timestamp TIMESTAMP NOT NULL,
            session_context TEXT,
            time_of_day_bucket INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (metric_type IN ('click_speed', 'click_latency', 'order_frequency', 'order_modification',
                   'position_size_variance', 'cancel_rate', 'tab_switches', 'inactivity_period',
                   'session_duration', 'config_change', 'focus_duration', 'distraction_score')),
            CHECK (session_context IN ('tired', 'alert', 'stressed') OR session_context IS NULL),
            CHECK (time_of_day_bucket >= 0 AND time_of_day_bucket <= 23 OR time_of_day_bucket IS NULL)
        )
    """
    )

    # Create indexes for behavioral_metrics
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavioral_metrics_profile ON behavioral_metrics(profile_id, timestamp)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavioral_metrics_analysis ON behavioral_metrics(profile_id, metric_type, time_of_day_bucket)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavioral_metrics_session ON behavioral_metrics(session_id)"
    )

    # config_changes table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS config_changes (
            change_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES tilt_profiles(profile_id),
            setting_name TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            changed_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Create indexes for config_changes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_config_changes_profile ON config_changes(profile_id, changed_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_config_changes_setting ON config_changes(setting_name, changed_at)"
    )

    # behavior_correlations table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS behavior_correlations (
            correlation_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL REFERENCES tilt_profiles(profile_id),
            behavior_type TEXT NOT NULL,
            correlation_coefficient TEXT NOT NULL,
            p_value TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            time_window_minutes INTEGER NOT NULL,
            calculated_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (p_value >= 0 AND p_value <= 1),
            CHECK (correlation_coefficient >= -1 AND correlation_coefficient <= 1)
        )
    """
    )

    # Create indexes for behavior_correlations
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavior_correlations_profile ON behavior_correlations(profile_id, calculated_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_behavior_correlations_type ON behavior_correlations(behavior_type, correlation_coefficient)"
    )


def downgrade():
    """Drop behavioral analytics tables."""
    op.execute("DROP TABLE IF EXISTS behavior_correlations")
    op.execute("DROP TABLE IF EXISTS config_changes")
    op.execute("DROP TABLE IF EXISTS behavioral_metrics")
