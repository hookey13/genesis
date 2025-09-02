"""Add 2FA tables and columns

Revision ID: 004_add_2fa
Revises: 003_sqlite_to_postgres
Create Date: 2025-09-02

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_add_2fa'
down_revision = '003_sqlite_to_postgres'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add 2FA columns to users table and create two_fa_attempts table."""
    
    # Add 2FA columns to users table
    op.add_column('users', sa.Column('two_fa_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('users', sa.Column('two_fa_required', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('users', sa.Column('two_fa_setup_at', sa.DateTime(timezone=True), nullable=True))
    
    # Create two_fa_attempts table for rate limiting
    op.create_table('two_fa_attempts',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('ip_address', postgresql.INET(), nullable=False),
        sa.Column('code_used', sa.String(length=10), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('attempted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index for efficient querying
    op.create_index('idx_2fa_attempts_user_time', 'two_fa_attempts', ['user_id', 'attempted_at'])


def downgrade() -> None:
    """Remove 2FA tables and columns."""
    
    # Drop index
    op.drop_index('idx_2fa_attempts_user_time', 'two_fa_attempts')
    
    # Drop two_fa_attempts table
    op.drop_table('two_fa_attempts')
    
    # Remove 2FA columns from users table
    op.drop_column('users', 'two_fa_setup_at')
    op.drop_column('users', 'two_fa_required')
    op.drop_column('users', 'two_fa_enabled')