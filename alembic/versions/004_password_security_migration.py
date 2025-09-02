"""Add password security fields and history table.

Revision ID: 004_password_security
Revises: 003_sqlite_to_postgres
Create Date: 2025-01-02

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_password_security'
down_revision = '003_sqlite_to_postgres'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema for password security."""
    
    # Add columns to users table for password migration tracking
    op.add_column('users', 
        sa.Column('sha256_migrated', sa.Boolean(), nullable=False, server_default='false')
    )
    op.add_column('users',
        sa.Column('password_changed_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column('users',
        sa.Column('old_sha256_hash', sa.String(64), nullable=True)
    )
    op.add_column('users',
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False, server_default='0')
    )
    op.add_column('users',
        sa.Column('last_failed_login', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column('users',
        sa.Column('last_successful_login', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column('users',
        sa.Column('is_locked', sa.Boolean(), nullable=False, server_default='false')
    )
    op.add_column('users',
        sa.Column('totp_secret', sa.String(32), nullable=True)
    )
    op.add_column('users',
        sa.Column('totp_enabled', sa.Boolean(), nullable=False, server_default='false')
    )
    
    # Create password history table
    op.create_table('user_password_history',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('idx_user_password_history_user_id', 'user_password_history', ['user_id'])
    op.create_index('idx_user_password_history_created', 'user_password_history', ['created_at'])
    op.create_index('idx_users_username_lower', 'users', [sa.text('lower(username)')])
    op.create_index('idx_users_email_lower', 'users', [sa.text('lower(email)')])
    
    # Create backup codes table for 2FA
    op.create_table('user_backup_codes',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('code_hash', sa.String(60), nullable=False),  # bcrypt hash
        sa.Column('used', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_user_backup_codes_user_id', 'user_backup_codes', ['user_id'])
    
    # Create password reset tokens table
    op.create_table('password_reset_tokens',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('token_hash', sa.String(64), nullable=False),  # SHA256 hash
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('used', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index('idx_password_reset_tokens_token_hash', 'password_reset_tokens', ['token_hash'])
    op.create_index('idx_password_reset_tokens_user_id', 'password_reset_tokens', ['user_id'])
    op.create_index('idx_password_reset_tokens_expires_at', 'password_reset_tokens', ['expires_at'])
    
    # Mark all existing users as needing migration from SHA256
    op.execute("""
        UPDATE users 
        SET sha256_migrated = false,
            old_sha256_hash = password_hash
        WHERE password_hash IS NOT NULL 
        AND LENGTH(password_hash) = 64
    """)
    
    # For any users with bcrypt hashes already (60 chars), mark as migrated
    op.execute("""
        UPDATE users 
        SET sha256_migrated = true
        WHERE password_hash IS NOT NULL 
        AND LENGTH(password_hash) = 60
    """)


def downgrade():
    """Downgrade database schema."""
    
    # Drop tables
    op.drop_table('password_reset_tokens')
    op.drop_table('user_backup_codes')
    op.drop_table('user_password_history')
    
    # Drop indexes
    op.drop_index('idx_users_email_lower', table_name='users')
    op.drop_index('idx_users_username_lower', table_name='users')
    
    # Remove columns from users table
    op.drop_column('users', 'totp_enabled')
    op.drop_column('users', 'totp_secret')
    op.drop_column('users', 'is_locked')
    op.drop_column('users', 'last_successful_login')
    op.drop_column('users', 'last_failed_login')
    op.drop_column('users', 'failed_login_attempts')
    op.drop_column('users', 'old_sha256_hash')
    op.drop_column('users', 'password_changed_at')
    op.drop_column('users', 'sha256_migrated')