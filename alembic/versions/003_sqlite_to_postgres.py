"""SQLite to PostgreSQL migration

Revision ID: 003
Revises: 002
Create Date: 2025-08-25

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    # Placeholder migration for future SQLite to PostgreSQL transition
    pass


def downgrade():
    # Placeholder migration
    pass