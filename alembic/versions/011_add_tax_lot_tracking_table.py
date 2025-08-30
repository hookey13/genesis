"""Add tax lot tracking table

Revision ID: 011
Revises: 010
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade():
    """Add tax lot tracking tables."""
    # Create tax lots table
    op.create_table(
        'tax_lots',
        sa.Column('lot_id', sa.String(36), primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('quantity', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('remaining_quantity', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('closed_quantity', sa.DECIMAL(20, 8), nullable=False, server_default='0'),
        sa.Column('cost_per_unit', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('acquired_at', sa.DateTime, nullable=False),
        sa.Column('account_id', sa.String(36), nullable=False),
        sa.Column('order_id', sa.String(36), nullable=False),
        sa.Column('status', sa.String(10), nullable=False, server_default='OPEN'),
        sa.Column('realized_pnl', sa.DECIMAL(20, 8), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime, nullable=True),
    )
    
    # Create indexes for tax lots
    op.create_index('idx_tax_lots_symbol', 'tax_lots', ['symbol'])
    op.create_index('idx_tax_lots_account', 'tax_lots', ['account_id'])
    op.create_index('idx_tax_lots_status', 'tax_lots', ['status'])
    op.create_index('idx_tax_lots_acquired', 'tax_lots', ['acquired_at'])
    
    # Create lot assignments table
    op.create_table(
        'lot_assignments',
        sa.Column('assignment_id', sa.String(36), primary_key=True),
        sa.Column('sale_id', sa.String(36), nullable=False),
        sa.Column('lot_id', sa.String(36), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('quantity', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('cost_per_unit', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('sale_price', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('sale_date', sa.DateTime, nullable=False),
        sa.Column('realized_pnl', sa.DECIMAL(20, 8), nullable=False),
        sa.Column('method_used', sa.String(10), nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['lot_id'], ['tax_lots.lot_id'], ),
    )
    
    # Create indexes for lot assignments
    op.create_index('idx_lot_assignments_sale', 'lot_assignments', ['sale_id'])
    op.create_index('idx_lot_assignments_lot', 'lot_assignments', ['lot_id'])
    op.create_index('idx_lot_assignments_date', 'lot_assignments', ['sale_date'])
    
    # Add lot tracking columns to orders table
    op.add_column('orders', sa.Column('tax_lot_id', sa.String(36), nullable=True))
    op.add_column('orders', sa.Column('lot_assignments', sa.JSON, nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_orders_tax_lot',
        'orders', 'tax_lots',
        ['tax_lot_id'], ['lot_id'],
        ondelete='SET NULL'
    )
    
    # Add tax tracking columns to positions table
    op.add_column('positions', sa.Column('tax_lot_id', sa.String(36), nullable=True))
    op.add_column('positions', sa.Column('acquisition_date', sa.DateTime, nullable=True))
    op.add_column('positions', sa.Column('cost_basis', sa.DECIMAL(20, 8), nullable=True))
    op.add_column('positions', sa.Column('tax_method', sa.String(20), nullable=True, server_default='FIFO'))
    
    # Create foreign key for positions
    op.create_foreign_key(
        'fk_positions_tax_lot',
        'positions', 'tax_lots',
        ['tax_lot_id'], ['lot_id'],
        ondelete='SET NULL'
    )


def downgrade():
    """Remove tax lot tracking tables."""
    # Drop foreign keys first
    op.drop_constraint('fk_positions_tax_lot', 'positions', type_='foreignkey')
    op.drop_constraint('fk_orders_tax_lot', 'orders', type_='foreignkey')
    
    # Remove columns from positions
    op.drop_column('positions', 'tax_method')
    op.drop_column('positions', 'cost_basis')
    op.drop_column('positions', 'acquisition_date')
    op.drop_column('positions', 'tax_lot_id')
    
    # Remove columns from orders
    op.drop_column('orders', 'lot_assignments')
    op.drop_column('orders', 'tax_lot_id')
    
    # Drop tables
    op.drop_table('lot_assignments')
    op.drop_table('tax_lots')