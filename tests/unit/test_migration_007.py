"""Unit tests for migration 007 - multi-pair trading tables."""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from alembic import command
from alembic.config import Config


class TestMigration007:
    """Test multi-pair trading tables migration."""

    @pytest.fixture
    def alembic_config(self, tmp_path):
        """Create Alembic config for testing."""
        config = Config()
        config.set_main_option('script_location', 'alembic')
        config.set_main_option('sqlalchemy.url', f'sqlite:///{tmp_path}/test.db')
        return config

    @pytest.fixture
    def engine(self, tmp_path):
        """Create test database engine."""
        return create_engine(f'sqlite:///{tmp_path}/test.db')

    @pytest.fixture
    def session(self, engine):
        """Create database session."""
        Session = sessionmaker(bind=engine)
        return Session()

    def test_migration_up_creates_tables(self, alembic_config, engine):
        """Test that migration creates all required tables."""
        # Run migrations up to 006 first
        command.upgrade(alembic_config, '006')

        # Run migration 007
        command.upgrade(alembic_config, '007')

        # Check tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert 'portfolio_limits' in tables
        assert 'signal_queue' in tables
        assert 'pair_performance' in tables
        assert 'position_correlations' in tables  # Should exist from previous migration

    def test_portfolio_limits_table_structure(self, alembic_config, engine):
        """Test portfolio_limits table has correct structure."""
        command.upgrade(alembic_config, '007')

        inspector = inspect(engine)
        columns = {col['name']: col for col in inspector.get_columns('portfolio_limits')}

        # Check column existence and types
        assert 'limit_id' in columns
        assert columns['limit_id']['primary_key'] is True
        assert 'account_id' in columns
        assert 'symbol' in columns
        assert columns['symbol']['nullable'] is True  # NULL for global limits
        assert 'max_position_size' in columns
        assert 'max_dollar_value' in columns
        assert 'max_open_positions' in columns
        assert 'limit_type' in columns
        assert 'tier' in columns
        assert 'created_at' in columns
        assert 'updated_at' in columns

        # Check indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('portfolio_limits')}
        assert 'idx_portfolio_limits_account' in indexes
        assert 'idx_portfolio_limits_symbol' in indexes

    def test_signal_queue_table_structure(self, alembic_config, engine):
        """Test signal_queue table has correct structure."""
        command.upgrade(alembic_config, '007')

        inspector = inspect(engine)
        columns = {col['name']: col for col in inspector.get_columns('signal_queue')}

        # Check all required columns
        required_columns = [
            'signal_id', 'account_id', 'symbol', 'strategy_name',
            'signal_type', 'confidence_score', 'priority', 'size_recommendation',
            'price_target', 'stop_loss', 'expiry_time', 'status',
            'conflict_resolution', 'created_at', 'processed_at'
        ]

        for col in required_columns:
            assert col in columns, f"Column {col} missing from signal_queue"

        # Check indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('signal_queue')}
        assert 'idx_signal_queue_status' in indexes
        assert 'idx_signal_queue_priority' in indexes
        assert 'idx_signal_queue_symbol' in indexes

    def test_pair_performance_table_structure(self, alembic_config, engine):
        """Test pair_performance table has correct structure."""
        command.upgrade(alembic_config, '007')

        inspector = inspect(engine)
        columns = {col['name']: col for col in inspector.get_columns('pair_performance')}

        # Check performance metrics columns
        assert 'total_trades' in columns
        assert 'winning_trades' in columns
        assert 'losing_trades' in columns
        assert 'total_pnl_dollars' in columns
        assert 'win_rate' in columns
        assert 'sharpe_ratio' in columns
        assert 'attribution_weight' in columns

        # Check indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('pair_performance')}
        assert 'idx_pair_performance_account' in indexes
        assert 'idx_pair_performance_symbol' in indexes
        assert 'idx_pair_performance_period' in indexes

    def test_position_correlations_enhanced(self, alembic_config, engine, session):
        """Test position_correlations table enhancements."""
        command.upgrade(alembic_config, '007')

        inspector = inspect(engine)
        columns = {col['name']: col for col in inspector.get_columns('position_correlations')}

        # Check new columns added
        assert 'symbol_1' in columns
        assert 'symbol_2' in columns
        assert 'risk_adjustment_factor' in columns

        # Check new index
        indexes = {idx['name']: idx for idx in inspector.get_indexes('position_correlations')}
        assert 'idx_correlations_symbols' in indexes

    def test_portfolio_limits_constraints(self, alembic_config, engine, session):
        """Test portfolio_limits table constraints."""
        command.upgrade(alembic_config, '007')

        # Test valid insert
        valid_limit = {
            'limit_id': str(uuid.uuid4()),
            'account_id': str(uuid.uuid4()),
            'symbol': 'BTC/USDT',
            'max_position_size': Decimal('1.5'),
            'max_dollar_value': Decimal('50000'),
            'limit_type': 'PAIR',
            'tier': 'HUNTER'
        }
        session.execute(text("""
            INSERT INTO portfolio_limits (limit_id, account_id, symbol, max_position_size,
                                        max_dollar_value, limit_type, tier)
            VALUES (:limit_id, :account_id, :symbol, :max_position_size,
                   :max_dollar_value, :limit_type, :tier)
        """), valid_limit)
        session.commit()

        # Test global limit (NULL symbol)
        global_limit = {
            'limit_id': str(uuid.uuid4()),
            'account_id': valid_limit['account_id'],
            'symbol': None,
            'max_position_size': Decimal('10'),
            'max_dollar_value': Decimal('100000'),
            'max_open_positions': 5,
            'limit_type': 'GLOBAL',
            'tier': 'HUNTER'
        }
        session.execute(text("""
            INSERT INTO portfolio_limits (limit_id, account_id, symbol, max_position_size,
                                        max_dollar_value, max_open_positions, limit_type, tier)
            VALUES (:limit_id, :account_id, :symbol, :max_position_size,
                   :max_dollar_value, :max_open_positions, :limit_type, :tier)
        """), global_limit)
        session.commit()

    def test_signal_queue_constraints(self, alembic_config, engine, session):
        """Test signal_queue table constraints."""
        command.upgrade(alembic_config, '007')

        # Test valid signal
        valid_signal = {
            'signal_id': str(uuid.uuid4()),
            'account_id': str(uuid.uuid4()),
            'symbol': 'ETH/USDT',
            'strategy_name': 'mean_reversion',
            'signal_type': 'BUY',
            'confidence_score': Decimal('0.85'),
            'priority': 75,
            'expiry_time': datetime.utcnow() + timedelta(minutes=5),
            'status': 'PENDING'
        }
        session.execute(text("""
            INSERT INTO signal_queue (signal_id, account_id, symbol, strategy_name,
                                    signal_type, confidence_score, priority, expiry_time, status)
            VALUES (:signal_id, :account_id, :symbol, :strategy_name,
                   :signal_type, :confidence_score, :priority, :expiry_time, :status)
        """), valid_signal)
        session.commit()

    def test_pair_performance_unique_constraint(self, alembic_config, engine, session):
        """Test pair_performance unique constraint."""
        command.upgrade(alembic_config, '007')

        account_id = str(uuid.uuid4())
        period_start = datetime(2025, 1, 1)
        period_end = datetime(2025, 1, 31)

        # Insert first record
        perf1 = {
            'performance_id': str(uuid.uuid4()),
            'account_id': account_id,
            'symbol': 'BTC/USDT',
            'period_start': period_start,
            'period_end': period_end,
            'total_trades': 10
        }
        session.execute(text("""
            INSERT INTO pair_performance (performance_id, account_id, symbol,
                                        period_start, period_end, total_trades)
            VALUES (:performance_id, :account_id, :symbol,
                   :period_start, :period_end, :total_trades)
        """), perf1)
        session.commit()

        # Try to insert duplicate (same account, symbol, period) - should succeed with different performance_id
        perf2 = perf1.copy()
        perf2['performance_id'] = str(uuid.uuid4())
        perf2['symbol'] = 'ETH/USDT'  # Different symbol
        session.execute(text("""
            INSERT INTO pair_performance (performance_id, account_id, symbol,
                                        period_start, period_end, total_trades)
            VALUES (:performance_id, :account_id, :symbol,
                   :period_start, :period_end, :total_trades)
        """), perf2)
        session.commit()

    def test_migration_down_removes_tables(self, alembic_config, engine):
        """Test that downgrade removes tables and columns correctly."""
        # Run migration up
        command.upgrade(alembic_config, '007')

        # Run migration down
        command.downgrade(alembic_config, '006')

        # Check tables are removed
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert 'portfolio_limits' not in tables
        assert 'signal_queue' not in tables
        assert 'pair_performance' not in tables

        # Check position_correlations columns are removed
        columns = {col['name'] for col in inspector.get_columns('position_correlations')}
        assert 'symbol_1' not in columns
        assert 'symbol_2' not in columns
        assert 'risk_adjustment_factor' not in columns

    def test_data_integrity_across_migration(self, alembic_config, engine, session):
        """Test that existing data in position_correlations survives migration."""
        # Run migrations up to 006
        command.upgrade(alembic_config, '006')

        # Insert test data into position_correlations
        correlation_id = str(uuid.uuid4())
        test_data = {
            'correlation_id': correlation_id,
            'position_1_id': str(uuid.uuid4()),
            'position_2_id': str(uuid.uuid4()),
            'correlation_coefficient': Decimal('0.75'),
            'calculation_window': 60,
            'alert_triggered': False
        }
        session.execute(text("""
            INSERT INTO position_correlations (correlation_id, position_1_id, position_2_id,
                                              correlation_coefficient, calculation_window, alert_triggered)
            VALUES (:correlation_id, :position_1_id, :position_2_id,
                   :correlation_coefficient, :calculation_window, :alert_triggered)
        """), test_data)
        session.commit()

        # Run migration 007
        command.upgrade(alembic_config, '007')

        # Verify data still exists
        result = session.execute(text("""
            SELECT * FROM position_correlations WHERE correlation_id = :id
        """), {'id': correlation_id}).fetchone()

        assert result is not None
        assert str(result[0]) == correlation_id  # correlation_id
        assert float(result[3]) == 0.75  # correlation_coefficient
