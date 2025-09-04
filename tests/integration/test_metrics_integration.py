"""Integration tests for performance metrics and reporting system.

Comprehensive tests to validate the complete metrics calculation pipeline
including edge cases, large datasets, and report generation.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import json
import numpy as np

from genesis.backtesting.performance_metrics import (
    PerformanceCalculator,
    PerformanceMetrics,
    TradeStatistics
)
from genesis.backtesting.report_generator import BacktestReportGenerator
from genesis.monitoring.performance_attribution import (
    PerformanceAttributor,
    AttributionResult
)


class TestMetricsIntegration:
    """Integration tests for complete metrics system."""
    
    @pytest.fixture
    def realistic_portfolio_data(self):
        """Generate realistic portfolio data for testing."""
        # Simulate 252 trading days (1 year)
        np.random.seed(42)
        
        initial_capital = Decimal('10000')
        equity_curve = [initial_capital]
        timestamps = []
        trades = []
        
        base_time = datetime(2024, 1, 1, 9, 30)
        current_equity = initial_capital
        
        # Generate realistic equity curve with volatility
        for day in range(252):
            # Daily return between -3% and 3%
            daily_return = np.random.normal(0.0005, 0.015)  # ~0.05% mean, 1.5% std dev
            current_equity = current_equity * Decimal(str(1 + daily_return))
            equity_curve.append(current_equity)
            timestamps.append(base_time + timedelta(days=day))
            
            # Generate trades (about 2 per week)
            if np.random.random() < 0.4:  # 40% chance of trade
                trade_pnl = float(current_equity) * np.random.normal(0.002, 0.01)
                trades.append({
                    'pnl': trade_pnl,
                    'duration': timedelta(hours=np.random.randint(1, 48)),
                    'entry_time': (base_time + timedelta(days=day)).isoformat(),
                    'exit_time': (base_time + timedelta(days=day, hours=4)).isoformat(),
                    'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
                    'side': np.random.choice(['LONG', 'SHORT']),
                    'quantity': np.random.uniform(0.01, 1.0),
                    'entry_price': np.random.uniform(20000, 50000),
                    'exit_price': np.random.uniform(20000, 50000)
                })
        
        return {
            'equity_curve': equity_curve,
            'timestamps': timestamps,
            'trades': trades,
            'initial_capital': initial_capital
        }
    
    def test_full_metrics_calculation_pipeline(self, realistic_portfolio_data):
        """Test complete metrics calculation with realistic data."""
        calculator = PerformanceCalculator(
            risk_free_rate=0.02,
            mar=0.0,
            periods_per_year=252
        )
        
        metrics = calculator.calculate_metrics(
            equity_curve=realistic_portfolio_data['equity_curve'],
            trades=realistic_portfolio_data['trades'],
            timestamps=realistic_portfolio_data['timestamps'],
            initial_capital=realistic_portfolio_data['initial_capital']
        )
        
        # Validate all metrics are calculated
        assert metrics.total_return is not None
        assert metrics.annualized_return != 0
        assert metrics.volatility > 0
        assert metrics.max_drawdown >= 0
        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.calmar_ratio is not None
        
        # Validate trade statistics if trades exist
        if realistic_portfolio_data['trades']:
            assert metrics.trade_stats is not None
            assert metrics.trade_stats.total_trades == len(realistic_portfolio_data['trades'])
            assert 0 <= metrics.trade_stats.win_rate <= 1
            assert metrics.trade_stats.profit_factor >= 0
    
    def test_report_generation_integration(self, realistic_portfolio_data, tmp_path):
        """Test complete report generation pipeline."""
        # Calculate metrics
        calculator = PerformanceCalculator()
        metrics = calculator.calculate_metrics(
            equity_curve=realistic_portfolio_data['equity_curve'],
            trades=realistic_portfolio_data['trades'],
            timestamps=realistic_portfolio_data['timestamps'],
            initial_capital=realistic_portfolio_data['initial_capital']
        )
        
        # Generate reports
        report_gen = BacktestReportGenerator(output_dir=tmp_path)
        report_path = report_gen.generate(
            metrics=metrics,
            portfolio_data=realistic_portfolio_data,
            strategy_name="IntegrationTest",
            backtest_params={
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'initial_capital': 10000,
                'frequency': '1d'
            },
            trades=realistic_portfolio_data['trades']
        )
        
        # Validate report files
        assert report_path.exists()
        assert report_path.suffix == '.html'
        
        # Check HTML content
        html_content = report_path.read_text()
        assert 'IntegrationTest' in html_content
        assert 'Executive Summary' in html_content
        assert 'svg' in html_content.lower()  # Charts are generated
        
        # Check JSON report
        json_path = report_path.with_suffix('.json')
        assert json_path.exists()
        
        with open(json_path) as f:
            json_data = json.load(f)
        
        assert json_data['strategy'] == 'IntegrationTest'
        assert 'metrics' in json_data
        assert 'portfolio' in json_data
        
        # Check text report
        text_path = report_path.with_suffix('.txt')
        assert text_path.exists()
        text_content = text_path.read_text()
        assert 'BACKTEST REPORT' in text_content
    
    def test_extreme_market_conditions(self):
        """Test metrics with extreme market conditions."""
        calculator = PerformanceCalculator()
        
        # Test massive drawdown
        crash_equity = [
            Decimal('10000'),
            Decimal('11000'),
            Decimal('12000'),
            Decimal('6000'),  # 50% crash
            Decimal('7000'),
            Decimal('8000')
        ]
        crash_timestamps = [datetime(2024, 1, i) for i in range(1, 7)]
        
        metrics = calculator.calculate_metrics(
            equity_curve=crash_equity,
            trades=[],
            timestamps=crash_timestamps,
            initial_capital=Decimal('10000')
        )
        
        assert metrics.max_drawdown >= Decimal('50')
        assert metrics.calmar_ratio < 0  # Negative return with drawdown
        
        # Test extreme volatility
        volatile_returns = np.array([0.5, -0.4, 0.3, -0.35, 0.45, -0.3])
        volatility = calculator._calculate_volatility(volatile_returns)
        assert volatility > 5  # Very high annualized volatility
    
    def test_rolling_metrics_accuracy(self):
        """Test rolling window calculations."""
        calculator = PerformanceCalculator()
        
        # Create steady uptrend with known characteristics
        equity_curve = [Decimal(10000 + i * 100) for i in range(100)]
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        
        rolling = calculator.calculate_rolling_metrics(
            equity_curve=equity_curve,
            timestamps=timestamps,
            window_size=30,
            step_size=10
        )
        
        assert len(rolling.sharpe_ratios) > 0
        assert len(rolling.returns) > 0
        assert len(rolling.volatilities) > 0
        
        # All windows should show positive returns for steady uptrend
        assert all(r > 0 for r in rolling.returns)
    
    def test_benchmark_comparison_integration(self):
        """Test benchmark comparison functionality."""
        calculator = PerformanceCalculator()
        
        # Portfolio returns
        equity_curve = [
            Decimal('10000'),
            Decimal('10500'),
            Decimal('10300'),
            Decimal('10800'),
            Decimal('11200')
        ]
        timestamps = [datetime(2024, 1, i) for i in range(1, 6)]
        
        # Benchmark returns (slightly lower)
        benchmark = [0.03, -0.01, 0.02, 0.015]
        
        metrics = calculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=[],
            timestamps=timestamps,
            benchmark=benchmark,
            initial_capital=Decimal('10000')
        )
        
        # Should have benchmark metrics
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.correlation is not None
        assert metrics.tracking_error is not None
        
        # Portfolio outperformed, so alpha should be positive
        assert metrics.alpha > 0
    
    def test_performance_attribution(self):
        """Test performance attribution analysis."""
        attributor = PerformanceAttributor()
        
        portfolio_returns = [0.02, 0.01, -0.005, 0.015, 0.025]
        portfolio_weights = [
            {'BTC': 0.5, 'ETH': 0.3, 'BNB': 0.2},
            {'BTC': 0.6, 'ETH': 0.25, 'BNB': 0.15},
            {'BTC': 0.55, 'ETH': 0.3, 'BNB': 0.15},
            {'BTC': 0.5, 'ETH': 0.35, 'BNB': 0.15},
            {'BTC': 0.45, 'ETH': 0.35, 'BNB': 0.2}
        ]
        asset_returns = {
            'BTC': [0.03, 0.015, -0.01, 0.02, 0.03],
            'ETH': [0.025, 0.005, 0.0, 0.01, 0.025],
            'BNB': [0.01, 0.01, -0.005, 0.015, 0.02]
        }
        timestamps = [datetime(2024, 1, i) for i in range(1, 6)]
        
        attribution = attributor.attribute_performance(
            portfolio_returns=portfolio_returns,
            portfolio_weights=portfolio_weights,
            asset_returns=asset_returns,
            timestamps=timestamps
        )
        
        assert isinstance(attribution, AttributionResult)
        assert attribution.total_return != 0
        assert len(attribution.asset_attribution) == 3
        assert 'BTC' in attribution.asset_attribution
    
    def test_edge_cases_comprehensive(self):
        """Test comprehensive edge cases."""
        calculator = PerformanceCalculator()
        
        # Test with single trade
        single_trade = [{'pnl': 100, 'duration': timedelta(hours=1)}]
        stats = calculator._calculate_trade_statistics(single_trade)
        assert stats.total_trades == 1
        assert stats.win_rate == 1.0
        assert stats.profit_factor == 0.0  # No losses
        
        # Test with all losing trades
        losing_trades = [
            {'pnl': -50},
            {'pnl': -100},
            {'pnl': -75}
        ]
        stats = calculator._calculate_trade_statistics(losing_trades)
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0
        
        # Test with zero volatility
        flat_equity = [Decimal('10000')] * 10
        timestamps = [datetime(2024, 1, i) for i in range(1, 11)]
        
        metrics = calculator.calculate_metrics(
            equity_curve=flat_equity,
            trades=[],
            timestamps=timestamps,
            initial_capital=Decimal('10000')
        )
        
        assert metrics.total_return == Decimal('0')
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_large_dataset_performance(self):
        """Test with large dataset for performance."""
        calculator = PerformanceCalculator()
        
        # Generate 5 years of daily data
        large_equity = []
        large_timestamps = []
        base_time = datetime(2020, 1, 1)
        current_value = Decimal('10000')
        
        for day in range(252 * 5):  # 5 years
            current_value *= Decimal(str(1 + np.random.normal(0.0003, 0.01)))
            large_equity.append(current_value)
            large_timestamps.append(base_time + timedelta(days=day))
        
        # Generate many trades
        large_trades = [
            {
                'pnl': np.random.normal(50, 100),
                'duration': timedelta(hours=np.random.randint(1, 72))
            }
            for _ in range(1000)
        ]
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        metrics = calculator.calculate_metrics(
            equity_curve=large_equity,
            trades=large_trades,
            timestamps=large_timestamps,
            initial_capital=Decimal('10000')
        )
        
        elapsed = time.time() - start_time
        
        # Should complete in under 5 seconds even with large dataset
        assert elapsed < 5.0
        assert metrics.trade_stats.total_trades == 1000
    
    def test_monthly_period_metrics(self):
        """Test monthly and period performance metrics."""
        calculator = PerformanceCalculator()
        
        # Create equity curve spanning multiple months
        equity_curve = []
        timestamps = []
        base_time = datetime(2024, 1, 1)
        
        # January - positive month
        for day in range(31):
            equity_curve.append(Decimal(10000 + day * 50))
            timestamps.append(base_time + timedelta(days=day))
        
        # February - negative month
        for day in range(29):
            equity_curve.append(Decimal(11550 - day * 30))
            timestamps.append(base_time + timedelta(days=31 + day))
        
        # March - recovery month
        for day in range(31):
            equity_curve.append(Decimal(10680 + day * 40))
            timestamps.append(base_time + timedelta(days=60 + day))
        
        metrics = calculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=[],
            timestamps=timestamps,
            initial_capital=Decimal('10000')
        )
        
        # Should have calculated best/worst months
        assert metrics.best_month > 0
        assert metrics.worst_month < 0
        assert metrics.positive_days > 0
        assert metrics.negative_days > 0
    
    def test_metrics_serialization(self):
        """Test that all metrics can be serialized properly."""
        calculator = PerformanceCalculator()
        
        equity_curve = [Decimal('10000'), Decimal('10500'), Decimal('11000')]
        timestamps = [datetime(2024, 1, i) for i in range(1, 4)]
        trades = [{'pnl': 250}, {'pnl': 250}]
        
        metrics = calculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=trades,
            timestamps=timestamps,
            initial_capital=Decimal('10000')
        )
        
        # Convert to summary dict
        summary = calculator.generate_performance_summary(metrics)
        
        # Should be JSON serializable
        json_str = json.dumps(summary, default=str)
        assert len(json_str) > 0
        
        # Deserialize and verify
        loaded = json.loads(json_str)
        assert 'returns' in loaded
        assert 'risk' in loaded
        assert 'risk_adjusted' in loaded