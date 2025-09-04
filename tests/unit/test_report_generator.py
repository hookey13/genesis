"""Unit tests for backtest report generation.

Tests report generation functionality including HTML, JSON, and text outputs.
"""

import json
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import shutil

from genesis.backtesting.report_generator import BacktestReportGenerator
from genesis.backtesting.performance_metrics import (
    PerformanceMetrics,
    TradeStatistics,
    DrawdownInfo
)


class TestBacktestReportGenerator:
    """Test BacktestReportGenerator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def report_generator(self, temp_dir):
        """Create report generator instance."""
        return BacktestReportGenerator(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        trade_stats = TradeStatistics(
            total_trades=50,
            winning_trades=30,
            losing_trades=18,
            breakeven_trades=2,
            win_rate=0.6,
            loss_rate=0.36,
            profit_factor=1.8,
            avg_win=Decimal('150.00'),
            avg_loss=Decimal('75.00'),
            largest_win=Decimal('500.00'),
            largest_loss=Decimal('200.00'),
            avg_trade=Decimal('45.00'),
            avg_trade_duration=timedelta(hours=2, minutes=30),
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            current_streak=2,
            expectancy=Decimal('45.00'),
            payoff_ratio=2.0
        )
        
        return PerformanceMetrics(
            total_return=Decimal('35.50'),
            annualized_return=0.425,
            compound_annual_growth_rate=0.40,
            volatility=0.22,
            downside_volatility=0.15,
            max_drawdown=Decimal('12.75'),
            max_drawdown_duration=timedelta(days=25),
            var_95=-0.025,
            cvar_95=-0.035,
            sharpe_ratio=1.75,
            sortino_ratio=2.50,
            calmar_ratio=3.14,
            information_ratio=1.20,
            trade_stats=trade_stats,
            best_day=Decimal('5.25'),
            worst_day=Decimal('-3.80'),
            best_month=Decimal('12.50'),
            worst_month=Decimal('-4.20'),
            positive_days=145,
            negative_days=107,
            beta=0.85,
            alpha=0.08,
            correlation=0.75,
            tracking_error=0.12,
            recovery_factor=2.78,
            ulcer_index=4.5,
            kelly_criterion=0.15
        )
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data."""
        return {
            'equity_curve': [10000, 10500, 10200, 10800, 11200, 10900, 11500],
            'timestamps': [
                datetime(2024, 1, i) for i in range(1, 8)
            ],
            'positions': []
        }
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            {
                'entry_time': '2024-01-01 09:30',
                'exit_time': '2024-01-01 11:30',
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'quantity': 0.5,
                'entry_price': 42000,
                'exit_price': 42500,
                'pnl': 250
            },
            {
                'entry_time': '2024-01-02 10:15',
                'exit_time': '2024-01-02 14:20',
                'symbol': 'ETHUSDT',
                'side': 'SHORT',
                'quantity': 5.0,
                'entry_price': 2200,
                'exit_price': 2180,
                'pnl': 100
            }
        ]
    
    @pytest.fixture
    def backtest_params(self):
        """Create sample backtest parameters."""
        return {
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 10000,
            'frequency': '1m',
            'commission': 0.001
        }
    
    def test_report_generator_initialization(self, temp_dir):
        """Test report generator initialization."""
        generator = BacktestReportGenerator(output_dir=temp_dir)
        
        assert generator.output_dir == temp_dir
        assert generator.output_dir.exists()
    
    def test_generate_complete_report(
        self, report_generator, sample_metrics, sample_portfolio_data,
        sample_trades, backtest_params
    ):
        """Test complete report generation."""
        report_path = report_generator.generate(
            metrics=sample_metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="TestStrategy",
            backtest_params=backtest_params,
            trades=sample_trades
        )
        
        assert report_path.exists()
        assert report_path.suffix == '.html'
        
        # Check other files were created
        json_path = report_path.with_suffix('.json')
        text_path = report_path.with_suffix('.txt')
        
        assert json_path.exists()
        assert text_path.exists()
    
    def test_html_report_content(
        self, report_generator, sample_metrics, sample_portfolio_data,
        backtest_params
    ):
        """Test HTML report content generation."""
        html_content = report_generator._render_html(
            metrics=sample_metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="TestStrategy",
            backtest_params=backtest_params,
            trades=None
        )
        
        # Check key sections are present
        assert "TestStrategy" in html_content
        assert "Executive Summary" in html_content
        assert "Performance Metrics" in html_content
        assert "Risk Metrics" in html_content
        assert "35.50%" in html_content  # Total return
        assert "1.75" in html_content  # Sharpe ratio
    
    def test_header_section(self, report_generator, backtest_params):
        """Test header section generation."""
        header = report_generator._create_header("TestStrategy", backtest_params)
        
        assert "TestStrategy" in header
        assert "2024-01-01" in header
        assert "2024-12-31" in header
        assert "10,000" in header  # Initial capital
    
    def test_summary_section_with_trades(self, report_generator, sample_metrics):
        """Test summary section with trade statistics."""
        summary = report_generator._create_summary_section(sample_metrics)
        
        assert "35.50%" in summary  # Total return
        assert "1.75" in summary  # Sharpe ratio
        assert "12.75%" in summary  # Max drawdown
        assert "60.0%" in summary  # Win rate
    
    def test_summary_section_no_trades(self, report_generator):
        """Test summary section without trade statistics."""
        metrics = PerformanceMetrics(
            total_return=Decimal('10.00'),
            annualized_return=0.12,
            compound_annual_growth_rate=0.11,
            volatility=0.15,
            downside_volatility=0.10,
            max_drawdown=Decimal('5.00'),
            max_drawdown_duration=timedelta(days=10),
            var_95=-0.01,
            cvar_95=-0.015,
            sharpe_ratio=0.8,
            sortino_ratio=1.2,
            calmar_ratio=2.2,
            information_ratio=0.5
        )
        
        summary = report_generator._create_summary_no_trades(metrics)
        
        assert "10.00%" in summary
        assert "12.00%" in summary  # Annualized return
        assert "15.00%" in summary  # Volatility
    
    def test_performance_section(self, report_generator, sample_metrics):
        """Test performance metrics section."""
        perf_section = report_generator._create_performance_section(sample_metrics)
        
        assert "Performance Metrics" in perf_section
        assert "35.50%" in perf_section  # Total return
        assert "42.50%" in perf_section  # Annualized return
        assert "1.750" in perf_section  # Sharpe ratio
        assert "5.25%" in perf_section  # Best day
        assert "-3.80%" in perf_section  # Worst day
    
    def test_risk_section(self, report_generator, sample_metrics):
        """Test risk metrics section."""
        risk_section = report_generator._create_risk_section(sample_metrics)
        
        assert "Risk Metrics" in risk_section
        assert "22.00%" in risk_section  # Volatility
        assert "12.75%" in risk_section  # Max drawdown
        assert "-2.50%" in risk_section  # VaR
        assert "2.78" in risk_section  # Recovery factor
    
    def test_trades_section(self, report_generator, sample_metrics, sample_trades):
        """Test trade statistics section."""
        trades_section = report_generator._create_trades_section(
            sample_metrics.trade_stats,
            sample_trades
        )
        
        assert "Trade Statistics" in trades_section
        assert "50" in trades_section  # Total trades
        assert "60.0%" in trades_section  # Win rate
        assert "1.80" in trades_section  # Profit factor
        assert "$150.00" in trades_section  # Average win
    
    def test_trade_log(self, report_generator):
        """Test trade log generation."""
        trades = [
            {
                'entry_time': '2024-01-01 09:30',
                'exit_time': '2024-01-01 11:30',
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'quantity': 0.5,
                'entry_price': 42000,
                'exit_price': 42500,
                'pnl': 250
            }
        ]
        
        trade_log = report_generator._create_trade_log(trades)
        
        assert "Recent Trades" in trade_log
        assert "BTCUSDT" in trade_log
        assert "LONG" in trade_log
        assert "$250.00" in trade_log
    
    def test_charts_section(self, report_generator, sample_portfolio_data, sample_metrics):
        """Test charts section generation."""
        charts = report_generator._create_charts_section(
            sample_portfolio_data,
            sample_metrics
        )
        
        assert "Performance Charts" in charts
        assert "Equity Curve" in charts
        assert "Drawdown" in charts
        assert "Returns Distribution" in charts
        assert "Rolling Sharpe Ratio" in charts
    
    def test_css_styles(self, report_generator):
        """Test CSS styles generation."""
        css = report_generator._get_css_styles()
        
        assert "body" in css
        assert "container" in css
        assert "positive" in css
        assert "negative" in css
        assert "@media" in css  # Responsive design
    
    def test_format_duration(self, report_generator):
        """Test duration formatting."""
        # Test days
        duration1 = timedelta(days=5, hours=3)
        assert report_generator._format_duration(duration1) == "5d 3h"
        
        # Test hours
        duration2 = timedelta(hours=8, minutes=30)
        assert report_generator._format_duration(duration2) == "8h 30m"
        
        # Test minutes only
        duration3 = timedelta(minutes=45)
        assert report_generator._format_duration(duration3) == "45m"
        
        # Test None
        assert report_generator._format_duration(None) == "N/A"
    
    def test_json_report_generation(
        self, report_generator, sample_metrics, sample_portfolio_data,
        sample_trades, backtest_params, temp_dir
    ):
        """Test JSON report generation."""
        json_path = report_generator._generate_json_report(
            metrics=sample_metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="TestStrategy",
            backtest_params=backtest_params,
            trades=sample_trades,
            report_name="test_report"
        )
        
        assert json_path.exists()
        
        # Load and validate JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['strategy'] == "TestStrategy"
        assert 'metrics' in data
        assert 'portfolio' in data
        assert 'trades' in data
        assert len(data['trades']) == 2
    
    def test_metrics_to_dict(self, report_generator, sample_metrics):
        """Test metrics to dictionary conversion."""
        metrics_dict = report_generator._metrics_to_dict(sample_metrics)
        
        assert 'returns' in metrics_dict
        assert 'risk' in metrics_dict
        assert 'risk_adjusted' in metrics_dict
        assert 'trades' in metrics_dict
        assert 'benchmark' in metrics_dict
        
        assert metrics_dict['returns']['total_return'] == 35.50
        assert metrics_dict['risk_adjusted']['sharpe_ratio'] == 1.75
        assert metrics_dict['trades']['win_rate'] == 0.6
        assert metrics_dict['benchmark']['beta'] == 0.85
    
    def test_text_summary_generation(
        self, report_generator, sample_metrics, backtest_params, temp_dir
    ):
        """Test text summary generation."""
        text_path = report_generator._generate_text_summary(
            metrics=sample_metrics,
            strategy_name="TestStrategy",
            backtest_params=backtest_params,
            report_name="test_report"
        )
        
        assert text_path.exists()
        
        content = text_path.read_text()
        
        assert "BACKTEST REPORT - TestStrategy" in content
        assert "Total Return: 35.50%" in content
        assert "Sharpe Ratio: 1.750" in content
        assert "Win Rate: 60.0%" in content
        assert "Project GENESIS Backtest Engine" in content
    
    def test_report_with_no_trades(
        self, report_generator, sample_portfolio_data, backtest_params, temp_dir
    ):
        """Test report generation with no trades."""
        metrics = PerformanceMetrics(
            total_return=Decimal('5.00'),
            annualized_return=0.06,
            compound_annual_growth_rate=0.055,
            volatility=0.10,
            downside_volatility=0.08,
            max_drawdown=Decimal('3.00'),
            max_drawdown_duration=timedelta(days=5),
            var_95=-0.01,
            cvar_95=-0.012,
            sharpe_ratio=0.4,
            sortino_ratio=0.6,
            calmar_ratio=1.8,
            information_ratio=0.3
        )
        
        report_path = report_generator.generate(
            metrics=metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="NoTradeStrategy",
            backtest_params=backtest_params,
            trades=None
        )
        
        assert report_path.exists()
        
        # Check content doesn't have trade sections
        content = report_path.read_text()
        assert "No trades executed" in content or "Trade Statistics" not in content
    
    def test_large_trade_log_truncation(self, report_generator):
        """Test that large trade logs are truncated."""
        # Create 150 trades
        trades = [
            {
                'entry_time': f'2024-01-{i:02d} 09:30',
                'exit_time': f'2024-01-{i:02d} 11:30',
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'quantity': 0.5,
                'entry_price': 42000 + i * 10,
                'exit_price': 42100 + i * 10,
                'pnl': 50 + i
            }
            for i in range(1, 151)
        ]
        
        trade_log = report_generator._create_trade_log(trades)
        
        # Should be empty for too many trades
        assert trade_log == ""
    
    def test_report_file_naming(
        self, report_generator, sample_metrics, sample_portfolio_data,
        backtest_params
    ):
        """Test report file naming convention."""
        report_path = report_generator.generate(
            metrics=sample_metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="TestStrategy",
            backtest_params=backtest_params,
            trades=None
        )
        
        # Check filename format
        filename = report_path.stem
        assert "TestStrategy" in filename
        assert len(filename.split('_')) >= 2  # Strategy_timestamp
    
    def test_empty_metrics_handling(
        self, report_generator, sample_portfolio_data, backtest_params
    ):
        """Test handling of minimal metrics."""
        metrics = PerformanceMetrics(
            total_return=Decimal('0'),
            annualized_return=0.0,
            compound_annual_growth_rate=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            max_drawdown=Decimal('0'),
            max_drawdown_duration=timedelta(0),
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0
        )
        
        # Should not raise an error
        report_path = report_generator.generate(
            metrics=metrics,
            portfolio_data=sample_portfolio_data,
            strategy_name="EmptyStrategy",
            backtest_params=backtest_params,
            trades=[]
        )
        
        assert report_path.exists()