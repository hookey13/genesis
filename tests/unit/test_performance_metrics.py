"""Unit tests for performance metrics calculation.

Tests all metric calculations including Sharpe ratio, Sortino ratio,
Calmar ratio, drawdown tracking, and trade statistics.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np

from genesis.backtesting.performance_metrics import (
    PerformanceCalculator,
    PerformanceMetrics,
    DrawdownInfo,
    TradeStatistics,
    RollingMetrics
)


class TestPerformanceCalculator:
    """Test PerformanceCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return PerformanceCalculator(
            risk_free_rate=0.02,
            mar=0.0,
            periods_per_year=252
        )
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve."""
        # Simulated equity curve with some volatility
        return [
            Decimal('10000'),
            Decimal('10200'),
            Decimal('10100'),
            Decimal('10300'),
            Decimal('10250'),
            Decimal('10500'),
            Decimal('10400'),
            Decimal('10600'),
            Decimal('10550'),
            Decimal('10700')
        ]
    
    @pytest.fixture
    def sample_timestamps(self):
        """Create sample timestamps."""
        base_time = datetime(2024, 1, 1)
        return [base_time + timedelta(days=i) for i in range(10)]
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            {'pnl': 100, 'duration': timedelta(hours=2)},
            {'pnl': -50, 'duration': timedelta(hours=1)},
            {'pnl': 200, 'duration': timedelta(hours=3)},
            {'pnl': -30, 'duration': timedelta(hours=1.5)},
            {'pnl': 150, 'duration': timedelta(hours=2.5)},
            {'pnl': -80, 'duration': timedelta(hours=1)},
            {'pnl': 120, 'duration': timedelta(hours=2)},
            {'pnl': 0, 'duration': timedelta(hours=1)},  # Breakeven
            {'pnl': 90, 'duration': timedelta(hours=1.5)},
            {'pnl': -40, 'duration': timedelta(hours=2)}
        ]
    
    def test_calculate_metrics_basic(
        self, calculator, sample_equity_curve, sample_timestamps, sample_trades
    ):
        """Test basic metrics calculation."""
        metrics = calculator.calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            timestamps=sample_timestamps,
            initial_capital=Decimal('10000')
        )
        
        assert metrics is not None
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == Decimal('7.00')  # 10700/10000 - 1 = 7%
        assert metrics.sharpe_ratio != 0
        assert metrics.sortino_ratio != 0
    
    def test_total_return_calculation(self, calculator):
        """Test total return calculation."""
        equity_curve = [Decimal('10000'), Decimal('11000')]
        
        total_return = calculator._total_return(
            equity_curve,
            Decimal('10000')
        )
        
        assert total_return == Decimal('10.00')  # 10% return
    
    def test_sharpe_ratio_calculation(self, calculator):
        """Test Sharpe ratio calculation."""
        # Returns with known Sharpe ratio
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.01])
        
        sharpe = calculator._sharpe_ratio(tuple(returns))
        
        # Sharpe should be positive for positive returns
        assert sharpe > 0
    
    def test_sharpe_ratio_zero_volatility(self, calculator):
        """Test Sharpe ratio with zero volatility."""
        # Constant returns (no volatility)
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        
        sharpe = calculator._sharpe_ratio(tuple(returns))
        
        # Should return 0 for zero volatility
        assert sharpe == 0.0
    
    def test_sortino_ratio_calculation(self, calculator):
        """Test Sortino ratio calculation."""
        # Returns with downside volatility
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.005])
        
        sortino = calculator._sortino_ratio(tuple(returns))
        
        # Sortino should be calculated
        assert sortino != 0
    
    def test_sortino_ratio_no_downside(self, calculator):
        """Test Sortino ratio with no downside returns."""
        # All positive returns
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.01])
        
        sortino = calculator._sortino_ratio(tuple(returns))
        
        # Should return 0 when no downside risk
        assert sortino == 0.0
    
    def test_calmar_ratio_calculation(self, calculator):
        """Test Calmar ratio calculation."""
        annualized_return = 0.15  # 15% annual return
        max_drawdown = Decimal('10')  # 10% drawdown
        
        calmar = calculator._calmar_ratio(annualized_return, max_drawdown)
        
        assert abs(calmar - 1.5) < 0.01  # 15% / 10% = 1.5
    
    def test_calmar_ratio_zero_drawdown(self, calculator):
        """Test Calmar ratio with zero drawdown."""
        annualized_return = 0.15
        max_drawdown = Decimal('0')
        
        calmar = calculator._calmar_ratio(annualized_return, max_drawdown)
        
        assert calmar == 0.0
    
    def test_drawdown_calculation(self, calculator):
        """Test maximum drawdown calculation."""
        # Equity curve with clear drawdown
        equity_curve = [
            Decimal('10000'),
            Decimal('11000'),  # Peak
            Decimal('10500'),  # Drawdown starts
            Decimal('10000'),  # Trough
            Decimal('10800'),  # Recovery
            Decimal('11000')   # Back to peak
        ]
        
        timestamps = [
            datetime(2024, 1, i) for i in range(1, 7)
        ]
        
        drawdown_info = calculator._calculate_drawdown(equity_curve, timestamps)
        
        # Allow for small rounding differences
        assert abs(float(drawdown_info.max_drawdown) - 9.09) < 0.01  # (11000-10000)/11000
        assert drawdown_info.peak_value == Decimal('11000')
        assert drawdown_info.trough_value == Decimal('10000')
        assert drawdown_info.recovery_time is not None
    
    def test_drawdown_no_recovery(self, calculator):
        """Test drawdown without recovery."""
        equity_curve = [
            Decimal('10000'),
            Decimal('11000'),  # Peak
            Decimal('10500'),
            Decimal('10000'),  # Trough - no recovery
        ]
        
        timestamps = [
            datetime(2024, 1, i) for i in range(1, 5)
        ]
        
        drawdown_info = calculator._calculate_drawdown(equity_curve, timestamps)
        
        # Allow for small rounding differences  
        assert abs(float(drawdown_info.max_drawdown) - 9.09) < 0.01
        assert drawdown_info.recovery_time is None
    
    def test_var_cvar_calculation(self, calculator):
        """Test VaR and CVaR calculation."""
        # Returns with known distribution
        returns = np.array([
            -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05
        ])
        
        var_95, cvar_95 = calculator._calculate_var_cvar(returns, confidence=0.95)
        
        # VaR should be around -0.04 (5th percentile)
        assert var_95 < 0
        # CVaR should be more negative than VaR
        assert cvar_95 <= var_95
    
    def test_trade_statistics(self, calculator):
        """Test trade statistics calculation."""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': -30},
            {'pnl': 150},
            {'pnl': 0}  # Breakeven
        ]
        
        stats = calculator._calculate_trade_statistics(trades)
        
        assert stats.total_trades == 6
        assert stats.winning_trades == 3
        assert stats.losing_trades == 2
        assert stats.breakeven_trades == 1
        assert stats.win_rate == 0.5  # 3/6
        assert stats.avg_win == Decimal('150.00')  # (100+200+150)/3
        assert stats.avg_loss == Decimal('40.00')  # (50+30)/2
        # Profit factor is 450/80 = 5.625, not 450/40
        assert abs(stats.profit_factor - 5.625) < 0.01  # 450/80
    
    def test_trade_statistics_no_losses(self, calculator):
        """Test trade statistics with no losing trades."""
        trades = [
            {'pnl': 100},
            {'pnl': 200},
            {'pnl': 150}
        ]
        
        stats = calculator._calculate_trade_statistics(trades)
        
        assert stats.winning_trades == 3
        assert stats.losing_trades == 0
        assert stats.win_rate == 1.0
        assert stats.profit_factor == 0.0  # No losses
    
    def test_consecutive_streaks(self, calculator):
        """Test consecutive win/loss streak calculation."""
        trades = [
            {'pnl': 100},   # Win
            {'pnl': 50},    # Win
            {'pnl': -30},   # Loss
            {'pnl': -40},   # Loss
            {'pnl': -50},   # Loss
            {'pnl': 100},   # Win
            {'pnl': 120},   # Win
        ]
        
        max_wins, max_losses, current = calculator._calculate_streaks(trades)
        
        assert max_wins == 2
        assert max_losses == 3
        assert current == 2  # Currently on 2-win streak
    
    def test_daily_returns(self, calculator):
        """Test daily returns calculation."""
        equity_curve = [
            Decimal('10000'),
            Decimal('10100'),
            Decimal('10200'),
            Decimal('10150'),
        ]
        
        timestamps = [
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 15, 0),
            datetime(2024, 1, 2, 9, 0),
            datetime(2024, 1, 2, 15, 0),
        ]
        
        daily_returns = calculator._calculate_daily_returns(equity_curve, timestamps)
        
        assert len(daily_returns) == 2
        # Allow for rounding differences
        assert abs(float(daily_returns[0][1]) - 1.00) < 0.01  # ~1% on day 1
        assert abs(float(daily_returns[1][1]) + 0.49) < 0.01  # ~-0.49% on day 2
    
    def test_rolling_metrics(self, calculator):
        """Test rolling window metrics calculation."""
        # Create longer equity curve
        equity_curve = [Decimal(10000 + i * 100) for i in range(50)]
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        
        rolling = calculator.calculate_rolling_metrics(
            equity_curve=equity_curve,
            timestamps=timestamps,
            window_size=10,
            step_size=5
        )
        
        assert isinstance(rolling, RollingMetrics)
        assert rolling.window_size == 10
        assert len(rolling.returns) > 0
        assert len(rolling.sharpe_ratios) > 0
        assert len(rolling.volatilities) > 0
    
    def test_benchmark_comparison(self, calculator):
        """Test benchmark comparison metrics."""
        equity_curve = [Decimal('10000'), Decimal('10500'), Decimal('11000')]
        timestamps = [datetime(2024, 1, i) for i in range(1, 4)]
        trades = []
        
        # Benchmark with lower returns
        benchmark = [0.02, 0.03]
        
        metrics = calculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=trades,
            timestamps=timestamps,
            benchmark=benchmark,
            initial_capital=Decimal('10000')
        )
        
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.correlation is not None
        assert metrics.tracking_error is not None
    
    def test_information_ratio(self, calculator):
        """Test information ratio calculation."""
        returns = (0.02, 0.01, 0.03, -0.01, 0.02)
        benchmark = (0.015, 0.01, 0.02, 0.005, 0.015)
        
        info_ratio = calculator._information_ratio(returns, benchmark)
        
        # Should calculate positive information ratio for outperformance
        assert info_ratio != 0
    
    def test_recovery_factor(self, calculator):
        """Test recovery factor calculation."""
        total_return = Decimal('20')  # 20% return
        max_drawdown = Decimal('10')  # 10% drawdown
        
        recovery = calculator._calculate_recovery_factor(total_return, max_drawdown)
        
        assert recovery == 2.0  # 20/10 = 2
    
    def test_ulcer_index(self, calculator):
        """Test Ulcer Index calculation."""
        # Equity curve with drawdowns
        equity_curve = [
            Decimal('10000'),
            Decimal('11000'),
            Decimal('10500'),  # Drawdown
            Decimal('10000'),  # More drawdown
            Decimal('10800'),
            Decimal('11000')
        ]
        
        ulcer = calculator._calculate_ulcer_index(equity_curve)
        
        # Ulcer index should be positive
        assert ulcer > 0
    
    def test_kelly_criterion(self, calculator):
        """Test Kelly Criterion calculation."""
        trade_stats = TradeStatistics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            breakeven_trades=0,
            win_rate=0.6,
            loss_rate=0.4,
            profit_factor=1.5,
            avg_win=Decimal('100'),
            avg_loss=Decimal('50'),
            largest_win=Decimal('300'),
            largest_loss=Decimal('100'),
            avg_trade=Decimal('20'),
            avg_trade_duration=timedelta(hours=2),
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            current_streak=2,
            expectancy=Decimal('20'),
            payoff_ratio=2.0
        )
        
        kelly = calculator._calculate_kelly_criterion(trade_stats)
        
        # Kelly should be positive for profitable system
        assert kelly > 0
        # Should be capped at 25%
        assert kelly <= 0.25
    
    def test_performance_summary(self, calculator):
        """Test performance summary generation."""
        metrics = PerformanceMetrics(
            total_return=Decimal('15.50'),
            annualized_return=0.185,
            compound_annual_growth_rate=0.175,
            volatility=0.20,
            downside_volatility=0.15,
            max_drawdown=Decimal('12.30'),
            max_drawdown_duration=timedelta(days=30),
            var_95=-0.02,
            cvar_95=-0.03,
            sharpe_ratio=1.25,
            sortino_ratio=1.50,
            calmar_ratio=1.42,
            information_ratio=0.85
        )
        
        summary = calculator.generate_performance_summary(metrics)
        
        assert 'returns' in summary
        assert 'risk' in summary
        assert 'risk_adjusted' in summary
        assert summary['returns']['total_return'] == '15.50%'
        assert summary['risk_adjusted']['sharpe_ratio'] == '1.25'
    
    def test_empty_data_handling(self, calculator):
        """Test handling of empty data."""
        metrics = calculator.calculate_metrics(
            equity_curve=[],
            trades=[],
            timestamps=[],
            initial_capital=Decimal('10000')
        )
        
        assert metrics.total_return == Decimal('0')
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
    
    def test_single_value_handling(self, calculator):
        """Test handling of single value inputs."""
        metrics = calculator.calculate_metrics(
            equity_curve=[Decimal('10000')],
            trades=[],
            timestamps=[datetime.now()],
            initial_capital=Decimal('10000')
        )
        
        assert metrics.total_return == Decimal('0')
        assert metrics.volatility == 0.0


class TestDrawdownInfo:
    """Test DrawdownInfo dataclass."""
    
    def test_drawdown_info_creation(self):
        """Test DrawdownInfo creation."""
        info = DrawdownInfo(
            max_drawdown=Decimal('15.5'),
            max_drawdown_duration=timedelta(days=20),
            current_drawdown=Decimal('5.0'),
            peak_value=Decimal('11000'),
            trough_value=Decimal('9350'),
            recovery_time=timedelta(days=10),
            drawdown_start=datetime(2024, 1, 1),
            drawdown_end=datetime(2024, 1, 20)
        )
        
        assert info.max_drawdown == Decimal('15.5')
        assert info.max_drawdown_duration.days == 20
        assert info.recovery_time.days == 10


class TestTradeStatistics:
    """Test TradeStatistics dataclass."""
    
    def test_trade_statistics_creation(self):
        """Test TradeStatistics creation."""
        stats = TradeStatistics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            breakeven_trades=0,
            win_rate=0.6,
            loss_rate=0.4,
            profit_factor=1.8,
            avg_win=Decimal('120'),
            avg_loss=Decimal('60'),
            largest_win=Decimal('500'),
            largest_loss=Decimal('200'),
            avg_trade=Decimal('36'),
            avg_trade_duration=timedelta(hours=3),
            max_consecutive_wins=8,
            max_consecutive_losses=5,
            current_streak=3,
            expectancy=Decimal('36'),
            payoff_ratio=2.0
        )
        
        assert stats.total_trades == 100
        assert stats.win_rate == 0.6
        assert stats.profit_factor == 1.8
        assert stats.expectancy == Decimal('36')


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            total_return=Decimal('25.50'),
            annualized_return=0.30,
            compound_annual_growth_rate=0.28,
            volatility=0.18,
            downside_volatility=0.12,
            max_drawdown=Decimal('10.0'),
            max_drawdown_duration=timedelta(days=15),
            var_95=-0.015,
            cvar_95=-0.020,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.8,
            information_ratio=0.9
        )
        
        assert metrics.total_return == Decimal('25.50')
        assert metrics.sharpe_ratio == 1.5
        assert metrics.calmar_ratio == 2.8
        
    def test_optional_fields(self):
        """Test optional fields in PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_return=Decimal('10'),
            annualized_return=0.12,
            compound_annual_growth_rate=0.11,
            volatility=0.15,
            downside_volatility=0.10,
            max_drawdown=Decimal('5'),
            max_drawdown_duration=timedelta(days=10),
            var_95=-0.01,
            cvar_95=-0.015,
            sharpe_ratio=0.8,
            sortino_ratio=1.2,
            calmar_ratio=2.2,
            information_ratio=0.5,
            beta=1.2,
            alpha=0.02,
            correlation=0.85
        )
        
        assert metrics.beta == 1.2
        assert metrics.alpha == 0.02
        assert metrics.correlation == 0.85