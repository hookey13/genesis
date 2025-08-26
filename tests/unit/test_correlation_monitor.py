"""Unit tests for CorrelationMonitor class."""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import UUID, uuid4
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from genesis.analytics.correlation import (
    CorrelationMonitor,
    MarketState,
    CorrelationAlert,
    TradeSuggestion,
    CorrelationImpact,
    StressTestResult
)
from genesis.core.models import Position, PositionSide
from genesis.core.constants import TradingTier
from genesis.core.events import Event, EventType, EventPriority
from genesis.engine.event_bus import EventBus


@pytest.fixture
def correlation_monitor():
    """Create a CorrelationMonitor instance for testing."""
    event_bus = Mock(spec=EventBus)
    event_bus.publish = AsyncMock()
    
    config = {
        'correlation_monitoring': {
            'thresholds': {
                'warning': 0.6,
                'critical': 0.8
            },
            'analysis': {
                'cache_ttl_seconds': 5
            },
            'alerting': {
                'alert_cooldown_minutes': 15,
                'max_alerts_per_day': 50
            }
        }
    }
    
    monitor = CorrelationMonitor(event_bus=event_bus, config=config)
    # Set tier to STRATEGIST for testing (higher than required HUNTER)
    from genesis.core.constants import TradingTier
    monitor.tier = TradingTier.STRATEGIST
    return monitor


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return [
        Position(
            position_id="11111111-1111-1111-1111-111111111111",
            account_id="22222222-2222-2222-2222-222222222222",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("2.0"),
            created_at=datetime.now(timezone.utc)
        ),
        Position(
            position_id="33333333-3333-3333-3333-333333333333",
            account_id="22222222-2222-2222-2222-222222222222",
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("3000"),
            current_price=Decimal("3100"),
            quantity=Decimal("5"),
            dollar_value=Decimal("15500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("3.33"),
            created_at=datetime.now(timezone.utc)
        ),
        Position(
            position_id="44444444-4444-4444-4444-444444444444",
            account_id="22222222-2222-2222-2222-222222222222",
            symbol="SOL/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("100"),
            current_price=Decimal("95"),
            quantity=Decimal("50"),
            dollar_value=Decimal("4750"),
            pnl_dollars=Decimal("250"),
            pnl_percent=Decimal("5.0"),
            created_at=datetime.now(timezone.utc)
        )
    ]


class TestCorrelationMonitor:
    """Test suite for CorrelationMonitor."""
    
    @pytest.mark.asyncio
    async def test_calculate_correlation_matrix_empty_positions(self, correlation_monitor):
        """Test correlation matrix calculation with empty positions."""
        result = await correlation_monitor.calculate_correlation_matrix([])
        assert result.size == 0
        assert isinstance(result, np.ndarray)
        
    @pytest.mark.asyncio
    async def test_calculate_correlation_matrix_single_position(self, correlation_monitor, sample_positions):
        """Test correlation matrix calculation with single position."""
        result = await correlation_monitor.calculate_correlation_matrix([sample_positions[0]])
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0
        
    @pytest.mark.asyncio
    async def test_calculate_correlation_matrix_multiple_positions(self, correlation_monitor, sample_positions):
        """Test correlation matrix calculation with multiple positions."""
        result = await correlation_monitor.calculate_correlation_matrix(sample_positions)
        assert result.shape == (3, 3)
        # Check diagonal is 1
        assert np.allclose(np.diag(result), 1.0)
        # Check matrix is symmetric
        assert np.allclose(result, result.T)
        # Check values are in range [-1, 1]
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
        
    @pytest.mark.asyncio
    async def test_correlation_matrix_caching(self, correlation_monitor, sample_positions):
        """Test that correlation matrix is cached properly."""
        # First call - calculates and caches
        result1 = await correlation_monitor.calculate_correlation_matrix(sample_positions[:2])
        
        # Second call - should return cached value
        result2 = await correlation_monitor.calculate_correlation_matrix(sample_positions[:2])
        
        assert np.array_equal(result1, result2)
        assert len(correlation_monitor.correlation_cache) == 1
        
    @pytest.mark.asyncio
    async def test_track_correlation_history(self, correlation_monitor, sample_positions):
        """Test correlation history tracking."""
        result = await correlation_monitor.track_correlation_history(sample_positions[:2], window_days=7)
        
        assert result["window_days"] == 7
        assert "start_date" in result
        assert "end_date" in result
        assert "daily_correlations" in result
        assert len(result["daily_correlations"]) == 8  # 7 days + today
        assert "average_period_correlation" in result
        
        # Check daily correlation structure
        for daily in result["daily_correlations"]:
            assert "date" in daily
            assert "average_correlation" in daily
            assert "max_correlation" in daily
            
    @pytest.mark.asyncio
    async def test_analyze_by_market_regime(self, correlation_monitor, sample_positions):
        """Test correlation analysis by market regime."""
        for market_state in MarketState:
            result = await correlation_monitor.analyze_by_market_regime(
                sample_positions[:2], 
                market_state
            )
            
            assert result["market_state"] == market_state.value
            assert "base_correlation" in result
            assert "adjusted_correlation" in result
            assert "regime_multiplier" in result
            assert "risk_assessment" in result
            
            # Check regime multiplier effects
            if market_state == MarketState.VOLATILE:
                assert result["regime_multiplier"] == 1.3
            elif market_state == MarketState.CALM:
                assert result["regime_multiplier"] == 0.8
                
    @pytest.mark.asyncio
    async def test_check_correlation_thresholds_no_alert(self, correlation_monitor, sample_positions):
        """Test correlation threshold checking with no alerts."""
        # Mock low correlation
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.3, 0.2],
                [0.3, 1.0, 0.4],
                [0.2, 0.4, 1.0]
            ])
            
            alerts = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts) == 0
            
    @pytest.mark.asyncio
    async def test_check_correlation_thresholds_with_alerts(self, correlation_monitor, sample_positions):
        """Test correlation threshold checking with alerts."""
        # Mock high correlation
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.85, 0.2],
                [0.85, 1.0, 0.65],
                [0.2, 0.65, 1.0]
            ])
            
            alerts = await correlation_monitor.check_correlation_thresholds(sample_positions)
            
            assert len(alerts) == 2  # Two pairs exceed threshold
            
            # Check first alert (BTC/ETH correlation)
            alert1 = alerts[0]
            assert alert1.correlation_level == Decimal("0.85")
            assert alert1.severity == "critical"
            assert len(alert1.affected_positions) == 2
            assert alert1.alert_id is not None
            
            # Check second alert (ETH/SOL correlation)
            alert2 = alerts[1]
            assert alert2.correlation_level == Decimal("0.65")
            assert alert2.severity == "warning"
            assert alert2.alert_id is not None
            
            # Verify event bus was called
            assert correlation_monitor.event_bus.publish.call_count == 2
            
    @pytest.mark.asyncio
    async def test_alert_cooldown(self, correlation_monitor, sample_positions):
        """Test alert cooldown functionality."""
        # Mock high correlation
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.85, 0.2],
                [0.85, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ])
            
            # First call should generate alerts
            alerts1 = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts1) == 1
            
            # Second call immediately after should not generate alerts due to cooldown
            alerts2 = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts2) == 0
            
            # Simulate time passing beyond cooldown
            pair_key = "_".join(sorted([str(sample_positions[0].position_id), str(sample_positions[1].position_id)]))
            correlation_monitor.alert_history[pair_key] = datetime.now(timezone.utc) - timedelta(minutes=20)
            
            # Third call should generate alerts again
            alerts3 = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts3) == 1
            
    @pytest.mark.asyncio
    async def test_daily_alert_limit(self, correlation_monitor, sample_positions):
        """Test daily alert limit enforcement."""
        # Set daily count near limit
        correlation_monitor.daily_alert_count = 49
        correlation_monitor.max_alerts_per_day = 50
        
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.85, 0.7],
                [0.85, 1.0, 0.75],
                [0.7, 0.75, 1.0]
            ])
            
            # Should generate only 1 alert before hitting limit
            alerts = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts) == 1
            assert correlation_monitor.daily_alert_count == 50
            
            # Next call should return empty due to daily limit
            alerts2 = await correlation_monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts2) == 0
            
    @pytest.mark.asyncio
    async def test_suggest_decorrelation_trades(self, correlation_monitor, sample_positions):
        """Test decorrelation trade suggestions."""
        # Mock high correlation
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.75, 0.3],
                [0.75, 1.0, 0.2],
                [0.3, 0.2, 1.0]
            ])
            
            suggestions = await correlation_monitor.suggest_decorrelation_trades(sample_positions)
            
            assert len(suggestions) >= 1
            
            # Check suggestion structure
            suggestion = suggestions[0]
            assert isinstance(suggestion, TradeSuggestion)
            assert suggestion.action == "reduce"
            assert suggestion.suggested_quantity > 0
            assert suggestion.expected_correlation_impact > 0
            assert suggestion.transaction_cost_estimate > 0
            assert suggestion.rationale != ""
            
    @pytest.mark.asyncio
    async def test_calculate_correlation_impact_no_existing(self, correlation_monitor, sample_positions):
        """Test correlation impact with no existing positions."""
        new_position = sample_positions[0]
        
        impact = await correlation_monitor.calculate_correlation_impact(new_position, [])
        
        assert impact.current_correlation == Decimal("0")
        assert impact.projected_correlation == Decimal("0")
        assert impact.correlation_change == Decimal("0")
        assert impact.risk_assessment == "low"
        assert "No existing positions" in impact.recommendation
        
    @pytest.mark.asyncio
    async def test_calculate_correlation_impact_with_existing(self, correlation_monitor, sample_positions):
        """Test correlation impact with existing positions."""
        new_position = sample_positions[2]
        existing = sample_positions[:2]
        
        with patch.object(correlation_monitor, 'calculate_correlation_matrix') as mock_calc:
            # Mock current correlation (2 positions)
            mock_calc.side_effect = [
                np.array([[1.0, 0.5], [0.5, 1.0]]),  # Current
                np.array([  # Projected with new position
                    [1.0, 0.5, 0.7],
                    [0.5, 1.0, 0.8],
                    [0.7, 0.8, 1.0]
                ])
            ]
            
            impact = await correlation_monitor.calculate_correlation_impact(new_position, existing)
            
            assert impact.current_correlation == Decimal("0.5")
            assert impact.projected_correlation > impact.current_correlation
            assert impact.correlation_change > 0
            assert impact.risk_assessment in ["low", "medium", "high"]
            assert impact.recommendation != ""
            
    @pytest.mark.asyncio
    async def test_run_stress_test_empty_portfolio(self, correlation_monitor):
        """Test stress test with empty portfolio."""
        result = await correlation_monitor.run_stress_test([])
        
        assert result.scenario == "Empty portfolio"
        assert result.portfolio_impact == Decimal("0")
        assert result.max_drawdown == Decimal("0")
        assert len(result.positions_at_risk) == 0
        
    @pytest.mark.asyncio
    async def test_run_stress_test_with_positions(self, correlation_monitor, sample_positions):
        """Test stress test with positions."""
        result = await correlation_monitor.run_stress_test(sample_positions, correlation_spike=0.9)
        
        assert result.scenario == "Correlation spike to 90%"
        assert result.correlation_spike == Decimal("0.9")
        assert result.portfolio_impact > 0
        assert result.max_drawdown > 0
        assert isinstance(result.positions_at_risk, list)
        assert isinstance(result.timestamp, datetime)
        
        # Check that high-value positions are identified as at-risk
        total_value = sum(p.dollar_value for p in sample_positions)
        assert result.max_drawdown <= total_value  # Drawdown shouldn't exceed total value
        
    def test_calculate_average_correlation(self, correlation_monitor):
        """Test average correlation calculation."""
        # Test empty matrix
        assert correlation_monitor._calculate_average_correlation(np.array([])) == 0.0
        
        # Test single element
        assert correlation_monitor._calculate_average_correlation(np.array([[1.0]])) == 0.0
        
        # Test 2x2 matrix
        matrix_2x2 = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert correlation_monitor._calculate_average_correlation(matrix_2x2) == 0.5
        
        # Test 3x3 matrix
        matrix_3x3 = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        avg = correlation_monitor._calculate_average_correlation(matrix_3x3)
        expected = (0.6 + 0.3 + 0.4) / 3
        assert abs(avg - expected) < 0.01
        
    def test_assess_regime_risk(self, correlation_monitor):
        """Test regime risk assessment."""
        # High correlation in volatile market
        high_corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        risk = correlation_monitor._assess_regime_risk(high_corr_matrix, MarketState.VOLATILE)
        assert "Very High" in risk
        
        # Low correlation in calm market
        low_corr_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
        risk = correlation_monitor._assess_regime_risk(low_corr_matrix, MarketState.CALM)
        assert "Low" in risk
        
        # Medium correlation in trending market
        med_corr_matrix = np.array([[1.0, 0.6], [0.6, 1.0]])
        risk = correlation_monitor._assess_regime_risk(med_corr_matrix, MarketState.TRENDING_UP)
        assert "Medium" in risk
        
    def test_get_cache_key(self, correlation_monitor, sample_positions):
        """Test cache key generation."""
        key1 = correlation_monitor._get_cache_key(sample_positions[:2])
        key2 = correlation_monitor._get_cache_key(sample_positions[:2][::-1])  # Reversed order
        
        # Cache key should be same regardless of order
        assert key1 == key2
        
        # Different positions should have different keys
        key3 = correlation_monitor._get_cache_key(sample_positions[1:3])
        assert key1 != key3
        
    @pytest.mark.asyncio
    async def test_tier_restrictions(self, correlation_monitor, sample_positions):
        """Test that methods require proper tier."""
        # Methods should be decorated with @requires_tier
        methods_to_check = [
            correlation_monitor.calculate_correlation_matrix,
            correlation_monitor.track_correlation_history,
            correlation_monitor.analyze_by_market_regime,
            correlation_monitor.check_correlation_thresholds,
            correlation_monitor.suggest_decorrelation_trades,
            correlation_monitor.calculate_correlation_impact,
            correlation_monitor.run_stress_test
        ]
        
        for method in methods_to_check:
            # Check that the method has tier decorator metadata
            assert hasattr(method, '__wrapped__'), f"{method.__name__} should have @requires_tier decorator"