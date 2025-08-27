"""Integration tests for correlation monitoring workflow."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from genesis.analytics.correlation import (
    CorrelationMonitor,
    MarketState,
)
from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Position
from genesis.data.correlation_repo import CorrelationRepository
from genesis.engine.event_bus import EventBus
from genesis.ui.widgets.correlation_heatmap import CorrelationHeatmap


@pytest.fixture
async def test_environment():
    """Set up test environment with all components."""
    # Create event bus
    event_bus = EventBus()

    # Load test configuration
    config = {
        'correlation_monitoring': {
            'thresholds': {
                'warning': 0.6,
                'critical': 0.8
            },
            'analysis': {
                'cache_ttl_seconds': 5,
                'historical_window_days': 30,
                'min_positions_for_analysis': 2
            },
            'alerting': {
                'persist_to_database': True,
                'alert_cooldown_minutes': 15,
                'max_alerts_per_day': 50
            },
            'decorrelation': {
                'suggestion_threshold': 0.7,
                'max_suggestions': 5,
                'min_position_size_percent': 10
            },
            'stress_testing': {
                'default_spike': 0.8,
                'volatility_assumption': 0.2
            }
        }
    }

    # Create components
    correlation_monitor = CorrelationMonitor(event_bus=event_bus, config=config)
    correlation_repo = Mock(spec=CorrelationRepository)
    correlation_repo.save_correlation = AsyncMock(return_value=uuid4())
    correlation_repo.save_correlation_matrix = AsyncMock(return_value=3)
    correlation_repo.save_correlation_alert = AsyncMock()
    correlation_repo.get_correlation = AsyncMock(return_value=None)
    correlation_repo.get_high_correlations = AsyncMock(return_value=[])

    # Create UI widget
    heatmap_widget = CorrelationHeatmap(
        correlation_monitor=correlation_monitor,
        event_bus=event_bus
    )

    return {
        'event_bus': event_bus,
        'correlation_monitor': correlation_monitor,
        'correlation_repo': correlation_repo,
        'heatmap_widget': heatmap_widget,
        'config': config
    }


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return [
        Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="BTC/USDT",
            side="LONG",
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("2.0"),
            opened_at=datetime.now(UTC),
            closed_at=None
        ),
        Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="ETH/USDT",
            side="LONG",
            entry_price=Decimal("3000"),
            current_price=Decimal("3100"),
            quantity=Decimal("5"),
            dollar_value=Decimal("15500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("3.33"),
            opened_at=datetime.now(UTC),
            closed_at=None
        ),
        Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="SOL/USDT",
            side="SHORT",
            entry_price=Decimal("100"),
            current_price=Decimal("95"),
            quantity=Decimal("50"),
            dollar_value=Decimal("4750"),
            pnl_dollars=Decimal("250"),
            pnl_percent=Decimal("5.0"),
            opened_at=datetime.now(UTC),
            closed_at=None
        )
    ]


class TestCorrelationWorkflow:
    """Test complete correlation monitoring workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_correlation_monitoring(self, test_environment, sample_positions):
        """Test complete correlation monitoring workflow."""
        env = await test_environment
        monitor = env['correlation_monitor']
        repo = env['correlation_repo']

        # Step 1: Calculate correlation matrix
        correlation_matrix = await monitor.calculate_correlation_matrix(sample_positions)
        assert correlation_matrix.shape == (3, 3)
        assert np.all(np.diag(correlation_matrix) == 1.0)

        # Step 2: Save correlation to database
        await repo.save_correlation_matrix(
            [p.position_id for p in sample_positions],
            correlation_matrix.tolist()
        )
        assert repo.save_correlation_matrix.called

        # Step 3: Check thresholds and generate alerts
        with patch.object(monitor, 'calculate_correlation_matrix') as mock_calc:
            # Mock high correlation
            mock_calc.return_value = np.array([
                [1.0, 0.85, 0.3],
                [0.85, 1.0, 0.2],
                [0.3, 0.2, 1.0]
            ])

            alerts = await monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts) > 0

            # Step 4: Save alerts to database
            for alert in alerts:
                await repo.save_correlation_alert(alert)
            assert repo.save_correlation_alert.call_count == len(alerts)

        # Step 5: Generate decorrelation suggestions
        suggestions = await monitor.suggest_decorrelation_trades(sample_positions)
        assert isinstance(suggestions, list)

        # Step 6: Perform stress test
        stress_result = await monitor.run_stress_test(sample_positions, correlation_spike=0.9)
        assert stress_result.correlation_spike == Decimal("0.9")
        assert stress_result.max_drawdown > 0

    @pytest.mark.asyncio
    async def test_real_time_updates_via_event_bus(self, test_environment, sample_positions):
        """Test real-time correlation updates through event bus."""
        env = await test_environment
        monitor = env['correlation_monitor']
        event_bus = env['event_bus']

        alerts_received = []

        # Subscribe to risk alerts
        async def alert_handler(event: Event):
            if event.type == EventType.RISK_ALERT:
                alerts_received.append(event.data['alert'])

        event_bus.subscribe(EventType.RISK_ALERT, alert_handler, EventPriority.HIGH)

        # Generate high correlation scenario
        with patch.object(monitor, 'calculate_correlation_matrix') as mock_calc:
            mock_calc.return_value = np.array([
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.1],
                [0.1, 0.1, 1.0]
            ])

            alerts = await monitor.check_correlation_thresholds(sample_positions)

            # Allow event processing
            await asyncio.sleep(0.1)

            # Verify alerts were published
            assert len(alerts_received) == len(alerts)
            assert alerts_received[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_correlation_caching_performance(self, test_environment, sample_positions):
        """Test correlation calculation caching for performance."""
        env = await test_environment
        monitor = env['correlation_monitor']

        # Mock expensive calculation
        call_count = 0
        original_get_historical = monitor._get_historical_prices

        def mock_get_historical(position):
            nonlocal call_count
            call_count += 1
            return original_get_historical(position)

        monitor._get_historical_prices = mock_get_historical

        # First call - should calculate
        result1 = await monitor.calculate_correlation_matrix(sample_positions)
        first_call_count = call_count

        # Second immediate call - should use cache
        result2 = await monitor.calculate_correlation_matrix(sample_positions)

        assert call_count == first_call_count  # No additional calls
        assert np.array_equal(result1, result2)

        # Wait for cache expiry
        await asyncio.sleep(6)  # Cache TTL is 5 seconds

        # Third call - should recalculate
        result3 = await monitor.calculate_correlation_matrix(sample_positions)
        assert call_count > first_call_count

    @pytest.mark.asyncio
    async def test_pre_trade_correlation_check(self, test_environment, sample_positions):
        """Test pre-trade correlation impact analysis."""
        env = await test_environment
        monitor = env['correlation_monitor']

        # Existing positions
        existing = sample_positions[:2]

        # New position to add
        new_position = Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="BTC/USDT",  # Same as existing - high correlation expected
            side="LONG",
            entry_price=Decimal("50500"),
            current_price=Decimal("51500"),
            quantity=Decimal("0.3"),
            dollar_value=Decimal("15450"),
            pnl_dollars=Decimal("300"),
            pnl_percent=Decimal("2.0"),
            opened_at=datetime.now(UTC),
            closed_at=None
        )

        # Check correlation impact
        impact = await monitor.calculate_correlation_impact(new_position, existing)

        assert impact.projected_correlation > impact.current_correlation
        assert impact.correlation_change > 0
        assert impact.risk_assessment in ["low", "medium", "high"]
        assert impact.recommendation != ""

    @pytest.mark.asyncio
    async def test_market_regime_analysis(self, test_environment, sample_positions):
        """Test correlation analysis under different market regimes."""
        env = await test_environment
        monitor = env['correlation_monitor']

        results = {}

        for regime in MarketState:
            result = await monitor.analyze_by_market_regime(sample_positions[:2], regime)
            results[regime] = result

            assert result['market_state'] == regime.value
            assert 'base_correlation' in result
            assert 'adjusted_correlation' in result
            assert 'regime_multiplier' in result
            assert 'risk_assessment' in result

        # Volatile markets should show higher adjusted correlation
        assert results[MarketState.VOLATILE]['adjusted_correlation'] > \
               results[MarketState.CALM]['adjusted_correlation']

    @pytest.mark.asyncio
    async def test_ui_widget_integration(self, test_environment, sample_positions):
        """Test UI widget integration with correlation monitor."""
        env = await test_environment
        widget = env['heatmap_widget']
        monitor = env['correlation_monitor']

        # Update widget with positions
        await widget.update_correlation(sample_positions)

        # Check widget state
        assert len(widget.positions) == len(sample_positions)
        assert widget.correlation_matrix.size > 0
        assert isinstance(widget.last_update, datetime)

        # Get summary
        summary = widget.get_correlation_summary()
        assert summary['positions'] == len(sample_positions)
        assert 'avg_correlation' in summary
        assert 'high_correlation_pairs' in summary

    @pytest.mark.asyncio
    async def test_database_persistence_recovery(self, test_environment, sample_positions):
        """Test saving and recovering correlation data from database."""
        env = await test_environment
        repo = env['correlation_repo']

        # Save correlation
        correlation_id = await repo.save_correlation(
            sample_positions[0].position_id,
            sample_positions[1].position_id,
            Decimal("0.75"),
            calculation_window=30,
            alert_triggered=True
        )

        assert repo.save_correlation.called
        assert isinstance(correlation_id, UUID)

        # Mock retrieval
        repo.get_correlation.return_value = {
            "correlation_id": correlation_id,
            "correlation_coefficient": Decimal("0.75"),
            "calculation_window": 30,
            "last_calculated": datetime.now(UTC),
            "alert_triggered": True
        }

        # Retrieve correlation
        retrieved = await repo.get_correlation(
            sample_positions[0].position_id,
            sample_positions[1].position_id
        )

        assert retrieved is not None
        assert retrieved["correlation_coefficient"] == Decimal("0.75")
        assert retrieved["alert_triggered"] is True

    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, test_environment, sample_positions):
        """Test alert cooldown and daily limit enforcement."""
        env = await test_environment
        monitor = env['correlation_monitor']

        with patch.object(monitor, 'calculate_correlation_matrix') as mock_calc:
            # High correlation that triggers alerts
            mock_calc.return_value = np.array([
                [1.0, 0.85, 0.1],
                [0.85, 1.0, 0.1],
                [0.1, 0.1, 1.0]
            ])

            # First alert should work
            alerts1 = await monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts1) == 1

            # Immediate second call - cooldown should prevent alert
            alerts2 = await monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts2) == 0

            # Test daily limit
            monitor.daily_alert_count = 49
            monitor.alert_history.clear()  # Clear cooldown

            alerts3 = await monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts3) == 1
            assert monitor.daily_alert_count == 50

            # Next should be blocked by daily limit
            monitor.alert_history.clear()  # Clear cooldown
            alerts4 = await monitor.check_correlation_thresholds(sample_positions)
            assert len(alerts4) == 0
