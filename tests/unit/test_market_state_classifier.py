"""
Unit tests for Market State Classifier module.

Tests market state classification logic, volatility calculations,
volume anomaly detection, state transitions, and position sizing adjustments.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import ANY, AsyncMock

import pytest

from genesis.analytics.market_state_classifier import (
    Candle,
    GlobalMarketState,
    GlobalMarketStateClassifier,
    MaintenanceMonitor,
    MarketState,
    MarketStateClassifier,
    MarketStateContext,
    PositionSizeAdjuster,
    StateTransitionManager,
    StrategyStateManager,
)
from genesis.analytics.volatility_calculator import VolatilityCalculator
from genesis.engine.event_bus import EventBus


@pytest.fixture
def event_bus():
    """Create mock event bus."""
    bus = AsyncMock(spec=EventBus)
    return bus


@pytest.fixture
def classifier(event_bus):
    """Create market state classifier instance."""
    return MarketStateClassifier(event_bus=event_bus)


@pytest.fixture
def volatility_calculator():
    """Create volatility calculator instance."""
    return VolatilityCalculator()


@pytest.fixture
def sample_candles():
    """Generate sample candle data."""
    candles = []
    base_price = Decimal("50000")

    for i in range(30):
        open_price = base_price + Decimal(str(i * 100))
        high = open_price + Decimal("500")
        low = open_price - Decimal("300")
        close = open_price + Decimal("200")

        candle = Candle(
            open_time=datetime.now(UTC) - timedelta(hours=30 - i),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=Decimal("1000") + Decimal(str(i * 10)),
            close_time=datetime.now(UTC) - timedelta(hours=29 - i),
            quote_volume=Decimal("50000000"),
            trades=1000 + i * 10,
        )
        candles.append(candle)

    return candles


class TestMarketStateClassifier:
    """Test MarketStateClassifier class."""

    @pytest.mark.asyncio
    async def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.event_bus is not None
        assert classifier.hysteresis_factor == Decimal("0.1")
        assert classifier.dead_volume_threshold == Decimal("0.2")
        assert classifier.panic_correlation_threshold == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_classify_market_state_default(self, classifier):
        """Test default market state classification."""
        state = await classifier.classify_market_state("BTCUSDT")
        assert state == MarketState.NORMAL

    def test_calculate_volatility_atr(self, classifier, sample_candles):
        """Test ATR volatility calculation."""
        atr = classifier.calculate_volatility_atr(sample_candles, period=14)

        assert isinstance(atr, Decimal)
        assert atr > 0
        # ATR should be reasonable for the sample data
        assert atr < Decimal("1000")

    def test_calculate_volatility_atr_insufficient_data(self, classifier):
        """Test ATR calculation with insufficient data."""
        candles = [
            Candle(
                open_time=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                close_time=datetime.now(UTC),
                quote_volume=Decimal("50000000"),
                trades=1000,
            )
        ]

        with pytest.raises(Exception):  # DataError
            classifier.calculate_volatility_atr(candles, period=14)

    def test_calculate_realized_volatility(self, classifier, sample_candles):
        """Test realized volatility calculation."""
        prices = [c.close for c in sample_candles]
        volatility = classifier.calculate_realized_volatility(prices, window=20)

        assert isinstance(volatility, Decimal)
        assert volatility > 0
        # Annualized volatility should be reasonable
        assert volatility < Decimal("10")  # Less than 1000% annualized

    def test_detect_volume_anomaly(self, classifier):
        """Test volume anomaly detection."""
        # Create historical volumes with some variation
        historical_volumes = [Decimal("1000") + Decimal(str(i * 10)) for i in range(20)]

        # Normal volume (within 2 std devs)
        is_anomaly = classifier.detect_volume_anomaly(
            Decimal("1100"), historical_volumes
        )
        assert not is_anomaly

        # Anomalous volume (>2 std devs)
        is_anomaly = classifier.detect_volume_anomaly(
            Decimal("5000"), historical_volumes
        )
        assert is_anomaly

    def test_update_volume_profile(self, classifier):
        """Test volume profile update."""
        symbol = "BTCUSDT"
        volume = Decimal("1000")
        timestamp = datetime.now(UTC)

        classifier.update_volume_profile(symbol, volume, timestamp)

        assert symbol in classifier._volume_profiles
        profile = classifier._volume_profiles[symbol]
        assert profile.symbol == symbol
        assert timestamp.hour in profile.hourly_averages
        assert timestamp.weekday() in profile.daily_averages

    def test_detect_volume_pattern_anomaly(self, classifier):
        """Test volume pattern anomaly detection."""
        symbol = "BTCUSDT"

        # Build volume profile
        for i in range(24):
            timestamp = datetime.now(UTC).replace(hour=i)
            classifier.update_volume_profile(symbol, Decimal("1000"), timestamp)

        # Test normal volume
        is_anomaly, reason = classifier.detect_volume_pattern_anomaly(
            symbol, Decimal("1100"), datetime.now(UTC)
        )
        assert not is_anomaly

        # Test anomalous volume
        is_anomaly, reason = classifier.detect_volume_pattern_anomaly(
            symbol, Decimal("10000"), datetime.now(UTC)
        )
        assert is_anomaly
        assert "deviates" in reason

    @pytest.mark.asyncio
    async def test_update_state(self, classifier):
        """Test market state update."""
        symbol = "BTCUSDT"

        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.5"),
            volume_24h=Decimal("1000000"),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False,
        )

        assert isinstance(context, MarketStateContext)
        assert context.symbol == symbol
        assert context.current_state in MarketState
        assert context.volatility_atr == Decimal("500")

    @pytest.mark.asyncio
    async def test_maintenance_detection(self, classifier):
        """Test maintenance state detection."""
        symbol = "BTCUSDT"

        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.5"),
            volume_24h=Decimal("1000000"),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=True,  # Maintenance detected
        )

        assert context.current_state == MarketState.MAINTENANCE
        assert "maintenance" in context.reason.lower()

    def test_hysteresis_application(self, classifier):
        """Test hysteresis to prevent state flapping."""
        current_state = MarketState.NORMAL
        new_state = MarketState.VOLATILE

        # Low confidence - should not transition
        should_transition = classifier._apply_hysteresis(
            current_state, new_state, Decimal("0.4")
        )
        assert not should_transition

        # High confidence - should transition
        should_transition = classifier._apply_hysteresis(
            current_state, new_state, Decimal("0.7")
        )
        assert should_transition

        # Always allow transition to MAINTENANCE
        should_transition = classifier._apply_hysteresis(
            current_state, MarketState.MAINTENANCE, Decimal("0.1")
        )
        assert should_transition


class TestVolatilityCalculator:
    """Test VolatilityCalculator class."""

    def test_calculate_atr(self, volatility_calculator):
        """Test ATR calculation."""
        high_prices = [Decimal(str(50000 + i * 100)) for i in range(20)]
        low_prices = [Decimal(str(49500 + i * 100)) for i in range(20)]
        close_prices = [Decimal(str(49750 + i * 100)) for i in range(20)]

        atr = volatility_calculator.calculate_atr(
            high_prices, low_prices, close_prices, period=14
        )

        assert isinstance(atr, Decimal)
        assert atr > 0

    def test_calculate_atr_percentage(self, volatility_calculator):
        """Test ATR percentage calculation."""
        atr = Decimal("500")
        current_price = Decimal("50000")

        atr_pct = volatility_calculator.calculate_atr_percentage(atr, current_price)

        assert atr_pct == Decimal("1.0")  # 500/50000 * 100 = 1%

    def test_calculate_realized_volatility(self, volatility_calculator):
        """Test realized volatility calculation."""
        prices = [Decimal(str(50000 + i * 100)) for i in range(30)]

        daily_vol, annual_vol = volatility_calculator.calculate_realized_volatility(
            prices, window=20
        )

        assert isinstance(daily_vol, Decimal)
        assert isinstance(annual_vol, Decimal)
        assert annual_vol > daily_vol  # Annualized should be larger

    def test_calculate_volatility_percentile(self, volatility_calculator):
        """Test volatility percentile calculation."""
        current_vol = Decimal("0.5")
        historical_vols = [Decimal(str(0.1 + i * 0.05)) for i in range(30)]

        percentile = volatility_calculator.calculate_volatility_percentile(
            current_vol, historical_vols, lookback_days=30
        )

        assert 0 <= percentile <= 100

    def test_detect_volatility_regime(self, volatility_calculator):
        """Test volatility regime detection."""
        historical_vols = [Decimal(str(0.1 + i * 0.01)) for i in range(30)]

        # Low volatility
        regime = volatility_calculator.detect_volatility_regime(
            Decimal("0.1"), historical_vols
        )
        assert regime == "LOW"

        # High volatility
        regime = volatility_calculator.detect_volatility_regime(
            Decimal("0.5"), historical_vols
        )
        assert regime == "HIGH"

        # Normal volatility
        regime = volatility_calculator.detect_volatility_regime(
            Decimal("0.25"), historical_vols
        )
        assert regime == "NORMAL"


class TestStateTransitionManager:
    """Test StateTransitionManager class."""

    @pytest.mark.asyncio
    async def test_transition_to_state(self, event_bus):
        """Test state transition execution."""
        manager = StateTransitionManager(event_bus=event_bus)

        await manager.transition_to_state(
            symbol="BTCUSDT",
            new_state=MarketState.VOLATILE,
            reason="High volatility detected",
        )

        assert len(manager._transition_history) == 1
        transition = manager._transition_history[0]
        assert transition["symbol"] == "BTCUSDT"
        assert transition["state"] == MarketState.VOLATILE.value

        # Check position sizing event was published
        event_bus.publish.assert_called()

    def test_position_multipliers(self):
        """Test position size multipliers by state."""
        manager = StateTransitionManager()

        assert manager.position_multipliers[MarketState.DEAD] == Decimal("0.5")
        assert manager.position_multipliers[MarketState.NORMAL] == Decimal("1.0")
        assert manager.position_multipliers[MarketState.VOLATILE] == Decimal("0.75")
        assert manager.position_multipliers[MarketState.PANIC] == Decimal("0.25")
        assert manager.position_multipliers[MarketState.MAINTENANCE] == Decimal("0")


class TestMaintenanceMonitor:
    """Test MaintenanceMonitor class."""

    @pytest.mark.asyncio
    async def test_schedule_maintenance(self, classifier):
        """Test scheduling maintenance window."""
        monitor = MaintenanceMonitor(classifier)

        scheduled_time = datetime.now(UTC) + timedelta(hours=1)
        monitor.schedule_maintenance("BTCUSDT", scheduled_time)

        assert "BTCUSDT" in monitor._scheduled_maintenances
        assert monitor._scheduled_maintenances["BTCUSDT"] == scheduled_time

    def test_cancel_scheduled_maintenance(self, classifier):
        """Test canceling scheduled maintenance."""
        monitor = MaintenanceMonitor(classifier)

        scheduled_time = datetime.now(UTC) + timedelta(hours=1)
        monitor.schedule_maintenance("BTCUSDT", scheduled_time)
        monitor.cancel_scheduled_maintenance("BTCUSDT")

        assert "BTCUSDT" not in monitor._scheduled_maintenances


class TestGlobalMarketStateClassifier:
    """Test GlobalMarketStateClassifier class."""

    @pytest.mark.asyncio
    async def test_classify_global_state(self, event_bus):
        """Test global market state classification."""
        classifier = GlobalMarketStateClassifier(event_bus=event_bus)

        btc_price = Decimal("50000")
        major_pairs = [
            {"symbol": "ETHUSDT", "change_percent": Decimal("-5")},
            {"symbol": "BNBUSDT", "change_percent": Decimal("-4")},
        ]

        state = await classifier.classify_global_state(
            btc_price, major_pairs, fear_greed_index=25
        )

        assert state in GlobalMarketState

    def test_calculate_price_change(self):
        """Test BTC price change calculation."""
        classifier = GlobalMarketStateClassifier()

        # Add price history
        for i in range(25):
            classifier._btc_price_history.append(Decimal(str(50000 + i * 100)))

        change = classifier._calculate_price_change()
        assert isinstance(change, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_correlation(self):
        """Test correlation calculation among major pairs."""
        classifier = GlobalMarketStateClassifier()

        major_pairs = [
            {"symbol": "ETHUSDT", "change_percent": Decimal("5")},
            {"symbol": "BNBUSDT", "change_percent": Decimal("4")},
            {"symbol": "ADAUSDT", "change_percent": Decimal("6")},
        ]

        correlation = await classifier._calculate_correlation(major_pairs)
        assert isinstance(correlation, Decimal)
        assert 0 <= correlation <= 1

    def test_determine_global_state(self):
        """Test global state determination logic."""
        classifier = GlobalMarketStateClassifier()

        # Test CRASH state
        state = classifier._determine_global_state(
            btc_change=Decimal("-0.20"),  # 20% drop
            correlation=Decimal("0.9"),
            panic_count=3,
            fear_greed_index=10,
        )
        assert state == GlobalMarketState.CRASH

        # Test BULL state
        state = classifier._determine_global_state(
            btc_change=Decimal("0.15"),  # 15% rise
            correlation=Decimal("0.5"),
            panic_count=0,
            fear_greed_index=75,
        )
        assert state == GlobalMarketState.BULL

        # Test CRAB state (sideways)
        state = classifier._determine_global_state(
            btc_change=Decimal("0.02"),  # 2% change
            correlation=Decimal("0.3"),
            panic_count=0,
            fear_greed_index=50,
        )
        assert state == GlobalMarketState.CRAB


class TestPositionSizeAdjuster:
    """Test PositionSizeAdjuster class."""

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, event_bus):
        """Test position size calculation with adjustments."""
        adjuster = PositionSizeAdjuster(event_bus=event_bus)

        base_size = Decimal("1000")

        # Test NORMAL state - no adjustment
        size, reason = await adjuster.calculate_position_size(
            symbol="BTCUSDT",
            base_size=base_size,
            market_state=MarketState.NORMAL,
            global_state=GlobalMarketState.CRAB,
            volatility_percentile=50,
        )
        assert size == base_size

        # Test VOLATILE state - 25% reduction
        size, reason = await adjuster.calculate_position_size(
            symbol="BTCUSDT",
            base_size=base_size,
            market_state=MarketState.VOLATILE,
            global_state=None,
            volatility_percentile=None,
        )
        assert size == Decimal("750")
        assert "VOLATILE" in reason

        # Test PANIC state with global CRASH - compound reduction
        size, reason = await adjuster.calculate_position_size(
            symbol="BTCUSDT",
            base_size=base_size,
            market_state=MarketState.PANIC,
            global_state=GlobalMarketState.CRASH,
            volatility_percentile=None,
        )
        assert size <= Decimal("100")  # Heavy reduction (minimum enforced at 10%)
        assert "PANIC" in reason and "CRASH" in reason

    def test_calculate_volatility_multiplier(self):
        """Test volatility-based multiplier calculation."""
        adjuster = PositionSizeAdjuster()

        # Extreme volatility
        mult = adjuster._calculate_volatility_multiplier(95)
        assert mult == Decimal("0.5")

        # High volatility
        mult = adjuster._calculate_volatility_multiplier(80)
        assert mult == Decimal("0.75")

        # Normal volatility
        mult = adjuster._calculate_volatility_multiplier(50)
        assert mult == Decimal("1.0")

        # Low volatility
        mult = adjuster._calculate_volatility_multiplier(10)
        assert mult == Decimal("1.1")


class TestStrategyStateManager:
    """Test StrategyStateManager class."""

    @pytest.mark.asyncio
    async def test_update_strategy_states(self, event_bus):
        """Test strategy state updates based on market conditions."""
        manager = StrategyStateManager(event_bus=event_bus)

        # Test NORMAL state - all strategies enabled
        states = await manager.update_strategy_states(MarketState.NORMAL)
        assert states["arbitrage"] is True
        assert states["grid_trading"] is True
        assert states["mean_reversion"] is True

        # Test VOLATILE state - selective strategies
        states = await manager.update_strategy_states(MarketState.VOLATILE)
        assert states["arbitrage"] is True
        assert states["grid_trading"] is False  # Disabled in volatile
        assert states["momentum"] is True

        # Test MAINTENANCE state - all disabled
        states = await manager.update_strategy_states(MarketState.MAINTENANCE)
        assert all(not enabled for enabled in states.values())

    def test_is_strategy_enabled(self):
        """Test checking if strategy is enabled."""
        manager = StrategyStateManager()
        manager._active_strategies = {"arbitrage": True, "grid_trading": False}

        assert manager.is_strategy_enabled("arbitrage") is True
        assert manager.is_strategy_enabled("grid_trading") is False
        assert manager.is_strategy_enabled("unknown") is False

    def test_get_enabled_strategies(self):
        """Test getting list of enabled strategies."""
        manager = StrategyStateManager()
        manager._active_strategies = {
            "arbitrage": True,
            "grid_trading": False,
            "momentum": True,
        }

        enabled = manager.get_enabled_strategies()
        assert "arbitrage" in enabled
        assert "momentum" in enabled
        assert "grid_trading" not in enabled

    @pytest.mark.asyncio
    async def test_emergency_disable_all(self, event_bus):
        """Test emergency disable of all strategies."""
        manager = StrategyStateManager(event_bus=event_bus)
        manager._active_strategies = {
            "arbitrage": True,
            "grid_trading": True,
            "momentum": True,
        }

        await manager.emergency_disable_all("System failure")

        assert all(not enabled for enabled in manager._active_strategies.values())
        event_bus.publish.assert_called_with(
            "EmergencyStrategyStopEvent", {"reason": "System failure", "timestamp": ANY}
        )
