"""Unit tests for Market Regime Detector"""

from datetime import UTC, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.engine.market_regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeIndicator,
)


@pytest.fixture
def mock_event_bus():
    """Create mock event bus"""
    event_bus = Mock(spec=EventBus)
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def detector(mock_event_bus):
    """Create market regime detector instance"""
    return MarketRegimeDetector(mock_event_bus)


class TestMarketRegimeDetector:
    """Test market regime detection functionality"""

    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.current_regime == MarketRegime.NEUTRAL
        assert len(detector.indicators) == 0
        assert detector.regime_history == []
        assert detector.confidence_threshold == Decimal("0.7")

    def test_add_indicator(self, detector):
        """Test adding regime indicators"""
        indicator = RegimeIndicator(
            name="volatility",
            value=Decimal("20"),
            weight=Decimal("0.3")
        )

        detector.add_indicator(indicator)
        assert "volatility" in detector.indicators
        assert detector.indicators["volatility"] == indicator

    def test_update_indicator(self, detector):
        """Test updating indicator values"""
        detector.add_indicator(RegimeIndicator(
            name="volatility",
            value=Decimal("20"),
            weight=Decimal("0.3")
        ))

        detector.update_indicator("volatility", Decimal("25"))
        assert detector.indicators["volatility"].value == Decimal("25")

    def test_update_nonexistent_indicator(self, detector):
        """Test updating nonexistent indicator"""
        with pytest.raises(KeyError):
            detector.update_indicator("nonexistent", Decimal("10"))

    def test_calculate_regime_score_bull(self, detector):
        """Test regime score calculation for bull market"""
        # Add indicators suggesting bull market
        detector.add_indicator(RegimeIndicator(
            name="trend",
            value=Decimal("1"),  # Uptrend
            weight=Decimal("0.4")
        ))
        detector.add_indicator(RegimeIndicator(
            name="momentum",
            value=Decimal("0.8"),  # Strong momentum
            weight=Decimal("0.3")
        ))
        detector.add_indicator(RegimeIndicator(
            name="sentiment",
            value=Decimal("75"),  # Greed
            weight=Decimal("0.3")
        ))

        score = detector._calculate_regime_score()
        assert score > Decimal("0.5")  # Bullish score

    def test_calculate_regime_score_bear(self, detector):
        """Test regime score calculation for bear market"""
        # Add indicators suggesting bear market
        detector.add_indicator(RegimeIndicator(
            name="trend",
            value=Decimal("-1"),  # Downtrend
            weight=Decimal("0.4")
        ))
        detector.add_indicator(RegimeIndicator(
            name="momentum",
            value=Decimal("-0.8"),  # Negative momentum
            weight=Decimal("0.3")
        ))
        detector.add_indicator(RegimeIndicator(
            name="sentiment",
            value=Decimal("25"),  # Fear
            weight=Decimal("0.3")
        ))

        score = detector._calculate_regime_score()
        assert score < Decimal("-0.5")  # Bearish score

    def test_determine_regime_from_score(self, detector):
        """Test regime determination from score"""
        # Bull market
        assert detector._determine_regime(Decimal("0.8")) == MarketRegime.BULL

        # Bear market
        assert detector._determine_regime(Decimal("-0.8")) == MarketRegime.BEAR

        # Neutral/Crab market
        assert detector._determine_regime(Decimal("0.2")) == MarketRegime.NEUTRAL

        # Crash
        assert detector._determine_regime(Decimal("-1.5")) == MarketRegime.CRASH

        # Recovery (after crash)
        detector.regime_history = [MarketRegime.CRASH]
        assert detector._determine_regime(Decimal("0.3")) == MarketRegime.RECOVERY

    @pytest.mark.asyncio
    async def test_detect_regime_change(self, detector, mock_event_bus):
        """Test detecting regime changes"""
        # Setup initial regime
        detector.current_regime = MarketRegime.NEUTRAL

        # Add indicators for bull market
        detector.add_indicator(RegimeIndicator("trend", Decimal("1"), Decimal("0.5")))
        detector.add_indicator(RegimeIndicator("momentum", Decimal("0.9"), Decimal("0.5")))

        # Detect regime
        new_regime = await detector.detect_regime()

        assert new_regime == MarketRegime.BULL
        assert detector.current_regime == MarketRegime.BULL
        assert len(detector.regime_history) == 1

        # Verify event published
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.GLOBAL_MARKET_STATE_CHANGE
        assert event.data["old_regime"] == MarketRegime.NEUTRAL.value
        assert event.data["new_regime"] == MarketRegime.BULL.value

    @pytest.mark.asyncio
    async def test_no_regime_change_event(self, detector, mock_event_bus):
        """Test no event when regime doesn't change"""
        detector.current_regime = MarketRegime.BULL
        detector.add_indicator(RegimeIndicator("trend", Decimal("0.9"), Decimal("1")))

        new_regime = await detector.detect_regime()

        assert new_regime == MarketRegime.BULL
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_from_market_data(self, detector):
        """Test updating indicators from market data"""
        market_data = {
            "price": Decimal("50000"),
            "volume": Decimal("1000000"),
            "volatility": Decimal("25"),
            "rsi": Decimal("65"),
            "fear_greed_index": Decimal("70")
        }

        await detector.update_from_market_data(market_data)

        # In a real implementation, this would update indicators
        # based on the market data
        assert detector.last_update is not None

    def test_get_regime_confidence(self, detector):
        """Test confidence calculation for regime detection"""
        # High confidence (strong indicators)
        detector.add_indicator(RegimeIndicator("trend", Decimal("1"), Decimal("0.5")))
        detector.add_indicator(RegimeIndicator("momentum", Decimal("0.9"), Decimal("0.5")))

        confidence = detector.get_regime_confidence()
        assert confidence > Decimal("0.8")

        # Low confidence (mixed indicators)
        detector.indicators["trend"].value = Decimal("0.1")
        detector.indicators["momentum"].value = Decimal("-0.1")

        confidence = detector.get_regime_confidence()
        assert confidence < Decimal("0.3")

    def test_regime_duration(self, detector):
        """Test regime duration tracking"""
        now = datetime.now(UTC)
        detector.regime_start_time = now
        detector.current_regime = MarketRegime.BULL

        # Simulate time passing
        with patch('genesis.engine.market_regime_detector.datetime') as mock_datetime:
            mock_datetime.now.return_value = now.replace(hour=now.hour + 2)
            mock_datetime.timezone = timezone

            duration = detector.get_regime_duration()
            assert duration.total_seconds() == 7200  # 2 hours

    def test_is_regime_stable(self, detector):
        """Test regime stability check"""
        # New regime - not stable
        detector.regime_start_time = datetime.now(UTC)
        assert not detector.is_regime_stable()

        # Old regime - stable
        with patch('genesis.engine.market_regime_detector.datetime') as mock_datetime:
            old_time = datetime.now(UTC)
            mock_datetime.now.return_value = old_time.replace(day=old_time.day + 2)
            mock_datetime.timezone = timezone
            detector.regime_start_time = old_time

            assert detector.is_regime_stable()

    @pytest.mark.asyncio
    async def test_crash_detection(self, detector, mock_event_bus):
        """Test crash detection with extreme indicators"""
        detector.current_regime = MarketRegime.BEAR

        # Extreme negative indicators
        detector.add_indicator(RegimeIndicator("trend", Decimal("-2"), Decimal("0.3")))
        detector.add_indicator(RegimeIndicator("momentum", Decimal("-1.5"), Decimal("0.3")))
        detector.add_indicator(RegimeIndicator("volume_spike", Decimal("3"), Decimal("0.4")))

        new_regime = await detector.detect_regime()

        assert new_regime == MarketRegime.CRASH
        mock_event_bus.publish.assert_called_once()

        event = mock_event_bus.publish.call_args[0][0]
        assert event.data["new_regime"] == MarketRegime.CRASH.value

    @pytest.mark.asyncio
    async def test_recovery_detection(self, detector, mock_event_bus):
        """Test recovery detection after crash"""
        # Set history with crash
        detector.current_regime = MarketRegime.CRASH
        detector.regime_history = [MarketRegime.BEAR, MarketRegime.CRASH]

        # Improving indicators
        detector.add_indicator(RegimeIndicator("trend", Decimal("0.3"), Decimal("0.5")))
        detector.add_indicator(RegimeIndicator("momentum", Decimal("0.4"), Decimal("0.5")))

        new_regime = await detector.detect_regime()

        assert new_regime == MarketRegime.RECOVERY
        assert detector.current_regime == MarketRegime.RECOVERY

    def test_regime_history_limit(self, detector):
        """Test regime history is limited in size"""
        # Add many regime changes
        for i in range(150):
            detector.regime_history.append(MarketRegime.BULL if i % 2 == 0 else MarketRegime.BEAR)

        # Should be truncated
        detector._truncate_history()
        assert len(detector.regime_history) <= 100

    def test_get_regime_statistics(self, detector):
        """Test regime statistics calculation"""
        # Add regime history
        detector.regime_history = [
            MarketRegime.BULL,
            MarketRegime.BULL,
            MarketRegime.NEUTRAL,
            MarketRegime.BEAR,
            MarketRegime.BULL
        ]

        stats = detector.get_regime_statistics()

        assert stats["bull_percentage"] == Decimal("60")  # 3/5
        assert stats["bear_percentage"] == Decimal("20")  # 1/5
        assert stats["neutral_percentage"] == Decimal("20")  # 1/5
        assert stats["total_changes"] == 5

    def test_regime_recommendation(self, detector):
        """Test strategy recommendations based on regime"""
        recommendations = {
            MarketRegime.BULL: detector.get_strategy_recommendation(MarketRegime.BULL),
            MarketRegime.BEAR: detector.get_strategy_recommendation(MarketRegime.BEAR),
            MarketRegime.NEUTRAL: detector.get_strategy_recommendation(MarketRegime.NEUTRAL),
            MarketRegime.CRASH: detector.get_strategy_recommendation(MarketRegime.CRASH),
            MarketRegime.RECOVERY: detector.get_strategy_recommendation(MarketRegime.RECOVERY)
        }

        assert "momentum" in recommendations[MarketRegime.BULL]
        assert "mean_reversion" in recommendations[MarketRegime.NEUTRAL]
        assert "defensive" in recommendations[MarketRegime.BEAR].lower() or "reduce" in recommendations[MarketRegime.BEAR].lower()
        assert "stop" in recommendations[MarketRegime.CRASH].lower() or "exit" in recommendations[MarketRegime.CRASH].lower()
        assert "cautious" in recommendations[MarketRegime.RECOVERY].lower() or "gradual" in recommendations[MarketRegime.RECOVERY].lower()

    @pytest.mark.asyncio
    async def test_auto_update_cycle(self, detector):
        """Test automatic regime update cycle"""
        update_count = 0

        async def mock_detect():
            nonlocal update_count
            update_count += 1
            if update_count >= 2:
                detector.auto_update = False
            return detector.current_regime

        detector.detect_regime = mock_detect
        detector.update_interval = 0.01  # 10ms for testing

        # Start auto update
        await detector.start_auto_update(interval=0.01)

        # Wait for updates
        await asyncio.sleep(0.05)

        assert update_count >= 2
        assert not detector.auto_update

    def test_serialize_regime_state(self, detector):
        """Test serialization of regime state"""
        detector.current_regime = MarketRegime.BULL
        detector.add_indicator(RegimeIndicator("trend", Decimal("0.8"), Decimal("0.5")))
        detector.regime_history = [MarketRegime.NEUTRAL, MarketRegime.BULL]

        state = detector.to_dict()

        assert state["current_regime"] == "bull"
        assert "indicators" in state
        assert "trend" in state["indicators"]
        assert state["indicators"]["trend"]["value"] == "0.8"
        assert len(state["regime_history"]) == 2

    def test_load_regime_state(self, detector):
        """Test loading regime state from dict"""
        state = {
            "current_regime": "bear",
            "indicators": {
                "trend": {"name": "trend", "value": "-0.5", "weight": "0.5"}
            },
            "regime_history": ["bull", "neutral", "bear"],
            "confidence_threshold": "0.8"
        }

        detector.from_dict(state)

        assert detector.current_regime == MarketRegime.BEAR
        assert "trend" in detector.indicators
        assert detector.indicators["trend"].value == Decimal("-0.5")
        assert len(detector.regime_history) == 3
        assert detector.confidence_threshold == Decimal("0.8")
