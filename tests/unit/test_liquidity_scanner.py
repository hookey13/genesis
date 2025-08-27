"""Unit tests for liquidity scanner module."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import aiohttp
import pytest

from genesis.analytics.liquidity_scanner import (
    HealthStatus,
    LiquidityMetrics,
    LiquidityScanner,
    LiquidityTier,
    PairHealthMonitor,
    PairRecommendationEngine,
    SpreadPersistenceTracker,
)


@pytest.fixture
def mock_session():
    """Mock aiohttp session."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session


@pytest.fixture
def liquidity_scanner(mock_session):
    """Create liquidity scanner with mock session."""
    return LiquidityScanner(session=mock_session)


@pytest.fixture
def spread_tracker():
    """Create spread persistence tracker."""
    return SpreadPersistenceTracker(window_hours=24)


@pytest.fixture
def recommendation_engine():
    """Create pair recommendation engine."""
    return PairRecommendationEngine()


@pytest.fixture
def health_monitor():
    """Create pair health monitor."""
    return PairHealthMonitor()


class TestLiquidityScanner:
    """Test liquidity scanner functionality."""

    def test_categorize_by_volume(self, liquidity_scanner):
        """Test volume categorization logic."""
        # Test LOW tier
        assert liquidity_scanner.categorize_by_volume(Decimal("50000")) == LiquidityTier.LOW
        assert liquidity_scanner.categorize_by_volume(Decimal("99999")) == LiquidityTier.LOW

        # Test MEDIUM tier
        assert liquidity_scanner.categorize_by_volume(Decimal("100000")) == LiquidityTier.MEDIUM
        assert liquidity_scanner.categorize_by_volume(Decimal("500000")) == LiquidityTier.MEDIUM
        assert liquidity_scanner.categorize_by_volume(Decimal("999999")) == LiquidityTier.MEDIUM

        # Test HIGH tier
        assert liquidity_scanner.categorize_by_volume(Decimal("1000000")) == LiquidityTier.HIGH
        assert liquidity_scanner.categorize_by_volume(Decimal("10000000")) == LiquidityTier.HIGH

        # Test edge cases
        assert liquidity_scanner.categorize_by_volume(Decimal("0")) == LiquidityTier.LOW
        assert liquidity_scanner.categorize_by_volume(Decimal("-1000")) == LiquidityTier.LOW

    @pytest.mark.asyncio
    async def test_analyze_order_book_depth(self, liquidity_scanner, mock_session):
        """Test order book depth analysis."""
        # Mock API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "bids": [
                ["50000", "1.5"],
                ["49999", "2.0"],
                ["49998", "1.0"]
            ],
            "asks": [
                ["50001", "1.2"],
                ["50002", "1.8"],
                ["50003", "2.5"]
            ]
        })
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await liquidity_scanner.analyze_order_book_depth("BTCUSDT", levels=3)

        assert result is not None
        assert "bid_depth" in result
        assert "ask_depth" in result
        assert result["levels"] == 3

        # Calculate expected depths
        expected_bid_depth = (Decimal("50000") * Decimal("1.5") +
                              Decimal("49999") * Decimal("2.0") +
                              Decimal("49998") * Decimal("1.0"))
        expected_ask_depth = (Decimal("50001") * Decimal("1.2") +
                              Decimal("50002") * Decimal("1.8") +
                              Decimal("50003") * Decimal("2.5"))

        assert result["bid_depth"] == expected_bid_depth
        assert result["ask_depth"] == expected_ask_depth

    @pytest.mark.asyncio
    async def test_analyze_order_book_depth_error(self, liquidity_scanner, mock_session):
        """Test order book depth analysis with API error."""
        # Mock API error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session.get.return_value.__aenter__.return_value = mock_response

        result = await liquidity_scanner.analyze_order_book_depth("BTCUSDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_scan_all_pairs(self, liquidity_scanner, mock_session):
        """Test scanning all pairs."""
        # Mock ticker data
        ticker_data = [
            {
                "symbol": "BTCUSDT",
                "quoteVolume": "5000000",
                "bidPrice": "50000",
                "askPrice": "50010"
            },
            {
                "symbol": "ETHUSDT",
                "quoteVolume": "2000000",
                "bidPrice": "3000",
                "askPrice": "3001"
            },
            {
                "symbol": "DOGEUSDT",
                "quoteVolume": "50000",
                "bidPrice": "0.10",
                "askPrice": "0.1001"
            },
            {
                "symbol": "BTCETH",  # Non-USDT pair, should be skipped
                "quoteVolume": "100000",
                "bidPrice": "16",
                "askPrice": "16.01"
            },
            {
                "symbol": "SHIBUSDT",  # Low volume, should be skipped
                "quoteVolume": "5000",
                "bidPrice": "0.00001",
                "askPrice": "0.000011"
            }
        ]

        # Mock ticker response
        ticker_response = AsyncMock()
        ticker_response.status = 200
        ticker_response.json = AsyncMock(return_value=ticker_data)

        # Mock depth response
        depth_response = AsyncMock()
        depth_response.status = 200
        depth_response.json = AsyncMock(return_value={
            "bids": [["1", "100"]],
            "asks": [["1.01", "100"]]
        })

        # Configure mock session
        mock_session.get.return_value.__aenter__.side_effect = [
            ticker_response,  # First call for ticker data
            depth_response,    # Subsequent calls for depth
            depth_response,
            depth_response
        ]

        results = await liquidity_scanner.scan_all_pairs()

        # Should have 3 results (BTCUSDT, ETHUSDT, DOGEUSDT)
        assert len(results) == 3
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert "DOGEUSDT" in results
        assert "BTCETH" not in results  # Non-USDT
        assert "SHIBUSDT" not in results  # Low volume

        # Check categorization
        assert results["BTCUSDT"].tier == LiquidityTier.HIGH
        assert results["ETHUSDT"].tier == LiquidityTier.HIGH
        assert results["DOGEUSDT"].tier == LiquidityTier.LOW


class TestSpreadPersistenceTracker:
    """Test spread persistence tracking."""

    def test_record_spread(self, spread_tracker):
        """Test recording spread observations."""
        now = datetime.now(UTC)

        # Record some spreads
        spread_tracker.record_spread("BTCUSDT", 15, now)
        spread_tracker.record_spread("BTCUSDT", 20, now + timedelta(minutes=5))
        spread_tracker.record_spread("BTCUSDT", 10, now + timedelta(minutes=10))

        assert "BTCUSDT" in spread_tracker.spread_history
        assert len(spread_tracker.spread_history["BTCUSDT"]) == 3

    def test_record_spread_window_cleanup(self, spread_tracker):
        """Test that old observations are removed."""
        now = datetime.now(UTC)

        # Record spread outside window
        spread_tracker.record_spread("BTCUSDT", 15, now - timedelta(hours=25))

        # Record recent spread
        spread_tracker.record_spread("BTCUSDT", 20, now)

        # Old spread should be removed
        assert len(spread_tracker.spread_history["BTCUSDT"]) == 1
        assert spread_tracker.spread_history["BTCUSDT"][0][1] == 20

    def test_calculate_spread_persistence_score(self, spread_tracker):
        """Test spread persistence score calculation."""
        now = datetime.now(UTC)

        # Test with no data
        score = spread_tracker.calculate_spread_persistence_score("BTCUSDT")
        assert score == Decimal("50")  # Default neutral score

        # Add profitable spreads (>= 10 bps)
        for i in range(10):
            spread_tracker.record_spread("BTCUSDT", 15, now + timedelta(minutes=i))

        score = spread_tracker.calculate_spread_persistence_score("BTCUSDT")
        assert score > Decimal("50")  # Should be high due to consistent profitable spreads

        # Add unprofitable spreads (< 10 bps)
        for i in range(10, 20):
            spread_tracker.record_spread("BTCUSDT", 5, now + timedelta(minutes=i))

        score = spread_tracker.calculate_spread_persistence_score("BTCUSDT")
        assert score < Decimal("75")  # Should decrease due to mixed profitability

    def test_persistence_score_with_high_variance(self, spread_tracker):
        """Test that high variance reduces persistence score."""
        now = datetime.now(UTC)

        # Add highly variable spreads
        spreads = [5, 50, 2, 100, 3, 80, 1, 90, 4, 70, 15, 25, 35, 45, 55]
        for i, spread in enumerate(spreads):
            spread_tracker.record_spread("BTCUSDT", spread, now + timedelta(minutes=i))

        score = spread_tracker.calculate_spread_persistence_score("BTCUSDT")
        # Score should be reduced due to high variance
        assert score < Decimal("80")


class TestPairRecommendationEngine:
    """Test pair recommendation engine."""

    def test_recommend_pairs_for_tier(self, recommendation_engine):
        """Test tier-appropriate pair recommendations."""
        # Create mock liquidity data
        liquidity_data = {
            "SHIBUSDT": LiquidityMetrics(
                symbol="SHIBUSDT",
                volume_24h=Decimal("50000"),
                spread_bps=8,
                bid_depth_10=Decimal("10000"),
                ask_depth_10=Decimal("10000"),
                tier=LiquidityTier.LOW,
                depth_score=Decimal("80"),
                timestamp=datetime.now(UTC)
            ),
            "DOGEUSDT": LiquidityMetrics(
                symbol="DOGEUSDT",
                volume_24h=Decimal("80000"),
                spread_bps=10,
                bid_depth_10=Decimal("15000"),
                ask_depth_10=Decimal("15000"),
                tier=LiquidityTier.LOW,
                depth_score=Decimal("75"),
                timestamp=datetime.now(UTC)
            ),
            "MATICUSDT": LiquidityMetrics(
                symbol="MATICUSDT",
                volume_24h=Decimal("500000"),
                spread_bps=5,
                bid_depth_10=Decimal("100000"),
                ask_depth_10=Decimal("100000"),
                tier=LiquidityTier.MEDIUM,
                depth_score=Decimal("90"),
                timestamp=datetime.now(UTC)
            ),
            "BTCUSDT": LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("10000000"),
                spread_bps=2,
                bid_depth_10=Decimal("1000000"),
                ask_depth_10=Decimal("1000000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("100"),
                timestamp=datetime.now(UTC)
            ),
        }

        # Test SNIPER tier recommendations
        sniper_recs = recommendation_engine.recommend_pairs_for_tier(
            "SNIPER", Decimal("1000"), liquidity_data
        )
        assert len(sniper_recs) == 2  # SHIBUSDT and DOGEUSDT
        assert "SHIBUSDT" in sniper_recs
        assert "DOGEUSDT" in sniper_recs

        # Test HUNTER tier recommendations
        hunter_recs = recommendation_engine.recommend_pairs_for_tier(
            "HUNTER", Decimal("5000"), liquidity_data
        )
        assert len(hunter_recs) == 1  # MATICUSDT
        assert "MATICUSDT" in hunter_recs

        # Test STRATEGIST tier recommendations
        strategist_recs = recommendation_engine.recommend_pairs_for_tier(
            "STRATEGIST", Decimal("15000"), liquidity_data
        )
        assert len(strategist_recs) == 1  # BTCUSDT
        assert "BTCUSDT" in strategist_recs

    def test_check_graduation_eligibility(self, recommendation_engine):
        """Test tier graduation eligibility checks."""
        # Test SNIPER to HUNTER graduation
        alert = recommendation_engine.check_graduation_eligibility(
            Decimal("2500"), "SNIPER"
        )
        assert alert is not None
        assert alert.current_tier == "SNIPER"
        assert alert.recommended_tier == "HUNTER"
        assert alert.current_capital == Decimal("2500")

        # Test no graduation needed
        alert = recommendation_engine.check_graduation_eligibility(
            Decimal("1500"), "SNIPER"
        )
        assert alert is None

        # Test HUNTER to STRATEGIST graduation
        alert = recommendation_engine.check_graduation_eligibility(
            Decimal("12000"), "HUNTER"
        )
        assert alert is not None
        assert alert.current_tier == "HUNTER"
        assert alert.recommended_tier == "STRATEGIST"

        # Test STRATEGIST (no further graduation)
        alert = recommendation_engine.check_graduation_eligibility(
            Decimal("100000"), "STRATEGIST"
        )
        assert alert is None


class TestPairHealthMonitor:
    """Test pair health monitoring."""

    def test_monitor_pair_health_healthy(self, health_monitor):
        """Test monitoring healthy pair."""
        current = LiquidityMetrics(
            symbol="BTCUSDT",
            volume_24h=Decimal("1000000"),
            spread_bps=5,
            bid_depth_10=Decimal("100000"),
            ask_depth_10=Decimal("100000"),
            tier=LiquidityTier.HIGH,
            depth_score=Decimal("90"),
            timestamp=datetime.now(UTC)
        )

        historical = [
            LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("950000"),
                spread_bps=5,
                bid_depth_10=Decimal("95000"),
                ask_depth_10=Decimal("95000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("88"),
                timestamp=datetime.now(UTC) - timedelta(hours=i)
            )
            for i in range(1, 6)
        ]

        status = health_monitor.monitor_pair_health("BTCUSDT", current, historical)
        assert status == HealthStatus.HEALTHY

    def test_monitor_pair_health_degrading(self, health_monitor):
        """Test monitoring degrading pair."""
        current = LiquidityMetrics(
            symbol="BTCUSDT",
            volume_24h=Decimal("700000"),  # 30% decline
            spread_bps=7,  # 40% widening
            bid_depth_10=Decimal("70000"),
            ask_depth_10=Decimal("70000"),
            tier=LiquidityTier.HIGH,
            depth_score=Decimal("65"),  # 28% reduction
            timestamp=datetime.now(UTC)
        )

        historical = [
            LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("1000000"),
                spread_bps=5,
                bid_depth_10=Decimal("100000"),
                ask_depth_10=Decimal("100000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("90"),
                timestamp=datetime.now(UTC) - timedelta(hours=i)
            )
            for i in range(1, 6)
        ]

        status = health_monitor.monitor_pair_health("BTCUSDT", current, historical)
        assert status == HealthStatus.DEGRADING

    def test_monitor_pair_health_unhealthy(self, health_monitor):
        """Test monitoring unhealthy pair."""
        current = LiquidityMetrics(
            symbol="BTCUSDT",
            volume_24h=Decimal("400000"),  # 60% decline
            spread_bps=10,  # 100% widening
            bid_depth_10=Decimal("40000"),
            ask_depth_10=Decimal("40000"),
            tier=LiquidityTier.HIGH,
            depth_score=Decimal("40"),  # 55% reduction
            timestamp=datetime.now(UTC)
        )

        historical = [
            LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("1000000"),
                spread_bps=5,
                bid_depth_10=Decimal("100000"),
                ask_depth_10=Decimal("100000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("90"),
                timestamp=datetime.now(UTC) - timedelta(hours=i)
            )
            for i in range(1, 6)
        ]

        status = health_monitor.monitor_pair_health("BTCUSDT", current, historical)
        assert status == HealthStatus.UNHEALTHY

    def test_auto_blacklisting(self, health_monitor):
        """Test automatic blacklisting after consecutive unhealthy days."""
        current = LiquidityMetrics(
            symbol="BTCUSDT",
            volume_24h=Decimal("100000"),
            spread_bps=20,
            bid_depth_10=Decimal("10000"),
            ask_depth_10=Decimal("10000"),
            tier=LiquidityTier.MEDIUM,
            depth_score=Decimal("20"),
            timestamp=datetime.now(UTC)
        )

        historical = [
            LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("1000000"),
                spread_bps=5,
                bid_depth_10=Decimal("100000"),
                ask_depth_10=Decimal("100000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("90"),
                timestamp=datetime.now(UTC) - timedelta(hours=i)
            )
            for i in range(1, 6)
        ]

        # Simulate 5 consecutive unhealthy days
        for _ in range(5):
            status = health_monitor.monitor_pair_health("BTCUSDT", current, historical)

        # Should be blacklisted after 5 unhealthy days
        assert health_monitor.is_blacklisted("BTCUSDT")
        status = health_monitor.monitor_pair_health("BTCUSDT", current, historical)
        assert status == HealthStatus.BLACKLISTED

    def test_blacklist_expiration(self, health_monitor):
        """Test blacklist expiration."""
        # Add to blacklist with past expiration
        health_monitor.blacklist["BTCUSDT"] = {
            "blacklist_id": "test-id",
            "reason": "test",
            "consecutive_losses": 5,
            "blacklisted_at": datetime.now(UTC) - timedelta(days=31),
            "expires_at": datetime.now(UTC) - timedelta(days=1)  # Expired yesterday
        }

        # Should not be blacklisted (expired)
        assert not health_monitor.is_blacklisted("BTCUSDT")
        # Expired entry should be removed
        assert "BTCUSDT" not in health_monitor.blacklist
