"""Integration tests for liquidity scanner workflow."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.analytics.liquidity_scanner import (
    HealthStatus,
    LiquidityMetrics,
    LiquidityScanner,
    LiquidityScannerJob,
    LiquidityTier,
    PairHealthMonitor,
    PairRecommendationEngine,
    SpreadPersistenceTracker,
)
from genesis.core.events import Event, EventType, TierGraduationEvent
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus


@pytest.fixture
async def test_repository():
    """Create test repository."""
    repo = SQLiteRepository(":memory:")
    await repo.initialize()
    yield repo
    await repo.shutdown()


@pytest.fixture
def event_bus():
    """Create test event bus."""
    return EventBus()


@pytest.fixture
def mock_api_data():
    """Mock API data for testing."""
    return {
        "ticker_data": [
            {
                "symbol": "BTCUSDT",
                "quoteVolume": "15000000",
                "bidPrice": "50000",
                "askPrice": "50010",
                "volume": "300"
            },
            {
                "symbol": "ETHUSDT",
                "quoteVolume": "8000000",
                "bidPrice": "3000",
                "askPrice": "3002",
                "volume": "2500"
            },
            {
                "symbol": "BNBUSDT",
                "quoteVolume": "3000000",
                "bidPrice": "400",
                "askPrice": "400.5",
                "volume": "7000"
            },
            {
                "symbol": "MATICUSDT",
                "quoteVolume": "500000",
                "bidPrice": "1.00",
                "askPrice": "1.002",
                "volume": "450000"
            },
            {
                "symbol": "DOGEUSDT",
                "quoteVolume": "250000",
                "bidPrice": "0.10",
                "askPrice": "0.1002",
                "volume": "2400000"
            },
            {
                "symbol": "SHIBUSDT",
                "quoteVolume": "80000",
                "bidPrice": "0.00001",
                "askPrice": "0.0000101",
                "volume": "7000000000"
            }
        ],
        "depth_data": {
            "bids": [
                ["1", "100"],
                ["0.99", "200"],
                ["0.98", "150"]
            ],
            "asks": [
                ["1.01", "100"],
                ["1.02", "200"],
                ["1.03", "150"]
            ]
        }
    }


class TestLiquidityScannerWorkflow:
    """Test complete liquidity scanner workflow."""

    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, test_repository, mock_api_data):
        """Test full scanning workflow with persistence."""
        # Create scanner with mock session
        mock_session = AsyncMock()
        scanner = LiquidityScanner(session=mock_session)

        # Mock API responses
        ticker_response = AsyncMock()
        ticker_response.status = 200
        ticker_response.json = AsyncMock(return_value=mock_api_data["ticker_data"])

        depth_response = AsyncMock()
        depth_response.status = 200
        depth_response.json = AsyncMock(return_value=mock_api_data["depth_data"])

        mock_session.get.return_value.__aenter__.side_effect = [
            ticker_response,  # Ticker data
            depth_response,   # Depth for each symbol
            depth_response,
            depth_response,
            depth_response,
            depth_response,
            depth_response
        ]

        # Perform scan
        results = await scanner.scan_all_pairs()

        # Verify results
        assert len(results) == 6
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert "SHIBUSDT" in results

        # Check tier categorization
        assert results["BTCUSDT"].tier == LiquidityTier.HIGH
        assert results["ETHUSDT"].tier == LiquidityTier.HIGH
        assert results["BNBUSDT"].tier == LiquidityTier.HIGH
        assert results["MATICUSDT"].tier == LiquidityTier.MEDIUM
        assert results["DOGEUSDT"].tier == LiquidityTier.MEDIUM
        assert results["SHIBUSDT"].tier == LiquidityTier.LOW

        # Save to repository
        for symbol, metrics in results.items():
            snapshot_data = {
                "symbol": metrics.symbol,
                "volume_24h": metrics.volume_24h,
                "liquidity_tier": metrics.tier.value,
                "spread_basis_points": metrics.spread_bps,
                "bid_depth_10": metrics.bid_depth_10,
                "ask_depth_10": metrics.ask_depth_10,
                "spread_persistence_score": Decimal("50"),  # Default
                "scanned_at": datetime.utcnow()
            }
            await test_repository.save_liquidity_snapshot(snapshot_data)

        # Verify persistence
        snapshots = await test_repository.get_liquidity_snapshots()
        assert len(snapshots) == 6

        # Verify we can query by tier
        low_tier = await test_repository.get_liquidity_snapshots(tier="LOW")
        assert len(low_tier) == 1
        assert low_tier[0]["symbol"] == "SHIBUSDT"

        medium_tier = await test_repository.get_liquidity_snapshots(tier="MEDIUM")
        assert len(medium_tier) == 2

        high_tier = await test_repository.get_liquidity_snapshots(tier="HIGH")
        assert len(high_tier) == 3

    @pytest.mark.asyncio
    async def test_recommendation_workflow(self, test_repository):
        """Test pair recommendation workflow."""
        engine = PairRecommendationEngine()

        # Create test liquidity data
        liquidity_data = {
            "SHIBUSDT": LiquidityMetrics(
                symbol="SHIBUSDT",
                volume_24h=Decimal("80000"),
                spread_bps=10,
                bid_depth_10=Decimal("8000"),
                ask_depth_10=Decimal("8000"),
                tier=LiquidityTier.LOW,
                depth_score=Decimal("70"),
                timestamp=datetime.now()
            ),
            "DOGEUSDT": LiquidityMetrics(
                symbol="DOGEUSDT",
                volume_24h=Decimal("90000"),
                spread_bps=8,
                bid_depth_10=Decimal("9000"),
                ask_depth_10=Decimal("9000"),
                tier=LiquidityTier.LOW,
                depth_score=Decimal("75"),
                timestamp=datetime.now()
            ),
            "MATICUSDT": LiquidityMetrics(
                symbol="MATICUSDT",
                volume_24h=Decimal("500000"),
                spread_bps=5,
                bid_depth_10=Decimal("50000"),
                ask_depth_10=Decimal("50000"),
                tier=LiquidityTier.MEDIUM,
                depth_score=Decimal("85"),
                timestamp=datetime.now()
            ),
            "ADAUSDT": LiquidityMetrics(
                symbol="ADAUSDT",
                volume_24h=Decimal("600000"),
                spread_bps=4,
                bid_depth_10=Decimal("60000"),
                ask_depth_10=Decimal("60000"),
                tier=LiquidityTier.MEDIUM,
                depth_score=Decimal("90"),
                timestamp=datetime.now()
            ),
            "BTCUSDT": LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("15000000"),
                spread_bps=2,
                bid_depth_10=Decimal("1500000"),
                ask_depth_10=Decimal("1500000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("100"),
                timestamp=datetime.now()
            ),
        }

        # Get recommendations for each tier
        sniper_capital = Decimal("1500")
        hunter_capital = Decimal("5000")
        strategist_capital = Decimal("15000")

        sniper_recs = engine.recommend_pairs_for_tier("SNIPER", sniper_capital, liquidity_data)
        hunter_recs = engine.recommend_pairs_for_tier("HUNTER", hunter_capital, liquidity_data)
        strategist_recs = engine.recommend_pairs_for_tier("STRATEGIST", strategist_capital, liquidity_data)

        # Save recommendations to repository
        for symbol in sniper_recs:
            rec_data = {
                "tier": "SNIPER",
                "symbol": symbol,
                "volume_24h": liquidity_data[symbol].volume_24h,
                "liquidity_score": liquidity_data[symbol].depth_score,
                "recommended_at": datetime.utcnow()
            }
            await test_repository.save_tier_recommendation(rec_data)

        for symbol in hunter_recs:
            rec_data = {
                "tier": "HUNTER",
                "symbol": symbol,
                "volume_24h": liquidity_data[symbol].volume_24h,
                "liquidity_score": liquidity_data[symbol].depth_score,
                "recommended_at": datetime.utcnow()
            }
            await test_repository.save_tier_recommendation(rec_data)

        for symbol in strategist_recs:
            rec_data = {
                "tier": "STRATEGIST",
                "symbol": symbol,
                "volume_24h": liquidity_data[symbol].volume_24h,
                "liquidity_score": liquidity_data[symbol].depth_score,
                "recommended_at": datetime.utcnow()
            }
            await test_repository.save_tier_recommendation(rec_data)

        # Verify recommendations were saved
        saved_sniper = await test_repository.get_tier_recommendations("SNIPER")
        saved_hunter = await test_repository.get_tier_recommendations("HUNTER")
        saved_strategist = await test_repository.get_tier_recommendations("STRATEGIST")

        assert len(saved_sniper) == len(sniper_recs)
        assert len(saved_hunter) == len(hunter_recs)
        assert len(saved_strategist) == len(strategist_recs)

        # Verify correct pairs for each tier
        sniper_symbols = {r["symbol"] for r in saved_sniper}
        assert "DOGEUSDT" in sniper_symbols or "SHIBUSDT" in sniper_symbols

        hunter_symbols = {r["symbol"] for r in saved_hunter}
        assert "MATICUSDT" in hunter_symbols or "ADAUSDT" in hunter_symbols

        strategist_symbols = {r["symbol"] for r in saved_strategist}
        assert "BTCUSDT" in strategist_symbols

    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self, test_repository):
        """Test pair health monitoring and blacklisting workflow."""
        monitor = PairHealthMonitor()

        # Create degrading metrics over time
        metrics_history = []
        for day in range(6):
            metrics = LiquidityMetrics(
                symbol="BADCOIN",
                volume_24h=Decimal(str(100000 * (0.5 ** day))),  # Halving each day
                spread_bps=int(10 * (2 ** day)),  # Doubling spread
                bid_depth_10=Decimal(str(10000 * (0.5 ** day))),
                ask_depth_10=Decimal(str(10000 * (0.5 ** day))),
                tier=LiquidityTier.LOW,
                depth_score=Decimal(str(80 * (0.5 ** day))),
                timestamp=datetime.now() - timedelta(days=5-day)
            )
            metrics_history.append(metrics)

        # Monitor health over time
        for i in range(1, len(metrics_history)):
            current = metrics_history[i]
            historical = metrics_history[:i]
            
            if len(historical) >= 5:
                status = monitor.monitor_pair_health("BADCOIN", current, historical[-5:])
            else:
                status = monitor.monitor_pair_health("BADCOIN", current, historical)

        # Should be blacklisted after consistent degradation
        assert monitor.is_blacklisted("BADCOIN")

        # Save blacklist to repository
        blacklist_data = {
            "symbol": "BADCOIN",
            "blacklist_reason": "5 consecutive days of unhealthy status",
            "consecutive_losses": 5,
            "blacklisted_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30)
        }
        await test_repository.save_pair_blacklist(blacklist_data)

        # Verify blacklist was saved
        assert await test_repository.is_pair_blacklisted("BADCOIN")

        blacklisted = await test_repository.get_blacklisted_pairs()
        assert len(blacklisted) == 1
        assert blacklisted[0]["symbol"] == "BADCOIN"

    @pytest.mark.asyncio
    async def test_graduation_alert_workflow(self, event_bus):
        """Test tier graduation alert workflow."""
        engine = PairRecommendationEngine()
        events_received = []

        # Subscribe to graduation events
        async def event_handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.TIER_GRADUATION, event_handler)

        # Check graduation eligibility
        alert = engine.check_graduation_eligibility(Decimal("2500"), "SNIPER")

        assert alert is not None
        assert alert.recommended_tier == "HUNTER"

        # Create and publish graduation event
        grad_event = TierGraduationEvent(
            current_tier=alert.current_tier,
            recommended_tier=alert.recommended_tier,
            current_capital=alert.current_capital,
            message=alert.message
        )

        await event_bus.publish(grad_event)
        
        # Give event bus time to process
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(events_received) == 1
        received = events_received[0]
        assert received.event_type == EventType.TIER_GRADUATION
        assert received.event_data["current_tier"] == "SNIPER"
        assert received.event_data["recommended_tier"] == "HUNTER"

    @pytest.mark.asyncio
    async def test_scheduled_scanning_job(self, test_repository):
        """Test scheduled scanning job."""
        # Create components
        mock_session = AsyncMock()
        scanner = LiquidityScanner(session=mock_session)
        persistence_tracker = SpreadPersistenceTracker()
        health_monitor = PairHealthMonitor()

        # Create scanner job
        scanner_job = LiquidityScannerJob(scanner, persistence_tracker, health_monitor)

        # Mock scan results
        mock_results = {
            "BTCUSDT": LiquidityMetrics(
                symbol="BTCUSDT",
                volume_24h=Decimal("10000000"),
                spread_bps=3,
                bid_depth_10=Decimal("1000000"),
                ask_depth_10=Decimal("1000000"),
                tier=LiquidityTier.HIGH,
                depth_score=Decimal("95"),
                timestamp=datetime.now()
            )
        }

        # Mock the scanner's scan_all_pairs method
        scanner.scan_all_pairs = AsyncMock(return_value=mock_results)

        # Run a manual scan
        results = await scanner_job.trigger_manual_scan()

        # Verify scan was executed
        assert scanner.scan_all_pairs.called
        assert len(results) == 1
        assert "BTCUSDT" in results

        # Verify spread persistence was updated
        assert "BTCUSDT" in persistence_tracker.spread_history
        assert len(persistence_tracker.spread_history["BTCUSDT"]) == 1

    @pytest.mark.asyncio
    async def test_spread_persistence_tracking(self):
        """Test spread persistence tracking over time."""
        tracker = SpreadPersistenceTracker(window_hours=1)  # 1 hour window for testing

        # Simulate spread observations over time
        base_time = datetime.now()
        
        # Add profitable spreads (>= 10 bps) for 30 minutes
        for i in range(6):  # 6 observations, 5 minutes apart
            tracker.record_spread(
                "ETHUSDT",
                15,  # 15 basis points
                base_time + timedelta(minutes=i*5)
            )

        # Calculate persistence score
        score = tracker.calculate_spread_persistence_score("ETHUSDT")
        assert score > Decimal("80")  # Should be high due to consistent profitable spreads

        # Add unprofitable spreads for next 30 minutes
        for i in range(6, 12):
            tracker.record_spread(
                "ETHUSDT",
                5,  # 5 basis points (unprofitable)
                base_time + timedelta(minutes=i*5)
            )

        # Recalculate score
        score = tracker.calculate_spread_persistence_score("ETHUSDT")
        assert score < Decimal("60")  # Should be lower due to mixed profitability

        # Wait for window to expire and add new data
        future_time = base_time + timedelta(hours=2)
        for i in range(6):
            tracker.record_spread(
                "ETHUSDT",
                20,  # 20 basis points
                future_time + timedelta(minutes=i*5)
            )

        # Old data should be removed, score should be high again
        score = tracker.calculate_spread_persistence_score("ETHUSDT")
        assert score > Decimal("80")

    @pytest.mark.asyncio
    async def test_database_persistence_integration(self, test_repository):
        """Test integration with database persistence."""
        # Save multiple snapshots
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        for symbol in symbols:
            for hour in range(3):
                snapshot_data = {
                    "symbol": symbol,
                    "volume_24h": Decimal("1000000") * (hour + 1),
                    "liquidity_tier": "HIGH",
                    "spread_basis_points": 5 + hour,
                    "bid_depth_10": Decimal("100000") * (hour + 1),
                    "ask_depth_10": Decimal("100000") * (hour + 1),
                    "spread_persistence_score": Decimal("80") + Decimal(str(hour * 5)),
                    "scanned_at": datetime.utcnow() - timedelta(hours=2-hour)
                }
                await test_repository.save_liquidity_snapshot(snapshot_data)

        # Query recent snapshots
        recent = await test_repository.get_liquidity_snapshots(hours_back=1)
        assert len(recent) == 3  # Only the most recent hour for each symbol

        # Query specific symbol
        btc_snapshots = await test_repository.get_liquidity_snapshots(symbol="BTCUSDT", hours_back=24)
        assert len(btc_snapshots) == 3
        assert all(s["symbol"] == "BTCUSDT" for s in btc_snapshots)

        # Test blacklist functionality
        await test_repository.save_pair_blacklist({
            "symbol": "SCAMCOIN",
            "blacklist_reason": "Rug pull detected",
            "consecutive_losses": 10,
            "expires_at": datetime.utcnow() + timedelta(days=90)
        })

        assert await test_repository.is_pair_blacklisted("SCAMCOIN")
        assert not await test_repository.is_pair_blacklisted("BTCUSDT")

        # Test tier recommendations
        for tier in ["SNIPER", "HUNTER", "STRATEGIST"]:
            await test_repository.save_tier_recommendation({
                "tier": tier,
                "symbol": f"TEST{tier}",
                "volume_24h": Decimal("1000000"),
                "liquidity_score": Decimal("85")
            })

        sniper_recs = await test_repository.get_tier_recommendations("SNIPER")
        assert len(sniper_recs) == 1
        assert sniper_recs[0]["symbol"] == "TESTSNIPER"