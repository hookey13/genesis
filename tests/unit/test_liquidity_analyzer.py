"""
Unit tests for Liquidity Analyzer
"""

import asyncio
import json
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genesis.analytics.liquidity_analyzer import (
    LiquidityAnalyzer,
    LiquidityAnalyzerConfig,
    LiquidityDepthMetrics,
    LiquidityScore,
    MarketImpactEstimate,
    MicrostructureAnomaly,
    OrderBookImbalance,
)
from genesis.core.exceptions import ValidationError


@pytest.fixture
def config():
    """Create test configuration"""
    return LiquidityAnalyzerConfig(
        depth_levels=[5, 10, 20],
        imbalance_threshold=Decimal("2.0"),
        anomaly_detection_window=50,
        market_impact_eta=Decimal("0.1"),
        market_impact_gamma=Decimal("0.05"),
        quote_stuffing_threshold=30
    )


@pytest.fixture
def analyzer(config):
    """Create liquidity analyzer instance"""
    return LiquidityAnalyzer(config)


@pytest.fixture
def sample_orderbook():
    """Generate sample orderbook data"""
    return {
        "bids": [
            [49995, 10],
            [49990, 15],
            [49985, 20],
            [49980, 25],
            [49975, 30],
            [49970, 35],
            [49965, 40],
            [49960, 45],
            [49955, 50],
            [49950, 55],
        ],
        "asks": [
            [50005, 12],
            [50010, 18],
            [50015, 22],
            [50020, 28],
            [50025, 32],
            [50030, 38],
            [50035, 42],
            [50040, 48],
            [50045, 52],
            [50050, 58],
        ]
    }


@pytest.fixture
def imbalanced_orderbook():
    """Generate imbalanced orderbook (bid heavy)"""
    return {
        "bids": [
            [49995, 100],  # Large bid volume
            [49990, 150],
            [49985, 200],
            [49980, 250],
            [49975, 300],
        ],
        "asks": [
            [50005, 10],  # Small ask volume
            [50010, 15],
            [50015, 20],
            [50020, 25],
            [50025, 30],
        ]
    }


class TestLiquidityDepthAssessment:
    """Test liquidity depth assessment"""

    @pytest.mark.asyncio
    async def test_depth_calculation_at_levels(self, analyzer, sample_orderbook):
        """Test depth calculation at multiple levels"""
        metrics = await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        # Check cumulative depth at each level
        assert metrics.bid_depth[5] == Decimal("100")  # Sum of first 5 bid levels
        assert metrics.bid_depth[10] == Decimal("325")  # Sum of first 10 bid levels
        
        assert metrics.ask_depth[5] == Decimal("112")  # Sum of first 5 ask levels
        assert metrics.ask_depth[10] == Decimal("350")  # Sum of first 10 ask levels

    @pytest.mark.asyncio
    async def test_total_volume_calculation(self, analyzer, sample_orderbook):
        """Test total bid/ask volume calculation"""
        metrics = await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        expected_bid_total = sum(level[1] for level in sample_orderbook["bids"])
        expected_ask_total = sum(level[1] for level in sample_orderbook["asks"])
        
        assert metrics.total_bid_volume == Decimal(str(expected_bid_total))
        assert metrics.total_ask_volume == Decimal(str(expected_ask_total))

    @pytest.mark.asyncio
    async def test_weighted_mid_price(self, analyzer, sample_orderbook):
        """Test weighted mid price calculation"""
        metrics = await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        # Weighted mid should be between best bid and ask
        best_bid = Decimal(str(sample_orderbook["bids"][0][0]))
        best_ask = Decimal(str(sample_orderbook["asks"][0][0]))
        
        assert best_bid < metrics.weighted_mid_price < best_ask

    @pytest.mark.asyncio
    async def test_liquidity_gap_detection(self, analyzer):
        """Test detection of liquidity gaps"""
        # Create orderbook with gaps
        orderbook_with_gaps = {
            "bids": [
                [50000, 10],
                [49995, 15],  # 5 point gap
                [49980, 20],  # 15 point gap (significant)
                [49975, 25],
            ],
            "asks": [
                [50010, 10],
                [50015, 15],
                [50030, 20],  # 15 point gap (significant)
                [50035, 25],
            ]
        }
        
        metrics = await analyzer.assess_depth("BTC/USDT", orderbook_with_gaps)
        
        # Should detect significant gaps
        assert len(metrics.bid_liquidity_gaps) > 0
        assert len(metrics.ask_liquidity_gaps) > 0

    @pytest.mark.asyncio
    async def test_invalid_orderbook_rejected(self, analyzer):
        """Test that invalid orderbook data is rejected"""
        # Empty orderbook
        with pytest.raises(ValidationError):
            await analyzer.assess_depth("BTC/USDT", {"bids": [], "asks": []})
        
        # Missing bids
        with pytest.raises(ValidationError):
            await analyzer.assess_depth("BTC/USDT", {"asks": [[50000, 10]]})


class TestMarketImpactEstimation:
    """Test market impact estimation"""

    @pytest.mark.asyncio
    async def test_market_impact_calculation(self, analyzer, sample_orderbook):
        """Test market impact estimation using square-root model"""
        # First assess depth to cache orderbook
        await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        # Estimate impact for various order sizes
        order_size = Decimal("100")
        adv = Decimal("10000")  # Average daily volume
        
        impact = await analyzer.estimate_impact(
            "BTC/USDT",
            order_size,
            "buy",
            adv
        )
        
        assert impact.temporary_impact_bps > Decimal("0")
        assert impact.permanent_impact_bps > Decimal("0")
        assert impact.total_impact_bps == impact.temporary_impact_bps + impact.permanent_impact_bps

    @pytest.mark.asyncio
    async def test_impact_increases_with_size(self, analyzer, sample_orderbook):
        """Test that impact increases with order size"""
        await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        adv = Decimal("10000")
        
        # Small order
        small_impact = await analyzer.estimate_impact(
            "BTC/USDT", Decimal("10"), "buy", adv
        )
        
        # Large order
        large_impact = await analyzer.estimate_impact(
            "BTC/USDT", Decimal("1000"), "buy", adv
        )
        
        assert large_impact.total_impact_bps > small_impact.total_impact_bps
        assert large_impact.expected_slippage > small_impact.expected_slippage

    @pytest.mark.asyncio
    async def test_buy_vs_sell_impact(self, analyzer, sample_orderbook):
        """Test market impact for buy vs sell orders"""
        await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        order_size = Decimal("100")
        adv = Decimal("10000")
        
        buy_impact = await analyzer.estimate_impact(
            "BTC/USDT", order_size, "buy", adv
        )
        
        sell_impact = await analyzer.estimate_impact(
            "BTC/USDT", order_size, "sell", adv
        )
        
        # Both should have impact
        assert buy_impact.total_impact_bps > Decimal("0")
        assert sell_impact.total_impact_bps > Decimal("0")
        
        # Execution prices should differ
        assert buy_impact.execution_price != sell_impact.execution_price

    @pytest.mark.asyncio
    async def test_confidence_interval(self, analyzer, sample_orderbook):
        """Test confidence interval calculation"""
        await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        
        impact = await analyzer.estimate_impact(
            "BTC/USDT", Decimal("100"), "buy", Decimal("10000")
        )
        
        lower, upper = impact.confidence_interval
        assert lower < impact.total_impact_bps < upper
        assert lower == impact.total_impact_bps * Decimal("0.7")
        assert upper == impact.total_impact_bps * Decimal("1.3")


class TestOrderBookImbalance:
    """Test order book imbalance detection"""

    @pytest.mark.asyncio
    async def test_balanced_orderbook(self, analyzer, sample_orderbook):
        """Test imbalance calculation for balanced orderbook"""
        imbalance = await analyzer.calculate_imbalance("BTC/USDT", sample_orderbook)
        
        assert imbalance.direction == "balanced"
        assert not imbalance.is_one_sided
        assert imbalance.severity == "low"

    @pytest.mark.asyncio
    async def test_bid_heavy_imbalance(self, analyzer, imbalanced_orderbook):
        """Test detection of bid-heavy imbalance"""
        imbalance = await analyzer.calculate_imbalance("BTC/USDT", imbalanced_orderbook)
        
        assert imbalance.direction == "bid_heavy"
        assert imbalance.is_one_sided
        assert imbalance.imbalance_ratio > Decimal("2.0")
        assert imbalance.flow_imbalance > Decimal("0")

    @pytest.mark.asyncio
    async def test_ask_heavy_imbalance(self, analyzer):
        """Test detection of ask-heavy imbalance"""
        # Create ask-heavy orderbook
        orderbook = {
            "bids": [[49995, 10], [49990, 15], [49985, 20]],
            "asks": [[50005, 100], [50010, 150], [50015, 200]]
        }
        
        imbalance = await analyzer.calculate_imbalance("BTC/USDT", orderbook)
        
        assert imbalance.direction == "ask_heavy"
        assert imbalance.is_one_sided
        assert imbalance.flow_imbalance < Decimal("0")

    @pytest.mark.asyncio
    async def test_persistent_imbalance_detection(self, analyzer, imbalanced_orderbook):
        """Test detection of persistent order imbalance"""
        # Create persistent imbalance
        for _ in range(10):
            await analyzer.calculate_imbalance("BTC/USDT", imbalanced_orderbook)
        
        # Check that history is maintained
        assert len(analyzer._imbalance_history["BTC/USDT"]) == 10
        
        # All should show bid-heavy
        for imb in analyzer._imbalance_history["BTC/USDT"]:
            assert imb.direction == "bid_heavy"


class TestMicrostructureAnomalies:
    """Test microstructure anomaly detection"""

    @pytest.mark.asyncio
    async def test_quote_stuffing_detection(self, analyzer, sample_orderbook):
        """Test detection of quote stuffing"""
        symbol = "BTC/USDT"
        
        # Simulate rapid updates (quote stuffing)
        for _ in range(40):  # More than threshold
            anomalies = await analyzer.detect_anomalies(symbol, sample_orderbook)
        
        # Should detect quote stuffing
        quote_stuffing = [a for a in anomalies if a.anomaly_type == "quote_stuffing"]
        assert len(quote_stuffing) > 0
        assert quote_stuffing[0].severity in ["medium", "high"]

    @pytest.mark.asyncio
    async def test_layering_detection(self, analyzer):
        """Test detection of layering/spoofing"""
        # Create orderbook with layering pattern
        orderbook = {
            "bids": [
                [49995, 10],   # Small at best
                [49990, 15],
                [49985, 100],  # Large away from best
                [49980, 150],  # Large away from best
                [49975, 200],  # Large away from best
            ],
            "asks": [[50005, 10], [50010, 15], [50015, 20]]
        }
        
        anomalies = await analyzer.detect_anomalies("BTC/USDT", orderbook)
        
        layering = [a for a in anomalies if a.anomaly_type == "layering"]
        assert len(layering) > 0

    @pytest.mark.asyncio
    async def test_unusual_spread_detection(self, analyzer):
        """Test detection of unusual spreads"""
        # Inverted spread
        inverted_orderbook = {
            "bids": [[50010, 10]],  # Bid > Ask
            "asks": [[50000, 10]]
        }
        
        anomalies = await analyzer.detect_anomalies("BTC/USDT", inverted_orderbook)
        
        unusual = [a for a in anomalies if a.anomaly_type == "unusual_spread"]
        assert len(unusual) > 0
        assert unusual[0].severity == "critical"
        assert unusual[0].details["type"] == "inverted_spread"

    @pytest.mark.asyncio
    async def test_wide_spread_detection(self, analyzer):
        """Test detection of extremely wide spreads"""
        # Wide spread (>1%)
        wide_orderbook = {
            "bids": [[49500, 10]],
            "asks": [[50500, 10]]  # 1000 point spread
        }
        
        anomalies = await analyzer.detect_anomalies("BTC/USDT", wide_orderbook)
        
        unusual = [a for a in anomalies if a.anomaly_type == "unusual_spread"]
        assert len(unusual) > 0
        assert unusual[0].details["type"] == "wide_spread"


class TestLiquidityScoring:
    """Test liquidity scoring system"""

    @pytest.mark.asyncio
    async def test_liquidity_score_calculation(self, analyzer, sample_orderbook):
        """Test composite liquidity score calculation"""
        score = await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            sample_orderbook,
            spread_bps=Decimal("10"),
            volatility=Decimal("0.02")
        )
        
        assert Decimal("0") <= score.overall_score <= Decimal("100")
        assert Decimal("0") <= score.depth_score <= Decimal("100")
        assert Decimal("0") <= score.spread_score <= Decimal("100")
        assert Decimal("0") <= score.stability_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_liquidity_grade_assignment(self, analyzer, sample_orderbook):
        """Test liquidity grade assignment"""
        # Good liquidity conditions
        good_score = await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            sample_orderbook,
            spread_bps=Decimal("5"),  # Tight spread
            volatility=Decimal("0.01")  # Low volatility
        )
        
        assert good_score.liquidity_grade in ["A", "B"]
        
        # Poor liquidity conditions
        poor_orderbook = {
            "bids": [[49990, 1], [49980, 1]],
            "asks": [[50010, 1], [50020, 1]]
        }
        
        poor_score = await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            poor_orderbook,
            spread_bps=Decimal("100"),  # Wide spread
            volatility=Decimal("0.5")  # High volatility
        )
        
        assert poor_score.liquidity_grade in ["D", "F"]

    @pytest.mark.asyncio
    async def test_time_of_day_adjustment(self, analyzer, sample_orderbook):
        """Test time of day adjustment to liquidity score"""
        score = await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            sample_orderbook,
            spread_bps=Decimal("10"),
            volatility=Decimal("0.02")
        )
        
        # Check that time adjustment is applied
        assert score.time_of_day_adjustment > Decimal("0")
        assert score.final_score == score.overall_score * score.time_of_day_adjustment

    @pytest.mark.asyncio
    async def test_score_caching(self, analyzer, sample_orderbook):
        """Test that liquidity scores are cached"""
        score1 = await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            sample_orderbook,
            spread_bps=Decimal("10"),
            volatility=Decimal("0.02")
        )
        
        # Check cache
        assert "BTC/USDT" in analyzer._liquidity_scores
        cached_score = analyzer._liquidity_scores["BTC/USDT"]
        assert cached_score.final_score == score1.final_score


class TestWeightedSpreadCalculations:
    """Test VWAP spread calculations"""

    @pytest.mark.asyncio
    async def test_vwap_spread_calculation(self, analyzer, sample_orderbook):
        """Test volume-weighted average spread calculation"""
        # Assess depth multiple times to build history
        for i in range(5):
            modified_orderbook = sample_orderbook.copy()
            # Vary volumes
            modified_orderbook["bids"][0][1] = 10 * (i + 1)
            modified_orderbook["asks"][0][1] = 12 * (i + 1)
            
            await analyzer.assess_depth("BTC/USDT", modified_orderbook)
        
        vwap_spread = await analyzer.calculate_vwap_spread(
            "BTC/USDT",
            sample_orderbook,
            lookback_seconds=300
        )
        
        assert vwap_spread > Decimal("0")

    @pytest.mark.asyncio
    async def test_vwap_with_no_history(self, analyzer, sample_orderbook):
        """Test VWAP calculation with no history"""
        vwap_spread = await analyzer.calculate_vwap_spread(
            "BTC/USDT",
            sample_orderbook,
            lookback_seconds=300
        )
        
        # Should calculate from current orderbook
        assert vwap_spread > Decimal("0")


class TestDataPersistence:
    """Test data persistence and loading"""

    @pytest.mark.asyncio
    async def test_save_metrics(self, analyzer, sample_orderbook):
        """Test saving metrics to file"""
        # Generate some data
        await analyzer.assess_depth("BTC/USDT", sample_orderbook)
        await analyzer.calculate_liquidity_score(
            "BTC/USDT",
            sample_orderbook,
            Decimal("10"),
            Decimal("0.02")
        )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            await analyzer.save_metrics("BTC/USDT", filepath)
            
            # Check file exists and contains data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert data["symbol"] == "BTC/USDT"
            assert "liquidity_score" in data
            assert data["depth_history_size"] > 0
            
        finally:
            Path(filepath).unlink()

    @pytest.mark.asyncio
    async def test_load_historical(self, analyzer):
        """Test loading historical metrics"""
        # Create test data file
        test_data = {
            "symbol": "BTC/USDT",
            "timestamp": datetime.now(UTC).isoformat(),
            "liquidity_score": {
                "final_score": 85.5,
                "grade": "B"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            filepath = f.name
        
        try:
            data = await analyzer.load_historical("BTC/USDT", filepath)
            
            assert data is not None
            assert data["symbol"] == "BTC/USDT"
            assert data["liquidity_score"]["final_score"] == 85.5
            
        finally:
            Path(filepath).unlink()


class TestConcurrency:
    """Test thread-safety with concurrent operations"""

    @pytest.mark.asyncio
    async def test_concurrent_depth_assessment(self, analyzer, sample_orderbook):
        """Test concurrent depth assessments"""
        async def assess(symbol, index):
            modified_orderbook = sample_orderbook.copy()
            modified_orderbook["bids"][0][1] = 10 + index
            return await analyzer.assess_depth(symbol, modified_orderbook)
        
        # Create concurrent tasks
        tasks = [assess("BTC/USDT", i) for i in range(10)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, LiquidityDepthMetrics)

    @pytest.mark.asyncio
    async def test_concurrent_anomaly_detection(self, analyzer, sample_orderbook):
        """Test concurrent anomaly detection"""
        tasks = [
            analyzer.detect_anomalies("BTC/USDT", sample_orderbook)
            for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_empty_orderbook_handling(self, analyzer):
        """Test handling of empty orderbook"""
        empty_orderbook = {"bids": [], "asks": []}
        
        with pytest.raises(ValidationError):
            await analyzer.assess_depth("BTC/USDT", empty_orderbook)

    @pytest.mark.asyncio
    async def test_single_level_orderbook(self, analyzer):
        """Test handling of single-level orderbook"""
        single_level = {
            "bids": [[49995, 10]],
            "asks": [[50005, 10]]
        }
        
        metrics = await analyzer.assess_depth("BTC/USDT", single_level)
        
        # Should handle gracefully
        assert metrics.bid_depth[5] == Decimal("10")
        assert metrics.ask_depth[5] == Decimal("10")

    @pytest.mark.asyncio
    async def test_zero_volume_levels(self, analyzer):
        """Test handling of zero volume levels"""
        orderbook = {
            "bids": [[49995, 0], [49990, 10]],
            "asks": [[50005, 0], [50010, 10]]
        }
        
        imbalance = await analyzer.calculate_imbalance("BTC/USDT", orderbook)
        
        # Should handle zero volumes
        assert imbalance.bid_pressure >= Decimal("0")
        assert imbalance.ask_pressure >= Decimal("0")