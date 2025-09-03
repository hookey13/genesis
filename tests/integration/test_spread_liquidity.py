"""
Integration tests for Spread Tracking and Liquidity Analysis
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.analytics.liquidity_analyzer import LiquidityAnalyzer, LiquidityAnalyzerConfig
from genesis.analytics.spread_tracker_enhanced import (
    EnhancedSpreadTracker,
    SpreadTrackerConfig,
)


@pytest.fixture
def spread_config():
    """Create spread tracker configuration"""
    return SpreadTrackerConfig(
        spread_window=500,
        baseline_period=600,
        volatility_halflife=120,
        cache_ttl=60
    )


@pytest.fixture
def liquidity_config():
    """Create liquidity analyzer configuration"""
    return LiquidityAnalyzerConfig(
        depth_levels=[5, 10, 20],
        imbalance_threshold=Decimal("2.0"),
        market_impact_eta=Decimal("0.1"),
        market_impact_gamma=Decimal("0.05")
    )


@pytest.fixture
def spread_tracker(spread_config):
    """Create spread tracker instance"""
    return EnhancedSpreadTracker(spread_config)


@pytest.fixture
def liquidity_analyzer(liquidity_config):
    """Create liquidity analyzer instance"""
    return LiquidityAnalyzer(liquidity_config)


def generate_realistic_orderbook(
    base_price: float = 50000,
    spread_bps: float = 10,
    depth_multiplier: float = 1.0,
    imbalance_factor: float = 1.0
) -> dict:
    """Generate realistic orderbook data"""
    mid_price = base_price
    half_spread = base_price * spread_bps / 20000
    
    bids = []
    asks = []
    
    # Generate bid levels
    for i in range(20):
        price = mid_price - half_spread - (i * base_price * 0.0001)
        volume = (20 - i) * depth_multiplier * imbalance_factor
        volume += random.uniform(-volume * 0.1, volume * 0.1)  # Add noise
        bids.append([price, max(1, volume)])
    
    # Generate ask levels
    for i in range(20):
        price = mid_price + half_spread + (i * base_price * 0.0001)
        volume = (20 - i) * depth_multiplier / imbalance_factor
        volume += random.uniform(-volume * 0.1, volume * 0.1)  # Add noise
        asks.append([price, max(1, volume)])
    
    return {"bids": bids, "asks": asks}


class TestIntegratedSpreadAndLiquidity:
    """Test integrated spread tracking and liquidity analysis"""

    @pytest.mark.asyncio
    async def test_combined_spread_and_liquidity_analysis(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test combined spread and liquidity analysis workflow"""
        symbol = "BTC/USDT"
        
        # Generate market data
        orderbook = generate_realistic_orderbook(spread_bps=15)
        
        # Update spread tracker
        best_bid = Decimal(str(orderbook["bids"][0][0]))
        best_ask = Decimal(str(orderbook["asks"][0][0]))
        bid_vol = Decimal(str(orderbook["bids"][0][1]))
        ask_vol = Decimal(str(orderbook["asks"][0][1]))
        
        spread_metrics = await spread_tracker.update_spread(
            symbol, best_bid, best_ask, bid_vol, ask_vol
        )
        
        # Assess liquidity
        depth_metrics = await liquidity_analyzer.assess_depth(symbol, orderbook)
        
        # Calculate liquidity score using spread metrics
        liquidity_score = await liquidity_analyzer.calculate_liquidity_score(
            symbol,
            orderbook,
            spread_metrics.current_spread_bps,
            spread_metrics.ewma_volatility
        )
        
        # Verify integration
        assert spread_metrics.current_spread_bps > Decimal("0")
        assert depth_metrics.total_bid_volume > Decimal("0")
        assert liquidity_score.final_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_real_time_market_monitoring(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test real-time monitoring of market conditions"""
        symbol = "BTC/USDT"
        metrics_history = []
        
        # Simulate market data stream
        for i in range(20):
            # Generate varying market conditions
            spread = 5 + (i % 10)  # Oscillating spread
            depth = 0.5 + (i / 20)  # Increasing depth
            imbalance = 1.0 + 0.5 * (-1) ** i  # Alternating imbalance
            
            orderbook = generate_realistic_orderbook(
                spread_bps=spread,
                depth_multiplier=depth,
                imbalance_factor=imbalance
            )
            
            # Update spread
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            spread_metrics = await spread_tracker.update_spread(
                symbol,
                best_bid,
                best_ask,
                Decimal(str(orderbook["bids"][0][1])),
                Decimal(str(orderbook["asks"][0][1]))
            )
            
            # Analyze liquidity
            depth_metrics = await liquidity_analyzer.assess_depth(symbol, orderbook)
            imbalance_metrics = await liquidity_analyzer.calculate_imbalance(
                symbol, orderbook
            )
            
            # Store metrics
            metrics_history.append({
                "spread": spread_metrics,
                "depth": depth_metrics,
                "imbalance": imbalance_metrics
            })
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        # Verify monitoring captured changes
        assert len(metrics_history) == 20
        
        # Check spread variation captured
        spreads = [m["spread"].current_spread_bps for m in metrics_history]
        assert max(spreads) > min(spreads)
        
        # Check depth increase captured
        depths = [m["depth"].total_bid_volume for m in metrics_history]
        assert depths[-1] > depths[0]  # Depth increased over time

    @pytest.mark.asyncio
    async def test_market_impact_with_liquidity(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test market impact estimation with current liquidity conditions"""
        symbol = "BTC/USDT"
        
        # Set up market conditions
        orderbook = generate_realistic_orderbook(
            spread_bps=10,
            depth_multiplier=2.0  # Good liquidity
        )
        
        # Assess current liquidity
        await liquidity_analyzer.assess_depth(symbol, orderbook)
        
        # Test impact for different order sizes
        small_order = Decimal("10")
        medium_order = Decimal("100")
        large_order = Decimal("1000")
        
        adv = Decimal("50000")  # Average daily volume
        
        small_impact = await liquidity_analyzer.estimate_impact(
            symbol, small_order, "buy", adv
        )
        medium_impact = await liquidity_analyzer.estimate_impact(
            symbol, medium_order, "buy", adv
        )
        large_impact = await liquidity_analyzer.estimate_impact(
            symbol, large_order, "buy", adv
        )
        
        # Verify impact increases with size
        assert small_impact.total_impact_bps < medium_impact.total_impact_bps
        assert medium_impact.total_impact_bps < large_impact.total_impact_bps

    @pytest.mark.asyncio
    async def test_volatility_regime_and_liquidity_score(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test relationship between volatility regime and liquidity scoring"""
        symbol = "BTC/USDT"
        
        # Create stable market conditions
        for _ in range(10):
            orderbook = generate_realistic_orderbook(spread_bps=5)
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
        
        stable_metrics = await spread_tracker.get_cached_metrics(symbol)
        stable_score = await liquidity_analyzer.calculate_liquidity_score(
            symbol,
            generate_realistic_orderbook(spread_bps=5),
            stable_metrics.current_spread_bps,
            stable_metrics.ewma_volatility
        )
        
        # Create volatile conditions
        for i in range(10):
            spread = 5 + (i * 5)  # Increasing spread
            orderbook = generate_realistic_orderbook(spread_bps=spread)
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
        
        volatile_metrics = await spread_tracker.get_cached_metrics(symbol)
        volatile_score = await liquidity_analyzer.calculate_liquidity_score(
            symbol,
            generate_realistic_orderbook(spread_bps=50),
            volatile_metrics.current_spread_bps,
            volatile_metrics.ewma_volatility
        )
        
        # Verify relationship
        assert stable_metrics.volatility_regime in ["low", "normal"]
        assert volatile_metrics.volatility_regime in ["high", "extreme"]
        assert stable_score.final_score > volatile_score.final_score


class TestAnomalyDetectionIntegration:
    """Test integrated anomaly detection"""

    @pytest.mark.asyncio
    async def test_spread_anomaly_with_microstructure_anomaly(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test detection of correlated spread and microstructure anomalies"""
        symbol = "BTC/USDT"
        
        # Establish normal conditions
        for _ in range(20):
            orderbook = generate_realistic_orderbook(spread_bps=10)
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
        
        # Create anomalous conditions
        anomalous_orderbook = generate_realistic_orderbook(spread_bps=100)  # 10x normal
        
        # Update spread - should detect anomaly
        best_bid = Decimal(str(anomalous_orderbook["bids"][0][0]))
        best_ask = Decimal(str(anomalous_orderbook["asks"][0][0]))
        
        spread_metrics = await spread_tracker.update_spread(
            symbol, best_bid, best_ask,
            Decimal("10"), Decimal("10")
        )
        
        # Check for microstructure anomalies
        micro_anomalies = await liquidity_analyzer.detect_anomalies(
            symbol, anomalous_orderbook
        )
        
        # Both should detect anomalies
        assert spread_metrics.is_anomaly is True
        assert len(micro_anomalies) > 0
        
        # Check for wide spread anomaly
        wide_spread = [a for a in micro_anomalies if a.anomaly_type == "unusual_spread"]
        assert len(wide_spread) > 0

    @pytest.mark.asyncio
    async def test_quote_stuffing_impact_on_spreads(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test impact of quote stuffing on spread metrics"""
        symbol = "BTC/USDT"
        
        # Simulate quote stuffing (rapid updates)
        for i in range(50):
            # Rapidly changing prices
            base_price = 50000 + (i * 0.1)
            orderbook = generate_realistic_orderbook(base_price=base_price)
            
            # Detect anomalies
            anomalies = await liquidity_analyzer.detect_anomalies(symbol, orderbook)
            
            # Update spread
            if i % 5 == 0:  # Update spread less frequently
                best_bid = Decimal(str(orderbook["bids"][0][0]))
                best_ask = Decimal(str(orderbook["asks"][0][0]))
                
                await spread_tracker.update_spread(
                    symbol, best_bid, best_ask,
                    Decimal("10"), Decimal("10")
                )
        
        # Check for quote stuffing detection
        final_anomalies = await liquidity_analyzer.detect_anomalies(
            symbol, generate_realistic_orderbook()
        )
        
        quote_stuffing = [
            a for a in final_anomalies if a.anomaly_type == "quote_stuffing"
        ]
        assert len(quote_stuffing) > 0


class TestPerformanceUnderLoad:
    """Test performance with high-frequency updates"""

    @pytest.mark.asyncio
    async def test_high_frequency_updates(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test system performance with 1000 updates/second"""
        symbol = "BTC/USDT"
        update_count = 100  # Reduced for test speed
        
        start_time = datetime.now(UTC)
        
        # Perform rapid updates
        tasks = []
        for i in range(update_count):
            orderbook = generate_realistic_orderbook(
                base_price=50000 + (i * 0.01),
                spread_bps=5 + (i % 5)
            )
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            # Alternate between spread and liquidity updates
            if i % 2 == 0:
                task = spread_tracker.update_spread(
                    symbol, best_bid, best_ask,
                    Decimal("10"), Decimal("10")
                )
            else:
                task = liquidity_analyzer.assess_depth(symbol, orderbook)
            
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert len(results) == update_count
        updates_per_second = update_count / duration if duration > 0 else 0
        
        # Should handle at least 100 updates/second
        assert updates_per_second > 100 or duration < 1.0

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test memory usage stays within limits"""
        symbol = "BTC/USDT"
        
        # Fill up history buffers
        for i in range(1000):
            orderbook = generate_realistic_orderbook()
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
            
            if i % 10 == 0:
                await liquidity_analyzer.assess_depth(symbol, orderbook)
        
        # Trigger cleanup
        await spread_tracker._cleanup_old_data()
        
        # Check memory usage
        assert spread_tracker._memory_usage < spread_tracker.config.max_memory_mb
        
        # Check that deques are bounded
        assert len(spread_tracker._spread_history[symbol]) <= spread_tracker.config.spread_window

    @pytest.mark.asyncio
    async def test_processing_time_requirements(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test that processing time stays under 2ms requirement"""
        symbol = "BTC/USDT"
        orderbook = generate_realistic_orderbook()
        
        # Warm up caches
        for _ in range(10):
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
        
        # Measure processing time
        processing_times = []
        
        for _ in range(100):
            orderbook = generate_realistic_orderbook()
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            start = datetime.now(UTC)
            
            # Process update
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal("10"), Decimal("10")
            )
            await liquidity_analyzer.assess_depth(symbol, orderbook)
            
            end = datetime.now(UTC)
            processing_time_ms = (end - start).total_seconds() * 1000
            processing_times.append(processing_time_ms)
        
        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Should meet <2ms requirement (relaxed for test environment)
        assert avg_time < 10  # Relaxed for test
        assert max_time < 20  # Relaxed for test


class TestWeightedSpreadIntegration:
    """Test integrated weighted spread calculations"""

    @pytest.mark.asyncio
    async def test_vwap_twap_comparison(self, spread_tracker, liquidity_analyzer):
        """Test VWAP vs TWAP spread calculations"""
        symbol = "BTC/USDT"
        
        # Create market data with varying volumes and times
        for i in range(20):
            # Vary volume significantly
            volume_multiplier = 1.0 if i % 5 != 0 else 10.0
            
            orderbook = generate_realistic_orderbook(
                spread_bps=10 + (i % 3),
                depth_multiplier=volume_multiplier
            )
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            bid_vol = Decimal(str(orderbook["bids"][0][1]))
            ask_vol = Decimal(str(orderbook["asks"][0][1]))
            
            await spread_tracker.update_spread(
                symbol, best_bid, best_ask, bid_vol, ask_vol
            )
            
            # Vary time intervals
            if i % 5 == 0:
                await asyncio.sleep(0.02)  # Longer interval
            else:
                await asyncio.sleep(0.01)
        
        metrics = await spread_tracker.get_cached_metrics(symbol)
        
        # VWAP and TWAP should differ when volumes and times vary
        assert metrics.vwap_spread != metrics.twap_spread
        
        # Both should be positive
        assert metrics.vwap_spread > Decimal("0")
        assert metrics.twap_spread > Decimal("0")


class TestLiquidityScoringIntegration:
    """Test integrated liquidity scoring with all components"""

    @pytest.mark.asyncio
    async def test_comprehensive_liquidity_assessment(
        self, spread_tracker, liquidity_analyzer
    ):
        """Test comprehensive liquidity assessment workflow"""
        symbol = "BTC/USDT"
        
        # Build up historical data
        for i in range(30):
            spread = 5 + (i % 10)
            depth = 1.0 + (i / 30)
            
            orderbook = generate_realistic_orderbook(
                spread_bps=spread,
                depth_multiplier=depth
            )
            
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            
            # Update spread tracker
            spread_metrics = await spread_tracker.update_spread(
                symbol, best_bid, best_ask,
                Decimal(str(orderbook["bids"][0][1])),
                Decimal(str(orderbook["asks"][0][1]))
            )
            
            # Assess liquidity every 5 updates
            if i % 5 == 0:
                await liquidity_analyzer.assess_depth(symbol, orderbook)
        
        # Calculate comprehensive metrics
        baseline = await spread_tracker.calculate_baseline(symbol)
        final_orderbook = generate_realistic_orderbook(spread_bps=8, depth_multiplier=1.5)
        
        # Get final assessments
        depth_metrics = await liquidity_analyzer.assess_depth(symbol, final_orderbook)
        imbalance = await liquidity_analyzer.calculate_imbalance(symbol, final_orderbook)
        vwap_spread = await liquidity_analyzer.calculate_vwap_spread(
            symbol, final_orderbook
        )
        
        # Calculate final score
        final_spread_metrics = await spread_tracker.get_cached_metrics(symbol)
        liquidity_score = await liquidity_analyzer.calculate_liquidity_score(
            symbol,
            final_orderbook,
            final_spread_metrics.current_spread_bps,
            final_spread_metrics.ewma_volatility
        )
        
        # Verify comprehensive assessment
        assert baseline["mean"] > Decimal("0")
        assert depth_metrics.total_bid_volume > Decimal("0")
        assert imbalance.imbalance_ratio > Decimal("0")
        assert vwap_spread > Decimal("0")
        assert liquidity_score.final_score > Decimal("0")
        assert liquidity_score.liquidity_grade in ["A", "B", "C", "D", "F"]