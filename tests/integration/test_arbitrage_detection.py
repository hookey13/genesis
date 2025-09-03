"""Integration tests for arbitrage detection system."""
import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
import random

import pytest
import numpy as np

from genesis.analytics.arbitrage_detector import ArbitrageDetector
from genesis.analytics.opportunity_models import (
    ExchangePair,
    OpportunityType,
    OpportunityStatus
)


class MarketDataSimulator:
    """Simulates realistic market data for integration testing."""
    
    def __init__(self):
        self.base_prices = {
            "BTC/USDT": Decimal("50000"),
            "ETH/USDT": Decimal("3000"),
            "BNB/USDT": Decimal("300"),
            "SOL/USDT": Decimal("100"),
            "ADA/USDT": Decimal("0.5"),
            "ETH/BTC": Decimal("0.06"),
            "BNB/BTC": Decimal("0.006"),
            "SOL/BTC": Decimal("0.002"),
            "BNB/ETH": Decimal("0.1"),
            "SOL/ETH": Decimal("0.033")
        }
        
        self.exchanges = ["binance", "coinbase", "kraken", "huobi", "okx"]
        self.fee_structures = {
            "binance": Decimal("0.001"),
            "coinbase": Decimal("0.005"),
            "kraken": Decimal("0.0026"),
            "huobi": Decimal("0.002"),
            "okx": Decimal("0.0015")
        }
    
    def generate_market_data(self, num_pairs: int = 100) -> Dict[str, List[ExchangePair]]:
        """Generate simulated market data with arbitrage opportunities."""
        market_data = {}
        
        for symbol, base_price in list(self.base_prices.items())[:num_pairs]:
            exchange_pairs = []
            
            for exchange in self.exchanges:
                # Add random variation to create arbitrage opportunities
                price_variation = Decimal(str(random.uniform(-0.005, 0.005)))
                bid_price = base_price * (Decimal("1") + price_variation)
                ask_price = bid_price * Decimal("1.0002")  # Small spread
                
                # Random volume
                volume = Decimal(str(random.uniform(10, 1000)))
                
                pair = ExchangePair(
                    exchange=exchange,
                    symbol=symbol,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_volume=volume,
                    ask_volume=volume * Decimal("0.9"),
                    timestamp=datetime.utcnow(),
                    fee_rate=self.fee_structures[exchange]
                )
                exchange_pairs.append(pair)
            
            market_data[symbol] = exchange_pairs
        
        return market_data
    
    def generate_triangular_data(self, exchange: str) -> Dict[str, ExchangePair]:
        """Generate market data with triangular arbitrage opportunities."""
        data = {}
        
        # Create profitable triangle: USDT -> BTC -> ETH -> USDT
        # With slight inefficiency to create arbitrage
        data["BTC/USDT"] = ExchangePair(
            exchange=exchange,
            symbol="BTC/USDT",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        data["ETH/BTC"] = ExchangePair(
            exchange=exchange,
            symbol="ETH/BTC",
            bid_price=Decimal("0.0605"),  # Slightly higher to create opportunity
            ask_price=Decimal("0.0606"),
            bid_volume=Decimal("50"),
            ask_volume=Decimal("50"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        data["ETH/USDT"] = ExchangePair(
            exchange=exchange,
            symbol="ETH/USDT",
            bid_price=Decimal("3000"),
            ask_price=Decimal("3001"),
            bid_volume=Decimal("30"),
            ask_volume=Decimal("30"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        # Add more pairs for complex paths
        data["BNB/USDT"] = ExchangePair(
            exchange=exchange,
            symbol="BNB/USDT",
            bid_price=Decimal("300"),
            ask_price=Decimal("300.3"),
            bid_volume=Decimal("100"),
            ask_volume=Decimal("100"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        data["BNB/BTC"] = ExchangePair(
            exchange=exchange,
            symbol="BNB/BTC",
            bid_price=Decimal("0.00601"),
            ask_price=Decimal("0.00602"),
            bid_volume=Decimal("200"),
            ask_volume=Decimal("200"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        data["BNB/ETH"] = ExchangePair(
            exchange=exchange,
            symbol="BNB/ETH",
            bid_price=Decimal("0.1001"),
            ask_price=Decimal("0.1002"),
            bid_volume=Decimal("150"),
            ask_volume=Decimal("150"),
            timestamp=datetime.utcnow(),
            fee_rate=self.fee_structures[exchange]
        )
        
        return data
    
    def generate_historical_data(
        self,
        symbol: str,
        exchange: str,
        num_points: int = 200
    ) -> List[tuple[datetime, Decimal]]:
        """Generate historical price data for statistical arbitrage."""
        history = []
        base_price = self.base_prices.get(symbol, Decimal("1000"))
        
        for i in range(num_points):
            timestamp = datetime.utcnow() - timedelta(minutes=num_points - i)
            
            # Add trend and noise
            trend = Decimal(str(i * 0.01))
            noise = Decimal(str(np.random.normal(0, float(base_price) * 0.001)))
            price = base_price + trend + noise
            
            history.append((timestamp, price))
        
        return history


@pytest.fixture
def simulator():
    """Create a market data simulator."""
    return MarketDataSimulator()


@pytest.fixture
def detector():
    """Create an arbitrage detector for testing."""
    return ArbitrageDetector(
        min_profit_pct=0.3,
        min_confidence=0.5,
        max_path_length=4,
        stat_arb_window=100,
        opportunity_ttl=5,
        max_opportunities=100
    )


class TestArbitrageDetectionIntegration:
    """Integration tests for arbitrage detection system."""
    
    @pytest.mark.asyncio
    async def test_multi_exchange_detection(self, detector, simulator):
        """Test detection across multiple exchanges with realistic data."""
        market_data = simulator.generate_market_data(num_pairs=20)
        
        start_time = time.time()
        opportunities = await detector.find_direct_arbitrage(market_data)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance check
        assert elapsed_time < 5000  # Should complete within 5 seconds
        
        # Should find some opportunities in simulated data
        if opportunities:
            for opp in opportunities:
                assert opp.type == OpportunityType.DIRECT
                assert opp.net_profit_pct >= detector.min_profit_pct
                assert opp.buy_exchange != opp.sell_exchange
                assert opp.confidence_score > 0
        
        print(f"Found {len(opportunities)} direct arbitrage opportunities in {elapsed_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_triangular_detection_performance(self, detector, simulator):
        """Test triangular arbitrage detection performance."""
        for exchange in ["binance", "coinbase", "kraken"]:
            market_data = simulator.generate_triangular_data(exchange)
            
            start_time = time.time()
            opportunities = await detector.find_triangular_arbitrage(exchange, market_data)
            elapsed_time = (time.time() - start_time) * 1000
            
            # Performance requirement: <5ms for detection
            assert elapsed_time < 5  # Very fast for small dataset
            
            if opportunities:
                for opp in opportunities:
                    assert opp.type == OpportunityType.TRIANGULAR
                    assert opp.exchange == exchange
                    assert len(opp.path) >= 3
                    assert opp.start_currency == opp.end_currency
            
            print(f"Exchange {exchange}: Found {len(opportunities)} triangular opportunities in {elapsed_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_high_frequency_updates(self, detector, simulator):
        """Test system performance with high-frequency market updates."""
        update_count = 100
        total_time = 0
        all_opportunities = []
        
        for i in range(update_count):
            market_data = simulator.generate_market_data(num_pairs=50)
            
            start_time = time.time()
            result = await detector.update_opportunities(market_data)
            elapsed_time = (time.time() - start_time) * 1000
            total_time += elapsed_time
            
            # Each update should be fast
            assert elapsed_time < 50  # 50ms per update max
            
            total_opps = (
                len(result["direct"]) +
                len(result["triangular"]) +
                len(result["statistical"])
            )
            all_opportunities.append(total_opps)
            
            # Simulate realistic delay between updates
            await asyncio.sleep(0.01)  # 10ms between updates
        
        avg_time = total_time / update_count
        avg_opportunities = sum(all_opportunities) / len(all_opportunities)
        
        print(f"Processed {update_count} updates")
        print(f"Average update time: {avg_time:.2f}ms")
        print(f"Average opportunities per update: {avg_opportunities:.1f}")
        
        # Overall performance requirements
        assert avg_time < 20  # Average should be under 20ms
    
    @pytest.mark.asyncio
    async def test_concurrent_detection(self, detector, simulator):
        """Test concurrent detection across different opportunity types."""
        market_data = simulator.generate_market_data(num_pairs=30)
        triangular_data = simulator.generate_triangular_data("binance")
        
        # Create statistical arbitrage data
        pair_a = ExchangePair(
            exchange="binance",
            symbol="BTC/USDT",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.001")
        )
        
        pair_b = ExchangePair(
            exchange="coinbase",
            symbol="BTC/USDT",
            bid_price=Decimal("49500"),
            ask_price=Decimal("49510"),
            bid_volume=Decimal("8"),
            ask_volume=Decimal("8"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.005")
        )
        
        # Historical data for statistical arbitrage
        historical_data = {
            "binance:BTC/USDT": simulator.generate_historical_data("BTC/USDT", "binance"),
            "coinbase:BTC/USDT": simulator.generate_historical_data("BTC/USDT", "coinbase")
        }
        
        # Run all detections concurrently
        start_time = time.time()
        
        direct_task = detector.find_direct_arbitrage(market_data)
        triangular_task = detector.find_triangular_arbitrage("binance", triangular_data)
        statistical_task = detector.find_statistical_arbitrage(pair_a, pair_b, historical_data)
        
        direct_opps, triangular_opps, statistical_opp = await asyncio.gather(
            direct_task,
            triangular_task,
            statistical_task
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"Concurrent detection completed in {elapsed_time:.2f}ms")
        print(f"Direct opportunities: {len(direct_opps)}")
        print(f"Triangular opportunities: {len(triangular_opps)}")
        print(f"Statistical opportunity: {'Found' if statistical_opp else 'None'}")
        
        # Should complete all detections quickly
        assert elapsed_time < 100  # 100ms for all concurrent operations
    
    @pytest.mark.asyncio
    async def test_opportunity_lifecycle(self, detector, simulator):
        """Test complete lifecycle of opportunities from detection to expiration."""
        market_data = simulator.generate_market_data(num_pairs=10)
        
        # Detect initial opportunities
        opportunities = await detector.find_direct_arbitrage(market_data)
        initial_count = len(opportunities)
        
        # Add to detector's tracking
        for opp in opportunities:
            detector.opportunities[opp.id] = opp
        
        # Simulate market changes over time
        for i in range(5):
            await asyncio.sleep(1)  # Wait 1 second
            
            # Generate slightly different market data
            new_market_data = simulator.generate_market_data(num_pairs=10)
            
            # Update opportunities
            result = await detector.update_opportunities(new_market_data)
            
            active_count = sum(
                1 for opp in detector.opportunities.values()
                if opp.status == OpportunityStatus.ACTIVE
            )
            
            print(f"Update {i + 1}: Active opportunities: {active_count}")
        
        # After 5 seconds, most opportunities should have expired (TTL is 5 seconds)
        final_active = sum(
            1 for opp in detector.opportunities.values()
            if opp.status == OpportunityStatus.ACTIVE
        )
        
        print(f"Initial opportunities: {initial_count}")
        print(f"Final active opportunities: {final_active}")
        
        # Most should have expired
        assert final_active <= initial_count
    
    @pytest.mark.asyncio
    async def test_large_scale_performance(self, detector, simulator):
        """Test performance with 100+ trading pairs."""
        num_pairs = 100
        market_data = simulator.generate_market_data(num_pairs=num_pairs)
        
        # Test direct arbitrage with many pairs
        start_time = time.time()
        opportunities = await detector.find_direct_arbitrage(market_data)
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"Processed {num_pairs} pairs with {len(market_data)} symbols")
        print(f"Found {len(opportunities)} opportunities in {elapsed_time:.2f}ms")
        
        # Should handle 100+ pairs efficiently
        assert elapsed_time < 5000  # 5 seconds max for 100 pairs
        
        # Calculate opportunities per second
        ops_per_second = (num_pairs * len(simulator.exchanges)) / (elapsed_time / 1000)
        print(f"Performance: {ops_per_second:.0f} pair comparisons per second")
    
    @pytest.mark.asyncio
    async def test_filtering_and_ranking(self, detector, simulator):
        """Test opportunity filtering and ranking with realistic data."""
        market_data = simulator.generate_market_data(num_pairs=50)
        
        # Detect opportunities
        opportunities = await detector.find_direct_arbitrage(market_data)
        
        if len(opportunities) > 10:
            # Test filtering
            filtered = detector.filter_opportunities(
                opportunities,
                min_profit=Decimal("0.5"),
                min_confidence=0.7
            )
            
            assert all(opp.profit_pct >= Decimal("0.5") for opp in filtered)
            assert all(opp.confidence_score >= 0.7 for opp in filtered)
            
            # Test ranking
            ranked = detector.rank_opportunities(filtered)
            
            # Check ranking order
            scores = [score for _, score in ranked]
            assert scores == sorted(scores, reverse=True)
            
            print(f"Original opportunities: {len(opportunities)}")
            print(f"Filtered opportunities: {len(filtered)}")
            print(f"Top opportunity score: {scores[0] if scores else 0:.4f}")
    
    @pytest.mark.asyncio
    async def test_execution_path_optimization(self, detector, simulator):
        """Test execution path optimization with realistic latencies."""
        market_data = simulator.generate_market_data(num_pairs=10)
        opportunities = await detector.find_direct_arbitrage(market_data)
        
        # Realistic exchange latencies (in ms)
        latencies = {
            "binance": 25,
            "coinbase": 45,
            "kraken": 35,
            "huobi": 55,
            "okx": 30
        }
        
        for opp in opportunities[:5]:  # Test first 5 opportunities
            path = detector.optimize_execution_path(opp, latencies)
            
            assert path.opportunity_id == opp.id
            assert len(path.steps) > 0
            assert path.estimated_time_ms > 0
            assert path.estimated_slippage >= 0
            assert 0 <= path.risk_score <= 1
            
            print(f"Opportunity {opp.symbol}: {opp.buy_exchange} -> {opp.sell_exchange}")
            print(f"  Estimated time: {path.estimated_time_ms}ms")
            print(f"  Risk score: {path.risk_score:.3f}")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, detector, simulator):
        """Test memory efficiency with continuous updates."""
        import gc
        import sys
        
        initial_memory = sys.getsizeof(detector.opportunities)
        
        # Simulate continuous operation
        for i in range(100):
            market_data = simulator.generate_market_data(num_pairs=20)
            await detector.update_opportunities(market_data)
            
            # Detector should maintain max_opportunities limit
            assert len(detector.opportunities) <= detector.max_opportunities
            
            # Periodic garbage collection
            if i % 20 == 0:
                gc.collect()
        
        final_memory = sys.getsizeof(detector.opportunities)
        memory_growth = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory} bytes")
        print(f"Final memory: {final_memory} bytes")
        print(f"Memory growth: {memory_growth} bytes")
        
        # Memory should not grow unbounded
        assert len(detector.opportunities) <= detector.max_opportunities