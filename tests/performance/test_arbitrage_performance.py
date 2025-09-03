"""Performance tests for arbitrage detection system."""
import asyncio
import time
import cProfile
import pstats
import io
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
import random

import pytest
import numpy as np

from genesis.analytics.arbitrage_detector import ArbitrageDetector
from genesis.analytics.opportunity_models import ExchangePair


def generate_large_market_data(num_pairs: int, num_exchanges: int) -> Dict[str, List[ExchangePair]]:
    """Generate large market dataset for performance testing."""
    market_data = {}
    symbols = [f"TOKEN{i}/USDT" for i in range(num_pairs)]
    exchanges = [f"exchange{i}" for i in range(num_exchanges)]
    
    for symbol in symbols:
        exchange_pairs = []
        base_price = Decimal(str(random.uniform(1, 10000)))
        
        for exchange in exchanges:
            variation = Decimal(str(random.uniform(-0.01, 0.01)))
            bid_price = base_price * (Decimal("1") + variation)
            ask_price = bid_price * Decimal("1.0002")
            
            pair = ExchangePair(
                exchange=exchange,
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_volume=Decimal(str(random.uniform(10, 1000))),
                ask_volume=Decimal(str(random.uniform(10, 1000))),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.001")
            )
            exchange_pairs.append(pair)
        
        market_data[symbol] = exchange_pairs
    
    return market_data


@pytest.fixture
def detector():
    """Create optimized detector for performance testing."""
    return ArbitrageDetector(
        min_profit_pct=0.3,
        min_confidence=0.5,
        max_path_length=3,  # Limit for performance
        stat_arb_window=50,  # Smaller window for speed
        opportunity_ttl=5,
        max_opportunities=100
    )


class TestArbitragePerformance:
    """Performance benchmarks for arbitrage detection."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_direct_arbitrage_performance_small(self, detector, benchmark):
        """Benchmark direct arbitrage with small dataset (10 pairs, 5 exchanges)."""
        market_data = generate_large_market_data(10, 5)
        
        async def run():
            return await detector.find_direct_arbitrage(market_data)
        
        result = await benchmark(run)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_direct_arbitrage_performance_medium(self, detector, benchmark):
        """Benchmark direct arbitrage with medium dataset (50 pairs, 5 exchanges)."""
        market_data = generate_large_market_data(50, 5)
        
        async def run():
            return await detector.find_direct_arbitrage(market_data)
        
        result = await benchmark(run)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_direct_arbitrage_performance_large(self, detector, benchmark):
        """Benchmark direct arbitrage with large dataset (100 pairs, 10 exchanges)."""
        market_data = generate_large_market_data(100, 10)
        
        async def run():
            return await detector.find_direct_arbitrage(market_data)
        
        result = await benchmark(run)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_detection_under_5ms(self, detector):
        """Verify detection completes under 5ms for typical workload."""
        market_data = generate_large_market_data(20, 5)
        
        # Warm up
        await detector.find_direct_arbitrage(market_data)
        
        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            await detector.find_direct_arbitrage(market_data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        p95_time = np.percentile(times, 95)
        
        print(f"Average time: {avg_time:.2f}ms")
        print(f"P95 time: {p95_time:.2f}ms")
        
        assert avg_time < 5  # Average under 5ms
        assert p95_time < 10  # 95th percentile under 10ms
    
    @pytest.mark.asyncio
    async def test_triangular_performance(self, detector):
        """Test triangular arbitrage performance."""
        # Generate connected graph of pairs
        market_data = {
            "BTC/USDT": ExchangePair(
                exchange="binance", symbol="BTC/USDT",
                bid_price=Decimal("50000"), ask_price=Decimal("50010"),
                bid_volume=Decimal("10"), ask_volume=Decimal("10"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            ),
            "ETH/USDT": ExchangePair(
                exchange="binance", symbol="ETH/USDT",
                bid_price=Decimal("3000"), ask_price=Decimal("3001"),
                bid_volume=Decimal("50"), ask_volume=Decimal("50"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            ),
            "ETH/BTC": ExchangePair(
                exchange="binance", symbol="ETH/BTC",
                bid_price=Decimal("0.06"), ask_price=Decimal("0.0601"),
                bid_volume=Decimal("100"), ask_volume=Decimal("100"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            ),
            "BNB/USDT": ExchangePair(
                exchange="binance", symbol="BNB/USDT",
                bid_price=Decimal("300"), ask_price=Decimal("300.3"),
                bid_volume=Decimal("100"), ask_volume=Decimal("100"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            ),
            "BNB/BTC": ExchangePair(
                exchange="binance", symbol="BNB/BTC",
                bid_price=Decimal("0.006"), ask_price=Decimal("0.00601"),
                bid_volume=Decimal("200"), ask_volume=Decimal("200"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            ),
            "BNB/ETH": ExchangePair(
                exchange="binance", symbol="BNB/ETH",
                bid_price=Decimal("0.1"), ask_price=Decimal("0.1001"),
                bid_volume=Decimal("150"), ask_volume=Decimal("150"),
                timestamp=datetime.utcnow(), fee_rate=Decimal("0.001")
            )
        }
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            await detector.find_triangular_arbitrage("binance", market_data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"Triangular detection average: {avg_time:.2f}ms")
        
        assert avg_time < 5  # Should be very fast
    
    @pytest.mark.asyncio
    async def test_concurrent_updates_performance(self, detector):
        """Test performance of concurrent opportunity updates."""
        # Create initial opportunities
        for i in range(50):
            opp_id = f"opp_{i}"
            detector.opportunities[opp_id] = None  # Placeholder
        
        market_data = generate_large_market_data(30, 5)
        
        async def update_batch():
            return await detector.update_opportunities(market_data)
        
        # Run concurrent updates
        start = time.perf_counter()
        tasks = [update_batch() for _ in range(10)]
        await asyncio.gather(*tasks)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"10 concurrent updates completed in {elapsed:.2f}ms")
        assert elapsed < 500  # Should handle 10 concurrent updates in under 500ms
    
    @pytest.mark.asyncio
    async def test_filtering_performance(self, detector):
        """Test performance of opportunity filtering with large dataset."""
        # Create many opportunities
        opportunities = []
        for i in range(1000):
            from genesis.analytics.opportunity_models import DirectArbitrageOpportunity, OpportunityType, OpportunityStatus
            
            opp = DirectArbitrageOpportunity(
                id=f"opp_{i}",
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(random.uniform(0.1, 2.0))),
                profit_amount=Decimal("100"),
                confidence_score=random.uniform(0.3, 1.0),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=10),
                status=OpportunityStatus.ACTIVE,
                buy_exchange="binance",
                sell_exchange="coinbase",
                symbol="BTC/USDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50100"),
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.1"),
                sell_fee=Decimal("0.5"),
                net_profit_pct=Decimal("0.4")
            )
            opportunities.append(opp)
        
        start = time.perf_counter()
        filtered = detector.filter_opportunities(
            opportunities,
            min_profit=Decimal("0.5"),
            min_confidence=0.7
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Filtered 1000 opportunities in {elapsed:.2f}ms")
        print(f"Result: {len(filtered)} opportunities passed filter")
        
        assert elapsed < 10  # Should filter 1000 items in under 10ms
    
    @pytest.mark.asyncio
    async def test_ranking_performance(self, detector):
        """Test performance of opportunity ranking."""
        # Create opportunities for ranking
        opportunities = []
        for i in range(500):
            from genesis.analytics.opportunity_models import DirectArbitrageOpportunity, OpportunityType, OpportunityStatus
            
            opp = DirectArbitrageOpportunity(
                id=f"opp_{i}",
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(random.uniform(0.3, 2.0))),
                profit_amount=Decimal("100"),
                confidence_score=random.uniform(0.5, 1.0),
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=10),
                status=OpportunityStatus.ACTIVE,
                buy_exchange="binance",
                sell_exchange="coinbase",
                symbol="BTC/USDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50100"),
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.1"),
                sell_fee=Decimal("0.5"),
                net_profit_pct=Decimal("0.4")
            )
            opportunities.append(opp)
        
        start = time.perf_counter()
        ranked = detector.rank_opportunities(opportunities)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Ranked 500 opportunities in {elapsed:.2f}ms")
        
        assert elapsed < 20  # Should rank 500 items in under 20ms
        assert len(ranked) == 500
    
    @pytest.mark.asyncio
    async def test_profiling_hot_paths(self, detector):
        """Profile hot paths to identify optimization opportunities."""
        market_data = generate_large_market_data(50, 5)
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run the main detection logic multiple times
        for _ in range(10):
            await detector.find_direct_arbitrage(market_data)
        
        profiler.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions
        
        profile_output = s.getvalue()
        print("\nTop 20 functions by cumulative time:")
        print(profile_output)
        
        # Check that no single function dominates
        lines = profile_output.split('\n')
        for line in lines:
            if 'cumulative' in line:
                continue
            if line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) > 3:
                    try:
                        cumulative_pct = float(parts[3].rstrip('%'))
                        # No single function should take more than 30% of time
                        assert cumulative_pct < 30, f"Function taking too much time: {line}"
                    except (ValueError, IndexError):
                        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, detector):
        """Test memory usage under sustained load."""
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        # Initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run many iterations
        for i in range(100):
            market_data = generate_large_market_data(20, 5)
            await detector.update_opportunities(market_data)
            
            if i % 20 == 0:
                gc.collect()
        
        # Final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate difference
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        print("\nTop 10 memory allocations:")
        for stat in top_stats[:10]:
            print(stat)
        
        # Check total memory growth
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
        
        # Memory should stay reasonable
        assert peak < 100 * 1024 * 1024  # Less than 100MB peak
    
    @pytest.mark.asyncio
    async def test_scalability(self, detector):
        """Test scalability with increasing load."""
        results = []
        
        for num_pairs in [10, 20, 50, 100, 200]:
            market_data = generate_large_market_data(num_pairs, 5)
            
            start = time.perf_counter()
            opportunities = await detector.find_direct_arbitrage(market_data)
            elapsed = (time.perf_counter() - start) * 1000
            
            results.append({
                'pairs': num_pairs,
                'time_ms': elapsed,
                'opportunities': len(opportunities),
                'time_per_pair': elapsed / num_pairs
            })
        
        print("\nScalability results:")
        print("Pairs | Time (ms) | Opportunities | Time/Pair (ms)")
        print("-" * 50)
        for r in results:
            print(f"{r['pairs']:5} | {r['time_ms']:9.2f} | {r['opportunities']:13} | {r['time_per_pair']:14.4f}")
        
        # Time should scale roughly linearly with number of pairs
        # Check that time per pair doesn't increase dramatically
        time_per_pair_values = [r['time_per_pair'] for r in results]
        assert max(time_per_pair_values) < min(time_per_pair_values) * 2  # Should not double