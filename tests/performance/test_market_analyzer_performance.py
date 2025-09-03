"""
Performance tests for MarketAnalyzer.

Tests latency requirements, throughput, and scalability.
"""

import asyncio
import time
import pytest
from decimal import Decimal
import statistics
import random

from genesis.analytics.market_analyzer import (
    MarketAnalyzer,
    MarketAnalyzerConfig
)


@pytest.fixture
def performance_config():
    """Performance test configuration"""
    return MarketAnalyzerConfig(
        min_profit_pct=Decimal('0.3'),
        min_confidence=0.7,
        max_latency_ms=10,
        orderbook_ttl_seconds=5,
        max_opportunities=10,
        spread_history_size=1000,
        max_cache_size_mb=100,
        enable_triangular=True,
        enable_statistical=True
    )


@pytest.fixture
def analyzer(performance_config):
    """Create analyzer for performance testing"""
    return MarketAnalyzer(performance_config)


def generate_orderbook_data(symbol='BTC/USDT', exchange='binance', levels=10):
    """Generate order book data for testing"""
    base_price = 50000 + random.uniform(-100, 100)
    
    bids = []
    asks = []
    
    for i in range(levels):
        bid_price = base_price - i - random.uniform(0.1, 1)
        ask_price = base_price + i + random.uniform(0.1, 1)
        
        bid_qty = random.uniform(0.1, 10)
        ask_qty = random.uniform(0.1, 10)
        
        bids.append([bid_price, bid_qty])
        asks.append([ask_price, ask_qty])
    
    return {
        'symbol': symbol,
        'exchange': exchange,
        'bids': bids,
        'asks': asks,
        'timestamp': time.time()
    }


class TestLatencyPerformance:
    """Test latency performance requirements"""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="latency")
    async def test_single_update_latency(self, analyzer, benchmark):
        """Benchmark single market update latency"""
        data = generate_orderbook_data()
        
        async def update():
            return await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
        
        result = await benchmark.pedantic(update, rounds=100, iterations=10)
        
        # Verify <10ms requirement
        assert benchmark.stats['mean'] < 0.01  # 10ms in seconds
    
    @pytest.mark.asyncio
    async def test_p95_latency(self, analyzer):
        """Test 95th percentile latency"""
        latencies = []
        
        for _ in range(1000):
            data = generate_orderbook_data()
            start = time.perf_counter()
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        assert p95 < 10  # P95 < 10ms
        assert p99 < 20  # P99 < 20ms
        
        print(f"P95 latency: {p95:.2f}ms, P99 latency: {p99:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_latency_with_full_cache(self, analyzer):
        """Test latency when cache is full"""
        # Fill cache with many pairs
        for i in range(100):
            data = generate_orderbook_data(f'PAIR{i}/USDT')
            await analyzer.analyze_market_data(f'PAIR{i}/USDT', 'binance', data)
        
        # Test latency with full cache
        latencies = []
        test_data = generate_orderbook_data('TEST/USDT')
        
        for _ in range(100):
            start = time.perf_counter()
            await analyzer.analyze_market_data('TEST/USDT', 'binance', test_data)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        assert avg_latency < 15  # Still fast with full cache


class TestThroughputPerformance:
    """Test throughput performance"""
    
    @pytest.mark.asyncio
    async def test_updates_per_second(self, analyzer):
        """Test maximum updates per second"""
        updates = 0
        start_time = time.perf_counter()
        
        # Process updates for 5 seconds
        while time.perf_counter() - start_time < 5:
            data = generate_orderbook_data()
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
            updates += 1
        
        updates_per_second = updates / 5
        
        # Should handle at least 100 updates/second
        assert updates_per_second > 100
        
        print(f"Throughput: {updates_per_second:.0f} updates/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_throughput(self, analyzer):
        """Test throughput with concurrent updates"""
        async def process_updates(pair_id):
            count = 0
            end_time = time.perf_counter() + 1  # Run for 1 second
            
            while time.perf_counter() < end_time:
                data = generate_orderbook_data(f'PAIR{pair_id}/USDT')
                await analyzer.analyze_market_data(f'PAIR{pair_id}/USDT', 'binance', data)
                count += 1
            
            return count
        
        # Run 10 concurrent tasks
        tasks = [process_updates(i) for i in range(10)]
        counts = await asyncio.gather(*tasks)
        
        total_updates = sum(counts)
        
        # Should handle many concurrent updates
        assert total_updates > 500  # 500+ updates in 1 second across 10 pairs
        
        print(f"Concurrent throughput: {total_updates} updates/second")


class TestScalabilityPerformance:
    """Test scalability with many pairs"""
    
    @pytest.mark.asyncio
    async def test_hundred_pairs_performance(self, analyzer):
        """Test performance with 100 trading pairs"""
        pairs = [f'TOKEN{i}/USDT' for i in range(100)]
        
        start_time = time.perf_counter()
        
        # Process one update for each pair
        tasks = []
        for pair in pairs:
            data = generate_orderbook_data(pair)
            task = analyzer.analyze_market_data(pair, 'binance', data)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        
        # Should complete 100 pairs quickly
        assert total_time < 2  # Less than 2 seconds for 100 pairs
        
        avg_time_per_pair = (total_time * 1000) / 100
        print(f"Average time per pair: {avg_time_per_pair:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, analyzer):
        """Test memory efficiency with many pairs"""
        import sys
        
        # Get initial memory
        initial_size = sys.getsizeof(analyzer._orderbook_cache)
        
        # Add 100 pairs
        for i in range(100):
            data = generate_orderbook_data(f'TOKEN{i}/USDT')
            await analyzer.analyze_market_data(f'TOKEN{i}/USDT', 'binance', data)
        
        # Check memory growth
        final_size = sys.getsizeof(analyzer._orderbook_cache)
        growth_mb = (final_size - initial_size) / (1024 * 1024)
        
        # Should be memory efficient
        assert growth_mb < 100  # Less than 100MB for 100 pairs
        
        metrics = analyzer.get_performance_metrics()
        print(f"Cache size: {metrics['cache_size_mb']:.2f}MB for {metrics['cache_size']} entries")


class TestOptimizationPerformance:
    """Test optimization and hot path performance"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, analyzer):
        """Test cache hit rate and performance"""
        # Warm up cache
        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        for pair in pairs:
            for exchange in ['binance', 'coinbase', 'kraken']:
                data = generate_orderbook_data(pair, exchange)
                await analyzer.analyze_market_data(pair, exchange, data)
        
        # Reset metrics
        analyzer._cache_hits = 0
        analyzer._cache_misses = 0
        
        # Test with warm cache
        for _ in range(100):
            pair = random.choice(pairs)
            exchange = random.choice(['binance', 'coinbase', 'kraken'])
            data = generate_orderbook_data(pair, exchange)
            await analyzer.analyze_market_data(pair, exchange, data)
        
        metrics = analyzer.get_performance_metrics()
        
        # Should have good cache hit rate
        assert metrics['cache_hit_rate'] > 0.5
        
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    
    @pytest.mark.asyncio
    async def test_opportunity_detection_speed(self, analyzer):
        """Test speed of opportunity detection"""
        # Create arbitrage scenario
        exchanges = ['binance', 'coinbase']
        
        # Setup orderbooks with arbitrage
        binance_data = {
            'bids': [[49900, 2.0]],
            'asks': [[50000, 2.0]],
            'timestamp': time.time()
        }
        
        coinbase_data = {
            'bids': [[50200, 2.0]],  # Arbitrage opportunity
            'asks': [[50300, 2.0]],
            'timestamp': time.time()
        }
        
        start_time = time.perf_counter()
        
        # Process both books
        await analyzer.analyze_market_data('BTC/USDT', 'binance', binance_data)
        opportunities = await analyzer.analyze_market_data('BTC/USDT', 'coinbase', coinbase_data)
        
        detection_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Should detect quickly
        assert detection_time_ms < 20
        assert len(opportunities) > 0
        
        print(f"Opportunity detection time: {detection_time_ms:.2f}ms")


class TestStressPerformance:
    """Stress test performance under extreme conditions"""
    
    @pytest.mark.asyncio
    async def test_burst_traffic(self, analyzer):
        """Test handling burst traffic"""
        burst_size = 1000
        
        start_time = time.perf_counter()
        
        # Send burst of updates
        tasks = []
        for i in range(burst_size):
            data = generate_orderbook_data(f'PAIR{i%20}/USDT')
            task = analyzer.analyze_market_data(f'PAIR{i%20}/USDT', 'binance', data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        burst_time = time.perf_counter() - start_time
        
        # Count successful updates
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        # Should handle most updates successfully
        assert successful > burst_size * 0.95
        
        print(f"Handled {successful}/{burst_size} updates in {burst_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, analyzer):
        """Test sustained high load"""
        duration_seconds = 10
        update_count = 0
        error_count = 0
        
        end_time = time.perf_counter() + duration_seconds
        
        while time.perf_counter() < end_time:
            try:
                data = generate_orderbook_data()
                await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
                update_count += 1
            except Exception:
                error_count += 1
            
            # No delay - maximum stress
        
        updates_per_second = update_count / duration_seconds
        error_rate = error_count / (update_count + error_count) if update_count + error_count > 0 else 0
        
        # Should maintain performance under load
        assert updates_per_second > 50
        assert error_rate < 0.01  # Less than 1% errors
        
        print(f"Sustained load: {updates_per_second:.0f} updates/s, {error_rate:.2%} error rate")


if __name__ == '__main__':
    # Run with: pytest test_market_analyzer_performance.py -v --benchmark-only
    pytest.main([__file__, '-v'])