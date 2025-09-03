"""
Integration tests for MarketAnalyzer with WebSocket and live market simulation.

Tests WebSocket data integration, live market data simulation, latency requirements,
and multi-pair concurrent processing.
"""

import asyncio
import json
import time
import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import random

from genesis.analytics.market_analyzer import (
    MarketAnalyzer,
    MarketAnalyzerConfig,
    OrderBook,
    OrderBookLevel,
    ArbitrageOpportunity,
    OpportunityType
)


@pytest.fixture
def integration_config():
    """Create integration test configuration"""
    return MarketAnalyzerConfig(
        min_profit_pct=Decimal('0.2'),  # Lower for testing
        min_confidence=0.6,
        max_latency_ms=10,  # Strict latency requirement
        orderbook_ttl_seconds=5,
        max_opportunities=20,
        spread_history_size=500,
        max_cache_size_mb=10,
        enable_triangular=True,
        enable_statistical=True,
        z_score_threshold=2.0,
        min_liquidity_usd=Decimal('50')
    )


@pytest.fixture
def analyzer(integration_config):
    """Create MarketAnalyzer for integration testing"""
    return MarketAnalyzer(integration_config)


class MarketDataSimulator:
    """Simulates realistic market data with volatility"""
    
    def __init__(self, base_price=50000, volatility=0.001):
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
        self.exchanges = ['binance', 'coinbase', 'kraken', 'huobi', 'okx']
        
    def generate_orderbook(self, symbol, exchange):
        """Generate realistic order book data"""
        # Add some price variation between exchanges
        exchange_offset = hash(exchange) % 100 - 50
        mid_price = self.current_price + exchange_offset
        
        # Random walk for price movement
        self.current_price += random.gauss(0, self.base_price * self.volatility)
        
        spread = random.uniform(0.5, 5)  # Random spread
        
        bids = []
        asks = []
        
        # Generate 10 levels of depth
        for i in range(10):
            bid_price = mid_price - spread/2 - i * 0.1
            ask_price = mid_price + spread/2 + i * 0.1
            
            bid_qty = random.uniform(0.1, 5.0)
            ask_qty = random.uniform(0.1, 5.0)
            
            bids.append([bid_price, bid_qty])
            asks.append([ask_price, ask_qty])
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }
    
    def generate_arbitrage_opportunity(self, symbol):
        """Generate order books with guaranteed arbitrage"""
        # Create price discrepancy between exchanges
        low_exchange = random.choice(self.exchanges)
        high_exchange = random.choice([e for e in self.exchanges if e != low_exchange])
        
        low_price = self.base_price
        high_price = self.base_price * 1.005  # 0.5% arbitrage
        
        low_book = {
            'symbol': symbol,
            'exchange': low_exchange,
            'bids': [[low_price - 1, 2.0], [low_price - 2, 3.0]],
            'asks': [[low_price, 2.0], [low_price + 1, 3.0]],
            'timestamp': time.time()
        }
        
        high_book = {
            'symbol': symbol,
            'exchange': high_exchange,
            'bids': [[high_price, 2.0], [high_price - 1, 3.0]],
            'asks': [[high_price + 1, 2.0], [high_price + 2, 3.0]],
            'timestamp': time.time()
        }
        
        return low_book, high_book


@pytest.fixture
def market_simulator():
    """Create market data simulator"""
    return MarketDataSimulator()


class MockWebSocketManager:
    """Mock WebSocket manager for testing"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.subscribers = []
        self.running = False
        
    async def connect(self):
        """Simulate connection"""
        self.running = True
        
    async def disconnect(self):
        """Simulate disconnection"""
        self.running = False
        
    async def subscribe(self, symbol, callback):
        """Subscribe to market data"""
        self.subscribers.append((symbol, callback))
        
    async def start_streaming(self):
        """Start streaming market data"""
        while self.running:
            for symbol, callback in self.subscribers:
                for exchange in self.simulator.exchanges:
                    data = self.simulator.generate_orderbook(symbol, exchange)
                    await callback(data)
            await asyncio.sleep(0.1)  # 100ms updates


class TestWebSocketIntegration:
    """Test WebSocket market data integration"""
    
    @pytest.mark.asyncio
    async def test_websocket_data_processing(self, analyzer, market_simulator):
        """Test processing WebSocket market data stream"""
        ws_manager = MockWebSocketManager(market_simulator)
        
        opportunities_found = []
        
        async def process_data(data):
            opps = await analyzer.analyze_market_data(
                data['symbol'],
                data['exchange'],
                data
            )
            opportunities_found.extend(opps)
        
        # Subscribe to market data
        await ws_manager.subscribe('BTC/USDT', process_data)
        
        # Start streaming
        ws_manager.running = True
        stream_task = asyncio.create_task(ws_manager.start_streaming())
        
        # Let it run for a short time
        await asyncio.sleep(0.5)
        
        # Stop streaming
        ws_manager.running = False
        await stream_task
        
        # Should have processed multiple updates
        assert len(analyzer._orderbook_cache) > 0
        
        # Check for any opportunities found
        if opportunities_found:
            for opp in opportunities_found:
                assert opp.symbol == 'BTC/USDT'
                assert opp.profit_pct > Decimal('0')
    
    @pytest.mark.asyncio
    async def test_multiple_websocket_connections(self, analyzer, market_simulator):
        """Test handling multiple WebSocket connections"""
        managers = [MockWebSocketManager(market_simulator) for _ in range(3)]
        
        all_tasks = []
        
        async def process_data(data):
            await analyzer.analyze_market_data(
                data['symbol'],
                data['exchange'],
                data
            )
        
        for i, manager in enumerate(managers):
            await manager.subscribe(f'ETH/USDT', process_data)
            manager.running = True
            task = asyncio.create_task(manager.start_streaming())
            all_tasks.append(task)
        
        # Run for a short time
        await asyncio.sleep(0.3)
        
        # Stop all
        for manager in managers:
            manager.running = False
        
        await asyncio.gather(*all_tasks)
        
        # Should handle multiple streams
        assert len(analyzer._orderbook_cache) > 0


class TestLiveMarketSimulation:
    """Test with simulated live market conditions"""
    
    @pytest.mark.asyncio
    async def test_continuous_market_updates(self, analyzer, market_simulator):
        """Test continuous market data updates"""
        update_count = 0
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        async def send_updates():
            nonlocal update_count
            for _ in range(50):  # 50 updates
                for symbol in symbols:
                    for exchange in market_simulator.exchanges[:3]:
                        data = market_simulator.generate_orderbook(symbol, exchange)
                        await analyzer.analyze_market_data(
                            symbol,
                            exchange,
                            data
                        )
                        update_count += 1
                await asyncio.sleep(0.01)  # 10ms between updates
        
        start_time = time.time()
        await send_updates()
        total_time = time.time() - start_time
        
        # Verify processing
        assert update_count > 0
        assert len(analyzer._orderbook_cache) > 0
        
        # Check performance
        metrics = analyzer.get_performance_metrics()
        assert metrics['avg_latency_ms'] < 50  # Should be fast
    
    @pytest.mark.asyncio
    async def test_arbitrage_detection_live(self, analyzer, market_simulator):
        """Test arbitrage detection with live-like data"""
        opportunities_detected = []
        
        # Generate some guaranteed arbitrage opportunities
        for _ in range(5):
            low_book, high_book = market_simulator.generate_arbitrage_opportunity('BTC/USDT')
            
            # Process both books
            opps1 = await analyzer.analyze_market_data(
                'BTC/USDT',
                low_book['exchange'],
                low_book
            )
            
            opps2 = await analyzer.analyze_market_data(
                'BTC/USDT',
                high_book['exchange'],
                high_book
            )
            
            opportunities_detected.extend(opps1)
            opportunities_detected.extend(opps2)
        
        # Should detect arbitrage opportunities
        assert len(opportunities_detected) > 0
        
        # Verify opportunity quality
        for opp in opportunities_detected:
            assert opp.opportunity_type == OpportunityType.DIRECT_ARBITRAGE
            assert opp.profit_pct >= analyzer.config.min_profit_pct
            assert opp.confidence >= analyzer.config.min_confidence
    
    @pytest.mark.asyncio
    async def test_market_volatility_handling(self, analyzer):
        """Test handling volatile market conditions"""
        # Create volatile simulator
        volatile_sim = MarketDataSimulator(base_price=50000, volatility=0.01)
        
        error_count = 0
        success_count = 0
        
        for _ in range(100):  # Many rapid updates
            try:
                data = volatile_sim.generate_orderbook('BTC/USDT', 'binance')
                await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
                success_count += 1
            except Exception:
                error_count += 1
        
        # Should handle volatility gracefully
        assert success_count > 95  # Allow a few errors
        assert error_count < 5


class TestLatencyRequirements:
    """Test latency requirements are met"""
    
    @pytest.mark.asyncio
    async def test_single_pair_latency(self, analyzer, market_simulator):
        """Test latency for single pair analysis"""
        data = market_simulator.generate_orderbook('BTC/USDT', 'binance')
        
        latencies = []
        
        for _ in range(100):
            start = time.time()
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        # Verify <10ms requirement
        assert avg_latency < 10
        assert p95_latency < 20  # P95 should be reasonable
    
    @pytest.mark.asyncio
    async def test_multi_pair_latency(self, analyzer, market_simulator):
        """Test latency with multiple pairs"""
        pairs = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 
            'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT'
        ]
        
        async def analyze_pair(pair):
            data = market_simulator.generate_orderbook(pair, 'binance')
            start = time.time()
            await analyzer.analyze_market_data(pair, 'binance', data)
            return (time.time() - start) * 1000
        
        # Analyze all pairs concurrently
        tasks = [analyze_pair(pair) for pair in pairs]
        latencies = await asyncio.gather(*tasks)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Should maintain low latency even with multiple pairs
        assert avg_latency < 15
    
    @pytest.mark.asyncio
    async def test_latency_under_load(self, analyzer, market_simulator):
        """Test latency under heavy load"""
        # Pre-fill cache with many order books
        for i in range(50):
            data = market_simulator.generate_orderbook(f'PAIR{i}/USDT', 'binance')
            await analyzer.analyze_market_data(f'PAIR{i}/USDT', 'binance', data)
        
        # Now test latency with full cache
        test_data = market_simulator.generate_orderbook('TEST/USDT', 'binance')
        
        latencies = []
        for _ in range(50):
            start = time.time()
            await analyzer.analyze_market_data('TEST/USDT', 'binance', test_data)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Should maintain performance under load
        assert avg_latency < 20


class TestMultiPairProcessing:
    """Test processing 100+ pairs simultaneously"""
    
    @pytest.mark.asyncio
    async def test_hundred_pairs_concurrent(self, analyzer, market_simulator):
        """Test with 100+ trading pairs"""
        # Generate 100 unique pairs
        pairs = [f'TOKEN{i}/USDT' for i in range(100)]
        
        async def process_pair(pair):
            data = market_simulator.generate_orderbook(pair, 'binance')
            return await analyzer.analyze_market_data(pair, 'binance', data)
        
        # Process all pairs concurrently
        start_time = time.time()
        tasks = [process_pair(pair) for pair in pairs]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert total_time < 5  # 5 seconds for 100 pairs
        
        # Verify all processed
        assert len(results) == 100
        
        # Check cache size
        assert len(analyzer._orderbook_cache) <= 100
    
    @pytest.mark.asyncio
    async def test_memory_usage_hundred_pairs(self, analyzer, market_simulator):
        """Test memory usage with 100 pairs"""
        initial_cache_size = analyzer._cache_size_bytes
        
        # Add 100 pairs with full order books
        for i in range(100):
            data = market_simulator.generate_orderbook(f'TOKEN{i}/USDT', 'binance')
            await analyzer.analyze_market_data(f'TOKEN{i}/USDT', 'binance', data)
        
        # Check memory usage
        metrics = analyzer.get_performance_metrics()
        cache_size_mb = metrics['cache_size_mb']
        
        # Should stay within limits (configured as 10MB for test)
        assert cache_size_mb < 10
        
        # Verify cache management is working
        assert len(analyzer._orderbook_cache) > 0
    
    @pytest.mark.asyncio
    async def test_cross_pair_correlation(self, analyzer, market_simulator):
        """Test correlation analysis across multiple pairs"""
        correlated_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BTC/ETH',
            'SOL/USDT', 'SOL/BTC', 'SOL/ETH'
        ]
        
        # Build spread history for correlation
        for _ in range(50):
            for pair in correlated_pairs:
                data = market_simulator.generate_orderbook(pair, 'binance')
                await analyzer.analyze_market_data(pair, 'binance', data)
            await asyncio.sleep(0.01)
        
        # Check spread histories built
        for pair in correlated_pairs:
            assert pair in analyzer._spread_history
            assert len(analyzer._spread_history[pair]) > 0


class TestRealTimeScenarios:
    """Test real-world trading scenarios"""
    
    @pytest.mark.asyncio
    async def test_flash_crash_detection(self, analyzer):
        """Test detection during flash crash"""
        normal_price = 50000
        crash_price = 45000  # 10% crash
        
        # Normal market
        normal_book = {
            'bids': [[normal_price - 1, 2.0]],
            'asks': [[normal_price, 2.0]],
            'timestamp': time.time()
        }
        
        # Build baseline
        for _ in range(50):
            await analyzer.analyze_market_data('BTC/USDT', 'binance', normal_book)
        
        # Flash crash
        crash_book = {
            'bids': [[crash_price - 1, 10.0]],  # High volume
            'asks': [[crash_price, 10.0]],
            'timestamp': time.time()
        }
        
        opportunities = await analyzer.analyze_market_data('BTC/USDT', 'binance', crash_book)
        
        # Should detect statistical anomaly
        if analyzer.config.enable_statistical and opportunities:
            stat_opps = [o for o in opportunities if o.opportunity_type == OpportunityType.STATISTICAL_ARBITRAGE]
            if stat_opps:
                assert stat_opps[0].statistical_significance > 2.0
    
    @pytest.mark.asyncio
    async def test_market_maker_detection(self, analyzer):
        """Test detection of market maker patterns"""
        # Simulate market maker with tight spreads
        mm_books = []
        for i in range(20):
            price = 50000 + i * 0.1
            mm_book = {
                'bids': [[price - 0.5, 100.0]],  # Large size, tight spread
                'asks': [[price + 0.5, 100.0]],
                'timestamp': time.time() + i
            }
            mm_books.append(mm_book)
        
        # Process market maker updates
        for book in mm_books:
            await analyzer.analyze_market_data('BTC/USDT', 'mm_exchange', book)
        
        # Check for consistent tight spreads
        if 'BTC/USDT' in analyzer._spread_history:
            spreads = [s['spread'] for s in analyzer._spread_history['BTC/USDT']]
            if spreads:
                avg_spread = sum(spreads) / len(spreads)
                assert avg_spread < 2  # Tight spread characteristic
    
    @pytest.mark.asyncio
    async def test_recovery_from_connection_loss(self, analyzer, market_simulator):
        """Test recovery after connection loss"""
        # Simulate normal operation
        for _ in range(10):
            data = market_simulator.generate_orderbook('BTC/USDT', 'binance')
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
        
        initial_cache_size = len(analyzer._orderbook_cache)
        
        # Simulate connection loss (no updates for TTL period)
        await asyncio.sleep(0.1)
        
        # Resume with new data
        for _ in range(10):
            data = market_simulator.generate_orderbook('BTC/USDT', 'binance')
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
        
        # Should continue working normally
        assert len(analyzer._orderbook_cache) > 0
        
        # Check metrics still working
        metrics = analyzer.get_performance_metrics()
        assert metrics['avg_latency_ms'] >= 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, analyzer, market_simulator):
        """Benchmark maximum throughput"""
        updates_processed = 0
        start_time = time.time()
        
        # Process as many updates as possible in 1 second
        while time.time() - start_time < 1.0:
            data = market_simulator.generate_orderbook('BTC/USDT', 'binance')
            await analyzer.analyze_market_data('BTC/USDT', 'binance', data)
            updates_processed += 1
        
        # Should process at least 100 updates per second
        assert updates_processed > 100
        
        print(f"Throughput: {updates_processed} updates/second")
    
    @pytest.mark.asyncio
    async def test_opportunity_detection_rate(self, analyzer, market_simulator):
        """Benchmark opportunity detection rate"""
        opportunities_found = 0
        
        # Generate many arbitrage opportunities
        for _ in range(20):
            low_book, high_book = market_simulator.generate_arbitrage_opportunity('BTC/USDT')
            
            opps1 = await analyzer.analyze_market_data(
                'BTC/USDT',
                low_book['exchange'],
                low_book
            )
            
            opps2 = await analyzer.analyze_market_data(
                'BTC/USDT',
                high_book['exchange'],
                high_book
            )
            
            opportunities_found += len(opps1) + len(opps2)
        
        # Should find most opportunities
        assert opportunities_found > 10
        
        # Check detection metrics
        assert analyzer._opportunities_detected == opportunities_found


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])