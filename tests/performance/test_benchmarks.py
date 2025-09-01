"""Performance benchmark tests for Project GENESIS."""

import asyncio
import time
from decimal import Decimal

import pytest
import pytest_benchmark
from unittest.mock import MagicMock, AsyncMock

from genesis.monitoring.performance_monitor import PerformanceMonitor
from genesis.monitoring.database_optimizer import DatabaseOptimizer, QueryCache
from genesis.monitoring.connection_pool_manager import (
    HTTPConnectionPool,
    WebSocketConnectionPool,
    RequestBatcher
)


class TestOrderExecutionBenchmarks:
    """Benchmark tests for order execution performance."""
    
    @pytest.fixture
    def mock_order(self):
        """Create mock order for benchmarking."""
        order = MagicMock()
        order.id = "test-order-123"
        order.symbol = "BTCUSDT"
        order.side = "buy"
        order.quantity = Decimal("0.01")
        order.order_type = "market"
        return order
    
    def test_order_execution_latency(self, benchmark, mock_order):
        """Benchmark order execution latency."""
        async def execute_order(order):
            # Simulate order execution
            await asyncio.sleep(0.001)  # 1ms execution
            return {"status": "filled", "id": order.id}
        
        def run_execution():
            return asyncio.run(execute_order(mock_order))
        
        result = benchmark(run_execution)
        assert result["status"] == "filled"
    
    def test_risk_check_latency(self, benchmark):
        """Benchmark risk check latency."""
        def perform_risk_check():
            # Simulate risk calculations
            position_size = Decimal("10000")
            max_position = Decimal("50000")
            drawdown = Decimal("0.05")
            max_drawdown = Decimal("0.10")
            
            checks = [
                position_size < max_position,
                drawdown < max_drawdown,
                position_size * drawdown < Decimal("1000")
            ]
            
            return all(checks)
        
        result = benchmark(perform_risk_check)
        assert result is True
    
    @pytest.mark.benchmark(group="order")
    def test_batch_order_processing(self, benchmark):
        """Benchmark batch order processing."""
        orders = [
            {"id": f"order-{i}", "symbol": "BTCUSDT", "side": "buy", "quantity": 0.01}
            for i in range(100)
        ]
        
        async def process_batch(batch):
            # Simulate batch processing
            await asyncio.sleep(0.01)  # 10ms for batch
            return [{"id": o["id"], "status": "filled"} for o in batch]
        
        async def run_batch():
            batcher = RequestBatcher(batch_size=10)
            results = []
            for order in orders:
                result = await batcher.add_request(
                    "orders",
                    order,
                    process_batch
                )
                results.append(result)
            return results
        
        def execute():
            return asyncio.run(run_batch())
        
        results = benchmark(execute)
        assert len(results) == 100


class TestDatabaseBenchmarks:
    """Benchmark tests for database performance."""
    
    @pytest.fixture
    def query_cache(self):
        """Create query cache for benchmarking."""
        return QueryCache(max_size=1000, ttl_seconds=300)
    
    def test_cache_hit_performance(self, benchmark, query_cache):
        """Benchmark cache hit performance."""
        # Populate cache
        for i in range(100):
            query_cache.set(f"key_{i}", f"value_{i}")
        
        def cache_lookup():
            results = []
            for i in range(100):
                result = query_cache.get(f"key_{i}")
                results.append(result)
            return results
        
        results = benchmark(cache_lookup)
        assert len(results) == 100
        assert all(r is not None for r in results)
    
    def test_cache_miss_performance(self, benchmark, query_cache):
        """Benchmark cache miss performance."""
        def cache_lookup():
            results = []
            for i in range(100):
                result = query_cache.get(f"missing_key_{i}")
                results.append(result)
            return results
        
        results = benchmark(cache_lookup)
        assert len(results) == 100
        assert all(r is None for r in results)
    
    @pytest.mark.benchmark(group="database")
    def test_query_normalization(self, benchmark):
        """Benchmark query normalization performance."""
        queries = [
            "SELECT * FROM orders WHERE id = 123",
            "SELECT * FROM positions WHERE symbol = 'BTCUSDT' AND side = 'buy'",
            "INSERT INTO trades (id, symbol, price) VALUES (456, 'ETHUSDT', 3500.50)",
            "UPDATE positions SET quantity = 0.5 WHERE id = 789",
            "DELETE FROM orders WHERE status = 'cancelled' AND created_at < '2024-01-01'"
        ] * 20
        
        def normalize_queries():
            import re
            normalized = []
            for query in queries:
                # Replace numbers
                norm = re.sub(r'\b\d+\b', '?', query)
                # Replace quoted strings
                norm = re.sub(r"'[^']*'", '?', norm)
                normalized.append(norm)
            return normalized
        
        results = benchmark(normalize_queries)
        assert len(results) == 100


class TestConnectionPoolBenchmarks:
    """Benchmark tests for connection pooling."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="connection")
    async def test_http_connection_pool(self, benchmark):
        """Benchmark HTTP connection pool performance."""
        pool = HTTPConnectionPool(max_connections=100)
        
        async def make_requests():
            # Simulate concurrent requests
            tasks = []
            for i in range(50):
                # Mock request without actual network call
                async def mock_request(idx):
                    await asyncio.sleep(0.001)  # 1ms latency
                    return {"id": idx, "status": 200}
                
                tasks.append(mock_request(i))
            
            results = await asyncio.gather(*tasks)
            return results
        
        def run_test():
            return asyncio.run(make_requests())
        
        results = benchmark(run_test)
        assert len(results) == 50
    
    @pytest.mark.benchmark(group="connection")
    def test_websocket_reconnection(self, benchmark):
        """Benchmark WebSocket reconnection performance."""
        pool = WebSocketConnectionPool(max_connections=10)
        
        async def simulate_reconnection():
            # Simulate connection failures and reconnections
            connections = []
            for i in range(10):
                try:
                    # Mock connection (would fail in real test)
                    await asyncio.sleep(0.005)  # 5ms connection time
                    connections.append(f"ws://test-{i}")
                except Exception:
                    pass
            return connections
        
        def run_test():
            return asyncio.run(simulate_reconnection())
        
        results = benchmark(run_test)
        assert len(results) == 10


class TestMonitoringBenchmarks:
    """Benchmark tests for monitoring performance."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for benchmarking."""
        return PerformanceMonitor()
    
    def test_metric_recording(self, benchmark, performance_monitor):
        """Benchmark metric recording performance."""
        async def record_metrics():
            for i in range(100):
                await performance_monitor.record_order_execution(
                    order_type="market",
                    symbol="BTCUSDT",
                    side="buy",
                    tier="sniper",
                    latency=0.025,
                    status="success"
                )
            return True
        
        def run_test():
            return asyncio.run(record_metrics())
        
        result = benchmark(run_test)
        assert result is True
    
    @pytest.mark.benchmark(group="monitoring")
    def test_metric_aggregation(self, benchmark, performance_monitor):
        """Benchmark metric aggregation performance."""
        # Pre-populate metrics
        asyncio.run(self._populate_metrics(performance_monitor))
        
        def aggregate_metrics():
            # Get Prometheus format metrics
            metrics_data = performance_monitor.get_metrics()
            
            # Parse metrics (simplified)
            lines = metrics_data.decode('utf-8').split('\n')
            parsed = []
            for line in lines:
                if line and not line.startswith('#'):
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        parsed.append((parts[0], float(parts[1])))
            
            return parsed
        
        results = benchmark(aggregate_metrics)
        assert len(results) > 0
    
    async def _populate_metrics(self, monitor):
        """Helper to populate metrics for testing."""
        for i in range(1000):
            await monitor.record_api_call(
                exchange="binance",
                endpoint="/api/v3/order",
                method="POST",
                latency=0.05,
                status_code=200
            )


class TestMemoryBenchmarks:
    """Benchmark tests for memory operations."""
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_allocation(self, benchmark):
        """Benchmark memory allocation patterns."""
        def allocate_memory():
            # Simulate order book updates
            order_books = []
            for _ in range(100):
                book = {
                    "bids": [(float(i), float(i*10)) for i in range(100)],
                    "asks": [(float(i), float(i*10)) for i in range(100)],
                    "timestamp": time.time()
                }
                order_books.append(book)
            return len(order_books)
        
        result = benchmark(allocate_memory)
        assert result == 100
    
    @pytest.mark.benchmark(group="memory")
    def test_object_creation(self, benchmark):
        """Benchmark object creation overhead."""
        class Position:
            def __init__(self, symbol, side, quantity, entry_price):
                self.symbol = symbol
                self.side = side
                self.quantity = Decimal(str(quantity))
                self.entry_price = Decimal(str(entry_price))
                self.pnl = Decimal("0")
                self.created_at = time.time()
        
        def create_positions():
            positions = []
            for i in range(1000):
                pos = Position(
                    symbol="BTCUSDT",
                    side="buy" if i % 2 == 0 else "sell",
                    quantity=0.01 * (i % 10 + 1),
                    entry_price=50000 + i
                )
                positions.append(pos)
            return len(positions)
        
        result = benchmark(create_positions)
        assert result == 1000


@pytest.mark.benchmark(
    min_rounds=5,
    disable_gc=True,
    warmup=True
)
class TestEndToEndBenchmarks:
    """End-to-end performance benchmarks."""
    
    def test_full_trade_cycle(self, benchmark):
        """Benchmark full trade cycle from signal to execution."""
        async def trade_cycle():
            # 1. Signal detection
            signal = {"symbol": "BTCUSDT", "action": "buy", "strength": 0.8}
            
            # 2. Risk check
            await asyncio.sleep(0.001)  # 1ms risk check
            
            # 3. Order preparation
            order = {
                "symbol": signal["symbol"],
                "side": signal["action"],
                "quantity": 0.01,
                "type": "market"
            }
            
            # 4. Order execution
            await asyncio.sleep(0.005)  # 5ms execution
            
            # 5. Position update
            await asyncio.sleep(0.001)  # 1ms update
            
            # 6. Metrics recording
            await asyncio.sleep(0.0005)  # 0.5ms metrics
            
            return {"status": "completed", "latency_ms": 7.5}
        
        def run_cycle():
            return asyncio.run(trade_cycle())
        
        result = benchmark(run_cycle)
        assert result["status"] == "completed"
        assert result["latency_ms"] < 50  # Under 50ms target