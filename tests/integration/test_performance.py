"""Performance benchmark tests for critical paths."""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self, name: str, target_ms: float):
        """Initialize benchmark."""
        self.name = name
        self.target_ms = target_ms
        self.results: list[float] = []

    def measure(self, func: Callable, *args, **kwargs) -> Any:
        """Measure function execution time."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        self.results.append(elapsed)
        return result

    async def measure_async(self, func: Callable, *args, **kwargs) -> Any:
        """Measure async function execution time."""
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        self.results.append(elapsed)
        return result

    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self.results:
            return {}

        return {
            "name": self.name,
            "runs": len(self.results),
            "min_ms": min(self.results),
            "max_ms": max(self.results),
            "mean_ms": statistics.mean(self.results),
            "median_ms": statistics.median(self.results),
            "stdev_ms": statistics.stdev(self.results) if len(self.results) > 1 else 0,
            "p95_ms": statistics.quantiles(self.results, n=20)[18] if len(self.results) > 1 else self.results[0],
            "target_ms": self.target_ms,
            "passed": statistics.mean(self.results) <= self.target_ms
        }


class TestCriticalPathPerformance:
    """Test performance of critical trading paths."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.benchmarks = []

    def teardown_method(self):
        """Report benchmark results."""
        if self.benchmarks:
            print("\n\n=== Performance Benchmark Results ===")
            for benchmark in self.benchmarks:
                stats = benchmark.get_stats()
                if stats:
                    status = "✅ PASS" if stats["passed"] else "❌ FAIL"
                    print(f"\n{stats['name']}: {status}")
                    print(f"  Target: {stats['target_ms']:.2f}ms")
                    print(f"  Mean: {stats['mean_ms']:.2f}ms")
                    print(f"  P95: {stats['p95_ms']:.2f}ms")
                    print(f"  Min/Max: {stats['min_ms']:.2f}ms / {stats['max_ms']:.2f}ms")

    def test_configuration_loading_performance(self):
        """Test configuration loading speed."""
        from config.settings import Settings

        benchmark = PerformanceBenchmark("Configuration Loading", target_ms=100)
        self.benchmarks.append(benchmark)

        for _ in range(10):
            benchmark.measure(Settings)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Configuration loading too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_decimal_calculation_performance(self):
        """Test Decimal arithmetic performance for money calculations."""
        benchmark = PerformanceBenchmark("Decimal Calculations", target_ms=1)
        self.benchmarks.append(benchmark)

        def calculate_position_size():
            balance = Decimal("10000.00")
            risk_percent = Decimal("0.02")
            entry_price = Decimal("45678.90")
            stop_loss = Decimal("45000.00")

            risk_amount = balance * risk_percent
            price_diff = entry_price - stop_loss
            position_size = risk_amount / price_diff
            return position_size

        for _ in range(100):
            benchmark.measure(calculate_position_size)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Decimal calculations too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    @pytest.mark.asyncio
    async def test_async_order_placement_performance(self):
        """Test async order placement simulation."""
        benchmark = PerformanceBenchmark("Async Order Placement", target_ms=50)
        self.benchmarks.append(benchmark)

        async def simulate_order_placement():
            # Simulate API call delay
            await asyncio.sleep(0.01)
            return {"order_id": "12345", "status": "filled"}

        for _ in range(20):
            await benchmark.measure_async(simulate_order_placement)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Order placement too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_risk_calculation_performance(self):
        """Test risk management calculation performance."""
        benchmark = PerformanceBenchmark("Risk Calculations", target_ms=5)
        self.benchmarks.append(benchmark)

        def calculate_risk_metrics():
            positions = [
                {"symbol": "BTC/USDT", "size": Decimal("0.5"), "entry": Decimal("45000")},
                {"symbol": "ETH/USDT", "size": Decimal("10"), "entry": Decimal("3000")},
            ]

            total_exposure = sum(p["size"] * p["entry"] for p in positions)
            max_position = max(positions, key=lambda p: p["size"] * p["entry"])
            risk_score = float(total_exposure) / 100000 * 100

            return {
                "total_exposure": total_exposure,
                "max_position": max_position,
                "risk_score": risk_score
            }

        for _ in range(100):
            benchmark.measure(calculate_risk_metrics)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Risk calculations too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_json_serialization_performance(self):
        """Test JSON serialization for API responses."""
        benchmark = PerformanceBenchmark("JSON Serialization", target_ms=2)
        self.benchmarks.append(benchmark)

        test_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": [
                {"symbol": "BTC/USDT", "size": 0.5, "pnl": 1234.56},
                {"symbol": "ETH/USDT", "size": 10, "pnl": -456.78},
            ],
            "balances": {"USDT": 10000, "BTC": 0.5, "ETH": 10},
            "metrics": {
                "total_trades": 150,
                "win_rate": 0.65,
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.15
            }
        }

        for _ in range(100):
            benchmark.measure(json.dumps, test_data)

        stats = benchmark.get_stats()
        assert stats["passed"], f"JSON serialization too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_logging_performance(self):
        """Test structured logging performance."""
        import structlog

        benchmark = PerformanceBenchmark("Structured Logging", target_ms=1)
        self.benchmarks.append(benchmark)

        logger = structlog.get_logger()

        def log_trade_event():
            logger.info(
                "trade_executed",
                symbol="BTC/USDT",
                side="buy",
                size=0.5,
                price=45678.90,
                timestamp=datetime.utcnow().isoformat()
            )

        for _ in range(100):
            benchmark.measure(log_trade_event)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Logging too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self):
        """Test performance of concurrent operations."""
        benchmark = PerformanceBenchmark("Concurrent Operations", target_ms=100)
        self.benchmarks.append(benchmark)

        async def fetch_data(exchange: str):
            await asyncio.sleep(0.01)  # Simulate API call
            return {"exchange": exchange, "price": 45678.90}

        async def run_concurrent():
            tasks = [fetch_data(ex) for ex in ["binance", "coinbase", "kraken"]]
            results = await asyncio.gather(*tasks)
            return results

        for _ in range(10):
            await benchmark.measure_async(run_concurrent)

        stats = benchmark.get_stats()
        assert stats["passed"], f"Concurrent ops too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_dataframe_operations_performance(self):
        """Test pandas DataFrame operations performance."""
        import numpy as np
        import pandas as pd

        benchmark = PerformanceBenchmark("DataFrame Operations", target_ms=10)
        self.benchmarks.append(benchmark)

        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'open': np.random.uniform(45000, 46000, 1000),
            'high': np.random.uniform(45500, 46500, 1000),
            'low': np.random.uniform(44500, 45500, 1000),
            'close': np.random.uniform(45000, 46000, 1000),
            'volume': np.random.uniform(100, 1000, 1000)
        })

        def calculate_indicators():
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            return df

        for _ in range(10):
            benchmark.measure(calculate_indicators)

        stats = benchmark.get_stats()
        assert stats["passed"], f"DataFrame ops too slow: {stats['mean_ms']:.2f}ms > {stats['target_ms']}ms"

    def test_performance_summary(self):
        """Generate and validate overall performance summary."""
        # This test runs last to ensure all benchmarks are complete
        if not self.benchmarks:
            pytest.skip("No benchmarks to summarize")

        failed_benchmarks = []
        for benchmark in self.benchmarks:
            stats = benchmark.get_stats()
            if stats and not stats["passed"]:
                failed_benchmarks.append(stats["name"])

        assert not failed_benchmarks, f"Performance benchmarks failed: {failed_benchmarks}"
