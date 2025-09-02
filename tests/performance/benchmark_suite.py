"""
Performance benchmark suite for Genesis trading system.
Establishes baselines and detects performance regressions.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
import statistics

import pytest
import pytest_benchmark
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_second: float
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
            
    def meets_target(self, target_ms: float) -> bool:
        """Check if benchmark meets performance target."""
        return self.p99_ms <= target_ms
        

class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str, target_ms: float = 50):
        self.name = name
        self.target_ms = target_ms
        self.results: List[float] = []
        
    async def setup(self):
        """Setup before benchmark (override if needed)."""
        pass
        
    async def teardown(self):
        """Cleanup after benchmark (override if needed)."""
        pass
        
    async def run_iteration(self):
        """Run single benchmark iteration (override in subclasses)."""
        raise NotImplementedError
        
    async def benchmark(self, iterations: int = 100) -> BenchmarkResult:
        """Run benchmark and collect results."""
        await self.setup()
        
        try:
            # Warmup
            for _ in range(min(10, iterations // 10)):
                await self.run_iteration()
                
            # Actual benchmark
            self.results.clear()
            
            for _ in range(iterations):
                start = time.perf_counter()
                await self.run_iteration()
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                self.results.append(elapsed)
                
            # Calculate statistics
            result = self._calculate_stats(iterations)
            
            # Log result
            logger.info(f"Benchmark '{self.name}': "
                       f"mean={result.mean_ms:.2f}ms, "
                       f"p99={result.p99_ms:.2f}ms, "
                       f"ops/s={result.ops_per_second:.2f}")
                       
            return result
            
        finally:
            await self.teardown()
            
    def _calculate_stats(self, iterations: int) -> BenchmarkResult:
        """Calculate statistics from results."""
        sorted_results = sorted(self.results)
        
        return BenchmarkResult(
            name=self.name,
            iterations=iterations,
            mean_ms=statistics.mean(self.results),
            median_ms=statistics.median(self.results),
            stddev_ms=statistics.stdev(self.results) if len(self.results) > 1 else 0,
            min_ms=sorted_results[0],
            max_ms=sorted_results[-1],
            p95_ms=sorted_results[int(len(sorted_results) * 0.95)],
            p99_ms=sorted_results[int(len(sorted_results) * 0.99)],
            ops_per_second=1000 / statistics.mean(self.results) if self.results else 0
        )


# Specific benchmark implementations

class OrderProcessingBenchmark(PerformanceBenchmark):
    """Benchmark order processing performance."""
    
    def __init__(self):
        super().__init__("order_processing", target_ms=50)
        self.order_id = 0
        
    async def setup(self):
        """Setup test data."""
        # Initialize mock order processor
        from genesis.engine.executor.market import MarketOrderExecutor
        self.executor = MarketOrderExecutor()
        
    async def run_iteration(self):
        """Process a single order."""
        self.order_id += 1
        
        order = {
            'order_id': f'test_{self.order_id}',
            'symbol': 'BTCUSDT',
            'side': 'buy' if self.order_id % 2 == 0 else 'sell',
            'type': 'market',
            'quantity': Decimal('0.001'),
            'timestamp': time.time()
        }
        
        # Simulate order processing
        await self.executor.execute(order)


class PositionCalculationBenchmark(PerformanceBenchmark):
    """Benchmark position calculation performance."""
    
    def __init__(self):
        super().__init__("position_calculation", target_ms=10)
        
    async def setup(self):
        """Setup test positions."""
        self.positions = [
            {
                'symbol': 'BTCUSDT',
                'quantity': Decimal('0.5'),
                'entry_price': Decimal('45000'),
                'current_price': Decimal('46000')
            }
            for _ in range(100)
        ]
        
    async def run_iteration(self):
        """Calculate P&L for all positions."""
        total_pnl = Decimal('0')
        
        for pos in self.positions:
            pnl = (pos['current_price'] - pos['entry_price']) * pos['quantity']
            total_pnl += pnl
            
        return total_pnl


class DatabaseQueryBenchmark(PerformanceBenchmark):
    """Benchmark database query performance."""
    
    def __init__(self):
        super().__init__("database_query", target_ms=5)
        
    async def setup(self):
        """Setup database connection."""
        from genesis.data.postgres_repo import PostgresRepository
        self.repo = PostgresRepository()
        await self.repo.connect()
        
    async def teardown(self):
        """Close database connection."""
        if hasattr(self, 'repo'):
            await self.repo.disconnect()
            
    async def run_iteration(self):
        """Execute database query."""
        # Simulate fetching recent orders
        query = """
            SELECT order_id, symbol, side, quantity, price, status
            FROM orders
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 100
        """
        
        results = await self.repo.execute_query(query)
        return results


class WebSocketMessageBenchmark(PerformanceBenchmark):
    """Benchmark WebSocket message processing."""
    
    def __init__(self):
        super().__init__("websocket_message", target_ms=10)
        
    async def setup(self):
        """Setup message processor."""
        self.messages = [
            json.dumps({
                'type': 'orderbook',
                'symbol': 'BTCUSDT',
                'bids': [[45000 + i, 0.1] for i in range(20)],
                'asks': [[45100 + i, 0.1] for i in range(20)],
                'timestamp': time.time()
            })
            for _ in range(100)
        ]
        
    async def run_iteration(self):
        """Process WebSocket message."""
        for message in self.messages:
            data = json.loads(message)
            
            # Simulate processing
            if data['type'] == 'orderbook':
                best_bid = data['bids'][0][0] if data['bids'] else 0
                best_ask = data['asks'][0][0] if data['asks'] else 0
                spread = best_ask - best_bid


class AuthenticationBenchmark(PerformanceBenchmark):
    """Benchmark authentication performance."""
    
    def __init__(self):
        super().__init__("authentication", target_ms=100)
        
    async def setup(self):
        """Setup auth components."""
        from genesis.security.jwt_manager import JWTManager
        from genesis.security.password_manager import PasswordManager
        
        self.jwt_manager = JWTManager()
        self.password_manager = PasswordManager()
        
        # Create test user
        self.test_password = "TestPassword123!"
        self.password_hash = await self.password_manager.hash_password(self.test_password)
        
    async def run_iteration(self):
        """Authenticate user and generate token."""
        # Verify password
        is_valid = await self.password_manager.verify_password(
            self.test_password,
            self.password_hash
        )
        
        if is_valid:
            # Generate JWT token
            token = self.jwt_manager.create_token({
                'user_id': 'test_user',
                'username': 'test',
                'tier': 'sniper'
            })
            
            # Verify token
            payload = self.jwt_manager.verify_token(token)
            
        return is_valid


class RateLimitBenchmark(PerformanceBenchmark):
    """Benchmark rate limiting performance."""
    
    def __init__(self):
        super().__init__("rate_limit_check", target_ms=1)
        
    async def setup(self):
        """Setup rate limiter."""
        self.rate_limits = {}
        self.window_size = 60  # 60 seconds
        self.max_requests = 1000
        
    async def run_iteration(self):
        """Check rate limit for request."""
        user_id = f"user_{time.time() % 100}"  # Simulate 100 users
        current_time = time.time()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
            
        # Remove old entries outside window
        self.rate_limits[user_id] = [
            t for t in self.rate_limits[user_id]
            if current_time - t < self.window_size
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[user_id]) >= self.max_requests:
            return False  # Rate limited
            
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True  # Allowed


class BenchmarkSuite:
    """Complete benchmark suite runner."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("test_results/benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks = [
            OrderProcessingBenchmark(),
            PositionCalculationBenchmark(),
            DatabaseQueryBenchmark(),
            WebSocketMessageBenchmark(),
            AuthenticationBenchmark(),
            RateLimitBenchmark()
        ]
        
        self.baseline: Optional[Dict] = None
        self.load_baseline()
        
    def load_baseline(self):
        """Load baseline results for comparison."""
        baseline_file = self.results_dir / "baseline.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline = json.load(f)
                logger.info(f"Loaded baseline from {baseline_file}")
        else:
            logger.info("No baseline found - will create new baseline")
            
    def save_baseline(self, results: List[BenchmarkResult]):
        """Save results as new baseline."""
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {r.name: asdict(r) for r in results}
        }
        
        baseline_file = self.results_dir / "baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
            
        logger.info(f"Saved new baseline to {baseline_file}")
        
    async def run_all(self, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Performance Benchmark Suite")
        logger.info(f"Iterations per benchmark: {iterations}")
        logger.info(f"{'='*60}\n")
        
        for benchmark in self.benchmarks:
            logger.info(f"Running: {benchmark.name}")
            
            try:
                result = await benchmark.benchmark(iterations)
                results.append(result)
                
                # Check against target
                if result.meets_target(benchmark.target_ms):
                    logger.info(f"✓ {benchmark.name} meets target ({benchmark.target_ms}ms)")
                else:
                    logger.warning(f"✗ {benchmark.name} exceeds target "
                                 f"({result.p99_ms:.2f}ms > {benchmark.target_ms}ms)")
                    
                # Check for regression
                if self.baseline:
                    self.check_regression(result)
                    
            except Exception as e:
                logger.error(f"Failed to run {benchmark.name}: {e}")
                
            logger.info("")  # Empty line between benchmarks
            
        return results
        
    def check_regression(self, result: BenchmarkResult) -> bool:
        """Check if result shows regression from baseline."""
        if not self.baseline or result.name not in self.baseline['benchmarks']:
            return False
            
        baseline_result = self.baseline['benchmarks'][result.name]
        baseline_p99 = baseline_result['p99_ms']
        
        # Allow 10% degradation before flagging as regression
        threshold = baseline_p99 * 1.1
        
        if result.p99_ms > threshold:
            degradation = ((result.p99_ms - baseline_p99) / baseline_p99) * 100
            logger.warning(f"⚠ REGRESSION DETECTED in {result.name}: "
                         f"{result.p99_ms:.2f}ms vs baseline {baseline_p99:.2f}ms "
                         f"({degradation:.1f}% degradation)")
            return True
            
        return False
        
    def generate_report(self, results: List[BenchmarkResult]) -> Path:
        """Generate benchmark report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_benchmarks': len(results),
                'passed': sum(1 for r in results if r.meets_target(
                    next(b.target_ms for b in self.benchmarks if b.name == r.name)
                )),
                'regressions': sum(1 for r in results if self.check_regression(r))
            },
            'results': [asdict(r) for r in results]
        }
        
        # Save JSON report
        report_file = self.results_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate comparison table
        self.print_comparison_table(results)
        
        return report_file
        
    def print_comparison_table(self, results: List[BenchmarkResult]):
        """Print comparison table with baseline."""
        print("\n" + "="*80)
        print("Performance Benchmark Results")
        print("="*80)
        print(f"{'Benchmark':<25} {'P99 (ms)':<12} {'Target (ms)':<12} {'Status':<10} {'vs Baseline':<15}")
        print("-"*80)
        
        for result in results:
            target = next(b.target_ms for b in self.benchmarks if b.name == result.name)
            status = "✓ PASS" if result.meets_target(target) else "✗ FAIL"
            
            # Compare with baseline
            baseline_comp = ""
            if self.baseline and result.name in self.baseline['benchmarks']:
                baseline_p99 = self.baseline['benchmarks'][result.name]['p99_ms']
                diff = result.p99_ms - baseline_p99
                diff_pct = (diff / baseline_p99) * 100 if baseline_p99 > 0 else 0
                
                if abs(diff_pct) < 5:
                    baseline_comp = "→ Similar"
                elif diff_pct < 0:
                    baseline_comp = f"↑ {abs(diff_pct):.1f}% faster"
                else:
                    baseline_comp = f"↓ {diff_pct:.1f}% slower"
                    
            print(f"{result.name:<25} {result.p99_ms:<12.2f} {target:<12} {status:<10} {baseline_comp:<15}")
            
        print("="*80)


async def main():
    """Run benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per benchmark')
    parser.add_argument('--save-baseline', action='store_true', help='Save results as new baseline')
    parser.add_argument('--benchmarks', nargs='+', help='Specific benchmarks to run')
    
    args = parser.parse_args()
    
    suite = BenchmarkSuite()
    
    # Filter benchmarks if specified
    if args.benchmarks:
        suite.benchmarks = [b for b in suite.benchmarks if b.name in args.benchmarks]
        
    # Run benchmarks
    results = await suite.run_all(args.iterations)
    
    # Save baseline if requested
    if args.save_baseline:
        suite.save_baseline(results)
        
    # Generate report
    report_file = suite.generate_report(results)
    logger.info(f"\nReport saved to: {report_file}")
    
    # Check for failures
    failures = sum(1 for r in results if not r.meets_target(
        next(b.target_ms for b in suite.benchmarks if b.name == r.name)
    ))
    
    if failures > 0:
        logger.error(f"\n{failures} benchmarks failed to meet performance targets")
        return 1
        
    logger.info("\nAll benchmarks passed!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)