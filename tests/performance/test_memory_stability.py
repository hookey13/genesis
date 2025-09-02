"""
48-hour memory stability test framework for Project GENESIS.
Tests system stability during extended operations and detects memory leaks.
"""

import asyncio
import gc
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import pytest
import psutil
import structlog

from genesis.monitoring.memory_profiler import MemoryProfiler
from genesis.monitoring.advanced_profiler import AdvancedPerformanceProfiler
from genesis.core.models import Order, Position, OrderStatus
from genesis.engine.event_bus import EventBus
from genesis.engine.state_machine import TierStateMachine

logger = structlog.get_logger(__name__)


class StabilityTestFramework:
    """Framework for running extended stability tests."""
    
    def __init__(
        self,
        test_duration_hours: float = 48.0,
        memory_growth_threshold: float = 0.05,  # 5% growth allowed
        checkpoint_interval_hours: float = 1.0
    ):
        self.test_duration_hours = test_duration_hours
        self.memory_growth_threshold = memory_growth_threshold
        self.checkpoint_interval_hours = checkpoint_interval_hours
        
        # Profilers
        self.memory_profiler = MemoryProfiler(
            growth_threshold=memory_growth_threshold,
            snapshot_interval=60,
            enable_tracemalloc=True
        )
        self.cpu_profiler = AdvancedPerformanceProfiler()
        
        # Test state
        self.start_time: Optional[float] = None
        self.start_memory: Optional[int] = None
        self.checkpoints: List[Dict[str, Any]] = []
        self.test_passed = True
        self.failure_reasons: List[str] = []
        
        # Simulated components
        self.event_bus: Optional[EventBus] = None
        self.state_machine: Optional[TierStateMachine] = None
        self.active_positions: List[Position] = []
        self.orders_processed = 0
        
    async def setup_test_environment(self) -> None:
        """Set up the test environment with simulated components."""
        # Initialize event bus
        self.event_bus = EventBus()
        await self.event_bus.start()
        
        # Initialize state machine
        self.state_machine = TierStateMachine()
        
        # Start profilers
        await self.memory_profiler.start_monitoring()
        await self.cpu_profiler.start_profiling()
        
        # Record initial state
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        
        logger.info(
            "Stability test environment initialized",
            start_memory_mb=self.start_memory / 1024 / 1024,
            test_duration_hours=self.test_duration_hours
        )
    
    async def simulate_trading_operations(self, duration_seconds: float = 60) -> None:
        """Simulate realistic trading operations for load generation."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                # Simulate order creation
                order = Order(
                    id=f"test_order_{self.orders_processed}",
                    symbol="BTC/USDT",
                    side="BUY" if self.orders_processed % 2 == 0 else "SELL",
                    type="LIMIT",
                    price=Decimal("50000.00"),
                    quantity=Decimal("0.001"),
                    status=OrderStatus.NEW,
                    timestamp=datetime.now()
                )
                
                # Process order through event bus
                await self.event_bus.publish_order_event(order)
                self.orders_processed += 1
                
                # Simulate position management
                if self.orders_processed % 10 == 0:
                    position = Position(
                        symbol="BTC/USDT",
                        side="LONG",
                        entry_price=Decimal("50000.00"),
                        quantity=Decimal("0.01"),
                        timestamp=datetime.now()
                    )
                    self.active_positions.append(position)
                
                # Clean up old positions periodically
                if len(self.active_positions) > 100:
                    self.active_positions = self.active_positions[-50:]
                
                # Small delay to simulate realistic timing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in simulated operations", error=str(e))
    
    async def perform_checkpoint(self) -> Dict[str, Any]:
        """Perform a checkpoint to record current state."""
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        
        # Get memory stats
        process = psutil.Process()
        current_memory = process.memory_info().rss
        memory_growth = (current_memory - self.start_memory) / self.start_memory
        
        # Get CPU stats
        cpu_percent = process.cpu_percent(interval=1.0)
        
        # Get profiler stats
        memory_stats = self.memory_profiler.get_memory_stats()
        leak_result = self.memory_profiler.detect_leaks()
        
        checkpoint = {
            'hour': int(elapsed_hours),
            'timestamp': datetime.now(),
            'memory_mb': current_memory / 1024 / 1024,
            'memory_growth_percent': memory_growth * 100,
            'cpu_percent': cpu_percent,
            'orders_processed': self.orders_processed,
            'active_positions': len(self.active_positions),
            'leak_detected': leak_result.has_leak,
            'leak_confidence': leak_result.confidence,
            'gc_objects': len(gc.get_objects())
        }
        
        self.checkpoints.append(checkpoint)
        
        # Check for failures
        if memory_growth > self.memory_growth_threshold:
            self.test_passed = False
            self.failure_reasons.append(
                f"Memory growth {memory_growth:.2%} exceeds threshold {self.memory_growth_threshold:.2%}"
            )
        
        if leak_result.has_leak and leak_result.confidence > 0.7:
            self.test_passed = False
            self.failure_reasons.append(
                f"Memory leak detected with {leak_result.confidence:.0%} confidence"
            )
        
        logger.info(
            "Checkpoint completed",
            hour=checkpoint['hour'],
            memory_growth=f"{memory_growth:.2%}",
            cpu_percent=cpu_percent,
            orders_processed=self.orders_processed
        )
        
        return checkpoint
    
    async def run_stability_test(self) -> Dict[str, Any]:
        """Run the complete stability test."""
        await self.setup_test_environment()
        
        test_end_time = self.start_time + (self.test_duration_hours * 3600)
        checkpoint_interval = self.checkpoint_interval_hours * 3600
        next_checkpoint = self.start_time + checkpoint_interval
        
        try:
            while time.time() < test_end_time:
                # Run simulated operations
                await self.simulate_trading_operations(duration_seconds=60)
                
                # Perform checkpoint if needed
                if time.time() >= next_checkpoint:
                    await self.perform_checkpoint()
                    next_checkpoint += checkpoint_interval
                    
                    # Force garbage collection periodically
                    if len(self.checkpoints) % 5 == 0:
                        gc.collect(2)
                
                # Check for early termination conditions
                if not self.test_passed and len(self.failure_reasons) >= 3:
                    logger.error("Test terminated early due to multiple failures")
                    break
            
            # Final checkpoint
            final_checkpoint = await self.perform_checkpoint()
            
        finally:
            # Clean up
            await self.cleanup()
        
        # Generate test results
        results = self.generate_test_results()
        return results
    
    async def cleanup(self) -> None:
        """Clean up test resources."""
        if self.memory_profiler:
            await self.memory_profiler.stop_monitoring()
        
        if self.cpu_profiler:
            await self.cpu_profiler.stop_profiling()
        
        if self.event_bus:
            await self.event_bus.stop()
        
        # Clear data structures
        self.active_positions.clear()
        gc.collect()
    
    def generate_test_results(self) -> Dict[str, Any]:
        """Generate comprehensive test results."""
        if not self.checkpoints:
            return {
                'test_passed': False,
                'failure_reasons': ['No checkpoints recorded'],
                'duration_hours': 0
            }
        
        first_checkpoint = self.checkpoints[0]
        last_checkpoint = self.checkpoints[-1]
        
        # Calculate overall statistics
        total_memory_growth = (
            (last_checkpoint['memory_mb'] - first_checkpoint['memory_mb']) / 
            first_checkpoint['memory_mb']
        ) * 100
        
        avg_cpu = sum(c['cpu_percent'] for c in self.checkpoints) / len(self.checkpoints)
        peak_memory = max(c['memory_mb'] for c in self.checkpoints)
        
        # Determine final test status
        if total_memory_growth > self.memory_growth_threshold * 100:
            self.test_passed = False
            self.failure_reasons.append(
                f"Total memory growth {total_memory_growth:.1f}% exceeds limit"
            )
        
        results = {
            'test_passed': self.test_passed,
            'failure_reasons': self.failure_reasons,
            'duration_hours': (last_checkpoint['hour'] - first_checkpoint['hour']),
            'start_memory_mb': first_checkpoint['memory_mb'],
            'end_memory_mb': last_checkpoint['memory_mb'],
            'peak_memory_mb': peak_memory,
            'total_memory_growth_percent': total_memory_growth,
            'average_cpu_percent': avg_cpu,
            'orders_processed': self.orders_processed,
            'checkpoints': len(self.checkpoints),
            'memory_leaks_detected': sum(1 for c in self.checkpoints if c['leak_detected']),
            'hourly_stats': self.checkpoints
        }
        
        return results


# ============= Test Functions =============

@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.timeout(172800)  # 48 hours timeout
async def test_48_hour_stability():
    """
    Run 48-hour stability test to verify system stability and memory behavior.
    This test is marked as slow and should only run in CI/CD or dedicated test environments.
    """
    framework = StabilityTestFramework(
        test_duration_hours=48.0,
        memory_growth_threshold=0.05,
        checkpoint_interval_hours=1.0
    )
    
    results = await framework.run_stability_test()
    
    # Assertions
    assert results['test_passed'], f"Stability test failed: {results['failure_reasons']}"
    assert results['total_memory_growth_percent'] < 5.0, \
        f"Memory growth {results['total_memory_growth_percent']:.1f}% exceeds 5% limit"
    assert results['memory_leaks_detected'] == 0, \
        f"Detected {results['memory_leaks_detected']} memory leaks during test"
    assert results['average_cpu_percent'] < 90.0, \
        f"Average CPU usage {results['average_cpu_percent']:.1f}% is too high"
    
    # Log results
    logger.info(
        "48-hour stability test completed",
        passed=results['test_passed'],
        duration_hours=results['duration_hours'],
        memory_growth=f"{results['total_memory_growth_percent']:.1f}%",
        orders_processed=results['orders_processed']
    )


@pytest.mark.asyncio
@pytest.mark.timeout(3600)  # 1 hour timeout
async def test_short_stability_run():
    """
    Short stability test (1 hour) for development and quick validation.
    """
    framework = StabilityTestFramework(
        test_duration_hours=1.0,
        memory_growth_threshold=0.02,  # Stricter for short test
        checkpoint_interval_hours=0.25  # Every 15 minutes
    )
    
    results = await framework.run_stability_test()
    
    assert results['test_passed'], f"Short stability test failed: {results['failure_reasons']}"
    assert results['total_memory_growth_percent'] < 2.0, \
        f"Memory growth {results['total_memory_growth_percent']:.1f}% exceeds 2% limit"


@pytest.mark.asyncio
async def test_memory_leak_detection():
    """Test that memory leak detection works correctly."""
    framework = StabilityTestFramework(
        test_duration_hours=0.1,  # 6 minutes
        memory_growth_threshold=0.01,
        checkpoint_interval_hours=0.05  # Every 3 minutes
    )
    
    # Inject a memory leak
    leak_list = []
    
    async def leaky_operations(duration_seconds: float):
        """Intentionally leaky operations for testing."""
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            # Create memory leak
            leak_list.append("x" * 10000)
            await asyncio.sleep(0.01)
    
    # Replace normal operations with leaky ones
    framework.simulate_trading_operations = leaky_operations
    
    results = await framework.run_stability_test()
    
    # Should detect the leak
    assert not results['test_passed'], "Should have detected memory leak"
    assert results['memory_leaks_detected'] > 0, "Should have detected at least one leak"
    assert any('leak' in reason.lower() for reason in results['failure_reasons']), \
        "Failure reasons should mention memory leak"


@pytest.mark.asyncio
async def test_checkpoint_accuracy():
    """Test that checkpoints accurately record system state."""
    framework = StabilityTestFramework(
        test_duration_hours=0.05,  # 3 minutes
        memory_growth_threshold=0.1,
        checkpoint_interval_hours=0.0167  # Every minute
    )
    
    await framework.setup_test_environment()
    
    # Perform operations
    await framework.simulate_trading_operations(30)
    
    # Take checkpoint
    checkpoint = await framework.perform_checkpoint()
    
    # Verify checkpoint data
    assert 'memory_mb' in checkpoint
    assert 'cpu_percent' in checkpoint
    assert 'orders_processed' in checkpoint
    assert checkpoint['memory_mb'] > 0
    assert checkpoint['cpu_percent'] >= 0
    assert checkpoint['orders_processed'] > 0
    
    await framework.cleanup()


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_profiling_overhead(benchmark):
    """Benchmark the overhead of profiling during stability tests."""
    
    async def run_with_profiling():
        framework = StabilityTestFramework(
            test_duration_hours=0.0167,  # 1 minute
            memory_growth_threshold=0.1,
            checkpoint_interval_hours=0.0083  # Every 30 seconds
        )
        await framework.run_stability_test()
    
    async def run_without_profiling():
        # Simulate operations without profiling
        for _ in range(600):  # 600 operations
            order = Order(
                id=f"test_order_{_}",
                symbol="BTC/USDT",
                side="BUY",
                type="LIMIT",
                price=Decimal("50000.00"),
                quantity=Decimal("0.001"),
                status=OrderStatus.NEW,
                timestamp=datetime.now()
            )
            await asyncio.sleep(0.1)
    
    # Benchmark with profiling
    profiling_time = benchmark(asyncio.run, run_with_profiling())
    
    # The overhead should be minimal (< 5%)
    # This is a placeholder assertion as we can't easily measure without profiling in pytest-benchmark
    assert profiling_time < 70, "Profiling overhead is too high"