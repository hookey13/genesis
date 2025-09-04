"""Performance validation for Genesis trading system."""

import asyncio
import psutil
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

from . import BaseValidator, ValidationIssue, ValidationSeverity


class PerformanceValidator(BaseValidator):
    """Validates system performance and benchmarks."""
    
    @property
    def name(self) -> str:
        return "performance"
    
    @property
    def description(self) -> str:
        return "Validates system performance, latency, throughput, and resource usage"
    
    async def _validate(self, mode: str):
        """Perform performance validation."""
        # Benchmark order processing speed
        await self._benchmark_order_processing()
        
        # Test system under load
        if mode in ["standard", "thorough"]:
            await self._test_system_load()
        
        # Verify memory usage limits
        await self._verify_memory_usage()
        
        # Check CPU utilization
        await self._check_cpu_utilization()
        
        # Test WebSocket message processing
        if mode == "thorough":
            await self._test_websocket_performance()
    
    async def _benchmark_order_processing(self):
        """Benchmark order processing speed."""
        try:
            from genesis.engine.executor.base import ExecutorFactory
            from genesis.core.models import Order
            
            factory = ExecutorFactory()
            executor = factory.get_executor("market")
            
            # Benchmark order creation and validation
            latencies = []
            
            for i in range(100):
                start = time.perf_counter()
                
                order = Order(
                    symbol="BTC/USDT",
                    side="buy",
                    quantity=Decimal("0.001"),
                    order_type="market",
                    client_order_id=f"perf_test_{i}"
                )
                
                # Validate order (without actual execution)
                order.validate()
                
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            
            # Check against thresholds
            self.check_threshold(
                avg_latency,
                10,
                "<",
                "Average order processing",
                "ms",
                ValidationSeverity.WARNING
            )
            
            self.check_threshold(
                p95_latency,
                20,
                "<",
                "P95 order processing",
                "ms",
                ValidationSeverity.WARNING
            )
            
            self.check_threshold(
                p99_latency,
                50,
                "<",
                "P99 order processing",
                "ms",
                ValidationSeverity.ERROR
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Order processing: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms",
                details={
                    "avg_ms": avg_latency,
                    "p95_ms": p95_latency,
                    "p99_ms": p99_latency,
                    "samples": len(latencies)
                }
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Order processing components not fully implemented",
                recommendation="Complete Order and ExecutorFactory implementation"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Order processing benchmark failed",
                details={"error": str(e)}
            ))
    
    async def _test_system_load(self):
        """Test system under load."""
        try:
            from genesis.engine.engine import TradingEngine
            from genesis.core.models import MarketData
            
            engine = TradingEngine()
            
            # Simulate market data stream
            market_updates = []
            for i in range(1000):
                market_updates.append(MarketData(
                    symbol="BTC/USDT",
                    bid=Decimal("50000") + Decimal(i),
                    ask=Decimal("50001") + Decimal(i),
                    timestamp=time.time() * 1000
                ))
            
            # Process updates and measure throughput
            start = time.perf_counter()
            processed = 0
            
            for update in market_updates:
                await engine.process_market_data(update)
                processed += 1
            
            duration = time.perf_counter() - start
            throughput = processed / duration
            
            self.check_threshold(
                throughput,
                1000,
                ">",
                "Market data throughput",
                " updates/sec",
                ValidationSeverity.WARNING
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Load test: processed {processed} updates in {duration:.2f}s ({throughput:.0f}/sec)",
                details={
                    "updates_processed": processed,
                    "duration_seconds": duration,
                    "throughput_per_second": throughput
                }
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Trading engine not fully implemented",
                recommendation="Complete TradingEngine implementation"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Load test failed",
                details={"error": str(e)}
            ))
    
    async def _verify_memory_usage(self):
        """Verify memory usage limits."""
        process = psutil.Process()
        
        # Get current memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        self.check_threshold(
            memory_mb,
            2048,  # 2GB limit
            "<",
            "Current memory usage",
            "MB",
            ValidationSeverity.WARNING
        )
        
        # Simulate memory-intensive operation
        test_data = []
        for i in range(10000):
            test_data.append({
                "symbol": "BTC/USDT",
                "price": 50000.0 + i,
                "volume": 100.0,
                "timestamp": time.time()
            })
        
        # Check memory after load
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_increase = memory_after - memory_mb
        
        self.check_threshold(
            memory_increase,
            100,  # Max 100MB increase
            "<",
            "Memory increase during load",
            "MB",
            ValidationSeverity.WARNING
        )
        
        # Clear test data
        test_data.clear()
        
        self.result.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=f"Memory usage: {memory_mb:.1f}MB (increase: {memory_increase:.1f}MB)",
            details={
                "initial_mb": memory_mb,
                "after_load_mb": memory_after,
                "increase_mb": memory_increase
            }
        ))
    
    async def _check_cpu_utilization(self):
        """Check CPU utilization."""
        # Get CPU usage over 1 second
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self.check_threshold(
            cpu_percent,
            50,
            "<",
            "CPU utilization",
            "%",
            ValidationSeverity.WARNING
        )
        
        # Get per-core usage
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        max_core = max(per_core)
        
        self.check_threshold(
            max_core,
            80,
            "<",
            "Max single core usage",
            "%",
            ValidationSeverity.ERROR
        )
        
        self.result.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=f"CPU usage: {cpu_percent:.1f}% (max core: {max_core:.1f}%)",
            details={
                "overall_percent": cpu_percent,
                "max_core_percent": max_core,
                "cores": len(per_core)
            }
        ))
    
    async def _test_websocket_performance(self):
        """Test WebSocket message processing performance."""
        try:
            from genesis.exchange.websocket_manager import WebSocketManager
            
            ws_manager = WebSocketManager()
            
            # Simulate message processing
            messages = []
            for i in range(1000):
                messages.append({
                    "e": "trade",
                    "s": "BTCUSDT",
                    "p": str(50000 + i),
                    "q": str(0.1),
                    "T": int(time.time() * 1000)
                })
            
            # Measure processing speed
            latencies = []
            
            for msg in messages[:100]:  # Test first 100 messages
                start = time.perf_counter()
                await ws_manager.process_message(msg)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
            
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            self.check_threshold(
                avg_latency,
                5,
                "<",
                "Average WebSocket processing",
                "ms",
                ValidationSeverity.WARNING
            )
            
            self.check_threshold(
                max_latency,
                10,
                "<",
                "Max WebSocket processing",
                "ms",
                ValidationSeverity.ERROR
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"WebSocket performance: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms",
                details={
                    "avg_ms": avg_latency,
                    "max_ms": max_latency,
                    "messages_tested": len(latencies)
                }
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="WebSocket manager not implemented",
                recommendation="Implement WebSocketManager for real-time data"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="WebSocket performance test failed",
                details={"error": str(e)}
            ))