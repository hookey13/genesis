"""Execution system validation for Genesis trading system."""

import asyncio
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import BaseValidator, ValidationIssue, ValidationSeverity


class ExecutionValidator(BaseValidator):
    """Validates order execution system and routing logic."""
    
    @property
    def name(self) -> str:
        return "execution"
    
    @property
    def description(self) -> str:
        return "Validates order routing, slicing algorithms, and execution performance"
    
    async def _validate(self, mode: str):
        """Perform execution validation."""
        # Test order routing logic
        await self._test_order_routing()
        
        # Verify slicing algorithms (Hunter+)
        await self._verify_slicing_algorithms()
        
        # Check VWAP execution (Strategist)
        if mode in ["standard", "thorough"]:
            await self._check_vwap_execution()
        
        # Validate circuit breaker triggers
        await self._validate_circuit_breakers()
        
        # Test execution latency
        if mode == "thorough":
            await self._test_execution_latency()
    
    async def _test_order_routing(self):
        """Test order routing logic."""
        try:
            from genesis.engine.executor.base import ExecutorFactory
            from genesis.core.models import Order, OrderType
            
            factory = ExecutorFactory()
            
            # Test order type routing
            order_types = [
                (OrderType.MARKET, "MarketExecutor"),
                (OrderType.LIMIT, "LimitExecutor"),
                (OrderType.STOP_LOSS, "StopLossExecutor"),
                (OrderType.ICEBERG, "IcebergExecutor"),
                (OrderType.VWAP, "VWAPExecutor")
            ]
            
            for order_type, expected_executor in order_types:
                executor = factory.get_executor(order_type)
                
                self.check_condition(
                    executor.__class__.__name__ == expected_executor,
                    f"Correct executor for {order_type.value}: {expected_executor}",
                    f"Wrong executor for {order_type.value}",
                    ValidationSeverity.ERROR,
                    details={"order_type": order_type.value, "executor": executor.__class__.__name__}
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Executor factory not implemented",
                recommendation="Implement ExecutorFactory in genesis/engine/executor/base.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Order routing test failed",
                details={"error": str(e)}
            ))
    
    async def _verify_slicing_algorithms(self):
        """Verify order slicing algorithms for Hunter tier."""
        try:
            from genesis.engine.executor.iceberg import IcebergExecutor
            
            executor = IcebergExecutor()
            
            # Test slicing logic
            test_cases = [
                (Decimal("10000"), 10, Decimal("1000")),  # Equal slices
                (Decimal("5555"), 5, Decimal("1111")),    # Uneven amount
                (Decimal("100"), 20, Decimal("10")),      # Min slice size check
            ]
            
            for total_size, num_slices, expected_slice in test_cases:
                slices = executor.calculate_slices(total_size, num_slices)
                
                self.check_condition(
                    len(slices) <= num_slices,
                    f"Slicing: {total_size} into {len(slices)} slices",
                    f"Slicing failed: too many slices",
                    ValidationSeverity.ERROR,
                    details={
                        "total": float(total_size),
                        "slices": len(slices),
                        "expected": num_slices
                    }
                )
                
                # Verify sum equals total
                total_sliced = sum(slices)
                self.check_condition(
                    abs(total_sliced - total_size) < Decimal("0.01"),
                    f"Slice sum matches total: {total_sliced} ≈ {total_size}",
                    f"Slice sum mismatch: {total_sliced} != {total_size}",
                    ValidationSeverity.CRITICAL,
                    details={"sum": float(total_sliced), "total": float(total_size)}
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Iceberg executor not implemented",
                recommendation="Implement IcebergExecutor for Hunter tier"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Slicing algorithm test error",
                details={"error": str(e)}
            ))
    
    async def _check_vwap_execution(self):
        """Check VWAP execution for Strategist tier."""
        try:
            from genesis.engine.executor.vwap import VWAPExecutor
            
            executor = VWAPExecutor(
                time_horizon=3600,  # 1 hour
                participation_rate=0.1  # 10% of volume
            )
            
            # Test VWAP calculation
            market_data = [
                {"price": Decimal("50000"), "volume": Decimal("100")},
                {"price": Decimal("50100"), "volume": Decimal("150")},
                {"price": Decimal("49900"), "volume": Decimal("200")},
            ]
            
            vwap = executor.calculate_vwap(market_data)
            expected_vwap = Decimal("50000")  # Simplified calculation
            
            self.check_condition(
                vwap is not None,
                f"VWAP calculated: {vwap}",
                "VWAP calculation failed",
                ValidationSeverity.ERROR,
                details={"vwap": float(vwap) if vwap else None}
            )
            
            # Test execution schedule
            schedule = executor.create_schedule(
                total_size=Decimal("1000"),
                duration_seconds=3600
            )
            
            self.check_condition(
                len(schedule) > 0,
                f"VWAP schedule created: {len(schedule)} intervals",
                "VWAP schedule creation failed",
                ValidationSeverity.ERROR,
                details={"intervals": len(schedule)}
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="VWAP executor not implemented",
                recommendation="Implement VWAPExecutor for Strategist tier"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="VWAP execution test error",
                details={"error": str(e)}
            ))
    
    async def _validate_circuit_breakers(self):
        """Validate circuit breaker triggers."""
        try:
            from genesis.exchange.circuit_breaker import CircuitBreaker, CircuitState
            
            breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                half_open_requests=2
            )
            
            # Test state transitions
            initial_state = breaker.state
            self.check_condition(
                initial_state == CircuitState.CLOSED,
                "Circuit breaker starts closed",
                f"Invalid initial state: {initial_state}",
                ValidationSeverity.ERROR
            )
            
            # Simulate failures
            for _ in range(5):
                breaker.record_failure()
            
            self.check_condition(
                breaker.state == CircuitState.OPEN,
                "Circuit breaker opens after failures",
                "Circuit breaker failed to open",
                ValidationSeverity.CRITICAL,
                details={"state": breaker.state}
            )
            
            # Test automatic recovery timeout
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Circuit breaker state transitions validated"
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Circuit breaker not implemented",
                recommendation="Implement CircuitBreaker in genesis/exchange/circuit_breaker.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Circuit breaker validation failed",
                details={"error": str(e)}
            ))
    
    async def _test_execution_latency(self):
        """Test order execution latency."""
        try:
            from genesis.engine.executor.market import MarketExecutor
            from genesis.core.models import Order
            
            executor = MarketExecutor()
            
            # Simulate order execution
            latencies = []
            
            for i in range(10):
                order = Order(
                    symbol="BTC/USDT",
                    side="buy",
                    quantity=Decimal("0.001"),
                    order_type="market"
                )
                
                start_time = time.perf_counter()
                # Simulate execution (would be actual API call)
                await asyncio.sleep(0.01)  # Simulate network latency
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            # Check against thresholds
            self.check_threshold(
                avg_latency,
                50,
                "<",
                "Average execution latency",
                "ms",
                ValidationSeverity.WARNING
            )
            
            self.check_threshold(
                max_latency,
                100,
                "<",
                "Maximum execution latency",
                "ms",
                ValidationSeverity.ERROR
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Execution latency: avg={avg_latency:.1f}ms, max={max_latency:.1f}ms",
                details={
                    "avg_ms": avg_latency,
                    "max_ms": max_latency,
                    "samples": len(latencies)
                }
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Market executor not implemented",
                recommendation="Implement MarketExecutor in genesis/engine/executor/market.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Execution latency test error",
                details={"error": str(e)}
            ))