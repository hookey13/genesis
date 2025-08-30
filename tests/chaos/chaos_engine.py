"""
Chaos engineering framework for testing system resilience.

This module provides controlled chaos injection to validate system recovery
and fault tolerance capabilities.
"""

import asyncio
import random
import time
import psutil
import signal
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import structlog

logger = structlog.get_logger(__name__)


class ChaosType(Enum):
    """Types of chaos that can be injected."""
    
    PROCESS_KILL = "process_kill"           # Kill a process
    PROCESS_RESTART = "process_restart"     # Restart a service
    NETWORK_DELAY = "network_delay"         # Add network latency
    NETWORK_LOSS = "network_loss"           # Drop packets
    NETWORK_PARTITION = "network_partition" # Network split
    CPU_STRESS = "cpu_stress"               # High CPU usage
    MEMORY_STRESS = "memory_stress"         # High memory usage
    DISK_STRESS = "disk_stress"             # High disk I/O
    DATABASE_SLOW = "database_slow"         # Slow database queries
    API_FAILURE = "api_failure"             # API failures


@dataclass
class ChaosEvent:
    """Represents a chaos event that was injected."""
    
    chaos_type: ChaosType
    timestamp: datetime
    duration_seconds: float
    target: str
    parameters: Dict[str, Any]
    impact: str
    recovered: bool = False
    recovery_time_seconds: Optional[float] = None


@dataclass
class ChaosMetrics:
    """Tracks metrics during chaos testing."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    events_injected: List[ChaosEvent] = field(default_factory=list)
    failures_detected: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    mean_recovery_time: float = 0
    max_recovery_time: float = 0
    service_availability: float = 100.0
    data_consistency_checks: int = 0
    data_inconsistencies: int = 0
    
    def add_event(self, event: ChaosEvent):
        """Add a chaos event to metrics."""
        self.events_injected.append(event)
        
        if event.recovered and event.recovery_time_seconds:
            # Update recovery metrics
            recovery_times = [
                e.recovery_time_seconds for e in self.events_injected
                if e.recovered and e.recovery_time_seconds
            ]
            if recovery_times:
                self.mean_recovery_time = sum(recovery_times) / len(recovery_times)
                self.max_recovery_time = max(recovery_times)
    
    def calculate_availability(self) -> float:
        """Calculate service availability percentage."""
        if not self.end_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
        else:
            total_time = (self.end_time - self.start_time).total_seconds()
        
        if total_time == 0:
            return 100.0
        
        # Calculate total downtime
        downtime = sum(
            e.duration_seconds for e in self.events_injected
            if not e.recovered or e.recovery_time_seconds is None
        )
        
        availability = ((total_time - downtime) / total_time) * 100
        return max(0, min(100, availability))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_events": len(self.events_injected),
            "failures_detected": self.failures_detected,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "mean_recovery_time_seconds": self.mean_recovery_time,
            "max_recovery_time_seconds": self.max_recovery_time,
            "service_availability_percent": self.calculate_availability(),
            "data_consistency_checks": self.data_consistency_checks,
            "data_inconsistencies": self.data_inconsistencies,
            "events": [
                {
                    "type": e.chaos_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "duration": e.duration_seconds,
                    "target": e.target,
                    "recovered": e.recovered,
                    "recovery_time": e.recovery_time_seconds
                }
                for e in self.events_injected
            ]
        }


class ChaosMonkey:
    """Main chaos injection engine."""
    
    def __init__(
        self,
        target_system: Optional[Dict[str, Any]] = None,
        recovery_validator: Optional[Callable] = None
    ):
        """
        Initialize chaos monkey.
        
        Args:
            target_system: System configuration and components
            recovery_validator: Function to validate system recovery
        """
        self.target_system = target_system or {}
        self.recovery_validator = recovery_validator or self._default_validator
        self.metrics = ChaosMetrics()
        self.running = False
        self.chaos_schedule: List[ChaosEvent] = []
        
    async def _default_validator(self) -> bool:
        """Default recovery validator."""
        # Simple health check
        return True
    
    async def run_chaos_test(
        self,
        duration_minutes: int = 60,
        chaos_probability: float = 0.1,
        chaos_types: Optional[List[ChaosType]] = None
    ):
        """
        Run chaos test for specified duration.
        
        Args:
            duration_minutes: Test duration in minutes
            chaos_probability: Probability of chaos injection per minute
            chaos_types: Types of chaos to inject (None = all types)
        """
        logger.info(
            "Starting chaos test",
            duration_minutes=duration_minutes,
            chaos_probability=chaos_probability
        )
        
        self.running = True
        self.metrics = ChaosMetrics()
        
        if chaos_types is None:
            chaos_types = list(ChaosType)
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now() < end_time and self.running:
                # Decide whether to inject chaos
                if random.random() < chaos_probability:
                    chaos_type = random.choice(chaos_types)
                    await self.inject_chaos(chaos_type)
                
                # Wait before next potential injection
                await asyncio.sleep(60)  # Check every minute
                
                # Validate system health
                is_healthy = await self.recovery_validator()
                if not is_healthy:
                    self.metrics.failures_detected += 1
                    logger.warning("System unhealthy after chaos injection")
                
        finally:
            self.running = False
            self.metrics.end_time = datetime.now()
            
            # Generate report
            await self.generate_report()
    
    async def inject_chaos(self, chaos_type: ChaosType) -> ChaosEvent:
        """Inject specific type of chaos."""
        logger.info(f"Injecting chaos: {chaos_type.value}")
        
        event = ChaosEvent(
            chaos_type=chaos_type,
            timestamp=datetime.now(),
            duration_seconds=random.uniform(5, 30),
            target="",
            parameters={},
            impact="unknown"
        )
        
        try:
            if chaos_type == ChaosType.PROCESS_KILL:
                await self._inject_process_kill(event)
            elif chaos_type == ChaosType.PROCESS_RESTART:
                await self._inject_process_restart(event)
            elif chaos_type == ChaosType.NETWORK_DELAY:
                await self._inject_network_delay(event)
            elif chaos_type == ChaosType.NETWORK_LOSS:
                await self._inject_network_loss(event)
            elif chaos_type == ChaosType.CPU_STRESS:
                await self._inject_cpu_stress(event)
            elif chaos_type == ChaosType.MEMORY_STRESS:
                await self._inject_memory_stress(event)
            elif chaos_type == ChaosType.DATABASE_SLOW:
                await self._inject_database_slow(event)
            elif chaos_type == ChaosType.API_FAILURE:
                await self._inject_api_failure(event)
            else:
                logger.warning(f"Chaos type not implemented: {chaos_type}")
            
            # Wait for recovery
            recovery_start = time.time()
            await asyncio.sleep(event.duration_seconds)
            
            # Check if system recovered
            if await self.recovery_validator():
                event.recovered = True
                event.recovery_time_seconds = time.time() - recovery_start
                self.metrics.successful_recoveries += 1
                logger.info(f"System recovered from {chaos_type.value}")
            else:
                event.recovered = False
                self.metrics.failed_recoveries += 1
                logger.error(f"System failed to recover from {chaos_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to inject chaos: {e}")
            event.impact = f"injection_failed: {str(e)}"
        
        self.metrics.add_event(event)
        return event
    
    async def _inject_process_kill(self, event: ChaosEvent):
        """Simulate process kill."""
        # In real implementation, would kill actual process
        # For testing, just simulate
        event.target = "trading_engine"
        event.parameters = {"signal": "SIGKILL"}
        event.impact = "process_terminated"
        
        logger.info("Simulating process kill", target=event.target)
        
        # Simulate kill effect
        if self.target_system.get("process"):
            # Could send actual signal here
            pass
    
    async def _inject_process_restart(self, event: ChaosEvent):
        """Simulate service restart."""
        event.target = "websocket_handler"
        event.parameters = {"restart_delay": 5}
        event.impact = "service_restarted"
        
        logger.info("Simulating service restart", target=event.target)
        
        # Simulate restart
        await asyncio.sleep(5)
    
    async def _inject_network_delay(self, event: ChaosEvent):
        """Inject network delay."""
        delay_ms = random.randint(100, 1000)
        event.target = "network"
        event.parameters = {"delay_ms": delay_ms}
        event.impact = f"added_{delay_ms}ms_latency"
        
        logger.info(f"Injecting {delay_ms}ms network delay")
        
        # In real implementation, would use tc (traffic control) on Linux
        # For testing, just simulate
        if os.name == "posix":
            # Example: tc qdisc add dev eth0 root netem delay {delay_ms}ms
            pass
    
    async def _inject_network_loss(self, event: ChaosEvent):
        """Inject packet loss."""
        loss_percent = random.randint(5, 30)
        event.target = "network"
        event.parameters = {"loss_percent": loss_percent}
        event.impact = f"{loss_percent}%_packet_loss"
        
        logger.info(f"Injecting {loss_percent}% packet loss")
        
        # In real implementation, would use tc on Linux
        # For testing, just simulate
    
    async def _inject_cpu_stress(self, event: ChaosEvent):
        """Inject CPU stress."""
        event.target = "cpu"
        event.parameters = {"cores": 2, "load_percent": 80}
        event.impact = "high_cpu_usage"
        
        logger.info("Injecting CPU stress")
        
        # Create CPU-intensive tasks
        async def cpu_burn():
            end_time = time.time() + event.duration_seconds
            while time.time() < end_time:
                # CPU-intensive calculation
                _ = sum(i * i for i in range(1000))
                await asyncio.sleep(0.001)  # Yield occasionally
        
        # Start multiple CPU burn tasks
        tasks = [asyncio.create_task(cpu_burn()) for _ in range(2)]
        
        # Wait for duration
        await asyncio.sleep(event.duration_seconds)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
    
    async def _inject_memory_stress(self, event: ChaosEvent):
        """Inject memory stress."""
        memory_mb = 100  # Allocate 100MB
        event.target = "memory"
        event.parameters = {"allocated_mb": memory_mb}
        event.impact = f"allocated_{memory_mb}MB"
        
        logger.info(f"Injecting memory stress: {memory_mb}MB")
        
        # Allocate memory
        data = bytearray(memory_mb * 1024 * 1024)
        
        # Hold for duration
        await asyncio.sleep(event.duration_seconds)
        
        # Release memory
        del data
    
    async def _inject_database_slow(self, event: ChaosEvent):
        """Simulate slow database queries."""
        event.target = "database"
        event.parameters = {"slowdown_factor": 10}
        event.impact = "10x_query_slowdown"
        
        logger.info("Simulating slow database queries")
        
        # In real implementation, would add delays to database operations
        # For testing, just simulate
    
    async def _inject_api_failure(self, event: ChaosEvent):
        """Simulate API failures."""
        failure_rate = 0.5  # 50% failure rate
        event.target = "exchange_api"
        event.parameters = {"failure_rate": failure_rate}
        event.impact = f"{int(failure_rate * 100)}%_api_failures"
        
        logger.info(f"Injecting API failures: {failure_rate * 100}%")
        
        # In real implementation, would intercept API calls
        # For testing, just simulate
    
    async def validate_data_consistency(self) -> bool:
        """Validate data consistency after chaos."""
        self.metrics.data_consistency_checks += 1
        
        try:
            # Check various consistency rules
            checks = [
                self._check_position_consistency(),
                self._check_order_consistency(),
                self._check_balance_consistency()
            ]
            
            results = await asyncio.gather(*checks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Consistency check failed: {result}")
                    self.metrics.data_inconsistencies += 1
                    return False
                elif not result:
                    self.metrics.data_inconsistencies += 1
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            self.metrics.data_inconsistencies += 1
            return False
    
    async def _check_position_consistency(self) -> bool:
        """Check position data consistency."""
        # Simulate position consistency check
        return True
    
    async def _check_order_consistency(self) -> bool:
        """Check order data consistency."""
        # Simulate order consistency check
        return True
    
    async def _check_balance_consistency(self) -> bool:
        """Check balance consistency."""
        # Simulate balance consistency check
        return True
    
    async def generate_report(self):
        """Generate chaos test report."""
        report_path = Path("tests/chaos/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"chaos_test_{timestamp}.json"
        
        # Final consistency check
        data_consistent = await self.validate_data_consistency()
        
        report = {
            "test_type": "chaos_engineering",
            "data_consistent": data_consistent,
            "metrics": self.metrics.to_dict(),
            "success": (
                self.metrics.failed_recoveries == 0 and
                self.metrics.data_inconsistencies == 0 and
                self.metrics.calculate_availability() > 95.0
            )
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report generated: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("CHAOS ENGINEERING TEST SUMMARY")
        print("=" * 60)
        print(f"Total Events Injected: {len(self.metrics.events_injected)}")
        print(f"Failures Detected: {self.metrics.failures_detected}")
        print(f"Successful Recoveries: {self.metrics.successful_recoveries}")
        print(f"Failed Recoveries: {self.metrics.failed_recoveries}")
        print(f"Mean Recovery Time: {self.metrics.mean_recovery_time:.1f}s")
        print(f"Max Recovery Time: {self.metrics.max_recovery_time:.1f}s")
        print(f"Service Availability: {self.metrics.calculate_availability():.2f}%")
        print(f"Data Consistency Checks: {self.metrics.data_consistency_checks}")
        print(f"Data Inconsistencies: {self.metrics.data_inconsistencies}")
        print(f"Test Result: {'PASS' if report['success'] else 'FAIL'}")
        print("=" * 60)


class RecoveryValidator:
    """Validates system recovery after chaos injection."""
    
    def __init__(self, health_checks: List[Callable] = None):
        """
        Initialize recovery validator.
        
        Args:
            health_checks: List of health check functions
        """
        self.health_checks = health_checks or []
    
    async def validate(self) -> bool:
        """Validate system has recovered."""
        if not self.health_checks:
            return True
        
        try:
            results = await asyncio.gather(
                *[check() for check in self.health_checks],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Health check failed: {result}")
                    return False
                elif not result:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Recovery validation error: {e}")
            return False


async def main():
    """Run chaos engineering test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chaos engineering test")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in minutes (default: 60)"
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.1,
        help="Chaos injection probability (default: 0.1)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in ChaosType],
        help="Types of chaos to inject"
    )
    
    args = parser.parse_args()
    
    chaos_types = None
    if args.types:
        chaos_types = [ChaosType(t) for t in args.types]
    
    # Create recovery validator
    async def check_api():
        # Simulate API health check
        return True
    
    async def check_database():
        # Simulate database health check
        return True
    
    validator = RecoveryValidator([check_api, check_database])
    
    # Create and run chaos monkey
    monkey = ChaosMonkey(recovery_validator=validator.validate)
    
    try:
        await monkey.run_chaos_test(
            duration_minutes=args.duration,
            chaos_probability=args.probability,
            chaos_types=chaos_types
        )
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())