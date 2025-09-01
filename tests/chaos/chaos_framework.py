"""
Chaos Engineering Framework for Project GENESIS.

Simulates various failure scenarios to test system resilience:
- Network failures and partitions
- Database failures and corruptions
- Exchange API outages
- Resource exhaustion
- Concurrent failure scenarios
"""

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class ChaosType(Enum):
    """Types of chaos events that can be injected."""
    
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    PACKET_LOSS = "packet_loss"
    DATABASE_FAILURE = "database_failure"
    DATABASE_CORRUPTION = "database_corruption"
    API_OUTAGE = "api_outage"
    API_RATE_LIMIT = "api_rate_limit"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    CLOCK_SKEW = "clock_skew"
    BYZANTINE_FAILURE = "byzantine_failure"


@dataclass
class ChaosEvent:
    """Represents a chaos event to be injected."""
    
    chaos_type: ChaosType
    duration_seconds: float
    severity: float  # 0.0 to 1.0
    target: Optional[str] = None  # Specific component to target
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class ChaosMonkey:
    """Main chaos injection engine."""
    
    def __init__(self):
        self.active_chaos: List[ChaosEvent] = []
        self.chaos_history: List[ChaosEvent] = []
        self.start_time = time.time()
        self.recovery_callbacks: Dict[ChaosType, List[Callable]] = {}
    
    async def inject_chaos(self, event: ChaosEvent):
        """Inject a chaos event into the system."""
        print(f"🐒 Chaos Monkey: Injecting {event.chaos_type.value} "
              f"(severity: {event.severity:.1%}, duration: {event.duration_seconds}s)")
        
        self.active_chaos.append(event)
        event_start = time.time()
        
        try:
            # Route to specific chaos injection method
            injector = getattr(self, f"_inject_{event.chaos_type.value}", None)
            if injector:
                await injector(event)
            else:
                await self._generic_chaos(event)
            
            # Wait for chaos duration
            await asyncio.sleep(event.duration_seconds)
            
        finally:
            # Remove chaos and trigger recovery
            self.active_chaos.remove(event)
            self.chaos_history.append(event)
            
            # Call recovery callbacks
            if event.chaos_type in self.recovery_callbacks:
                for callback in self.recovery_callbacks[event.chaos_type]:
                    await callback(event)
            
            elapsed = time.time() - event_start
            print(f"✅ Chaos Monkey: Recovered from {event.chaos_type.value} "
                  f"(actual duration: {elapsed:.1f}s)")
    
    async def _inject_network_latency(self, event: ChaosEvent):
        """Inject network latency into API calls."""
        base_latency = event.parameters.get("base_latency_ms", 100)
        jitter = event.parameters.get("jitter_ms", 50)
        
        async def delayed_call(original_func):
            """Wrapper to add latency to network calls."""
            async def wrapper(*args, **kwargs):
                delay_ms = base_latency + random.uniform(-jitter, jitter)
                delay_ms *= event.severity  # Scale by severity
                await asyncio.sleep(delay_ms / 1000)
                return await original_func(*args, **kwargs)
            return wrapper
        
        # Patch network calls with delay
        with patch("genesis.exchange.gateway.BinanceGateway._make_request") as mock:
            mock.side_effect = delayed_call(mock.return_value)
    
    async def _inject_network_partition(self, event: ChaosEvent):
        """Simulate network partition."""
        affected_hosts = event.parameters.get("hosts", ["api.binance.com"])
        
        async def connection_error(*args, **kwargs):
            raise ConnectionError(f"Network partition: Cannot reach {affected_hosts}")
        
        with patch("genesis.exchange.gateway.BinanceGateway._make_request") as mock:
            if random.random() < event.severity:
                mock.side_effect = connection_error
    
    async def _inject_packet_loss(self, event: ChaosEvent):
        """Simulate packet loss."""
        loss_rate = event.severity
        
        async def maybe_drop_packet(original_func):
            async def wrapper(*args, **kwargs):
                if random.random() < loss_rate:
                    raise asyncio.TimeoutError("Packet lost")
                return await original_func(*args, **kwargs)
            return wrapper
        
        with patch("genesis.exchange.websocket_manager.WebSocketManager._send") as mock:
            mock.side_effect = maybe_drop_packet(mock.return_value)
    
    async def _inject_database_failure(self, event: ChaosEvent):
        """Simulate database connection failure."""
        failure_type = event.parameters.get("failure_type", "connection")
        
        if failure_type == "connection":
            error = ConnectionError("Database connection lost")
        elif failure_type == "lock":
            error = TimeoutError("Database lock timeout")
        else:
            error = Exception("Database failure")
        
        with patch("genesis.data.repository.Repository.execute") as mock:
            if random.random() < event.severity:
                mock.side_effect = error
    
    async def _inject_database_corruption(self, event: ChaosEvent):
        """Simulate data corruption."""
        corruption_type = event.parameters.get("corruption_type", "value")
        
        async def corrupt_data(original_func):
            async def wrapper(*args, **kwargs):
                result = await original_func(*args, **kwargs)
                
                if corruption_type == "value" and isinstance(result, dict):
                    # Randomly corrupt numeric values
                    for key in result:
                        if isinstance(result[key], (int, float, Decimal)):
                            if random.random() < event.severity * 0.1:
                                result[key] = result[key] * Decimal("-1")
                
                elif corruption_type == "missing" and isinstance(result, list):
                    # Randomly remove items
                    if random.random() < event.severity:
                        result = result[::2]  # Remove every other item
                
                return result
            return wrapper
        
        with patch("genesis.data.repository.Repository.fetch") as mock:
            mock.side_effect = corrupt_data(mock.return_value)
    
    async def _inject_api_outage(self, event: ChaosEvent):
        """Simulate complete API outage."""
        with patch("genesis.exchange.gateway.BinanceGateway.is_connected", False):
            with patch("genesis.exchange.gateway.BinanceGateway._make_request") as mock:
                mock.side_effect = Exception("Service unavailable")
    
    async def _inject_api_rate_limit(self, event: ChaosEvent):
        """Simulate hitting rate limits."""
        from genesis.core.exceptions import RateLimitExceeded
        
        burst_size = event.parameters.get("burst_size", 10)
        request_count = 0
        
        async def rate_limited_call(original_func):
            nonlocal request_count
            async def wrapper(*args, **kwargs):
                request_count += 1
                if request_count > burst_size:
                    raise RateLimitExceeded("Rate limit exceeded", retry_after=60)
                return await original_func(*args, **kwargs)
            return wrapper
        
        with patch("genesis.exchange.gateway.BinanceGateway._make_request") as mock:
            mock.side_effect = rate_limited_call(mock.return_value)
    
    async def _inject_memory_leak(self, event: ChaosEvent):
        """Simulate memory leak."""
        leak_rate_mb = event.parameters.get("leak_rate_mb", 10)
        leaked_memory = []
        
        async def leak_memory():
            while event in self.active_chaos:
                # Allocate memory that won't be freed
                chunk_size = int(leak_rate_mb * event.severity * 1024 * 1024)
                leaked_memory.append(bytearray(chunk_size))
                await asyncio.sleep(1)
        
        asyncio.create_task(leak_memory())
    
    async def _inject_cpu_spike(self, event: ChaosEvent):
        """Simulate CPU spike."""
        intensity = event.severity
        
        async def cpu_intensive_task():
            while event in self.active_chaos:
                # Perform CPU-intensive calculation
                start = time.time()
                while time.time() - start < intensity:
                    _ = sum(i**2 for i in range(10000))
                await asyncio.sleep(0.1)
        
        asyncio.create_task(cpu_intensive_task())
    
    async def _inject_clock_skew(self, event: ChaosEvent):
        """Simulate clock synchronization issues."""
        skew_seconds = event.parameters.get("skew_seconds", 30) * event.severity
        
        def skewed_time():
            return time.time() + skew_seconds
        
        with patch("time.time", skewed_time):
            with patch("genesis.exchange.time_sync.TimeSync.get_server_time") as mock:
                mock.return_value = datetime.now() + timedelta(seconds=skew_seconds)
    
    async def _inject_byzantine_failure(self, event: ChaosEvent):
        """Simulate Byzantine failures (inconsistent behavior)."""
        
        async def byzantine_behavior(original_func):
            async def wrapper(*args, **kwargs):
                choice = random.random()
                
                if choice < event.severity * 0.3:
                    # Return wrong result
                    result = await original_func(*args, **kwargs)
                    if isinstance(result, dict) and "price" in result:
                        result["price"] = result["price"] * Decimal("2")
                    return result
                
                elif choice < event.severity * 0.6:
                    # Timeout randomly
                    await asyncio.sleep(random.uniform(5, 10))
                    raise asyncio.TimeoutError("Byzantine timeout")
                
                else:
                    # Work normally
                    return await original_func(*args, **kwargs)
            
            return wrapper
        
        with patch("genesis.exchange.gateway.BinanceGateway.get_ticker") as mock:
            mock.side_effect = byzantine_behavior(mock.return_value)
    
    async def _generic_chaos(self, event: ChaosEvent):
        """Generic chaos for unimplemented types."""
        print(f"Generic chaos: {event.chaos_type.value}")
        await asyncio.sleep(event.duration_seconds)
    
    def register_recovery(self, chaos_type: ChaosType, callback: Callable):
        """Register a recovery callback for a chaos type."""
        if chaos_type not in self.recovery_callbacks:
            self.recovery_callbacks[chaos_type] = []
        self.recovery_callbacks[chaos_type].append(callback)
    
    def get_chaos_report(self) -> Dict:
        """Generate a report of chaos events."""
        return {
            "total_events": len(self.chaos_history),
            "active_chaos": len(self.active_chaos),
            "uptime_seconds": time.time() - self.start_time,
            "events_by_type": {
                chaos_type.value: sum(
                    1 for e in self.chaos_history if e.chaos_type == chaos_type
                )
                for chaos_type in ChaosType
            },
            "average_severity": sum(e.severity for e in self.chaos_history) / len(self.chaos_history)
            if self.chaos_history else 0,
        }


class ChaosScenario:
    """Predefined chaos scenarios for testing."""
    
    @staticmethod
    def market_crash() -> List[ChaosEvent]:
        """Simulate market crash conditions."""
        return [
            ChaosEvent(
                ChaosType.API_RATE_LIMIT,
                duration_seconds=30,
                severity=0.9,
                parameters={"burst_size": 5}
            ),
            ChaosEvent(
                ChaosType.NETWORK_LATENCY,
                duration_seconds=60,
                severity=0.8,
                parameters={"base_latency_ms": 500, "jitter_ms": 200}
            ),
            ChaosEvent(
                ChaosType.PACKET_LOSS,
                duration_seconds=20,
                severity=0.3
            ),
        ]
    
    @staticmethod
    def infrastructure_failure() -> List[ChaosEvent]:
        """Simulate infrastructure failures."""
        return [
            ChaosEvent(
                ChaosType.DATABASE_FAILURE,
                duration_seconds=10,
                severity=1.0,
                parameters={"failure_type": "connection"}
            ),
            ChaosEvent(
                ChaosType.NETWORK_PARTITION,
                duration_seconds=30,
                severity=0.7,
                parameters={"hosts": ["api.binance.com"]}
            ),
            ChaosEvent(
                ChaosType.MEMORY_LEAK,
                duration_seconds=120,
                severity=0.5,
                parameters={"leak_rate_mb": 50}
            ),
        ]
    
    @staticmethod
    def cascading_failure() -> List[ChaosEvent]:
        """Simulate cascading failures."""
        return [
            ChaosEvent(
                ChaosType.CPU_SPIKE,
                duration_seconds=30,
                severity=0.8
            ),
            ChaosEvent(
                ChaosType.MEMORY_LEAK,
                duration_seconds=60,
                severity=0.6,
                parameters={"leak_rate_mb": 30}
            ),
            ChaosEvent(
                ChaosType.DATABASE_FAILURE,
                duration_seconds=15,
                severity=0.9,
                parameters={"failure_type": "lock"}
            ),
            ChaosEvent(
                ChaosType.API_OUTAGE,
                duration_seconds=20,
                severity=1.0
            ),
        ]
    
    @staticmethod
    def data_corruption() -> List[ChaosEvent]:
        """Simulate data corruption scenarios."""
        return [
            ChaosEvent(
                ChaosType.DATABASE_CORRUPTION,
                duration_seconds=45,
                severity=0.3,
                parameters={"corruption_type": "value"}
            ),
            ChaosEvent(
                ChaosType.BYZANTINE_FAILURE,
                duration_seconds=30,
                severity=0.5
            ),
            ChaosEvent(
                ChaosType.CLOCK_SKEW,
                duration_seconds=60,
                severity=0.7,
                parameters={"skew_seconds": 120}
            ),
        ]


@asynccontextmanager
async def chaos_test(scenario: List[ChaosEvent], delay_between: float = 5.0):
    """Context manager for running chaos tests."""
    monkey = ChaosMonkey()
    tasks = []
    
    try:
        # Start chaos events with delays
        for i, event in enumerate(scenario):
            if i > 0:
                await asyncio.sleep(delay_between)
            task = asyncio.create_task(monkey.inject_chaos(event))
            tasks.append(task)
        
        yield monkey
        
    finally:
        # Wait for all chaos to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate report
        report = monkey.get_chaos_report()
        print("\n" + "="*50)
        print("CHAOS TEST REPORT")
        print("="*50)
        print(f"Total events: {report['total_events']}")
        print(f"Average severity: {report['average_severity']:.1%}")
        print("Events by type:")
        for event_type, count in report['events_by_type'].items():
            if count > 0:
                print(f"  - {event_type}: {count}")
        print("="*50)