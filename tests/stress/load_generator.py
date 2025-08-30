"""
Load testing infrastructure for simulating 100x normal message volume.

This module provides comprehensive load testing to validate system performance
under extreme conditions with configurable load profiles.
"""

import asyncio
import json
import time
import random
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class LoadProfile(Enum):
    """Predefined load profiles for different market conditions."""
    
    NORMAL = "normal"           # Standard trading day
    VOLATILE = "volatile"       # High volatility period
    NEWS_SPIKE = "news_spike"   # Sudden spike from news event
    OPENING_BELL = "opening_bell"  # Market opening rush
    FLASH_CRASH = "flash_crash"    # Extreme market movement
    GRADUAL_RAMP = "gradual_ramp"  # Slowly increasing load


@dataclass
class LoadMetrics:
    """Tracks performance metrics during load testing."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    messages_sent: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    total_latency_ms: float = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0
    peak_messages_per_second: float = 0
    current_messages_per_second: float = 0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    
    def add_latency(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies_ms.append(latency_ms)
        self.total_latency_ms += latency_ms
    
    def get_percentile(self, percentile: float) -> float:
        """Get latency percentile (e.g., 50 for median, 95 for p95)."""
        if not self.latencies_ms:
            return 0
        
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def get_average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if self.messages_processed == 0:
            return 0
        return self.total_latency_ms / self.messages_processed
    
    def get_throughput(self) -> float:
        """Get average throughput in messages per second."""
        if not self.end_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        else:
            duration = (self.end_time - self.start_time).total_seconds()
        
        if duration == 0:
            return 0
        return self.messages_processed / duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time or datetime.now() - self.start_time).total_seconds(),
            "messages_sent": self.messages_sent,
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "drop_rate": (self.messages_dropped / self.messages_sent * 100) if self.messages_sent > 0 else 0,
            "errors": self.errors,
            "error_rate": (self.errors / self.messages_sent * 100) if self.messages_sent > 0 else 0,
            "average_latency_ms": self.get_average_latency(),
            "p50_latency_ms": self.get_percentile(50),
            "p95_latency_ms": self.get_percentile(95),
            "p99_latency_ms": self.get_percentile(99),
            "throughput_msg_per_sec": self.get_throughput(),
            "peak_messages_per_second": self.peak_messages_per_second
        }


class WebSocketMessageGenerator:
    """Generates realistic WebSocket messages at high volume."""
    
    def __init__(self, base_rate: int = 10):
        """
        Initialize message generator.
        
        Args:
            base_rate: Base messages per second (will be multiplied)
        """
        self.base_rate = base_rate
        self.multiplier = 1
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT", "AVAX/USDT"]
        
    def set_multiplier(self, multiplier: int):
        """Set the message rate multiplier."""
        self.multiplier = multiplier
        logger.info(f"Message rate set to {self.base_rate * multiplier} msg/sec")
    
    async def generate_ticker_update(self, symbol: str) -> Dict[str, Any]:
        """Generate a ticker update message."""
        base_price = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 100,
            "MATIC/USDT": 1,
            "AVAX/USDT": 35
        }.get(symbol, 100)
        
        price = base_price * (1 + random.gauss(0, 0.002))
        
        return {
            "type": "ticker",
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "data": {
                "bid": price - (price * 0.0001),
                "ask": price + (price * 0.0001),
                "last": price,
                "volume": random.uniform(100, 10000)
            }
        }
    
    async def generate_trade(self, symbol: str) -> Dict[str, Any]:
        """Generate a trade message."""
        base_price = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 100,
            "MATIC/USDT": 1,
            "AVAX/USDT": 35
        }.get(symbol, 100)
        
        return {
            "type": "trade",
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "data": {
                "id": str(int(time.time() * 1000000)),
                "price": base_price * (1 + random.gauss(0, 0.001)),
                "amount": random.uniform(0.001, 10),
                "side": random.choice(["buy", "sell"])
            }
        }
    
    async def generate_orderbook_update(self, symbol: str) -> Dict[str, Any]:
        """Generate an order book update message."""
        base_price = {
            "BTC/USDT": 50000,
            "ETH/USDT": 3000,
            "SOL/USDT": 100,
            "MATIC/USDT": 1,
            "AVAX/USDT": 35
        }.get(symbol, 100)
        
        bids = []
        asks = []
        
        for i in range(5):
            bid_price = base_price * (1 - 0.0001 * (i + 1))
            ask_price = base_price * (1 + 0.0001 * (i + 1))
            
            bids.append([bid_price, random.uniform(0.1, 50)])
            asks.append([ask_price, random.uniform(0.1, 50)])
        
        return {
            "type": "orderbook",
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "data": {
                "bids": bids,
                "asks": asks,
                "sequence": int(time.time() * 1000)
            }
        }
    
    async def generate_stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Generate a stream of messages at the configured rate."""
        messages_per_batch = max(1, self.base_rate * self.multiplier // 10)
        batch_delay = 0.1  # 100ms between batches
        
        while True:
            batch = []
            
            for _ in range(messages_per_batch):
                symbol = random.choice(self.symbols)
                message_type = random.choices(
                    ["ticker", "trade", "orderbook"],
                    weights=[0.3, 0.5, 0.2]
                )[0]
                
                if message_type == "ticker":
                    msg = await self.generate_ticker_update(symbol)
                elif message_type == "trade":
                    msg = await self.generate_trade(symbol)
                else:
                    msg = await self.generate_orderbook_update(symbol)
                
                batch.append(msg)
            
            for msg in batch:
                yield msg
            
            await asyncio.sleep(batch_delay)


class LoadGenerator:
    """Main load testing orchestrator."""
    
    def __init__(self, target_system: Optional[Callable] = None):
        """
        Initialize load generator.
        
        Args:
            target_system: Async function to process messages
        """
        self.target_system = target_system or self._default_processor
        self.message_generator = WebSocketMessageGenerator()
        self.metrics = LoadMetrics()
        self.running = False
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
    async def _default_processor(self, message: Dict[str, Any]) -> bool:
        """Default message processor for testing without a real system."""
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.001, 0.01))
        return True
    
    async def run_load_test(
        self,
        duration_seconds: int = 300,
        multiplier: int = 100,
        profile: LoadProfile = LoadProfile.NORMAL
    ):
        """
        Run load test with specified parameters.
        
        Args:
            duration_seconds: Test duration in seconds
            multiplier: Message rate multiplier (e.g., 100 for 100x)
            profile: Load profile to use
        """
        logger.info(
            "Starting load test",
            duration=duration_seconds,
            multiplier=multiplier,
            profile=profile.value
        )
        
        self.running = True
        self.metrics = LoadMetrics()
        self.message_generator.set_multiplier(multiplier)
        
        # Start tasks
        producer_task = asyncio.create_task(self._produce_messages(profile))
        consumer_task = asyncio.create_task(self._consume_messages())
        monitor_task = asyncio.create_task(self._monitor_performance())
        
        try:
            # Run for specified duration
            await asyncio.sleep(duration_seconds)
            
        finally:
            self.running = False
            
            # Wait for tasks to complete
            await asyncio.gather(
                producer_task,
                consumer_task,
                monitor_task,
                return_exceptions=True
            )
            
            self.metrics.end_time = datetime.now()
            
            # Generate report
            await self.generate_report()
    
    async def _produce_messages(self, profile: LoadProfile):
        """Produce messages according to load profile."""
        try:
            if profile == LoadProfile.GRADUAL_RAMP:
                # Gradually increase load
                for multiplier in range(1, 101, 10):
                    if not self.running:
                        break
                    self.message_generator.set_multiplier(multiplier)
                    await asyncio.sleep(10)
            
            elif profile == LoadProfile.NEWS_SPIKE:
                # Sudden spike after 30 seconds
                await asyncio.sleep(30)
                self.message_generator.set_multiplier(500)
                await asyncio.sleep(10)
                self.message_generator.set_multiplier(100)
            
            elif profile == LoadProfile.FLASH_CRASH:
                # Extreme spike for short period
                await asyncio.sleep(20)
                self.message_generator.set_multiplier(1000)
                await asyncio.sleep(5)
                self.message_generator.set_multiplier(100)
            
            # Generate messages
            async for message in self.message_generator.generate_stream():
                if not self.running:
                    break
                
                try:
                    # Track send time for latency calculation
                    message["send_time"] = time.time()
                    
                    # Try to put message in queue (non-blocking)
                    try:
                        self.message_queue.put_nowait(message)
                        self.metrics.messages_sent += 1
                    except asyncio.QueueFull:
                        self.metrics.messages_dropped += 1
                        logger.warning("Queue full, dropping message")
                        
                except Exception as e:
                    logger.error(f"Error producing message: {e}")
                    self.metrics.errors += 1
                    
        except Exception as e:
            logger.error(f"Producer failed: {e}")
    
    async def _consume_messages(self):
        """Consume and process messages."""
        batch_size = 100
        batch = []
        
        while self.running or not self.message_queue.empty():
            try:
                # Get message with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    continue
                
                # Process batch when full or on timeout
                if len(batch) >= batch_size or not self.running:
                    for msg in batch:
                        # Calculate latency
                        if "send_time" in msg:
                            latency_ms = (time.time() - msg["send_time"]) * 1000
                            self.metrics.add_latency(latency_ms)
                        
                        # Process message
                        try:
                            success = await self.target_system(msg)
                            if success:
                                self.metrics.messages_processed += 1
                            else:
                                self.metrics.errors += 1
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.metrics.errors += 1
                    
                    batch = []
                    
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                self.metrics.errors += 1
    
    async def _monitor_performance(self):
        """Monitor performance metrics during test."""
        import psutil
        
        window_size = 10  # 10 second window for rate calculation
        message_counts = []
        
        while self.running:
            try:
                # Track messages per second
                current_count = self.metrics.messages_processed
                message_counts.append((time.time(), current_count))
                
                # Keep only recent counts
                cutoff_time = time.time() - window_size
                message_counts = [
                    (t, c) for t, c in message_counts
                    if t > cutoff_time
                ]
                
                # Calculate current rate
                if len(message_counts) >= 2:
                    time_diff = message_counts[-1][0] - message_counts[0][0]
                    msg_diff = message_counts[-1][1] - message_counts[0][1]
                    
                    if time_diff > 0:
                        current_rate = msg_diff / time_diff
                        self.metrics.current_messages_per_second = current_rate
                        self.metrics.peak_messages_per_second = max(
                            self.metrics.peak_messages_per_second,
                            current_rate
                        )
                
                # System metrics
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent(interval=1)
                
                self.metrics.memory_usage_mb.append(memory_mb)
                self.metrics.cpu_usage_percent.append(cpu_percent)
                
                # Log status
                logger.info(
                    "Load test status",
                    messages_sent=self.metrics.messages_sent,
                    messages_processed=self.metrics.messages_processed,
                    messages_dropped=self.metrics.messages_dropped,
                    current_rate=f"{self.metrics.current_messages_per_second:.1f} msg/s",
                    avg_latency=f"{self.metrics.get_average_latency():.2f} ms",
                    memory_mb=f"{memory_mb:.1f}",
                    cpu_percent=f"{cpu_percent:.1f}"
                )
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1)
    
    async def generate_report(self):
        """Generate load test report."""
        report_path = Path("tests/stress/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"load_test_{timestamp}.json"
        
        report = {
            "test_type": "load_test",
            "metrics": self.metrics.to_dict(),
            "success": (
                self.metrics.get_percentile(95) < 100 and  # p95 < 100ms
                self.metrics.messages_dropped < self.metrics.messages_sent * 0.01  # <1% drop rate
            )
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report generated: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("LOAD TEST SUMMARY")
        print("=" * 60)
        print(f"Duration: {report['metrics']['duration_seconds']:.1f} seconds")
        print(f"Messages Sent: {self.metrics.messages_sent:,}")
        print(f"Messages Processed: {self.metrics.messages_processed:,}")
        print(f"Messages Dropped: {self.metrics.messages_dropped:,} ({report['metrics']['drop_rate']:.2f}%)")
        print(f"Errors: {self.metrics.errors:,} ({report['metrics']['error_rate']:.2f}%)")
        print(f"Throughput: {report['metrics']['throughput_msg_per_sec']:.1f} msg/s")
        print(f"Peak Rate: {self.metrics.peak_messages_per_second:.1f} msg/s")
        print(f"Average Latency: {report['metrics']['average_latency_ms']:.2f} ms")
        print(f"P50 Latency: {report['metrics']['p50_latency_ms']:.2f} ms")
        print(f"P95 Latency: {report['metrics']['p95_latency_ms']:.2f} ms")
        print(f"P99 Latency: {report['metrics']['p99_latency_ms']:.2f} ms")
        print(f"Test Result: {'PASS' if report['success'] else 'FAIL'}")
        print("=" * 60)


async def main():
    """Run load test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load testing for Genesis")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=100,
        help="Message rate multiplier (default: 100x)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=[p.value for p in LoadProfile],
        default=LoadProfile.NORMAL.value,
        help="Load profile to use"
    )
    
    args = parser.parse_args()
    
    generator = LoadGenerator()
    
    try:
        await generator.run_load_test(
            duration_seconds=args.duration,
            multiplier=args.multiplier,
            profile=LoadProfile(args.profile)
        )
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())