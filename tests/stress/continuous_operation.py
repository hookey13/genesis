"""
24-hour continuous operation test framework for production validation.

This module provides comprehensive testing for system stability over extended periods
with paper trading mode to validate all trading operations without risk.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import psutil
import structlog
from unittest.mock import AsyncMock, MagicMock

logger = structlog.get_logger(__name__)


@dataclass
class StabilityMetrics:
    """Tracks stability metrics during continuous operation."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    uptime_seconds: float = 0
    total_errors: int = 0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    positions_opened: int = 0
    positions_closed: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    websocket_reconnects: int = 0
    database_errors: int = 0
    api_errors: int = 0
    
    def calculate_uptime(self) -> float:
        """Calculate total uptime in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_average_memory(self) -> float:
        """Get average memory usage in MB."""
        return sum(self.memory_usage_mb) / len(self.memory_usage_mb) if self.memory_usage_mb else 0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return max(self.memory_usage_mb) if self.memory_usage_mb else 0
    
    def get_average_cpu(self) -> float:
        """Get average CPU usage percentage."""
        return sum(self.cpu_usage_percent) / len(self.cpu_usage_percent) if self.cpu_usage_percent else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "uptime_seconds": self.calculate_uptime(),
            "uptime_hours": self.calculate_uptime() / 3600,
            "total_errors": self.total_errors,
            "average_memory_mb": self.get_average_memory(),
            "peak_memory_mb": self.get_peak_memory(),
            "average_cpu_percent": self.get_average_cpu(),
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "websocket_reconnects": self.websocket_reconnects,
            "database_errors": self.database_errors,
            "api_errors": self.api_errors,
            "error_rate_per_hour": (self.total_errors / (self.calculate_uptime() / 3600)) if self.calculate_uptime() > 0 else 0
        }


class MockExchangeData:
    """Generates realistic mock exchange data for paper trading."""
    
    def __init__(self):
        self.base_price = Decimal("50000")  # BTC starting price
        self.volatility = Decimal("0.002")  # 0.2% volatility
        self.trend = Decimal("0")  # Neutral trend
        
    async def get_ticker(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """Generate realistic ticker data."""
        # Add some random walk to price
        import random
        price_change = self.base_price * self.volatility * Decimal(str(random.gauss(0, 1)))
        self.base_price += price_change + (self.base_price * self.trend)
        
        return {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.now().isoformat(),
            "high": float(self.base_price * Decimal("1.01")),
            "low": float(self.base_price * Decimal("0.99")),
            "bid": float(self.base_price - Decimal("10")),
            "bidVolume": float(Decimal(str(random.uniform(0.1, 2.0)))),
            "ask": float(self.base_price + Decimal("10")),
            "askVolume": float(Decimal(str(random.uniform(0.1, 2.0)))),
            "last": float(self.base_price),
            "close": float(self.base_price),
            "baseVolume": float(Decimal(str(random.uniform(1000, 5000)))),
            "quoteVolume": float(self.base_price * Decimal(str(random.uniform(1000, 5000)))),
            "info": {}
        }
    
    async def get_order_book(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """Generate realistic order book data."""
        import random
        
        bids = []
        asks = []
        
        # Generate 20 levels of bids and asks
        for i in range(20):
            bid_price = self.base_price - Decimal(str(10 * (i + 1)))
            ask_price = self.base_price + Decimal(str(10 * (i + 1)))
            
            bids.append([
                float(bid_price),
                float(Decimal(str(random.uniform(0.1, 5.0))))
            ])
            asks.append([
                float(ask_price),
                float(Decimal(str(random.uniform(0.1, 5.0))))
            ])
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.now().isoformat(),
            "nonce": None
        }
    
    async def stream_trades(self, symbol: str = "BTC/USDT"):
        """Stream realistic trade data."""
        import random
        
        while True:
            # Generate 1-10 trades per second
            num_trades = random.randint(1, 10)
            
            for _ in range(num_trades):
                side = random.choice(["buy", "sell"])
                price = self.base_price + Decimal(str(random.uniform(-50, 50)))
                
                yield {
                    "id": str(int(time.time() * 1000000)),
                    "timestamp": int(time.time() * 1000),
                    "datetime": datetime.now().isoformat(),
                    "symbol": symbol,
                    "type": None,
                    "side": side,
                    "price": float(price),
                    "amount": float(Decimal(str(random.uniform(0.001, 1.0)))),
                    "cost": None,
                    "fee": None
                }
            
            await asyncio.sleep(1)


class ContinuousOperationTest:
    """24-hour continuous operation test harness."""
    
    def __init__(self, duration_hours: int = 24):
        self.duration_hours = duration_hours
        self.metrics = StabilityMetrics()
        self.mock_exchange = MockExchangeData()
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.trading_task: Optional[asyncio.Task] = None
        self.positions: Dict[str, Any] = {}
        self.orders: Dict[str, Any] = {}
        
    async def start(self):
        """Start the continuous operation test."""
        logger.info("Starting continuous operation test", duration_hours=self.duration_hours)
        
        self.running = True
        self.metrics.start_time = datetime.now()
        
        # Start monitoring and trading tasks
        self.monitor_task = asyncio.create_task(self._monitor_system())
        self.trading_task = asyncio.create_task(self._simulate_trading())
        
        # Run for specified duration
        end_time = datetime.now() + timedelta(hours=self.duration_hours)
        
        try:
            while datetime.now() < end_time and self.running:
                await asyncio.sleep(60)  # Check every minute
                
                # Log progress
                elapsed = (datetime.now() - self.metrics.start_time).total_seconds() / 3600
                remaining = self.duration_hours - elapsed
                logger.info(
                    "Test progress",
                    elapsed_hours=f"{elapsed:.2f}",
                    remaining_hours=f"{remaining:.2f}",
                    errors=self.metrics.total_errors
                )
        
        except Exception as e:
            logger.error("Test failed", error=str(e))
            self.metrics.total_errors += 1
            raise
        
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the continuous operation test."""
        logger.info("Stopping continuous operation test")
        
        self.running = False
        self.metrics.end_time = datetime.now()
        
        # Cancel monitoring and trading tasks
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.trading_task:
            self.trading_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.monitor_task,
            self.trading_task,
            return_exceptions=True
        )
        
        # Generate report
        await self.generate_report()
    
    async def _monitor_system(self):
        """Monitor system metrics during test."""
        while self.running:
            try:
                # Collect system metrics
                process = psutil.Process()
                
                # Memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.memory_usage_mb.append(memory_mb)
                
                # CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                self.metrics.cpu_usage_percent.append(cpu_percent)
                
                # Check for anomalies
                if memory_mb > 1000:  # Alert if memory exceeds 1GB
                    logger.warning("High memory usage detected", memory_mb=memory_mb)
                
                if cpu_percent > 80:  # Alert if CPU exceeds 80%
                    logger.warning("High CPU usage detected", cpu_percent=cpu_percent)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                self.metrics.total_errors += 1
                await asyncio.sleep(5)
    
    async def _simulate_trading(self):
        """Simulate paper trading operations."""
        import random
        
        while self.running:
            try:
                # Simulate various trading operations
                operation = random.choice([
                    "place_order",
                    "cancel_order",
                    "check_positions",
                    "get_ticker",
                    "get_orderbook"
                ])
                
                if operation == "place_order":
                    await self._place_mock_order()
                elif operation == "cancel_order":
                    await self._cancel_mock_order()
                elif operation == "check_positions":
                    await self._check_positions()
                elif operation == "get_ticker":
                    await self.mock_exchange.get_ticker()
                elif operation == "get_orderbook":
                    await self.mock_exchange.get_order_book()
                
                # Random delay between operations (100ms to 2s)
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Trading simulation error", error=str(e))
                self.metrics.total_errors += 1
                self.metrics.api_errors += 1
                await asyncio.sleep(1)
    
    async def _place_mock_order(self):
        """Place a mock order."""
        import random
        
        order_id = f"order_{int(time.time() * 1000000)}"
        ticker = await self.mock_exchange.get_ticker()
        
        order = {
            "id": order_id,
            "symbol": "BTC/USDT",
            "type": random.choice(["limit", "market"]),
            "side": random.choice(["buy", "sell"]),
            "price": ticker["last"],
            "amount": random.uniform(0.001, 0.1),
            "status": "open",
            "timestamp": time.time()
        }
        
        self.orders[order_id] = order
        self.metrics.orders_placed += 1
        
        # Simulate order fill after random delay
        await asyncio.sleep(random.uniform(0.5, 3.0))
        if order_id in self.orders and random.random() > 0.1:  # 90% fill rate
            self.orders[order_id]["status"] = "closed"
            self.metrics.orders_filled += 1
            
            # Create position if buy order
            if order["side"] == "buy":
                position_id = f"pos_{order_id}"
                self.positions[position_id] = {
                    "id": position_id,
                    "symbol": order["symbol"],
                    "amount": order["amount"],
                    "entry_price": order["price"],
                    "timestamp": time.time()
                }
                self.metrics.positions_opened += 1
    
    async def _cancel_mock_order(self):
        """Cancel a mock order."""
        open_orders = [
            oid for oid, order in self.orders.items()
            if order["status"] == "open"
        ]
        
        if open_orders:
            import random
            order_id = random.choice(open_orders)
            self.orders[order_id]["status"] = "cancelled"
            self.metrics.orders_cancelled += 1
    
    async def _check_positions(self):
        """Check and potentially close positions."""
        import random
        
        for position_id in list(self.positions.keys()):
            # Randomly close positions (5% chance)
            if random.random() < 0.05:
                del self.positions[position_id]
                self.metrics.positions_closed += 1
    
    async def validate_consistency(self) -> bool:
        """Validate position and order consistency."""
        try:
            # Check for orphaned orders
            orphaned_orders = [
                order for order in self.orders.values()
                if order["status"] == "open" and 
                (time.time() - order["timestamp"]) > 3600  # Open for more than 1 hour
            ]
            
            if orphaned_orders:
                logger.warning(f"Found {len(orphaned_orders)} orphaned orders")
                return False
            
            # Check position values
            for position in self.positions.values():
                if position["amount"] <= 0:
                    logger.error(f"Invalid position amount: {position}")
                    return False
            
            logger.info("Consistency validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Consistency validation error: {e}")
            return False
    
    async def generate_report(self):
        """Generate test report."""
        report_path = Path("tests/stress/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"continuous_operation_{timestamp}.json"
        
        # Validate consistency
        consistency_valid = await self.validate_consistency()
        
        report = {
            "test_type": "continuous_operation",
            "duration_hours": self.duration_hours,
            "consistency_valid": consistency_valid,
            "metrics": self.metrics.to_dict(),
            "final_positions": len(self.positions),
            "final_orders": len(self.orders),
            "success": self.metrics.total_errors == 0 and consistency_valid
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report generated: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("CONTINUOUS OPERATION TEST SUMMARY")
        print("=" * 60)
        print(f"Duration: {self.metrics.calculate_uptime() / 3600:.2f} hours")
        print(f"Total Errors: {self.metrics.total_errors}")
        print(f"Error Rate: {report['metrics']['error_rate_per_hour']:.2f} per hour")
        print(f"Average Memory: {self.metrics.get_average_memory():.2f} MB")
        print(f"Peak Memory: {self.metrics.get_peak_memory():.2f} MB")
        print(f"Average CPU: {self.metrics.get_average_cpu():.2f}%")
        print(f"Orders Placed: {self.metrics.orders_placed}")
        print(f"Orders Filled: {self.metrics.orders_filled}")
        print(f"Positions Opened: {self.metrics.positions_opened}")
        print(f"Positions Closed: {self.metrics.positions_closed}")
        print(f"Consistency Valid: {consistency_valid}")
        print(f"Test Result: {'PASS' if report['success'] else 'FAIL'}")
        print("=" * 60)


async def main():
    """Run 24-hour continuous operation test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="24-hour continuous operation test")
    parser.add_argument(
        "--duration",
        type=int,
        default=24,
        help="Test duration in hours (default: 24)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick 1-hour test instead of full duration"
    )
    
    args = parser.parse_args()
    
    duration = 1 if args.quick else args.duration
    
    test = ContinuousOperationTest(duration_hours=duration)
    
    try:
        await test.start()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        await test.stop()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await test.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())