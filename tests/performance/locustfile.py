"""
Performance test suite using Locust for Project GENESIS.

Tests system performance under various load conditions including:
- Order execution latency
- Market data processing throughput
- Concurrent user handling
- API response times
"""

import random
import time
from decimal import Decimal
from typing import Dict, List

from locust import HttpUser, between, events, task
from locust.env import Environment
from locust.stats import stats_printer, stats_history

# Performance requirements from architecture
LATENCY_P99_REQUIREMENT_MS = 50  # <50ms p99 latency
THROUGHPUT_ORDERS_PER_SEC = 100  # 100 orders/second
MAX_CONCURRENT_USERS = 1000  # Support 1000 concurrent connections


class TradingUser(HttpUser):
    """Simulates a trading client interacting with the Genesis API."""
    
    wait_time = between(0.1, 1.0)  # Wait 100ms to 1s between requests
    
    def on_start(self):
        """Initialize user session."""
        # Authenticate and get session token
        self.client.headers.update({
            "Authorization": f"Bearer test-token-{self.environment.runner.user_count}",
            "Content-Type": "application/json",
        })
        
        # Track user metrics
        self.order_count = 0
        self.position_count = 0
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    
    @task(10)
    def get_ticker(self):
        """Fetch market ticker (high frequency operation)."""
        symbol = random.choice(self.symbols)
        with self.client.get(
            f"/api/v1/ticker/{symbol}",
            name="/api/v1/ticker/[symbol]",
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() * 1000 > LATENCY_P99_REQUIREMENT_MS:
                response.failure(f"Latency {response.elapsed.total_seconds()*1000:.2f}ms > {LATENCY_P99_REQUIREMENT_MS}ms")
            elif response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
            else:
                response.success()
    
    @task(5)
    def get_order_book(self):
        """Fetch order book (medium frequency)."""
        symbol = random.choice(self.symbols)
        depth = random.choice([5, 10, 20, 50])
        with self.client.get(
            f"/api/v1/orderbook/{symbol}?depth={depth}",
            name="/api/v1/orderbook/[symbol]",
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() * 1000 > LATENCY_P99_REQUIREMENT_MS * 2:
                response.failure(f"Latency too high for order book")
            else:
                response.success()
    
    @task(3)
    def place_order(self):
        """Place a trading order (critical path)."""
        order_data = {
            "symbol": random.choice(self.symbols),
            "side": random.choice(["BUY", "SELL"]),
            "type": "LIMIT" if random.random() < 0.7 else "MARKET",
            "quantity": str(random.uniform(0.001, 0.1)),
            "price": str(random.uniform(40000, 60000)) if random.random() < 0.7 else None,
        }
        
        start_time = time.time()
        with self.client.post(
            "/api/v1/orders",
            json=order_data,
            name="/api/v1/orders",
            catch_response=True
        ) as response:
            latency_ms = (time.time() - start_time) * 1000
            
            # Strict latency requirement for orders
            if latency_ms > LATENCY_P99_REQUIREMENT_MS:
                response.failure(f"Order latency {latency_ms:.2f}ms exceeds requirement")
            elif response.status_code not in [200, 201]:
                response.failure(f"Order failed with status {response.status_code}")
            else:
                self.order_count += 1
                response.success()
    
    @task(2)
    def get_positions(self):
        """Get current positions."""
        with self.client.get("/api/v1/positions") as response:
            if response.status_code == 200:
                self.position_count = len(response.json().get("positions", []))
    
    @task(2)
    def get_account_balance(self):
        """Get account balance."""
        self.client.get("/api/v1/account/balance")
    
    @task(1)
    def cancel_order(self):
        """Cancel an order (less frequent)."""
        if self.order_count > 0:
            order_id = f"order-{random.randint(1, self.order_count)}"
            self.client.delete(f"/api/v1/orders/{order_id}")
    
    @task(1)
    def get_pnl(self):
        """Get P&L summary."""
        self.client.get("/api/v1/account/pnl")
    
    def on_stop(self):
        """Clean up user session."""
        # Cancel all open orders
        if self.order_count > 0:
            self.client.delete("/api/v1/orders/all")


class MarketDataUser(HttpUser):
    """Simulates a user primarily consuming market data."""
    
    wait_time = between(0.05, 0.2)  # Very frequent requests
    
    def on_start(self):
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        self.client.headers.update({"Authorization": "Bearer market-data-token"})
    
    @task(20)
    def stream_ticker(self):
        """Simulate WebSocket-like ticker streaming."""
        for symbol in self.symbols:
            self.client.get(
                f"/api/v1/ticker/{symbol}",
                name="/api/v1/ticker/stream"
            )
    
    @task(10)
    def get_trades(self):
        """Get recent trades."""
        symbol = random.choice(self.symbols)
        self.client.get(f"/api/v1/trades/{symbol}?limit=50")
    
    @task(5)
    def get_klines(self):
        """Get candlestick data."""
        symbol = random.choice(self.symbols)
        interval = random.choice(["1m", "5m", "15m", "1h"])
        self.client.get(f"/api/v1/klines/{symbol}?interval={interval}&limit=100")


class HighFrequencyTrader(HttpUser):
    """Simulates a high-frequency trading bot."""
    
    wait_time = between(0.01, 0.05)  # Very fast requests
    host = "http://localhost:8000"
    
    def on_start(self):
        self.client.headers.update({
            "Authorization": "Bearer hft-token",
            "X-Client-Type": "HFT",
        })
        self.active_orders = []
    
    @task(30)
    def rapid_order_placement(self):
        """Place orders rapidly."""
        order_batch = []
        for _ in range(random.randint(1, 5)):
            order = {
                "symbol": "BTC/USDT",
                "side": random.choice(["BUY", "SELL"]),
                "type": "LIMIT",
                "quantity": str(random.uniform(0.001, 0.01)),
                "price": str(random.uniform(49000, 51000)),
            }
            order_batch.append(order)
        
        # Batch order placement
        with self.client.post(
            "/api/v1/orders/batch",
            json={"orders": order_batch},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.active_orders.extend(response.json().get("order_ids", []))
                response.success()
            else:
                response.failure(f"Batch order failed: {response.status_code}")
    
    @task(20)
    def modify_orders(self):
        """Modify existing orders."""
        if self.active_orders:
            order_id = random.choice(self.active_orders)
            new_price = str(random.uniform(49000, 51000))
            self.client.patch(
                f"/api/v1/orders/{order_id}",
                json={"price": new_price}
            )
    
    @task(10)
    def cancel_orders(self):
        """Cancel orders frequently."""
        if len(self.active_orders) > 10:
            # Cancel oldest orders
            to_cancel = self.active_orders[:5]
            for order_id in to_cancel:
                self.client.delete(f"/api/v1/orders/{order_id}")
                self.active_orders.remove(order_id)


# Custom event handlers for performance metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize performance tracking."""
    print(f"Starting performance test with target: {THROUGHPUT_ORDERS_PER_SEC} orders/sec")
    print(f"P99 latency requirement: <{LATENCY_P99_REQUIREMENT_MS}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Track individual request metrics."""
    # Custom tracking for critical endpoints
    if "orders" in name and response_time > LATENCY_P99_REQUIREMENT_MS:
        print(f"⚠️  Order endpoint latency violation: {response_time:.2f}ms")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Generate performance report."""
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)
    
    # Calculate p99 latency
    if environment.stats.total.response_times:
        p99 = environment.stats.total.get_response_time_percentile(0.99)
        print(f"P99 Latency: {p99:.2f}ms (Requirement: <{LATENCY_P99_REQUIREMENT_MS}ms)")
        
        if p99 <= LATENCY_P99_REQUIREMENT_MS:
            print("✅ P99 latency requirement PASSED")
        else:
            print(f"❌ P99 latency requirement FAILED ({p99:.2f}ms > {LATENCY_P99_REQUIREMENT_MS}ms)")
    
    # Calculate throughput
    if environment.stats.total.num_requests > 0:
        duration = time.time() - environment.stats.start_time
        throughput = environment.stats.total.num_requests / duration
        print(f"Throughput: {throughput:.2f} requests/sec")
        
        # Check order-specific throughput
        order_stats = environment.stats.get("/api/v1/orders", "POST")
        if order_stats:
            order_throughput = order_stats.num_requests / duration
            print(f"Order Throughput: {order_throughput:.2f} orders/sec")
            
            if order_throughput >= THROUGHPUT_ORDERS_PER_SEC:
                print("✅ Order throughput requirement PASSED")
            else:
                print(f"❌ Order throughput requirement FAILED ({order_throughput:.2f} < {THROUGHPUT_ORDERS_PER_SEC})")
    
    # Error rate
    error_rate = environment.stats.total.fail_ratio
    print(f"Error Rate: {error_rate:.2%}")
    
    if error_rate < 0.01:  # Less than 1% errors
        print("✅ Error rate acceptable")
    else:
        print(f"⚠️  High error rate: {error_rate:.2%}")
    
    print("="*50)


# Scenario definitions for different load patterns
class StepLoadShape(LoadShape):
    """Gradually increase load to find breaking point."""
    
    step_time = 60  # 60 seconds per step
    step_users = 10  # Add 10 users per step
    max_users = 100
    
    def tick(self):
        run_time = self.get_run_time()
        current_step = run_time // self.step_time
        
        if current_step * self.step_users > self.max_users:
            return None
        
        return (current_step * self.step_users, self.step_users)


class SpikeLoadShape(LoadShape):
    """Simulate sudden traffic spikes."""
    
    stages = [
        {"duration": 60, "users": 10},   # Normal load
        {"duration": 10, "users": 100},  # Sudden spike
        {"duration": 60, "users": 10},   # Return to normal
        {"duration": 10, "users": 200},  # Larger spike
        {"duration": 60, "users": 20},   # Slightly elevated
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["users"] / stage["duration"])
            run_time -= stage["duration"]
        
        return None


# Import load shape classes if running with custom scenarios
try:
    from locust import LoadShape
except ImportError:
    LoadShape = object  # Fallback for older Locust versions