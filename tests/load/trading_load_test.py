"""
Comprehensive Locust load testing suite for Genesis trading system.
Tests realistic trading patterns, order processing, and system limits.
"""

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import random
import json
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrader(FastHttpUser):
    """Base class for all trading user simulations."""
    
    abstract = True
    
    # Trading symbols to use in tests
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
    
    def on_start(self):
        """Initialize user session with authentication."""
        self.login()
        self.setup_trading_data()
        
    def login(self):
        """Authenticate user and obtain JWT token."""
        # Generate test user credentials
        self.username = f"trader_{random.randint(1, 10000)}"
        self.password = "TestPassword123!"
        
        # Attempt login with retry logic
        for attempt in range(3):
            try:
                response = self.client.post(
                    "/api/auth/login",
                    json={
                        "username": self.username,
                        "password": self.password,
                        "totp_code": "123456"  # Test TOTP code
                    },
                    catch_response=True
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get('access_token')
                    self.refresh_token = data.get('refresh_token')
                    self.headers = {'Authorization': f'Bearer {self.token}'}
                    response.success()
                    logger.info(f"User {self.username} logged in successfully")
                    break
                elif response.status_code == 404:
                    # Create user if not exists (for test environment)
                    self.register_user()
                else:
                    response.failure(f"Login failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                    
    def register_user(self):
        """Register a new test user."""
        response = self.client.post(
            "/api/auth/register",
            json={
                "username": self.username,
                "password": self.password,
                "email": f"{self.username}@test.genesis.com"
            }
        )
        
        if response.status_code == 201:
            logger.info(f"User {self.username} registered successfully")
            self.login()  # Login after registration
            
    def setup_trading_data(self):
        """Initialize trading-specific data for the user."""
        self.active_orders = []
        self.positions = {}
        self.balance = 10000.0  # Starting balance in USDT
        
    def refresh_auth_token(self):
        """Refresh JWT token when needed."""
        response = self.client.post(
            "/api/auth/refresh",
            json={"refresh_token": self.refresh_token},
            headers=self.headers
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get('access_token')
            self.headers = {'Authorization': f'Bearer {self.token}'}
            

class NormalTrader(BaseTrader):
    """Simulates normal trading behavior with reasonable patterns."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    weight = 60  # 60% of users will be normal traders
    
    @task(20)
    def place_market_order(self):
        """Place a market order with realistic parameters."""
        symbol = random.choice(self.symbols)
        side = random.choice(['buy', 'sell'])
        quantity = round(random.uniform(0.001, 0.1), 4)
        
        order_data = {
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'quantity': quantity,
            'client_order_id': f"{self.username}_{int(time.time() * 1000)}"
        }
        
        with self.client.post(
            "/api/trading/orders",
            json=order_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                order = response.json()
                self.active_orders.append(order['order_id'])
                
                # Validate response time
                if response.elapsed.total_seconds() > 0.050:
                    response.failure(f"Order too slow: {response.elapsed.total_seconds()}s")
                else:
                    response.success()
            else:
                response.failure(f"Order failed: {response.text}")
                
    @task(15)
    def place_limit_order(self):
        """Place a limit order with price near market."""
        symbol = random.choice(self.symbols)
        side = random.choice(['buy', 'sell'])
        quantity = round(random.uniform(0.001, 0.05), 4)
        
        # Get current price (simplified for testing)
        base_price = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'BNBUSDT': 400}
        price = base_price.get(symbol, 100) * random.uniform(0.995, 1.005)
        
        order_data = {
            'symbol': symbol,
            'side': side,
            'type': 'limit',
            'quantity': quantity,
            'price': round(price, 2),
            'time_in_force': 'GTC',
            'client_order_id': f"{self.username}_{int(time.time() * 1000)}"
        }
        
        with self.client.post(
            "/api/trading/orders",
            json=order_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                order = response.json()
                self.active_orders.append(order['order_id'])
                response.success()
            else:
                response.failure(f"Limit order failed: {response.text}")
                
    @task(10)
    def get_positions(self):
        """Retrieve current positions."""
        with self.client.get(
            "/api/trading/positions",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.positions = response.json()
                
                # Validate response time for position queries
                if response.elapsed.total_seconds() > 0.010:
                    response.failure(f"Position query too slow: {response.elapsed.total_seconds()}s")
                else:
                    response.success()
            else:
                response.failure(f"Failed to get positions: {response.status_code}")
                
    @task(8)
    def get_orderbook(self):
        """Fetch orderbook for a symbol."""
        symbol = random.choice(self.symbols)
        
        with self.client.get(
            f"/api/market/orderbook/{symbol}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate orderbook structure
                if 'bids' in data and 'asks' in data:
                    response.success()
                else:
                    response.failure("Invalid orderbook structure")
            else:
                response.failure(f"Failed to get orderbook: {response.status_code}")
                
    @task(5)
    def cancel_order(self):
        """Cancel a random active order."""
        if not self.active_orders:
            return
            
        order_id = random.choice(self.active_orders)
        
        with self.client.delete(
            f"/api/trading/orders/{order_id}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 204]:
                self.active_orders.remove(order_id)
                response.success()
            elif response.status_code == 404:
                # Order already filled or cancelled
                if order_id in self.active_orders:
                    self.active_orders.remove(order_id)
                response.success()
            else:
                response.failure(f"Failed to cancel order: {response.status_code}")
                
    @task(3)
    def get_account_balance(self):
        """Check account balance."""
        with self.client.get(
            "/api/account/balance",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.balance = data.get('usdt_balance', 0)
                response.success()
            else:
                response.failure(f"Failed to get balance: {response.status_code}")
                
    @task(2)
    def get_trade_history(self):
        """Retrieve recent trade history."""
        with self.client.get(
            "/api/trading/trades?limit=50",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get trade history: {response.status_code}")


class HighFrequencyTrader(BaseTrader):
    """Aggressive high-frequency trading simulation."""
    
    wait_time = between(0.01, 0.1)  # Very short wait times
    weight = 20  # 20% of users will be HFT
    
    def on_start(self):
        """Initialize HFT-specific settings."""
        super().on_start()
        self.max_orders = 50  # Maximum concurrent orders
        self.order_pairs = []  # Track order pairs for quick cancellation
        
    @task(30)
    def rapid_order_placement(self):
        """Place orders rapidly with immediate cancellation intent."""
        if len(self.active_orders) >= self.max_orders:
            return
            
        symbol = random.choice(self.symbols[:2])  # Focus on liquid pairs
        
        # Place buy and sell orders simultaneously
        for side in ['buy', 'sell']:
            quantity = round(random.uniform(0.001, 0.01), 6)
            
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': 'limit',
                'quantity': quantity,
                'price': round(random.uniform(40000, 50000), 2) if symbol == 'BTCUSDT' else round(random.uniform(2800, 3200), 2),
                'time_in_force': 'IOC',  # Immediate or cancel
                'client_order_id': f"hft_{self.username}_{int(time.time() * 1000000)}"
            }
            
            response = self.client.post(
                "/api/trading/orders",
                json=order_data,
                headers=self.headers
            )
            
            if response.status_code == 201:
                order = response.json()
                self.active_orders.append(order['order_id'])
                
    @task(20)
    def quick_cancellation(self):
        """Cancel orders quickly after placement."""
        if len(self.active_orders) > 10:
            # Cancel multiple orders at once
            orders_to_cancel = random.sample(
                self.active_orders, 
                min(5, len(self.active_orders))
            )
            
            for order_id in orders_to_cancel:
                self.client.delete(
                    f"/api/trading/orders/{order_id}",
                    headers=self.headers
                )
                if order_id in self.active_orders:
                    self.active_orders.remove(order_id)
                    
    @task(10)
    def market_microstructure_probe(self):
        """Probe market microstructure with small orders."""
        symbol = 'BTCUSDT'  # Focus on most liquid pair
        
        # Place multiple small orders at different price levels
        for i in range(5):
            side = 'buy' if i % 2 == 0 else 'sell'
            price_offset = i * 10
            
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': 'limit',
                'quantity': 0.001,
                'price': 45000 + price_offset if side == 'buy' else 45000 - price_offset,
                'time_in_force': 'FOK',  # Fill or kill
                'client_order_id': f"probe_{self.username}_{int(time.time() * 1000000)}_{i}"
            }
            
            self.client.post(
                "/api/trading/orders",
                json=order_data,
                headers=self.headers
            )


class MarketDataConsumer(BaseTrader):
    """User focused on consuming market data streams."""
    
    wait_time = between(0.5, 2)
    weight = 15  # 15% of users will be data consumers
    
    @task(40)
    def stream_orderbook_updates(self):
        """Simulate orderbook streaming consumption."""
        symbols = random.sample(self.symbols, min(3, len(self.symbols)))
        
        for symbol in symbols:
            with self.client.get(
                f"/api/market/orderbook/{symbol}?depth=50",
                headers=self.headers,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate data freshness
                    if 'timestamp' in data:
                        data_age = time.time() - data['timestamp'] / 1000
                        if data_age > 1.0:
                            response.failure(f"Stale orderbook data: {data_age}s old")
                        else:
                            response.success()
                else:
                    response.failure(f"Failed to get orderbook stream: {response.status_code}")
                    
    @task(30)
    def get_ticker_data(self):
        """Fetch ticker data for multiple symbols."""
        with self.client.get(
            "/api/market/tickers",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get tickers: {response.status_code}")
                
    @task(20)
    def get_kline_data(self):
        """Retrieve historical kline/candlestick data."""
        symbol = random.choice(self.symbols)
        interval = random.choice(['1m', '5m', '15m', '1h'])
        
        with self.client.get(
            f"/api/market/klines/{symbol}?interval={interval}&limit=100",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate kline data structure
                if isinstance(data, list) and len(data) > 0:
                    response.success()
                else:
                    response.failure("Invalid kline data structure")
            else:
                response.failure(f"Failed to get klines: {response.status_code}")
                
    @task(10)
    def get_market_depth(self):
        """Get aggregated market depth."""
        symbol = random.choice(self.symbols)
        
        with self.client.get(
            f"/api/market/depth/{symbol}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get market depth: {response.status_code}")


class StressTestUser(BaseTrader):
    """User designed to stress test the system."""
    
    wait_time = between(0.001, 0.01)  # Minimal wait time
    weight = 5  # 5% of users for stress testing
    
    @task(50)
    def burst_orders(self):
        """Send burst of orders to test rate limiting."""
        burst_size = random.randint(10, 20)
        
        for _ in range(burst_size):
            order_data = {
                'symbol': random.choice(self.symbols),
                'side': random.choice(['buy', 'sell']),
                'type': 'market',
                'quantity': 0.001,
                'client_order_id': f"stress_{self.username}_{int(time.time() * 1000000)}"
            }
            
            # Fire and forget - don't wait for response
            self.client.post(
                "/api/trading/orders",
                json=order_data,
                headers=self.headers,
                name="/api/trading/orders_burst"
            )
            
    @task(30)
    def large_order_attempt(self):
        """Attempt to place unusually large orders."""
        order_data = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'type': 'market',
            'quantity': random.uniform(100, 1000),  # Large quantity
            'client_order_id': f"large_{self.username}_{int(time.time() * 1000)}"
        }
        
        with self.client.post(
            "/api/trading/orders",
            json=order_data,
            headers=self.headers,
            catch_response=True,
            name="/api/trading/orders_large"
        ) as response:
            # Expect rejection for risk limits
            if response.status_code == 400:
                response.success()  # Proper rejection is success
            elif response.status_code == 201:
                response.failure("Large order should have been rejected")
            else:
                response.failure(f"Unexpected response: {response.status_code}")
                
    @task(20)
    def invalid_request_flood(self):
        """Send invalid requests to test error handling."""
        invalid_requests = [
            {'symbol': 'INVALID', 'side': 'buy', 'type': 'market'},
            {'symbol': 'BTCUSDT', 'side': 'invalid', 'type': 'market', 'quantity': 0.1},
            {'symbol': 'BTCUSDT', 'side': 'buy', 'type': 'invalid', 'quantity': 0.1},
            {'symbol': 'BTCUSDT', 'side': 'buy', 'type': 'limit'},  # Missing price
            {},  # Empty request
        ]
        
        request = random.choice(invalid_requests)
        
        with self.client.post(
            "/api/trading/orders",
            json=request,
            headers=self.headers,
            catch_response=True,
            name="/api/trading/orders_invalid"
        ) as response:
            # Should properly reject invalid requests
            if response.status_code in [400, 422]:
                response.success()
            else:
                response.failure(f"Invalid request not properly rejected: {response.status_code}")


# Event handlers for distributed testing
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    logger.info("Load test starting...")
    logger.info(f"Target host: {environment.host}")
    logger.info(f"Total users: {environment.parsed_options.users if environment.parsed_options else 'Not specified'}")
    

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Clean up after test completion."""
    logger.info("Load test completed")
    
    # Log final statistics
    logger.info(f"Total requests: {environment.stats.total.num_requests}")
    logger.info(f"Total failures: {environment.stats.total.num_failures}")
    logger.info(f"Average response time: {environment.stats.total.avg_response_time}ms")
    logger.info(f"Requests per second: {environment.stats.total.current_rps}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Custom request event handler for detailed metrics."""
    if exception:
        logger.error(f"Request failed - {name}: {exception}")
    elif response_time > 100:  # Log slow requests
        logger.warning(f"Slow request - {name}: {response_time}ms")


# Performance test scenarios
class PerformanceScenarios:
    """Predefined test scenarios for different load patterns."""
    
    @staticmethod
    def normal_trading_day():
        """Simulate a normal trading day pattern."""
        return {
            'users': 100,
            'spawn_rate': 2,
            'run_time': '1h',
            'user_classes': [NormalTrader, MarketDataConsumer]
        }
    
    @staticmethod
    def high_volume_period():
        """Simulate high volume trading period."""
        return {
            'users': 500,
            'spawn_rate': 10,
            'run_time': '30m',
            'user_classes': [NormalTrader, HighFrequencyTrader, MarketDataConsumer]
        }
    
    @staticmethod
    def stress_test():
        """Maximum stress test scenario."""
        return {
            'users': 1000,
            'spawn_rate': 50,
            'run_time': '15m',
            'user_classes': [NormalTrader, HighFrequencyTrader, StressTestUser, MarketDataConsumer]
        }
    
    @staticmethod
    def endurance_test():
        """Long-running stability test."""
        return {
            'users': 200,
            'spawn_rate': 5,
            'run_time': '48h',
            'user_classes': [NormalTrader, MarketDataConsumer]
        }