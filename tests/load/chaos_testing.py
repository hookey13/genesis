"""
Chaos engineering tests for Genesis trading system.
Simulates failures, network issues, and edge cases to test resilience.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosType(Enum):
    """Types of chaos to inject."""
    NETWORK_PARTITION = "network_partition"
    DATABASE_FAILURE = "database_failure"
    SERVICE_CRASH = "service_crash"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LATENCY_INJECTION = "latency_injection"
    PACKET_LOSS = "packet_loss"
    CLOCK_SKEW = "clock_skew"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_BREACH = "rate_limit_breach"
    DATA_CORRUPTION = "data_corruption"


class ChaosScenario:
    """Base class for chaos scenarios."""
    
    def __init__(self, name: str, duration: float, probability: float = 1.0):
        self.name = name
        self.duration = duration
        self.probability = probability
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.affected_users = 0
        self.errors_caused = 0
        
    def should_trigger(self) -> bool:
        """Determine if chaos should trigger based on probability."""
        return random.random() < self.probability
        
    async def inject(self, user: 'ChaosUser'):
        """Inject chaos for a specific user."""
        if not self.should_trigger():
            return
            
        self.start_time = time.time()
        self.affected_users += 1
        
        logger.info(f"[CHAOS] Injecting {self.name} for user {user.username}")
        
        try:
            await self._execute_chaos(user)
        except Exception as e:
            logger.error(f"[CHAOS] Error during {self.name}: {e}")
            self.errors_caused += 1
        finally:
            self.end_time = time.time()
            
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Execute the actual chaos logic (override in subclasses)."""
        raise NotImplementedError
        
    def get_metrics(self) -> Dict:
        """Get chaos scenario metrics."""
        return {
            'name': self.name,
            'duration': self.duration,
            'probability': self.probability,
            'affected_users': self.affected_users,
            'errors_caused': self.errors_caused,
            'execution_time': (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        }


class NetworkPartitionChaos(ChaosScenario):
    """Simulate network partition by failing requests."""
    
    def __init__(self, duration: float = 30):
        super().__init__("Network Partition", duration, probability=0.1)
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Make all requests fail for duration."""
        original_client = user.client
        
        class FailingClient:
            def __getattr__(self, name):
                def method(*args, **kwargs):
                    raise Exception("Network partition - connection failed")
                return method
                
        user.client = FailingClient()
        await asyncio.sleep(self.duration)
        user.client = original_client


class LatencyInjectionChaos(ChaosScenario):
    """Inject artificial latency into requests."""
    
    def __init__(self, min_latency: float = 0.5, max_latency: float = 5.0, duration: float = 60):
        super().__init__("Latency Injection", duration, probability=0.2)
        self.min_latency = min_latency
        self.max_latency = max_latency
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Add random latency to requests."""
        original_request = user.client.request
        
        async def delayed_request(*args, **kwargs):
            delay = random.uniform(self.min_latency, self.max_latency)
            await asyncio.sleep(delay)
            return await original_request(*args, **kwargs)
            
        user.client.request = delayed_request
        await asyncio.sleep(self.duration)
        user.client.request = original_request


class AuthenticationFailureChaos(ChaosScenario):
    """Simulate authentication failures by invalidating tokens."""
    
    def __init__(self, duration: float = 10):
        super().__init__("Authentication Failure", duration, probability=0.05)
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Invalidate user's auth token."""
        original_token = user.token
        user.token = "invalid_token_chaos"
        user.headers = {'Authorization': f'Bearer {user.token}'}
        
        await asyncio.sleep(self.duration)
        
        # Attempt to re-authenticate
        user.login()


class RateLimitChaos(ChaosScenario):
    """Trigger rate limiting by sending burst requests."""
    
    def __init__(self, burst_size: int = 100):
        super().__init__("Rate Limit Breach", 0, probability=0.1)
        self.burst_size = burst_size
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Send burst of requests to trigger rate limiting."""
        tasks = []
        
        for _ in range(self.burst_size):
            task = user.client.post(
                "/api/trading/orders",
                json={
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'type': 'market',
                    'quantity': 0.001
                },
                headers=user.headers,
                name="/api/trading/orders_burst"
            )
            tasks.append(task)
            
        # Fire all requests simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count rate limit responses
        rate_limited = sum(1 for r in results if hasattr(r, 'status_code') and r.status_code == 429)
        logger.info(f"[CHAOS] Rate limit triggered: {rate_limited}/{self.burst_size} requests limited")


class DataCorruptionChaos(ChaosScenario):
    """Send corrupted data to test validation."""
    
    def __init__(self):
        super().__init__("Data Corruption", 0, probability=0.15)
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Send various forms of corrupted data."""
        corruption_types = [
            # Invalid JSON
            lambda: "not json{]",
            
            # Missing required fields
            lambda: json.dumps({'side': 'buy'}),
            
            # Invalid field types
            lambda: json.dumps({
                'symbol': 12345,  # Should be string
                'side': True,  # Should be 'buy' or 'sell'
                'quantity': "not_a_number"
            }),
            
            # Extreme values
            lambda: json.dumps({
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'type': 'market',
                'quantity': 999999999.99999999
            }),
            
            # SQL injection attempt
            lambda: json.dumps({
                'symbol': "'; DROP TABLE orders; --",
                'side': 'buy',
                'type': 'market',
                'quantity': 0.1
            }),
            
            # XSS attempt
            lambda: json.dumps({
                'symbol': '<script>alert("XSS")</script>',
                'side': 'buy',
                'type': 'market',
                'quantity': 0.1
            })
        ]
        
        for corruption in corruption_types:
            try:
                data = corruption()
                
                response = await user.client.post(
                    "/api/trading/orders",
                    data=data,
                    headers={**user.headers, 'Content-Type': 'application/json'},
                    name="/api/trading/orders_corrupted"
                )
                
                # System should reject with 400 or 422
                if response.status_code not in [400, 422]:
                    logger.error(f"[CHAOS] Corrupted data not rejected! Status: {response.status_code}")
                    self.errors_caused += 1
                    
            except Exception as e:
                logger.debug(f"[CHAOS] Expected error handling corruption: {e}")


class ResourceExhaustionChaos(ChaosScenario):
    """Attempt to exhaust system resources."""
    
    def __init__(self):
        super().__init__("Resource Exhaustion", 0, probability=0.05)
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Try to exhaust various resources."""
        
        # 1. Connection exhaustion - open many connections
        connections = []
        try:
            for _ in range(100):
                conn = user.client.get(
                    "/api/health",
                    headers=user.headers,
                    stream=True,  # Keep connection open
                    name="/api/health_exhaust"
                )
                connections.append(conn)
                
        except Exception as e:
            logger.info(f"[CHAOS] Connection exhaustion handled: {e}")
            
        finally:
            # Clean up connections
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
                    
        # 2. Memory exhaustion - request large data
        try:
            response = await user.client.get(
                "/api/trading/trades?limit=1000000",  # Request excessive data
                headers=user.headers,
                name="/api/trades_large"
            )
            
            if response.status_code == 200:
                logger.warning("[CHAOS] System allowed excessive data request!")
                self.errors_caused += 1
                
        except Exception as e:
            logger.info(f"[CHAOS] Large data request handled: {e}")


class ClockSkewChaos(ChaosScenario):
    """Simulate clock skew by manipulating timestamps."""
    
    def __init__(self):
        super().__init__("Clock Skew", 0, probability=0.1)
        
    async def _execute_chaos(self, user: 'ChaosUser'):
        """Send requests with skewed timestamps."""
        
        # Future timestamp
        future_time = datetime.now() + timedelta(hours=1)
        
        response = await user.client.post(
            "/api/trading/orders",
            json={
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'type': 'market',
                'quantity': 0.001,
                'timestamp': int(future_time.timestamp() * 1000)
            },
            headers=user.headers,
            name="/api/orders_future_time"
        )
        
        if response.status_code == 201:
            logger.warning("[CHAOS] System accepted future timestamp!")
            self.errors_caused += 1
            
        # Past timestamp (expired)
        past_time = datetime.now() - timedelta(hours=1)
        
        response = await user.client.post(
            "/api/trading/orders",
            json={
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'type': 'market',
                'quantity': 0.001,
                'timestamp': int(past_time.timestamp() * 1000)
            },
            headers=user.headers,
            name="/api/orders_past_time"
        )
        
        if response.status_code == 201:
            logger.warning("[CHAOS] System accepted expired timestamp!")
            self.errors_caused += 1


class ChaosUser(FastHttpUser):
    """User that injects chaos during testing."""
    
    wait_time = between(0.5, 2)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.username = f"chaos_user_{random.randint(1, 10000)}"
        self.chaos_scenarios: List[ChaosScenario] = []
        self.chaos_enabled = True
        
    def on_start(self):
        """Initialize user and chaos scenarios."""
        self.login()
        self.setup_chaos_scenarios()
        
    def login(self):
        """Authenticate user."""
        for attempt in range(3):
            try:
                response = self.client.post(
                    "/api/auth/login",
                    json={
                        "username": self.username,
                        "password": "ChaosTest123!",
                        "totp_code": "123456"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get('access_token')
                    self.headers = {'Authorization': f'Bearer {self.token}'}
                    break
                    
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to login chaos user: {e}")
                    
    def setup_chaos_scenarios(self):
        """Initialize chaos scenarios to inject."""
        self.chaos_scenarios = [
            NetworkPartitionChaos(),
            LatencyInjectionChaos(),
            AuthenticationFailureChaos(),
            RateLimitChaos(),
            DataCorruptionChaos(),
            ResourceExhaustionChaos(),
            ClockSkewChaos()
        ]
        
    @task(1)
    async def inject_random_chaos(self):
        """Randomly inject a chaos scenario."""
        if not self.chaos_enabled or not self.chaos_scenarios:
            return
            
        scenario = random.choice(self.chaos_scenarios)
        await scenario.inject(self)
        
    @task(5)
    def normal_trading(self):
        """Perform normal trading operations between chaos."""
        try:
            # Place order
            response = self.client.post(
                "/api/trading/orders",
                json={
                    'symbol': random.choice(['BTCUSDT', 'ETHUSDT']),
                    'side': random.choice(['buy', 'sell']),
                    'type': 'market',
                    'quantity': round(random.uniform(0.001, 0.01), 4)
                },
                headers=self.headers,
                catch_response=True
            )
            
            if response.status_code != 201:
                response.failure(f"Normal order failed during chaos: {response.status_code}")
            else:
                response.success()
                
        except Exception as e:
            logger.debug(f"Expected failure during chaos: {e}")
            
    @task(3)
    def check_system_health(self):
        """Check if system is still healthy during chaos."""
        try:
            response = self.client.get(
                "/api/health",
                headers=self.headers,
                catch_response=True,
                timeout=5
            )
            
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            else:
                response.success()
                
        except Exception as e:
            logger.warning(f"Health check failed during chaos: {e}")


class ChaosOrchestrator:
    """Orchestrate complex chaos scenarios."""
    
    def __init__(self):
        self.scenarios: List[ChaosScenario] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_errors = 0
        
    async def run_cascade_failure(self, users: List[ChaosUser]):
        """Simulate cascading failures across services."""
        logger.info("[CHAOS] Starting cascade failure scenario")
        
        # Phase 1: Database slowdown
        for user in users[:len(users)//3]:
            await LatencyInjectionChaos(min_latency=2, max_latency=5).inject(user)
            
        await asyncio.sleep(10)
        
        # Phase 2: Authentication service failure
        for user in users[len(users)//3:2*len(users)//3]:
            await AuthenticationFailureChaos().inject(user)
            
        await asyncio.sleep(10)
        
        # Phase 3: Complete network partition
        for user in users[2*len(users)//3:]:
            await NetworkPartitionChaos().inject(user)
            
        logger.info("[CHAOS] Cascade failure scenario complete")
        
    async def run_recovery_test(self, users: List[ChaosUser]):
        """Test system recovery after chaos."""
        logger.info("[CHAOS] Starting recovery test")
        
        # Inject multiple failures
        tasks = []
        for user in users:
            scenario = random.choice([
                NetworkPartitionChaos(duration=5),
                LatencyInjectionChaos(duration=5),
                AuthenticationFailureChaos(duration=5)
            ])
            tasks.append(scenario.inject(user))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for chaos to end
        await asyncio.sleep(10)
        
        # Check system recovery
        logger.info("[CHAOS] Checking system recovery...")
        
        recovery_success = 0
        recovery_failed = 0
        
        for user in users:
            try:
                response = user.client.get(
                    "/api/health",
                    headers=user.headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    recovery_success += 1
                else:
                    recovery_failed += 1
                    
            except Exception:
                recovery_failed += 1
                
        logger.info(f"[CHAOS] Recovery results: {recovery_success} success, {recovery_failed} failed")
        
        return {
            'recovery_success': recovery_success,
            'recovery_failed': recovery_failed,
            'recovery_rate': recovery_success / len(users) if users else 0
        }


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize chaos testing."""
    logger.info("=" * 80)
    logger.info("CHAOS ENGINEERING TEST STARTING")
    logger.info("Warning: This test will intentionally cause failures!")
    logger.info("=" * 80)
    

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report chaos testing results."""
    logger.info("=" * 80)
    logger.info("CHAOS ENGINEERING TEST COMPLETE")
    logger.info("=" * 80)
    
    # Collect and report chaos metrics
    if hasattr(environment, 'chaos_metrics'):
        metrics = environment.chaos_metrics
        
        logger.info("\nChaos Injection Summary:")
        for scenario_type, count in metrics.items():
            logger.info(f"  {scenario_type}: {count} injections")
            
        logger.info("\nSystem Resilience:")
        total_requests = environment.stats.total.num_requests
        total_failures = environment.stats.total.num_failures
        
        if total_requests > 0:
            success_rate = (1 - total_failures / total_requests) * 100
            logger.info(f"  Success rate under chaos: {success_rate:.2f}%")
            logger.info(f"  Total requests: {total_requests}")
            logger.info(f"  Total failures: {total_failures}")
            
            if success_rate >= 99:
                logger.info("  ✓ EXCELLENT: System maintained 99%+ success rate")
            elif success_rate >= 95:
                logger.info("  ✓ GOOD: System maintained 95%+ success rate")
            elif success_rate >= 90:
                logger.info("  ⚠ FAIR: System maintained 90%+ success rate")
            else:
                logger.error("  ✗ POOR: System success rate below 90%")


if __name__ == "__main__":
    logger.info("Chaos testing module loaded. Use with Locust for chaos injection.")