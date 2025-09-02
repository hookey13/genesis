"""
Pytest configuration and fixtures for Project GENESIS tests.
Includes memory leak detection and performance regression tracking.
"""

import asyncio
import gc
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genesis.monitoring.memory_profiler import MemoryProfiler


@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Set up test environment variables."""
    test_env_vars = {
        "BINANCE_API_KEY": "test_api_key",
        "BINANCE_API_SECRET": "test_api_secret",
        "API_SECRET_KEY": "dGVzdF9zZWNyZXRfa2V5X2Zvcl90ZXN0aW5nX3B1cnBvc2VzX29ubHk=",  # Base64 encoded test key
        "BINANCE_TESTNET": "true",
        "DEPLOYMENT_ENV": "development",
        "DEBUG": "true",
        "TEST_MODE": "true",
        "USE_MOCK_EXCHANGE": "true",
        "DATABASE_URL": "sqlite:///test.db",
        "LOG_LEVEL": "DEBUG",
    }

    # Save original environment
    original_env = os.environ.copy()

    # Set test environment
    os.environ.update(test_env_vars)

    yield test_env_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings object for testing."""
    settings = MagicMock()

    # Mock exchange settings
    settings.exchange = MagicMock()
    settings.exchange.binance_api_key = MagicMock()
    settings.exchange.binance_api_key.get_secret_value = MagicMock(
        return_value="test_api_key"
    )
    settings.exchange.binance_api_secret = MagicMock()
    settings.exchange.binance_api_secret.get_secret_value = MagicMock(
        return_value="test_api_secret"
    )
    settings.exchange.binance_testnet = True
    settings.exchange.exchange_rate_limit = 1200

    # Mock trading settings
    settings.trading = MagicMock()
    settings.trading.trading_pairs = ["BTC/USDT", "ETH/USDT"]
    settings.trading.trading_tier = "sniper"
    settings.trading.max_position_size_usdt = Decimal("100.0")

    # Mock development settings
    settings.development = MagicMock()
    settings.development.use_mock_exchange = True

    return settings


@pytest.fixture
def market_data():
    """Load market data fixtures."""
    fixture_path = Path(__file__).parent / "fixtures" / "market_data.json"
    with open(fixture_path) as f:
        data = json.load(f)

    # Convert numeric values to Decimal
    def convert_to_decimal(obj):
        if isinstance(obj, dict):
            return {k: convert_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_decimal(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return Decimal(str(obj))
        return obj

    return convert_to_decimal(data)


@pytest_asyncio.fixture
async def mock_exchange():
    """Create a mock exchange instance."""
    from genesis.exchange.mock_exchange import MockExchange

    exchange = MockExchange(
        initial_balance={
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("5.0"),
        }
    )
    exchange.market_prices = {"BTC/USDT": Decimal("50000"), "ETH/USDT": Decimal("3000")}
    return exchange


@pytest_asyncio.fixture
async def gateway(mock_settings):
    """Create a BinanceGateway instance."""
    from genesis.exchange.gateway import BinanceGateway

    with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
        gateway = BinanceGateway(mock_mode=True)
        await gateway.initialize()
        yield gateway
        await gateway.close()


@pytest.fixture(scope="function")
def rate_limiter():
    """Create a RateLimiter instance."""
    from genesis.exchange.rate_limiter import RateLimiter

    return RateLimiter(max_weight=1200, window_seconds=60)


@pytest.fixture(scope="function")
def circuit_breaker():
    """Create a CircuitBreaker instance."""
    from genesis.exchange.circuit_breaker import CircuitBreaker

    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        failure_window_seconds=10,
        recovery_timeout_seconds=60,  # Long timeout to prevent automatic transitions during tests
    )


@pytest.fixture
def health_monitor():
    """Create a HealthMonitor instance."""
    from genesis.exchange.health_monitor import HealthMonitor

    return HealthMonitor(
        check_interval_seconds=5,
        window_size=10,
        degraded_threshold=0.95,
        unhealthy_threshold=0.80,
    )


@pytest.fixture
def mock_ccxt_exchange():
    """Create a mock ccxt exchange."""
    mock = AsyncMock()

    # Mock common methods
    mock.load_markets = AsyncMock(return_value=True)
    mock.fetch_balance = AsyncMock(
        return_value={
            "info": {
                "balances": {
                    "USDT": {"free": "10000", "locked": "0"},
                    "BTC": {"free": "0.5", "locked": "0"},
                }
            }
        }
    )
    mock.create_order = AsyncMock(
        return_value={
            "id": "12345",
            "info": {"orderId": "BINANCE_12345"},
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "price": 50000,
            "amount": 0.001,
            "filled": 0,
            "timestamp": 1700000000000,
        }
    )
    mock.cancel_order = AsyncMock(return_value={"status": "canceled"})
    mock.fetch_order = AsyncMock(
        return_value={
            "id": "12345",
            "info": {"orderId": "BINANCE_12345"},
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "filled",
            "price": 50000,
            "amount": 0.001,
            "filled": 0.001,
            "timestamp": 1700000000000,
        }
    )
    mock.fetch_order_book = AsyncMock(
        return_value={
            "bids": [[50000, 1.5], [49999, 2.0]],
            "asks": [[50001, 1.2], [50002, 1.8]],
            "timestamp": 1700000000000,
        }
    )
    mock.fetch_ohlcv = AsyncMock(
        return_value=[
            [1700000000000, 50000, 50100, 49900, 50050, 100],
            [1699999940000, 49950, 50050, 49850, 50000, 95],
        ]
    )
    mock.fetch_ticker = AsyncMock(
        return_value={
            "symbol": "BTC/USDT",
            "bid": 50000,
            "ask": 50001,
            "last": 50000.5,
            "baseVolume": 1500,
            "timestamp": 1700000000000,
        }
    )
    mock.fetch_time = AsyncMock(return_value=1700000000000)
    mock.close = AsyncMock(return_value=None)

    return mock


@pytest.fixture
def temp_project_structure(tmp_path):
    """Create a temporary project structure for testing."""
    # Create directories
    dirs = [
        "genesis/core",
        "genesis/engine",
        "genesis/strategies/sniper",
        "genesis/exchange",
        "genesis/data",
        "genesis/ui",
        "genesis/utils",
        "tests/unit",
        "tests/integration",
        "config",
        "scripts",
        "docker",
        ".genesis/data",
        ".genesis/logs",
    ]

    for dir_path in dirs:
        (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    init_files = [
        "genesis/__init__.py",
        "genesis/core/__init__.py",
        "genesis/engine/__init__.py",
        "genesis/strategies/__init__.py",
        "genesis/strategies/sniper/__init__.py",
        "config/__init__.py",
        "tests/__init__.py",
    ]

    for init_file in init_files:
        (tmp_path / init_file).touch()

    return tmp_path


@pytest.fixture
def sample_env_file(tmp_path):
    """Create a sample .env file for testing."""
    env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_api_secret
TRADING_TIER=sniper
MAX_POSITION_SIZE_USDT=100.0
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content.strip())
    return env_file


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean imports between tests to avoid module caching issues."""
    yield
    # Remove our modules from sys.modules
    modules_to_remove = [
        key for key in sys.modules.keys() if key.startswith(("genesis", "config"))
    ]
    for module in modules_to_remove:
        del sys.modules[module]


# ============= Pytest Configuration for Memory Leak Detection =============

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--memory-threshold",
        action="store",
        default="0.05",
        help="Memory growth threshold for leak detection (default: 5%)"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests including 48-hour stability tests"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "memory_intensive: marks tests that use significant memory")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ============= Memory Leak Detection Fixtures =============

class MemoryTracker:
    """Track memory usage for leak detection."""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.process = psutil.Process()
        self.start_memory = None
        self.end_memory = None
    
    def start(self):
        """Start memory tracking."""
        gc.collect()  # Force GC before measurement
        self.start_memory = self.process.memory_info().rss
    
    def stop(self):
        """Stop memory tracking and check for leaks."""
        gc.collect()  # Force GC before measurement
        self.end_memory = self.process.memory_info().rss
        
        if self.start_memory and self.end_memory:
            growth = (self.end_memory - self.start_memory) / self.start_memory
            if growth > self.threshold:
                raise AssertionError(
                    f"Memory leak detected: {growth:.2%} growth "
                    f"(threshold: {self.threshold:.2%})\n"
                    f"Start: {self.start_memory / 1024 / 1024:.2f} MB\n"
                    f"End: {self.end_memory / 1024 / 1024:.2f} MB"
                )


@pytest.fixture
def memory_tracker(request):
    """Fixture for memory leak detection."""
    threshold = float(request.config.getoption("--memory-threshold"))
    tracker = MemoryTracker(threshold=threshold)
    tracker.start()
    yield tracker
    tracker.stop()


@pytest.fixture
def assert_no_memory_leak():
    """Fixture to assert no memory leaks in a test."""
    def _assert(threshold: float = 0.05):
        tracker = MemoryTracker(threshold=threshold)
        tracker.start()
        return tracker
    return _assert


@pytest.fixture
def performance_baseline():
    """Load or create performance baseline for regression testing."""
    baseline_file = Path(".performance-baseline/baseline.json")
    
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
    else:
        baseline = {
            "memory_mb": 100,
            "cpu_percent": 50,
            "response_time_ms": 100
        }
    
    return baseline


@pytest.fixture
def assert_no_regression(performance_baseline):
    """Fixture to assert no performance regression."""
    def _assert(current_metrics: Dict[str, float], threshold: float = 0.15):
        """Assert that current metrics don't regress more than threshold."""
        for metric, current_value in current_metrics.items():
            if metric in performance_baseline:
                baseline_value = performance_baseline[metric]
                if baseline_value > 0:
                    regression = (current_value - baseline_value) / baseline_value
                    assert regression <= threshold, (
                        f"Performance regression in {metric}: "
                        f"{regression:.2%} increase (threshold: {threshold:.2%})\n"
                        f"Baseline: {baseline_value}\n"
                        f"Current: {current_value}"
                    )
    return _assert
