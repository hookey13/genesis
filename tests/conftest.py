"""
Pytest configuration and fixtures for Project GENESIS tests.
"""

import asyncio
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
    settings.exchange.binance_api_key.get_secret_value = MagicMock(return_value="test_api_key")
    settings.exchange.binance_api_secret = MagicMock()
    settings.exchange.binance_api_secret.get_secret_value = MagicMock(return_value="test_api_secret")
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
            "ETH": Decimal("5.0")
        }
    )
    exchange.market_prices = {
        "BTC/USDT": Decimal("50000"),
        "ETH/USDT": Decimal("3000")
    }
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
        recovery_timeout_seconds=60  # Long timeout to prevent automatic transitions during tests
    )


@pytest.fixture
def health_monitor():
    """Create a HealthMonitor instance."""
    from genesis.exchange.health_monitor import HealthMonitor
    return HealthMonitor(
        check_interval_seconds=5,
        window_size=10,
        degraded_threshold=0.95,
        unhealthy_threshold=0.80
    )


@pytest.fixture
def mock_ccxt_exchange():
    """Create a mock ccxt exchange."""
    mock = AsyncMock()

    # Mock common methods
    mock.load_markets = AsyncMock(return_value=True)
    mock.fetch_balance = AsyncMock(return_value={
        "info": {
            "balances": {
                "USDT": {"free": "10000", "locked": "0"},
                "BTC": {"free": "0.5", "locked": "0"}
            }
        }
    })
    mock.create_order = AsyncMock(return_value={
        "id": "12345",
        "info": {"orderId": "BINANCE_12345"},
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "status": "open",
        "price": 50000,
        "amount": 0.001,
        "filled": 0,
        "timestamp": 1700000000000
    })
    mock.cancel_order = AsyncMock(return_value={"status": "canceled"})
    mock.fetch_order = AsyncMock(return_value={
        "id": "12345",
        "info": {"orderId": "BINANCE_12345"},
        "symbol": "BTC/USDT",
        "side": "buy",
        "type": "limit",
        "status": "filled",
        "price": 50000,
        "amount": 0.001,
        "filled": 0.001,
        "timestamp": 1700000000000
    })
    mock.fetch_order_book = AsyncMock(return_value={
        "bids": [[50000, 1.5], [49999, 2.0]],
        "asks": [[50001, 1.2], [50002, 1.8]],
        "timestamp": 1700000000000
    })
    mock.fetch_ohlcv = AsyncMock(return_value=[
        [1700000000000, 50000, 50100, 49900, 50050, 100],
        [1699999940000, 49950, 50050, 49850, 50000, 95]
    ])
    mock.fetch_ticker = AsyncMock(return_value={
        "symbol": "BTC/USDT",
        "bid": 50000,
        "ask": 50001,
        "last": 50000.5,
        "baseVolume": 1500,
        "timestamp": 1700000000000
    })
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
