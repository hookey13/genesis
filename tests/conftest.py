"""
Pytest configuration and fixtures for Project GENESIS tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    test_env_vars = {
        "BINANCE_API_KEY": "test_api_key",
        "BINANCE_API_SECRET": "test_api_secret",
        "API_SECRET_KEY": "test_secret_key_for_testing",
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
def mock_settings():
    """Mock settings object for testing."""
    from config.settings import ExchangeSettings, Settings, TradingSettings

    settings = MagicMock(spec=Settings)
    settings.exchange = MagicMock(spec=ExchangeSettings)
    settings.exchange.binance_api_key = "test_key"
    settings.exchange.binance_api_secret = "test_secret"
    settings.exchange.binance_testnet = True

    settings.trading = MagicMock(spec=TradingSettings)
    settings.trading.trading_tier = "sniper"
    settings.trading.max_position_size_usdt = 100.0

    return settings


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
