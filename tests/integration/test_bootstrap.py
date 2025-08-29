"""
Integration tests for application bootstrap sequence.

These tests verify the complete startup flow including
configuration, database, and connectivity checks.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from genesis.__main__ import GenesisApplication


@pytest.fixture
def temp_database():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def test_env(temp_database):
    """Set up test environment variables."""
    env_vars = {
        "BINANCE_API_KEY": "test_api_key",
        "BINANCE_API_SECRET": "test_api_secret",
        "BINANCE_TESTNET": "true",
        "DATABASE_URL": temp_database,
        "LOG_LEVEL": "INFO",
        "LOG_FILE_PATH": ".genesis/test/logs/trading.log",
        "MAX_CLOCK_DRIFT_MS": "5000",
        "SYNC_CHECK_INTERVAL_SECONDS": "300",
        "NTP_SERVERS": "time.google.com,pool.ntp.org",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.mark.integration
class TestBootstrapSequence:
    """Test the complete application bootstrap sequence."""

    def test_successful_bootstrap(self, test_env):
        """Test successful application initialization."""
        app = GenesisApplication()

        with patch("genesis.__main__.asyncio.run") as mock_async_run:
            with patch.object(app, "_test_database_connection", return_value=True):
                with patch.object(app, "_run_migrations", return_value=True):
                    # Mock async functions to return successful results
                    mock_async_run.side_effect = [
                        True,  # REST connectivity
                        MagicMock(
                            is_acceptable=True, drift_ms=100, source="binance"
                        ),  # Clock drift
                        True,  # WebSocket connectivity
                    ]

                    result = app.initialize()
                    assert result is True
                    assert app.settings is not None
                    assert app.logger is not None

    def test_bootstrap_fails_on_database_error(self, test_env):
        """Test bootstrap fails gracefully on database connection error."""
        app = GenesisApplication()

        with patch.object(app, "_test_database_connection", return_value=False):
            result = app.initialize()
            assert result is False

    def test_bootstrap_fails_on_migration_error(self, test_env):
        """Test bootstrap fails gracefully on migration error."""
        app = GenesisApplication()

        with patch.object(app, "_test_database_connection", return_value=True):
            with patch.object(app, "_run_migrations", return_value=False):
                result = app.initialize()
                assert result is False

    def test_bootstrap_fails_on_clock_drift_exceeded(self, test_env):
        """Test bootstrap fails when clock drift exceeds threshold."""
        app = GenesisApplication()

        with patch("genesis.__main__.asyncio.run") as mock_async_run:
            with patch.object(app, "_test_database_connection", return_value=True):
                with patch.object(app, "_run_migrations", return_value=True):
                    # Mock async functions - clock drift exceeds threshold
                    mock_async_run.side_effect = [
                        True,  # REST connectivity
                        MagicMock(
                            is_acceptable=False,
                            drift_ms=6000,
                            source="binance",
                            max_clock_drift_ms=5000,
                        ),  # Clock drift exceeded
                    ]

                    result = app.initialize()
                    assert result is False

    def test_bootstrap_warns_on_websocket_failure(self, test_env, capsys):
        """Test bootstrap continues with warning on WebSocket failure."""
        app = GenesisApplication()

        with patch("genesis.__main__.asyncio.run") as mock_async_run:
            with patch.object(app, "_test_database_connection", return_value=True):
                with patch.object(app, "_run_migrations", return_value=True):
                    # Mock async functions - WebSocket fails but doesn't block
                    mock_async_run.side_effect = [
                        True,  # REST connectivity
                        MagicMock(
                            is_acceptable=True, drift_ms=100, source="binance"
                        ),  # Clock drift OK
                        False,  # WebSocket connectivity fails
                    ]

                    result = app.initialize()
                    assert result is True  # Should still succeed

                    captured = capsys.readouterr()
                    assert "WebSocket connection failed" in captured.out
                    assert "service degraded" in captured.out


@pytest.mark.integration
class TestDatabaseOperations:
    """Test database connectivity and migration operations."""

    def test_database_connection(self, temp_database):
        """Test database connection check."""
        app = GenesisApplication()
        app.settings = MagicMock()
        app.settings.database.database_url = temp_database

        result = app._test_database_connection()
        assert result is True

    def test_database_connection_with_invalid_url(self):
        """Test database connection with invalid URL."""
        app = GenesisApplication()
        app.settings = MagicMock()
        app.settings.database.database_url = "invalid://database/url"

        result = app._test_database_connection()
        assert result is False

    @pytest.mark.skipif(
        not Path("alembic.ini").exists(), reason="Alembic config not found"
    )
    def test_run_migrations(self, temp_database):
        """Test Alembic migration execution."""
        app = GenesisApplication()
        app.settings = MagicMock()
        app.settings.database.database_url = temp_database

        # This will create tables if models are properly configured
        result = app._run_migrations()
        assert result is True

        # Verify tables were created
        engine = create_engine(temp_database)
        with engine.connect() as conn:
            # Check if at least the alembic version table exists
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
                )
            )
            assert result.fetchone() is not None


@pytest.mark.integration
@pytest.mark.asyncio
class TestConnectivityChecks:
    """Test exchange connectivity checks."""

    async def test_rest_connectivity_success(self):
        """Test successful REST API connectivity check."""
        app = GenesisApplication()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session.get = AsyncMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock()

            result = await app._test_rest_connectivity()
            assert result is True

    async def test_rest_connectivity_failure(self):
        """Test REST API connectivity check failure."""
        app = GenesisApplication()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=TimeoutError())
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock()

            result = await app._test_rest_connectivity()
            assert result is False

    async def test_websocket_connectivity_success(self):
        """Test successful WebSocket connectivity check."""
        app = GenesisApplication()

        with patch("websockets.connect") as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.ping = AsyncMock()
            mock_connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_connect.return_value.__aexit__ = AsyncMock()

            result = await app._test_websocket_connectivity()
            assert result is True

    async def test_websocket_connectivity_failure(self):
        """Test WebSocket connectivity check failure."""
        app = GenesisApplication()

        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            result = await app._test_websocket_connectivity()
            assert result is False
