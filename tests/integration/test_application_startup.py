"""
Integration tests for application startup and initialization.

These tests verify that the complete application can start up properly
with various configurations and handle errors gracefully.
"""

import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genesis.__main__ import GenesisApplication


class TestApplicationStartup:
    """Test application initialization and startup."""

    @pytest.fixture
    def temp_env_file(self, tmp_path):
        """Create a temporary .env file for testing."""
        env_content = """
BINANCE_API_KEY=test_api_key_123
BINANCE_API_SECRET=test_api_secret_456
API_SECRET_KEY=test_secret_key_789_secure_random_string
TRADING_TIER=sniper
DEPLOYMENT_ENV=development
DEBUG=true
BINANCE_TESTNET=true
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        # Temporarily change to tmp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        yield env_file

        # Restore original directory
        os.chdir(original_cwd)

    def test_application_initialization_success(self, temp_env_file, monkeypatch):
        """Test successful application initialization."""
        # Create necessary directories
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        app = GenesisApplication()

        # Mock signal handlers to avoid interference
        with patch("signal.signal"):
            result = app.initialize()

        assert result is True
        assert app.settings is not None
        assert app.logger is not None
        assert app.settings.trading.trading_tier == "sniper"
        assert app.settings.deployment.deployment_env == "development"

    def test_application_initialization_missing_env(self):
        """Test application initialization with missing .env file."""
        # Ensure .env doesn't exist
        if Path(".env").exists():
            Path(".env").unlink()

        app = GenesisApplication()

        # Should fail gracefully
        with patch("signal.signal"):
            result = app.initialize()

        # Should handle missing required env vars
        assert result is False

    def test_application_run_with_mock_settings(self, temp_env_file):
        """Test application run method with mocked settings."""
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        app = GenesisApplication()

        with patch("signal.signal"):
            # Initialize first
            init_result = app.initialize()
            assert init_result is True

            # Test run (should return 0 for success)
            exit_code = app.run()
            assert exit_code == 0

    def test_application_shutdown(self, temp_env_file):
        """Test application shutdown process."""
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        app = GenesisApplication()

        with patch("signal.signal"):
            app.initialize()
            app.running = True

            # Test shutdown
            app.shutdown()

            assert app.running is False

    def test_signal_handler_setup(self, temp_env_file):
        """Test that signal handlers are properly configured."""
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        app = GenesisApplication()

        with patch("signal.signal") as mock_signal:
            app.setup_signal_handlers()

            # Verify SIGINT and SIGTERM handlers were set
            assert mock_signal.call_count >= 2

            # Check that SIGINT was registered
            sigint_calls = [
                call
                for call in mock_signal.call_args_list
                if call[0][0] == signal.SIGINT
            ]
            assert len(sigint_calls) > 0

            # Check that SIGTERM was registered
            sigterm_calls = [
                call
                for call in mock_signal.call_args_list
                if call[0][0] == signal.SIGTERM
            ]
            assert len(sigterm_calls) > 0

    def test_configuration_validation_warnings(self, tmp_path):
        """Test that configuration validation produces appropriate warnings."""
        # Create env with potential issues
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=short_key
DEPLOYMENT_ENV=production
DEBUG=true
BINANCE_TESTNET=true
TRADING_TIER=sniper
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            log_dir = Path(".genesis/logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            from config.settings import validate_configuration

            # This should produce warnings about debug in production
            report = validate_configuration()

            assert "DEBUG mode enabled in production!" in report["warnings"]
            assert "Using testnet in production environment" in report["warnings"]

        finally:
            os.chdir(original_cwd)


class TestEmergencyClosureIntegration:
    """Test emergency closure script integration."""

    def test_emergency_closure_initialization(self, tmp_path, monkeypatch):
        """Test emergency closure handler initialization."""
        # Setup test environment
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_api_secret_key_long_enough
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            from scripts.emergency_close import EmergencyClosureHandler

            handler = EmergencyClosureHandler(testnet=True)
            result = handler.initialize()

            assert result is True
            assert handler.testnet is True
            assert handler.logger is not None

        finally:
            os.chdir(original_cwd)

    def test_emergency_closure_confirmation_prompt(self):
        """Test emergency closure confirmation mechanism."""
        from scripts.emergency_close import EmergencyClosureHandler

        handler = EmergencyClosureHandler(testnet=True)

        # Test rejection
        with patch("builtins.input", return_value="no"):
            result = handler.get_confirmation()
            assert result is False

        # Test acceptance
        with patch("builtins.input", return_value="CLOSE ALL"):
            result = handler.get_confirmation()
            assert result is True

    def test_emergency_closure_mock_execution(self, tmp_path):
        """Test emergency closure with mock exchange."""
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_api_secret_key_long_enough
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            from scripts.emergency_close import EmergencyClosureHandler

            handler = EmergencyClosureHandler(testnet=True)

            # Initialize
            handler.initialize()

            # Test mock exchange connection
            exchange = handler.connect_exchange()
            assert exchange is not None
            assert exchange["connected"] is True
            assert exchange["mock"] is True

            # Test fetching mock positions
            positions = handler.fetch_open_positions(exchange)
            assert isinstance(positions, list)

            # Test fetching mock orders
            orders = handler.fetch_open_orders(exchange)
            assert isinstance(orders, list)

        finally:
            os.chdir(original_cwd)


class TestLoggingIntegration:
    """Test logging system integration."""

    def test_logging_setup(self, tmp_path):
        """Test logging system setup."""
        from genesis.utils.logger import LoggerType, get_logger, setup_logging

        log_dir = tmp_path / "logs"

        setup_logging(log_level="DEBUG", log_dir=log_dir, enable_json=True)

        # Verify log files were created
        assert (log_dir / "trading.log").exists()
        assert (log_dir / "audit.log").exists()
        assert (log_dir / "tilt.log").exists()
        assert (log_dir / "system.log").exists()

        # Test getting different logger types
        system_logger = get_logger("test", LoggerType.SYSTEM)
        assert system_logger is not None

        trading_logger = get_logger("test", LoggerType.TRADING)
        assert trading_logger is not None

    def test_sensitive_data_redaction(self, tmp_path):
        """Test that sensitive data is redacted in logs."""
        from genesis.utils.logger import (
            redact_sensitive_data,
        )

        # Test redaction function directly
        event_dict = {
            "message": "test",
            "api_key": "secret123",
            "password": "mypass",
            "normal_field": "visible",
            "nested": {"api_secret": "hidden", "public": "shown"},
        }

        redacted = redact_sensitive_data(None, "info", event_dict)

        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["password"] == "***REDACTED***"
        assert redacted["normal_field"] == "visible"
        assert redacted["nested"]["api_secret"] == "***REDACTED***"
        assert redacted["nested"]["public"] == "shown"

    def test_performance_logger(self, tmp_path):
        """Test performance logging context manager."""
        import time

        from genesis.utils.logger import PerformanceLogger, get_logger, setup_logging

        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)

        logger = get_logger("test")

        # Test successful operation
        with PerformanceLogger(logger, "test_operation", warn_threshold_ms=50):
            time.sleep(0.01)  # 10ms

        # Test slow operation (should log warning)
        with PerformanceLogger(logger, "slow_operation", warn_threshold_ms=5):
            time.sleep(0.01)  # 10ms > 5ms threshold

        # Test operation with error
        try:
            with PerformanceLogger(logger, "failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

    def test_log_context_manager(self, tmp_path):
        """Test log context manager for temporary context."""
        from genesis.utils.logger import LogContext, get_logger, setup_logging

        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir)

        logger = get_logger("test")

        # Use context manager to add temporary context
        with LogContext(logger, request_id="123", user="trader1") as ctx_logger:
            ctx_logger.info("test_message")
            # Context should be included in this message

        # Context should be removed after exiting
        logger.info("message_without_context")


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_tier_feature_validation(self, tmp_path):
        """Test that tier features are properly validated."""
        # Test Sniper tier restrictions
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_secret_key_long_enough_32chars
TRADING_TIER=sniper
ENABLE_MULTI_PAIR_TRADING=true
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            from config.settings import Settings

            # Should raise error for invalid feature
            with pytest.raises(ValueError, match="Multi-pair trading requires Hunter"):
                Settings()

        finally:
            os.chdir(original_cwd)

    def test_settings_singleton_pattern(self, tmp_path):
        """Test that settings use singleton pattern."""
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_secret_key_long_enough_32chars
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Clear any existing instance
            import config.settings
            from config.settings import get_settings

            config.settings._settings_instance = None

            # Get settings twice
            settings1 = get_settings()
            settings2 = get_settings()

            # Should be the same instance
            assert settings1 is settings2

            # Force reload should create new instance
            settings3 = get_settings(reload=True)
            assert settings3 is not settings2

        finally:
            os.chdir(original_cwd)

    def test_configuration_validation_report(self, tmp_path):
        """Test configuration validation report generation."""
        env_content = """
BINANCE_API_KEY=test_key
BINANCE_API_SECRET=test_secret
API_SECRET_KEY=test_secret_key_long_enough_32chars
TRADING_TIER=strategist
ENABLE_STATISTICAL_ARB=true
ALERT_ON_ERROR=true
"""
        env_file = tmp_path / ".env"
        env_file.write_text(env_content.strip())

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            from config.settings import validate_configuration

            report = validate_configuration()

            assert report["valid"] is True
            assert report["tier"] == "strategist"
            assert "Backup credentials not configured" in report["warnings"]
            assert "Error alerts enabled but SMTP not configured" in report["warnings"]

        finally:
            os.chdir(original_cwd)
