"""
Unit tests for settings validation and configuration.
"""

import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from config.settings import (
    ExchangeSettings,
    Settings,
    TimeSyncSettings,
    TradingSettings,
    TradingTier,
    ValidationError,
)


class TestTimeSyncSettings:
    """Tests for time synchronization settings."""

    def test_parse_ntp_servers_from_string(self):
        """Test parsing comma-separated NTP servers."""
        with patch.dict(
            os.environ,
            {"NTP_SERVERS": "time1.google.com,pool.ntp.org,time.windows.com"},
        ):
            settings = TimeSyncSettings()
            assert settings.ntp_servers == [
                "time1.google.com",
                "pool.ntp.org",
                "time.windows.com",
            ]

    def test_parse_ntp_servers_handles_whitespace(self):
        """Test parsing handles whitespace correctly."""
        with patch.dict(
            os.environ,
            {"NTP_SERVERS": "time1.google.com , pool.ntp.org , time.windows.com"},
        ):
            settings = TimeSyncSettings()
            assert settings.ntp_servers == [
                "time1.google.com",
                "pool.ntp.org",
                "time.windows.com",
            ]

    def test_default_ntp_servers(self):
        """Test default NTP servers are set."""
        settings = TimeSyncSettings()
        assert settings.ntp_servers == ["time.google.com", "pool.ntp.org"]

    def test_max_clock_drift_validation(self):
        """Test max clock drift must be positive."""
        with patch.dict(os.environ, {"MAX_CLOCK_DRIFT_MS": "0"}):
            with pytest.raises(ValidationError):
                TimeSyncSettings()

    def test_sync_interval_validation(self):
        """Test sync interval must be positive."""
        with patch.dict(os.environ, {"SYNC_CHECK_INTERVAL_SECONDS": "-1"}):
            with pytest.raises(ValidationError):
                TimeSyncSettings()


class TestTradingSettings:
    """Tests for trading settings validation."""

    def test_parse_trading_pairs_from_string(self):
        """Test parsing comma-separated trading pairs."""
        with patch.dict(os.environ, {"TRADING_PAIRS": "BTC/USDT,ETH/USDT,SOL/USDT"}):
            settings = TradingSettings()
            assert settings.trading_pairs == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_decimal_conversion(self):
        """Test that numeric values are converted to Decimal."""
        with patch.dict(
            os.environ,
            {"MAX_POSITION_SIZE_USDT": "1000.50", "MAX_DAILY_LOSS_USDT": "100.25"},
        ):
            settings = TradingSettings()
            assert settings.max_position_size_usdt == Decimal("1000.50")
            assert settings.max_daily_loss_usdt == Decimal("100.25")

    def test_tier_validation(self):
        """Test trading tier enum validation."""
        with patch.dict(os.environ, {"TRADING_TIER": "invalid_tier"}):
            with pytest.raises(ValidationError):
                TradingSettings()


class TestExchangeSettings:
    """Tests for exchange settings validation."""

    def test_api_credential_placeholder_rejection(self):
        """Test that placeholder API credentials are rejected."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "your_api_key_here",
                "BINANCE_API_SECRET": "valid_secret",
            },
        ):
            with pytest.raises(ValidationError, match="placeholder value"):
                ExchangeSettings()

    def test_empty_api_credentials_rejected(self):
        """Test that empty API credentials are rejected."""
        with patch.dict(
            os.environ, {"BINANCE_API_KEY": "", "BINANCE_API_SECRET": "valid_secret"}
        ):
            with pytest.raises(ValidationError):
                ExchangeSettings()

    def test_rate_limit_bounds(self):
        """Test rate limit must be within bounds."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "valid_key",
                "BINANCE_API_SECRET": "valid_secret",
                "EXCHANGE_RATE_LIMIT": "5000",
            },
        ):
            with pytest.raises(ValidationError):
                ExchangeSettings()


class TestSettingsRedaction:
    """Tests for configuration redaction."""

    def test_redacted_dict_masks_sensitive_values(self):
        """Test that sensitive values are redacted."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "actual_api_key",
                "BINANCE_API_SECRET": "actual_secret",
                "API_SECRET_KEY": "jwt_secret",
                "SMTP_PASSWORD": "email_password",
            },
        ):
            settings = Settings()
            redacted = settings.redacted_dict()

            # Check exchange credentials are redacted
            assert redacted["exchange"]["binance_api_key"] == "***REDACTED***"
            assert redacted["exchange"]["binance_api_secret"] == "***REDACTED***"

            # Check security keys are redacted
            assert redacted["security"]["api_secret_key"] == "***REDACTED***"

            # Check non-sensitive values are not redacted
            assert redacted["trading"]["trading_tier"] == "sniper"
            assert isinstance(redacted["trading"]["max_position_size_usdt"], str)

    def test_redacted_dict_preserves_structure(self):
        """Test that redacted dict preserves configuration structure."""
        settings = Settings()
        redacted = settings.redacted_dict()

        # Check all main sections exist
        assert "exchange" in redacted
        assert "trading" in redacted
        assert "risk" in redacted
        assert "database" in redacted
        assert "logging" in redacted
        assert "time_sync" in redacted
        assert "features" in redacted


class TestTierFeatureValidation:
    """Tests for tier-based feature validation."""

    def test_sniper_tier_restrictions(self):
        """Test that Sniper tier cannot enable advanced features."""
        with patch.dict(
            os.environ, {"TRADING_TIER": "sniper", "ENABLE_MULTI_PAIR_TRADING": "true"}
        ):
            with pytest.raises(ValueError, match="Multi-pair trading requires Hunter"):
                Settings()

    def test_hunter_tier_allows_multi_pair(self):
        """Test that Hunter tier can enable multi-pair trading."""
        with patch.dict(
            os.environ,
            {
                "TRADING_TIER": "hunter",
                "ENABLE_MULTI_PAIR_TRADING": "true",
                "ENABLE_STATISTICAL_ARB": "false",
            },
        ):
            settings = Settings()
            assert settings.features.enable_multi_pair_trading is True

    def test_strategist_tier_allows_all_features(self):
        """Test that Strategist tier can enable all features."""
        with patch.dict(
            os.environ,
            {
                "TRADING_TIER": "strategist",
                "ENABLE_MULTI_PAIR_TRADING": "true",
                "ENABLE_STATISTICAL_ARB": "true",
            },
        ):
            settings = Settings()
            assert settings.features.enable_multi_pair_trading is True
            assert settings.features.enable_statistical_arb is True


class TestConfigurationValidation:
    """Tests for overall configuration validation."""

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        from config.settings import validate_configuration

        with patch("config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.trading.trading_tier = TradingTier.SNIPER
            mock_settings.deployment.deployment_env = "development"
            mock_settings.exchange.binance_testnet = True
            mock_settings.development.debug = False
            mock_settings.development.use_mock_exchange = False
            mock_settings.security.session_timeout_minutes = 30
            mock_settings.backup.do_spaces_key = "key"
            mock_settings.notifications.smtp_host = "smtp.gmail.com"
            mock_settings.notifications.alert_on_error = False
            mock_get_settings.return_value = mock_settings

            report = validate_configuration()
            assert report["valid"] is True
            assert report["tier"] == TradingTier.SNIPER
            assert report["environment"] == "development"

    def test_validate_configuration_warnings(self):
        """Test configuration validation generates appropriate warnings."""
        from config.settings import DeploymentEnv, validate_configuration

        with patch("config.settings.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.trading.trading_tier = TradingTier.SNIPER
            mock_settings.deployment.deployment_env = DeploymentEnv.PRODUCTION
            mock_settings.exchange.binance_testnet = (
                True  # Warning: testnet in production
            )
            mock_settings.development.debug = True  # Warning: debug in production
            mock_settings.development.use_mock_exchange = False
            mock_settings.security.session_timeout_minutes = (
                120  # Warning: long timeout
            )
            mock_settings.backup.do_spaces_key = None  # Warning: no backup
            mock_settings.notifications.smtp_host = None
            mock_settings.notifications.alert_on_error = (
                True  # Warning: alerts but no SMTP
            )
            mock_get_settings.return_value = mock_settings

            report = validate_configuration()
            assert report["valid"] is True
            assert len(report["warnings"]) >= 4  # Should have multiple warnings
