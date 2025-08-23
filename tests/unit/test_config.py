"""
Unit tests for configuration management.
"""

import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError


def test_settings_load_from_env(test_env):
    """Test that settings load correctly from environment variables."""
    from config.settings import Settings
    
    settings = Settings()
    
    # Check exchange settings
    assert settings.exchange.binance_api_key == "test_api_key"
    assert settings.exchange.binance_api_secret == "test_api_secret"
    assert settings.exchange.binance_testnet is True
    
    # Check deployment settings
    assert settings.deployment.deployment_env == "development"
    assert settings.development.debug is True
    assert settings.development.test_mode is True


def test_trading_settings_decimal_conversion(test_env):
    """Test that trading amounts are converted to Decimal."""
    from config.settings import TradingSettings
    
    settings = TradingSettings()
    
    assert isinstance(settings.max_position_size_usdt, Decimal)
    assert isinstance(settings.max_daily_loss_usdt, Decimal)
    assert isinstance(settings.stop_loss_percentage, Decimal)
    assert isinstance(settings.take_profit_percentage, Decimal)


def test_trading_pairs_parsing(test_env):
    """Test parsing of comma-separated trading pairs."""
    os.environ["TRADING_PAIRS"] = "BTC/USDT,ETH/USDT,BNB/USDT"
    
    from config.settings import TradingSettings
    
    settings = TradingSettings()
    
    assert settings.trading_pairs == ["BTC/USDT", "ETH/USDT", "BNB/USDT"]


def test_database_url_validation(test_env):
    """Test database URL validation."""
    from config.settings import DatabaseSettings
    
    # Valid SQLite URL
    os.environ["DATABASE_URL"] = "sqlite:///test.db"
    settings = DatabaseSettings()
    assert settings.database_url == "sqlite:///test.db"
    
    # Valid PostgreSQL URL
    os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/db"
    settings = DatabaseSettings()
    assert settings.database_url == "postgresql://user:pass@localhost/db"
    
    # Invalid URL
    os.environ["DATABASE_URL"] = "mysql://localhost/db"
    with pytest.raises(ValidationError):
        DatabaseSettings()


def test_tier_feature_validation(test_env):
    """Test that feature flags are validated against tier restrictions."""
    from config.settings import Settings
    
    # Sniper tier cannot use multi-pair trading
    os.environ["TRADING_TIER"] = "sniper"
    os.environ["ENABLE_MULTI_PAIR_TRADING"] = "true"
    
    with pytest.raises(ValueError, match="Multi-pair trading requires Hunter"):
        Settings()
    
    # Sniper tier cannot use statistical arbitrage
    os.environ["ENABLE_MULTI_PAIR_TRADING"] = "false"
    os.environ["ENABLE_STATISTICAL_ARB"] = "true"
    
    with pytest.raises(ValueError, match="Statistical arbitrage requires Strategist"):
        Settings()
    
    # Hunter tier can use multi-pair but not statistical arb
    os.environ["TRADING_TIER"] = "hunter"
    os.environ["ENABLE_MULTI_PAIR_TRADING"] = "true"
    os.environ["ENABLE_STATISTICAL_ARB"] = "true"
    
    with pytest.raises(ValueError, match="Statistical arbitrage requires Strategist"):
        Settings()
    
    # Strategist tier can use all features
    os.environ["TRADING_TIER"] = "strategist"
    settings = Settings()  # Should not raise
    assert settings.features.enable_multi_pair_trading is True
    assert settings.features.enable_statistical_arb is True


def test_logging_settings_paths(test_env):
    """Test that logging paths are correctly parsed."""
    from config.settings import LoggingSettings
    
    settings = LoggingSettings()
    
    assert isinstance(settings.log_file_path, Path)
    assert isinstance(settings.audit_log_path, Path)
    assert isinstance(settings.tilt_log_path, Path)
    assert settings.log_max_bytes == 10485760  # 10MB
    assert settings.log_backup_count == 5


def test_tilt_detection_thresholds(test_env):
    """Test tilt detection threshold validation."""
    from config.settings import TiltDetectionSettings
    
    settings = TiltDetectionSettings()
    
    assert settings.tilt_click_speed_threshold == 5.0
    assert settings.tilt_cancel_rate_threshold == 0.5
    assert settings.tilt_revenge_trade_threshold == 3
    assert settings.tilt_position_size_variance == 0.3
    
    # Test validation ranges
    os.environ["TILT_CANCEL_RATE_THRESHOLD"] = "1.5"  # Invalid: > 1
    with pytest.raises(ValidationError):
        TiltDetectionSettings()


def test_performance_settings(test_env):
    """Test performance tuning settings."""
    from config.settings import PerformanceSettings
    
    settings = PerformanceSettings()
    
    assert settings.connection_pool_size == 10
    assert settings.connection_timeout_seconds == 30
    assert settings.cache_ttl_seconds == 60
    assert settings.event_bus_buffer_size == 1000
    assert settings.event_bus_worker_threads == 4


def test_get_settings_function(test_env):
    """Test the get_settings helper function."""
    from config.settings import get_settings
    
    settings = get_settings()
    
    assert settings is not None
    assert settings.exchange.binance_api_key == "test_api_key"
    assert settings.trading.trading_tier == "sniper"


def test_settings_missing_required_fields():
    """Test that missing required fields raise appropriate errors."""
    from config.settings import ExchangeSettings, SecuritySettings
    
    # Clear required environment variables
    env_backup = os.environ.copy()
    os.environ.clear()
    
    try:
        # Missing API key should raise
        with pytest.raises(ValidationError):
            ExchangeSettings()
        
        # Missing security key should raise
        with pytest.raises(ValidationError):
            SecuritySettings()
    finally:
        # Restore environment
        os.environ.update(env_backup)


def test_deployment_environment_enum(test_env):
    """Test deployment environment enum validation."""
    from config.settings import DeploymentSettings
    
    # Valid environments
    for env in ["development", "staging", "production"]:
        os.environ["DEPLOYMENT_ENV"] = env
        settings = DeploymentSettings()
        assert settings.deployment_env == env
    
    # Invalid environment
    os.environ["DEPLOYMENT_ENV"] = "invalid"
    with pytest.raises(ValidationError):
        DeploymentSettings()


def test_notification_settings_optional_fields(test_env):
    """Test that notification settings handle optional fields correctly."""
    from config.settings import NotificationSettings
    
    # Remove optional fields from environment
    for key in ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "ALERT_EMAIL_TO"]:
        os.environ.pop(key, None)
    
    settings = NotificationSettings()
    
    assert settings.smtp_host is None
    assert settings.smtp_username is None
    assert settings.smtp_password is None
    assert settings.alert_email_to is None
    assert settings.alert_on_tilt is True  # Default value


def test_ui_settings(test_env):
    """Test UI configuration settings."""
    from config.settings import UISettings
    
    settings = UISettings()
    
    assert settings.ui_theme == "zen_garden"
    assert settings.ui_refresh_rate_ms == 100
    assert settings.ui_show_debug_panel is False


def test_time_sync_settings(test_env):
    """Test time synchronization settings."""
    from config.settings import TimeSyncSettings
    
    settings = TimeSyncSettings()
    
    assert settings.ntp_server == "pool.ntp.org"
    assert settings.time_sync_interval_seconds == 3600