"""
Configuration management using Pydantic for Project GENESIS.

This module handles all environment variables and configuration validation,
ensuring type safety and providing defaults where appropriate.

Key Features:
    - Type-safe configuration with Pydantic validation
    - Tier-based feature validation
    - Environment-specific settings
    - Comprehensive error messages for misconfiguration
    - Secure handling of sensitive credentials

Example:
    >>> from config.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.trading.trading_tier)
    TradingTier.SNIPER
"""

import logging
import sys
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings

# Configure module logger
logger = logging.getLogger(__name__)


class TradingTier(str, Enum):
    """Trading tier levels."""

    SNIPER = "sniper"
    HUNTER = "hunter"
    STRATEGIST = "strategist"


class DeploymentEnv(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExchangeSettings(BaseSettings):
    """
    Exchange-related configuration.

    Attributes:
        binance_api_key: API key for Binance exchange (stored securely)
        binance_api_secret: API secret for Binance exchange (stored securely)
        binance_testnet: Whether to use testnet (True) or mainnet (False)
        exchange_rate_limit: Maximum requests per minute to respect rate limits

    Raises:
        ValidationError: If API credentials are invalid or rate limit is non-positive
    """

    binance_api_key: SecretStr = Field(..., env="BINANCE_API_KEY", min_length=1)
    binance_api_secret: SecretStr = Field(..., env="BINANCE_API_SECRET", min_length=1)
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")
    exchange_rate_limit: int = Field(1200, env="EXCHANGE_RATE_LIMIT", gt=0, le=2400)

    @field_validator("binance_api_key", "binance_api_secret")
    @classmethod
    def validate_api_credentials(cls, v: SecretStr, info) -> SecretStr:
        """Validate API credentials are not placeholder values."""
        if v and v.get_secret_value() in [
            "your_api_key_here",
            "your_api_secret_here",
            "",
        ]:
            raise ValueError(
                f"{info.field_name} contains placeholder value - please set actual credentials"
            )
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class TradingSettings(BaseSettings):
    """Trading configuration."""

    trading_tier: TradingTier = Field(TradingTier.SNIPER, env="TRADING_TIER")
    max_position_size_usdt: Decimal = Field(
        Decimal("100.0"), env="MAX_POSITION_SIZE_USDT", gt=0
    )
    max_daily_loss_usdt: Decimal = Field(
        Decimal("50.0"), env="MAX_DAILY_LOSS_USDT", gt=0
    )
    stop_loss_percentage: Decimal = Field(
        Decimal("2.0"), env="STOP_LOSS_PERCENTAGE", gt=0
    )
    take_profit_percentage: Decimal = Field(
        Decimal("3.0"), env="TAKE_PROFIT_PERCENTAGE", gt=0
    )
    trading_pairs: list[str] = Field(["BTC/USDT"], env="TRADING_PAIRS")

    @field_validator("trading_pairs", mode="before")
    @classmethod
    def parse_trading_pairs(cls, v):
        """Parse comma-separated trading pairs."""
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(",")]
        return v

    @field_validator(
        "max_position_size_usdt",
        "max_daily_loss_usdt",
        "stop_loss_percentage",
        "take_profit_percentage",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Ensure values are Decimal type."""
        return Decimal(str(v))

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class RiskSettings(BaseSettings):
    """Risk management configuration."""
    
    position_risk_percent: Decimal = Field(
        Decimal("5.0"), env="POSITION_RISK_PERCENT", gt=0, le=100
    )
    minimum_position_size: Decimal = Field(
        Decimal("10.0"), env="MINIMUM_POSITION_SIZE", gt=0
    )
    account_sync_interval: int = Field(
        60, env="ACCOUNT_SYNC_INTERVAL", gt=0
    )
    daily_loss_limit_sniper: Decimal = Field(
        Decimal("25.0"), env="DAILY_LOSS_LIMIT_SNIPER", gt=0
    )
    daily_loss_limit_hunter: Decimal = Field(
        Decimal("100.0"), env="DAILY_LOSS_LIMIT_HUNTER", gt=0
    )
    daily_loss_limit_strategist: Decimal = Field(
        Decimal("500.0"), env="DAILY_LOSS_LIMIT_STRATEGIST", gt=0
    )
    correlation_alert_threshold: Decimal = Field(
        Decimal("0.7"), env="CORRELATION_ALERT_THRESHOLD", ge=0, le=1
    )
    max_positions_sniper: int = Field(1, env="MAX_POSITIONS_SNIPER", gt=0)
    max_positions_hunter: int = Field(3, env="MAX_POSITIONS_HUNTER", gt=0)
    max_positions_strategist: int = Field(5, env="MAX_POSITIONS_STRATEGIST", gt=0)
    
    @field_validator(
        "position_risk_percent",
        "minimum_position_size",
        "daily_loss_limit_sniper",
        "daily_loss_limit_hunter",
        "daily_loss_limit_strategist",
        "correlation_alert_threshold"
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Ensure values are Decimal type."""
        return Decimal(str(v))
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    database_url: str = Field("sqlite:///.genesis/data/genesis.db", env="DATABASE_URL")

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not (v.startswith("sqlite://") or v.startswith("postgresql://")):
            raise ValueError("Database URL must be SQLite or PostgreSQL")
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_file_path: Path = Field(Path(".genesis/logs/trading.log"), env="LOG_FILE_PATH")
    audit_log_path: Path = Field(Path(".genesis/logs/audit.log"), env="AUDIT_LOG_PATH")
    tilt_log_path: Path = Field(Path(".genesis/logs/tilt.log"), env="TILT_LOG_PATH")
    log_max_bytes: int = Field(10485760, env="LOG_MAX_BYTES", gt=0)  # 10MB
    log_backup_count: int = Field(5, env="LOG_BACKUP_COUNT", ge=0)

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class TiltDetectionSettings(BaseSettings):
    """Tilt detection thresholds."""

    tilt_click_speed_threshold: float = Field(
        5.0, env="TILT_CLICK_SPEED_THRESHOLD", gt=0
    )
    tilt_cancel_rate_threshold: float = Field(
        0.5, env="TILT_CANCEL_RATE_THRESHOLD", ge=0, le=1
    )
    tilt_revenge_trade_threshold: int = Field(
        3, env="TILT_REVENGE_TRADE_THRESHOLD", gt=0
    )
    tilt_position_size_variance: float = Field(
        0.3, env="TILT_POSITION_SIZE_VARIANCE", ge=0, le=1
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class BackupSettings(BaseSettings):
    """Backup configuration."""

    do_spaces_key: Optional[str] = Field(None, env="DO_SPACES_KEY")
    do_spaces_secret: Optional[str] = Field(None, env="DO_SPACES_SECRET")
    do_spaces_region: str = Field("sgp1", env="DO_SPACES_REGION")
    do_spaces_bucket: str = Field("genesis-backups", env="DO_SPACES_BUCKET")
    backup_schedule: str = Field("0 */4 * * *", env="BACKUP_SCHEDULE")  # Cron format

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class DeploymentSettings(BaseSettings):
    """Deployment configuration."""

    server_host: str = Field("0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(8000, env="SERVER_PORT", gt=0, le=65535)
    deployment_env: DeploymentEnv = Field(
        DeploymentEnv.DEVELOPMENT, env="DEPLOYMENT_ENV"
    )
    supervisor_config_path: Path = Field(
        Path("/etc/supervisor/conf.d/genesis.conf"), env="SUPERVISOR_CONFIG_PATH"
    )
    process_auto_restart: bool = Field(True, env="PROCESS_AUTO_RESTART")
    process_restart_delay_seconds: int = Field(
        5, env="PROCESS_RESTART_DELAY_SECONDS", ge=0
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class NotificationSettings(BaseSettings):
    """Notification configuration."""

    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: int = Field(587, env="SMTP_PORT", gt=0, le=65535)
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    alert_email_to: Optional[str] = Field(None, env="ALERT_EMAIL_TO")

    alert_on_tilt: bool = Field(True, env="ALERT_ON_TILT")
    alert_on_daily_loss_limit: bool = Field(True, env="ALERT_ON_DAILY_LOSS_LIMIT")
    alert_on_error: bool = Field(True, env="ALERT_ON_ERROR")
    alert_on_tier_change: bool = Field(True, env="ALERT_ON_TIER_CHANGE")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class DevelopmentSettings(BaseSettings):
    """Development and testing settings."""

    debug: bool = Field(False, env="DEBUG")
    test_mode: bool = Field(False, env="TEST_MODE")
    use_mock_exchange: bool = Field(False, env="USE_MOCK_EXCHANGE")
    enable_profiling: bool = Field(False, env="ENABLE_PROFILING")
    profile_output_dir: Path = Field(
        Path(".genesis/profiling"), env="PROFILE_OUTPUT_DIR"
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class SecuritySettings(BaseSettings):
    """
    Security configuration for internal API and session management.

    Attributes:
        api_secret_key: Secret key for API authentication (min 32 chars recommended)
        jwt_expiration_hours: JWT token expiration time in hours
        session_timeout_minutes: User session timeout in minutes

    Raises:
        ValidationError: If secret key is too short or timing values are invalid
    """

    api_secret_key: SecretStr = Field(..., env="API_SECRET_KEY", min_length=16)
    jwt_expiration_hours: int = Field(
        24, env="JWT_EXPIRATION_HOURS", gt=0, le=168
    )  # Max 1 week
    session_timeout_minutes: int = Field(
        30, env="SESSION_TIMEOUT_MINUTES", gt=5, le=1440
    )  # 5min-24hr

    @field_validator("api_secret_key")
    @classmethod
    def validate_secret_key_strength(cls, v: SecretStr) -> SecretStr:
        """Ensure secret key is strong enough."""
        secret = v.get_secret_value()
        if len(secret) < 32:
            logger.warning("API secret key is shorter than recommended 32 characters")
        if secret in [
            "generate_a_secure_random_key_here",
            "test_secret_key_for_testing",
        ]:
            raise ValueError(
                "API secret key contains placeholder value - generate a secure key"
            )
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class FeatureFlags(BaseSettings):
    """Feature flags for conditional functionality."""

    enable_paper_trading: bool = Field(True, env="ENABLE_PAPER_TRADING")
    enable_backtesting: bool = Field(False, env="ENABLE_BACKTESTING")
    enable_multi_pair_trading: bool = Field(False, env="ENABLE_MULTI_PAIR_TRADING")
    enable_statistical_arb: bool = Field(False, env="ENABLE_STATISTICAL_ARB")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class TimeSyncSettings(BaseSettings):
    """Time synchronization settings."""

    ntp_server: str = Field("pool.ntp.org", env="NTP_SERVER")
    time_sync_interval_seconds: int = Field(
        3600, env="TIME_SYNC_INTERVAL_SECONDS", gt=0
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class UISettings(BaseSettings):
    """Terminal UI configuration."""

    ui_theme: str = Field("zen_garden", env="UI_THEME")
    ui_refresh_rate_ms: int = Field(100, env="UI_REFRESH_RATE_MS", gt=0)
    ui_show_debug_panel: bool = Field(False, env="UI_SHOW_DEBUG_PANEL")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class PerformanceSettings(BaseSettings):
    """Performance tuning configuration."""

    connection_pool_size: int = Field(10, env="CONNECTION_POOL_SIZE", gt=0)
    connection_timeout_seconds: int = Field(30, env="CONNECTION_TIMEOUT_SECONDS", gt=0)
    cache_ttl_seconds: int = Field(60, env="CACHE_TTL_SECONDS", ge=0)
    cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE", gt=0)
    event_bus_buffer_size: int = Field(1000, env="EVENT_BUS_BUFFER_SIZE", gt=0)
    event_bus_worker_threads: int = Field(4, env="EVENT_BUS_WORKER_THREADS", gt=0)

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars meant for other settings
    }


class Settings:
    """Main settings class that aggregates all configuration sections."""

    def __init__(self):
        """Initialize all settings sections from environment."""
        self.exchange = ExchangeSettings()
        self.trading = TradingSettings()
        self.risk = RiskSettings()
        self.database = DatabaseSettings()
        self.logging = LoggingSettings()
        self.tilt = TiltDetectionSettings()
        self.backup = BackupSettings()
        self.deployment = DeploymentSettings()
        self.notifications = NotificationSettings()
        self.development = DevelopmentSettings()
        self.security = SecuritySettings()
        self.features = FeatureFlags()
        self.time_sync = TimeSyncSettings()
        self.ui = UISettings()
        self.performance = PerformanceSettings()

        # Validate tier features after initialization
        self.validate_tier_features()

    def validate_tier_features(self) -> None:
        """Validate that feature flags match tier restrictions."""
        tier = self.trading.trading_tier

        if tier == TradingTier.SNIPER:
            if self.features.enable_multi_pair_trading:
                raise ValueError("Multi-pair trading requires Hunter tier or higher")
            if self.features.enable_statistical_arb:
                raise ValueError("Statistical arbitrage requires Strategist tier")

        elif tier == TradingTier.HUNTER:
            if self.features.enable_statistical_arb:
                raise ValueError("Statistical arbitrage requires Strategist tier")


_settings_instance: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get validated settings instance (singleton pattern).

    Args:
        reload: Force reload of settings from environment

    Returns:
        Settings: Validated configuration object

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If .env file specified but not found

    Example:
        >>> settings = get_settings()
        >>> api_key = settings.exchange.binance_api_key.get_secret_value()
    """
    global _settings_instance

    if _settings_instance is None or reload:
        try:
            _settings_instance = Settings()
            logger.info(
                f"Configuration loaded successfully for {_settings_instance.deployment.deployment_env} environment"
            )
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            # Provide helpful error messages for common issues
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                logger.error(f"  - {field}: {msg}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            logger.info("Copy .env.example to .env and configure your settings")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise

    return _settings_instance


def validate_configuration() -> dict[str, Any]:
    """
    Validate the current configuration and return a status report.

    Returns:
        Dict containing validation status and any warnings
    """
    report = {
        "valid": True,
        "warnings": [],
        "info": [],
        "tier": None,
        "environment": None,
    }

    try:
        settings = get_settings()
        report["tier"] = settings.trading.trading_tier
        report["environment"] = settings.deployment.deployment_env

        # Check for development settings in production
        if settings.deployment.deployment_env == DeploymentEnv.PRODUCTION:
            if settings.development.debug:
                report["warnings"].append("DEBUG mode enabled in production!")
            if settings.exchange.binance_testnet:
                report["warnings"].append("Using testnet in production environment")
            if settings.development.use_mock_exchange:
                report["warnings"].append("Mock exchange enabled in production!")

        # Check security settings
        if settings.deployment.deployment_env != DeploymentEnv.DEVELOPMENT:
            if settings.security.session_timeout_minutes > 60:
                report["warnings"].append(
                    "Session timeout longer than 1 hour in non-dev environment"
                )

        # Validate tier features
        settings.validate_tier_features()
        report["info"].append(
            f"Tier {settings.trading.trading_tier} features validated"
        )

        # Check backup configuration
        if not settings.backup.do_spaces_key:
            report["warnings"].append("Backup credentials not configured")

        # Check notification configuration
        if (
            not settings.notifications.smtp_host
            and settings.notifications.alert_on_error
        ):
            report["warnings"].append("Error alerts enabled but SMTP not configured")

    except Exception as e:
        report["valid"] = False
        report["error"] = str(e)

    return report


if __name__ == "__main__":
    # Enhanced configuration testing with detailed output

    try:
        # Load and validate settings
        settings = get_settings()
        print("✓ Configuration loaded successfully!")
        print(f"  Trading Tier: {settings.trading.trading_tier}")
        print(f"  Environment: {settings.deployment.deployment_env}")
        print(f"  Database: {settings.database.database_url}")
        print(f"  Testnet: {settings.exchange.binance_testnet}")

        # Run validation report
        print("\n📋 Validation Report:")
        report = validate_configuration()

        if report["valid"]:
            print("✓ Configuration is valid")

            if report["warnings"]:
                print("\n⚠️  Warnings:")
                for warning in report["warnings"]:
                    print(f"  - {warning}")

            if report["info"]:
                print("\n📌 Info:")
                for info in report["info"]:
                    print(f"  - {info}")
        else:
            print(f"✗ Configuration invalid: {report.get('error', 'Unknown error')}")

    except ValidationError as e:
        print("✗ Configuration validation failed:")
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            print(f"  - {field}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)
