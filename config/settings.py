"""
Configuration management using Pydantic for Project GENESIS.

This module handles all environment variables and configuration validation,
ensuring type safety and providing defaults where appropriate.
"""

from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


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
    """Exchange-related configuration."""
    binance_api_key: str = Field(..., env="BINANCE_API_KEY")
    binance_api_secret: str = Field(..., env="BINANCE_API_SECRET")
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")
    exchange_rate_limit: int = Field(1200, env="EXCHANGE_RATE_LIMIT", gt=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


class TradingSettings(BaseSettings):
    """Trading configuration."""
    trading_tier: TradingTier = Field(TradingTier.SNIPER, env="TRADING_TIER")
    max_position_size_usdt: Decimal = Field(Decimal("100.0"), env="MAX_POSITION_SIZE_USDT", gt=0)
    max_daily_loss_usdt: Decimal = Field(Decimal("50.0"), env="MAX_DAILY_LOSS_USDT", gt=0)
    stop_loss_percentage: Decimal = Field(Decimal("2.0"), env="STOP_LOSS_PERCENTAGE", gt=0)
    take_profit_percentage: Decimal = Field(Decimal("3.0"), env="TAKE_PROFIT_PERCENTAGE", gt=0)
    trading_pairs: List[str] = Field(["BTC/USDT"], env="TRADING_PAIRS")

    @validator("trading_pairs", pre=True)
    def parse_trading_pairs(cls, v):
        """Parse comma-separated trading pairs."""
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(",")]
        return v

    @validator("max_position_size_usdt", "max_daily_loss_usdt", "stop_loss_percentage", "take_profit_percentage")
    def ensure_decimal(cls, v):
        """Ensure values are Decimal type."""
        return Decimal(str(v))

    class Config:
        env_file = ".env"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    database_url: str = Field("sqlite:///.genesis/data/genesis.db", env="DATABASE_URL")

    @validator("database_url")
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not (v.startswith("sqlite://") or v.startswith("postgresql://")):
            raise ValueError("Database URL must be SQLite or PostgreSQL")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_file_path: Path = Field(Path(".genesis/logs/trading.log"), env="LOG_FILE_PATH")
    audit_log_path: Path = Field(Path(".genesis/logs/audit.log"), env="AUDIT_LOG_PATH")
    tilt_log_path: Path = Field(Path(".genesis/logs/tilt.log"), env="TILT_LOG_PATH")
    log_max_bytes: int = Field(10485760, env="LOG_MAX_BYTES", gt=0)  # 10MB
    log_backup_count: int = Field(5, env="LOG_BACKUP_COUNT", ge=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


class TiltDetectionSettings(BaseSettings):
    """Tilt detection thresholds."""
    tilt_click_speed_threshold: float = Field(5.0, env="TILT_CLICK_SPEED_THRESHOLD", gt=0)
    tilt_cancel_rate_threshold: float = Field(0.5, env="TILT_CANCEL_RATE_THRESHOLD", ge=0, le=1)
    tilt_revenge_trade_threshold: int = Field(3, env="TILT_REVENGE_TRADE_THRESHOLD", gt=0)
    tilt_position_size_variance: float = Field(0.3, env="TILT_POSITION_SIZE_VARIANCE", ge=0, le=1)

    class Config:
        env_file = ".env"
        case_sensitive = False


class BackupSettings(BaseSettings):
    """Backup configuration."""
    do_spaces_key: Optional[str] = Field(None, env="DO_SPACES_KEY")
    do_spaces_secret: Optional[str] = Field(None, env="DO_SPACES_SECRET")
    do_spaces_region: str = Field("sgp1", env="DO_SPACES_REGION")
    do_spaces_bucket: str = Field("genesis-backups", env="DO_SPACES_BUCKET")
    backup_schedule: str = Field("0 */4 * * *", env="BACKUP_SCHEDULE")  # Cron format

    class Config:
        env_file = ".env"
        case_sensitive = False


class DeploymentSettings(BaseSettings):
    """Deployment configuration."""
    server_host: str = Field("0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(8000, env="SERVER_PORT", gt=0, le=65535)
    deployment_env: DeploymentEnv = Field(DeploymentEnv.DEVELOPMENT, env="DEPLOYMENT_ENV")
    supervisor_config_path: Path = Field(
        Path("/etc/supervisor/conf.d/genesis.conf"),
        env="SUPERVISOR_CONFIG_PATH"
    )
    process_auto_restart: bool = Field(True, env="PROCESS_AUTO_RESTART")
    process_restart_delay_seconds: int = Field(5, env="PROCESS_RESTART_DELAY_SECONDS", ge=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


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

    class Config:
        env_file = ".env"
        case_sensitive = False


class DevelopmentSettings(BaseSettings):
    """Development and testing settings."""
    debug: bool = Field(False, env="DEBUG")
    test_mode: bool = Field(False, env="TEST_MODE")
    use_mock_exchange: bool = Field(False, env="USE_MOCK_EXCHANGE")
    enable_profiling: bool = Field(False, env="ENABLE_PROFILING")
    profile_output_dir: Path = Field(Path(".genesis/profiling"), env="PROFILE_OUTPUT_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security configuration."""
    api_secret_key: str = Field(..., env="API_SECRET_KEY")
    jwt_expiration_hours: int = Field(24, env="JWT_EXPIRATION_HOURS", gt=0)
    session_timeout_minutes: int = Field(30, env="SESSION_TIMEOUT_MINUTES", gt=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


class FeatureFlags(BaseSettings):
    """Feature flags for conditional functionality."""
    enable_paper_trading: bool = Field(True, env="ENABLE_PAPER_TRADING")
    enable_backtesting: bool = Field(False, env="ENABLE_BACKTESTING")
    enable_multi_pair_trading: bool = Field(False, env="ENABLE_MULTI_PAIR_TRADING")
    enable_statistical_arb: bool = Field(False, env="ENABLE_STATISTICAL_ARB")

    class Config:
        env_file = ".env"
        case_sensitive = False


class TimeSyncSettings(BaseSettings):
    """Time synchronization settings."""
    ntp_server: str = Field("pool.ntp.org", env="NTP_SERVER")
    time_sync_interval_seconds: int = Field(3600, env="TIME_SYNC_INTERVAL_SECONDS", gt=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


class UISettings(BaseSettings):
    """Terminal UI configuration."""
    ui_theme: str = Field("zen_garden", env="UI_THEME")
    ui_refresh_rate_ms: int = Field(100, env="UI_REFRESH_RATE_MS", gt=0)
    ui_show_debug_panel: bool = Field(False, env="UI_SHOW_DEBUG_PANEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


class PerformanceSettings(BaseSettings):
    """Performance tuning configuration."""
    connection_pool_size: int = Field(10, env="CONNECTION_POOL_SIZE", gt=0)
    connection_timeout_seconds: int = Field(30, env="CONNECTION_TIMEOUT_SECONDS", gt=0)
    cache_ttl_seconds: int = Field(60, env="CACHE_TTL_SECONDS", ge=0)
    cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE", gt=0)
    event_bus_buffer_size: int = Field(1000, env="EVENT_BUS_BUFFER_SIZE", gt=0)
    event_bus_worker_threads: int = Field(4, env="EVENT_BUS_WORKER_THREADS", gt=0)

    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration sections."""
    
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tilt: TiltDetectionSettings = Field(default_factory=TiltDetectionSettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)
    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    time_sync: TimeSyncSettings = Field(default_factory=TimeSyncSettings)
    ui: UISettings = Field(default_factory=UISettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    class Config:
        env_file = ".env"
        case_sensitive = False

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

    def __init__(self, **kwargs):
        """Initialize settings and validate tier features."""
        super().__init__(**kwargs)
        self.validate_tier_features()


def get_settings() -> Settings:
    """Get validated settings instance."""
    return Settings()


if __name__ == "__main__":
    # Test configuration loading
    try:
        settings = get_settings()
        print(f"Configuration loaded successfully!")
        print(f"Trading Tier: {settings.trading.trading_tier}")
        print(f"Deployment Environment: {settings.deployment.deployment_env}")
        print(f"Database URL: {settings.database.database_url}")
    except Exception as e:
        print(f"Configuration error: {e}")