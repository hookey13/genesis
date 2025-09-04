"""
Configuration management for Project GENESIS.

Loads and validates trading rules and tier limits from YAML configuration.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from genesis.core.models import TradingTier
from genesis.security.secrets_manager import SecretBackend, SecretsManager
from genesis.security.vault_client import VaultClient


class TierLimits(BaseModel):
    """Tier-specific trading limits and parameters."""

    max_position_size: Decimal = Field(
        gt=0, description="Maximum position size in USDT"
    )
    daily_loss_limit: Decimal = Field(
        gt=0, le=100, description="Daily loss limit as percentage"
    )
    position_risk_percent: Decimal = Field(
        gt=0, le=100, description="Risk per position as percentage"
    )
    max_positions: int = Field(gt=0, description="Maximum concurrent positions")
    stop_loss_percent: Decimal = Field(
        gt=0, le=100, description="Stop loss as percentage"
    )
    allowed_strategies: list[str] = Field(
        min_length=1, description="Allowed strategy names"
    )
    max_leverage: Decimal = Field(ge=1, description="Maximum leverage allowed")
    min_win_rate: Decimal = Field(
        ge=0, le=1, description="Minimum win rate to maintain"
    )
    max_drawdown: Decimal = Field(
        gt=0, le=100, description="Maximum drawdown as percentage"
    )
    recovery_lockout_hours: Decimal = Field(
        ge=0, description="Lockout hours after tilt level 3"
    )
    tilt_thresholds: dict[str, int] = Field(description="Tilt detection thresholds")
    kelly_sizing: dict[str, Any] | None = Field(
        default=None, description="Kelly sizing parameters"
    )
    conviction_multipliers: dict[str, Decimal] | None = Field(
        default=None, description="Conviction level multipliers"
    )

    @field_validator("tilt_thresholds")
    @classmethod
    def validate_tilt_thresholds(cls, v: dict) -> dict:
        """Validate tilt thresholds are properly ordered."""
        required_levels = ["level_1", "level_2", "level_3"]
        for level in required_levels:
            if level not in v:
                raise ValueError(f"Missing tilt threshold: {level}")

        if not (v["level_1"] < v["level_2"] < v["level_3"]):
            raise ValueError("Tilt thresholds must be in ascending order")

        return v


class GlobalRules(BaseModel):
    """Global risk rules that apply to all tiers."""

    min_position_size: Decimal = Field(
        gt=0, description="Minimum position size in USDT"
    )
    max_portfolio_exposure: Decimal = Field(
        gt=0, le=100, description="Max portfolio exposure as percentage"
    )
    max_correlation_exposure: Decimal = Field(
        gt=0, le=100, description="Max correlated exposure as percentage"
    )
    force_close_loss_percent: Decimal = Field(
        gt=0, le=100, description="Force close at loss percentage"
    )
    max_daily_trades: int = Field(gt=0, description="Maximum trades per day")
    max_slippage_percent: Decimal = Field(
        ge=0, description="Maximum slippage as percentage"
    )
    order_timeout_seconds: int = Field(gt=0, description="Order timeout in seconds")


class ExchangeLimits(BaseModel):
    """Exchange-specific rate limits and constraints."""

    rate_limit_per_second: int = Field(gt=0, description="API calls per second")
    max_order_size_usdt: Decimal = Field(gt=0, description="Maximum single order size")
    min_order_size_usdt: Decimal = Field(gt=0, description="Minimum order size")
    max_websocket_connections: int = Field(
        gt=0, description="Maximum WebSocket connections"
    )


class MarketConditionAdjustment(BaseModel):
    """Adjustments based on market conditions."""

    position_size_multiplier: Decimal = Field(
        gt=0, description="Position size adjustment"
    )
    stop_loss_multiplier: Decimal = Field(gt=0, description="Stop loss adjustment")
    max_positions_multiplier: Decimal = Field(
        gt=0, description="Max positions adjustment"
    )


class TierGate(BaseModel):
    """Requirements for tier progression."""

    capital_required: Decimal = Field(gt=0, description="Capital requirement in USDT")
    trades_required: int = Field(gt=0, description="Number of trades required")
    win_rate_required: Decimal = Field(ge=0, le=1, description="Win rate requirement")
    days_required: int = Field(gt=0, description="Days of trading required")
    max_drawdown_allowed: Decimal = Field(
        gt=0, le=100, description="Maximum drawdown allowed"
    )


class TradingRulesConfig(BaseModel):
    """Complete trading rules configuration."""

    tiers: dict[str, TierLimits]
    global_rules: GlobalRules
    exchange_limits: dict[str, ExchangeLimits]
    market_conditions: dict[str, MarketConditionAdjustment]
    tier_gates: dict[str, TierGate]


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Trading rules configuration
    trading_rules_path: Path = Field(
        default=Path(__file__).parent / "trading_rules.yaml",
        description="Path to trading rules YAML file",
    )

    # Database settings
    database_url: str = Field(
        default="sqlite:///genesis.db", description="Database connection URL"
    )

    # Exchange settings (deprecated - use Vault)
    exchange_api_key: str | None = Field(
        default=None, description="Exchange API key (deprecated)"
    )
    exchange_api_secret: str | None = Field(
        default=None, description="Exchange API secret (deprecated)"
    )
    exchange_testnet: bool = Field(
        default=True, description="Use testnet instead of mainnet"
    )

    # Vault settings
    vault_url: str | None = Field(default=None, description="HashiCorp Vault URL")
    vault_token: str | None = Field(default=None, description="HashiCorp Vault token")
    use_vault: bool = Field(
        default=True, description="Use Vault for secrets (False for dev)"
    )

    # Secrets management settings
    secrets_backend: str = Field(
        default="vault", description="Secrets backend: vault, aws, local"
    )
    enable_key_rotation: bool = Field(
        default=True, description="Enable automatic API key rotation"
    )
    rotation_interval_days: int = Field(
        default=30, description="API key rotation interval in days"
    )

    # HSM settings
    enable_hsm: bool = Field(
        default=False, description="Enable Hardware Security Module"
    )
    hsm_type: str = Field(
        default="simulator", description="HSM type: simulator, softhsm, yubihsm"
    )
    hsm_library_path: str | None = Field(
        default=None, description="Path to HSM PKCS#11 library"
    )

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    # System settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    dry_run: bool = Field(default=True, description="Dry run mode (no real trades)")

    class Config:
        env_file = ".env"
        env_prefix = "GENESIS_"
        case_sensitive = False

    _vault_client: VaultClient | None = None
    _secrets_manager: SecretsManager | None = None

    def get_vault_client(self) -> VaultClient:
        """Get or create enhanced Vault client instance with SecretsManager.

        Returns:
            VaultClient instance
        """
        if self._vault_client is None:
            # Determine backend type
            backend_map = {
                "vault": SecretBackend.VAULT,
                "aws": SecretBackend.AWS,
                "local": SecretBackend.LOCAL,
            }
            backend_type = backend_map.get(self.secrets_backend.lower())

            self._vault_client = VaultClient(
                vault_url=self.vault_url,
                vault_token=self.vault_token,
                use_vault=self.use_vault,
                backend_type=backend_type,
                enable_rotation=self.enable_key_rotation,
            )
        return self._vault_client

    def get_secrets_manager(self) -> SecretsManager:
        """Get or create SecretsManager instance.

        Returns:
            SecretsManager instance
        """
        if self._secrets_manager is None:
            backend_map = {
                "vault": SecretBackend.VAULT,
                "aws": SecretBackend.AWS,
                "local": SecretBackend.LOCAL,
            }
            backend = backend_map.get(self.secrets_backend.lower(), SecretBackend.LOCAL)

            config = {"vault_url": self.vault_url, "vault_token": self.vault_token}

            self._secrets_manager = SecretsManager(backend=backend, config=config)
        return self._secrets_manager

    def get_exchange_credentials(
        self, read_only: bool = False
    ) -> dict[str, str] | None:
        """Get exchange API credentials from Vault or environment.

        Args:
            read_only: Whether to get read-only credentials

        Returns:
            Dictionary with 'api_key' and 'api_secret' or None
        """
        vault_client = self.get_vault_client()

        # Try to get from Vault first
        credentials = vault_client.get_exchange_api_keys(read_only=read_only)

        # Fall back to environment variables if not using Vault
        if not credentials and not self.use_vault:
            if read_only:
                # Try read-only env vars first
                api_key = self.exchange_api_key or None
                api_secret = self.exchange_api_secret or None
            else:
                api_key = self.exchange_api_key
                api_secret = self.exchange_api_secret

            if api_key and api_secret:
                credentials = {"api_key": api_key, "api_secret": api_secret}

        return credentials

    def get_database_encryption_key(self) -> str | None:
        """Get database encryption key from Vault or environment.

        Returns:
            Encryption key or None
        """
        vault_client = self.get_vault_client()
        return vault_client.get_database_encryption_key()

    def load_trading_rules(self) -> TradingRulesConfig:
        """
        Load and validate trading rules from YAML file.

        Returns:
            Validated trading rules configuration

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        if not self.trading_rules_path.exists():
            raise FileNotFoundError(
                f"Trading rules file not found: {self.trading_rules_path}"
            )

        with open(self.trading_rules_path) as f:
            raw_config = yaml.safe_load(f)

        return TradingRulesConfig(**raw_config)

    def get_tier_limits(self, tier: TradingTier) -> dict[str, Any]:
        """
        Get tier-specific limits.

        Args:
            tier: Trading tier

        Returns:
            Dictionary of tier limits
        """
        config = self.load_trading_rules()
        tier_config = config.tiers.get(tier.value)

        if not tier_config:
            raise ValueError(f"No configuration found for tier: {tier.value}")

        # Convert to dictionary with proper types for risk engine
        return {
            "max_position_size": tier_config.max_position_size,
            "daily_loss_limit": tier_config.daily_loss_limit,
            "position_risk_percent": tier_config.position_risk_percent,
            "max_positions": tier_config.max_positions,
            "stop_loss_percent": tier_config.stop_loss_percent,
            "allowed_strategies": tier_config.allowed_strategies,
            "max_leverage": tier_config.max_leverage,
        }


# Create global settings instance
settings = Settings()


def load_tier_configuration(tier: TradingTier) -> dict[str, Any]:
    """
    Load tier configuration from YAML file.

    Args:
        tier: Trading tier to load configuration for

    Returns:
        Dictionary containing tier limits and parameters
    """
    return settings.get_tier_limits(tier)
