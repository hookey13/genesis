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


class TierLimits(BaseModel):
    """Tier-specific trading limits and parameters."""

    max_position_size: Decimal = Field(gt=0, description="Maximum position size in USDT")
    daily_loss_limit: Decimal = Field(gt=0, le=100, description="Daily loss limit as percentage")
    position_risk_percent: Decimal = Field(gt=0, le=100, description="Risk per position as percentage")
    max_positions: int = Field(gt=0, description="Maximum concurrent positions")
    stop_loss_percent: Decimal = Field(gt=0, le=100, description="Stop loss as percentage")
    allowed_strategies: list[str] = Field(min_length=1, description="Allowed strategy names")
    max_leverage: Decimal = Field(ge=1, description="Maximum leverage allowed")
    min_win_rate: Decimal = Field(ge=0, le=1, description="Minimum win rate to maintain")
    max_drawdown: Decimal = Field(gt=0, le=100, description="Maximum drawdown as percentage")
    recovery_lockout_hours: Decimal = Field(ge=0, description="Lockout hours after tilt level 3")
    tilt_thresholds: dict[str, int] = Field(description="Tilt detection thresholds")
    kelly_sizing: dict[str, Any] | None = Field(default=None, description="Kelly sizing parameters")
    conviction_multipliers: dict[str, Decimal] | None = Field(default=None, description="Conviction level multipliers")

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

    min_position_size: Decimal = Field(gt=0, description="Minimum position size in USDT")
    max_portfolio_exposure: Decimal = Field(gt=0, le=100, description="Max portfolio exposure as percentage")
    max_correlation_exposure: Decimal = Field(gt=0, le=100, description="Max correlated exposure as percentage")
    force_close_loss_percent: Decimal = Field(gt=0, le=100, description="Force close at loss percentage")
    max_daily_trades: int = Field(gt=0, description="Maximum trades per day")
    max_slippage_percent: Decimal = Field(ge=0, description="Maximum slippage as percentage")
    order_timeout_seconds: int = Field(gt=0, description="Order timeout in seconds")


class ExchangeLimits(BaseModel):
    """Exchange-specific rate limits and constraints."""

    rate_limit_per_second: int = Field(gt=0, description="API calls per second")
    max_order_size_usdt: Decimal = Field(gt=0, description="Maximum single order size")
    min_order_size_usdt: Decimal = Field(gt=0, description="Minimum order size")
    max_websocket_connections: int = Field(gt=0, description="Maximum WebSocket connections")


class MarketConditionAdjustment(BaseModel):
    """Adjustments based on market conditions."""

    position_size_multiplier: Decimal = Field(gt=0, description="Position size adjustment")
    stop_loss_multiplier: Decimal = Field(gt=0, description="Stop loss adjustment")
    max_positions_multiplier: Decimal = Field(gt=0, description="Max positions adjustment")


class TierGate(BaseModel):
    """Requirements for tier progression."""

    capital_required: Decimal = Field(gt=0, description="Capital requirement in USDT")
    trades_required: int = Field(gt=0, description="Number of trades required")
    win_rate_required: Decimal = Field(ge=0, le=1, description="Win rate requirement")
    days_required: int = Field(gt=0, description="Days of trading required")
    max_drawdown_allowed: Decimal = Field(gt=0, le=100, description="Maximum drawdown allowed")


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
        description="Path to trading rules YAML file"
    )

    # Database settings
    database_url: str = Field(
        default="sqlite:///genesis.db",
        description="Database connection URL"
    )

    # Exchange settings
    exchange_api_key: str | None = Field(default=None, description="Exchange API key")
    exchange_api_secret: str | None = Field(default=None, description="Exchange API secret")
    exchange_testnet: bool = Field(default=True, description="Use testnet instead of mainnet")

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
            raise FileNotFoundError(f"Trading rules file not found: {self.trading_rules_path}")

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
