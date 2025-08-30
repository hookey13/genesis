"""
Paper trading configuration for validation testing.

Defines test parameters for paper trading validation including
position sizing, trade count requirements, and duration settings.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class PaperTradingConfig(BaseModel):
    """Configuration for paper trading test suite."""

    # Session Configuration
    session_duration_hours: int = Field(default=24, description="Paper trading session duration")
    min_trades_required: int = Field(default=10, description="Minimum trades for validation")

    # Position Sizing (SNIPER tier limits)
    max_position_size_usdt: Decimal = Field(
        default=Decimal("100"),
        description="Maximum position size in USDT"
    )
    min_position_size_usdt: Decimal = Field(
        default=Decimal("10"),
        description="Minimum position size in USDT"
    )

    # Risk Settings
    stop_loss_percent: Decimal = Field(
        default=Decimal("2.0"),
        description="Stop loss percentage"
    )
    slippage_percent: Decimal = Field(
        default=Decimal("0.1"),
        description="Simulated slippage for paper trades"
    )

    # P&L Requirements
    pnl_accuracy_decimals: int = Field(
        default=2,
        description="Required P&L accuracy in decimal places"
    )

    # Trading Pairs for Testing
    test_symbols: list[str] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        description="Symbols to trade during testing"
    )

    # Initial Balances
    initial_balance_usdt: Decimal = Field(
        default=Decimal("10000"),
        description="Starting USDT balance for paper trading"
    )

    # UI Update Settings
    ui_refresh_interval_ms: int = Field(
        default=1000,
        description="UI refresh interval in milliseconds"
    )

    # Monitoring Settings
    heartbeat_interval_seconds: int = Field(
        default=30,
        description="Heartbeat monitoring interval"
    )
    health_check_interval_seconds: int = Field(
        default=60,
        description="Health check monitoring interval"
    )

    # Performance Thresholds
    max_latency_ms: int = Field(
        default=500,
        description="Maximum acceptable latency"
    )

    # Continuous Operation Settings
    auto_reconnect: bool = Field(
        default=True,
        description="Enable automatic reconnection on disconnect"
    )
    max_reconnect_attempts: int = Field(
        default=10,
        description="Maximum reconnection attempts"
    )

    # Logging Settings
    log_performance_interval_seconds: int = Field(
        default=60,
        description="Performance metrics logging interval"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


def get_paper_trading_config() -> PaperTradingConfig:
    """
    Get paper trading configuration instance.
    
    Returns:
        Paper trading configuration
    """
    return PaperTradingConfig()


def get_test_validation_criteria() -> dict[str, Any]:
    """
    Get validation criteria for paper trading tests.
    
    Returns:
        Dictionary of validation criteria
    """
    config = get_paper_trading_config()

    return {
        "min_trades": config.min_trades_required,
        "min_duration_hours": 24,
        "pnl_accuracy_decimals": config.pnl_accuracy_decimals,
        "required_uptime_percent": 99.0,  # 99% uptime for 24 hours
        "max_manual_interventions": 0,  # No manual intervention allowed
        "ui_responsiveness": True,  # UI must show live updates
        "position_tracking": True,  # All positions must be tracked
        "pnl_tracking": True,  # P&L must be accurate
        "acceptance_criteria": {
            "AC1": "10 successful round-trip trades completed",
            "AC2": "P&L calculation accurate to 2 decimal places",
            "AC3": "24-hour continuous operation achieved",
            "AC4": "UI showing live positions and P&L",
            "AC5": "No manual intervention required",
        }
    }
