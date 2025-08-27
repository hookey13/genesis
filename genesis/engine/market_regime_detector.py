"""
Market regime detection for adaptive strategy control.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus
from typing import Optional

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    CRASH = "crash"
    RECOVERY = "recovery"


class RegimeIndicator(str, Enum):
    """Indicators for regime detection."""
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    CORRELATION = "correlation"
    VOLUME = "volume"
    SPREAD = "spread"


class MarketRegimeDetector:
    """Detect and classify market regimes for strategy adaptation."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.current_regime = MarketRegime.NORMAL
        self.regime_confidence = Decimal("1.0")
        self.last_detection = datetime.now(UTC)

    async def start(self) -> None:
        """Start regime detector."""
        logger.info("Market regime detector started")

    async def stop(self) -> None:
        """Stop regime detector."""
        logger.info("Market regime detector stopped")

    async def detect_regime(
        self,
        market_data: Optional[dict] = None
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            market_data: Optional market data for detection
            
        Returns:
            Detected market regime
        """
        # Simplified regime detection
        # In production, would analyze volatility, trends, correlations, etc.

        if market_data:
            volatility = market_data.get("volatility", 0.2)
            trend = market_data.get("trend", 0)

            if volatility > 0.5:
                self.current_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.1:
                self.current_regime = MarketRegime.LOW_VOLATILITY
            elif trend > 0.2:
                self.current_regime = MarketRegime.TRENDING_UP
            elif trend < -0.2:
                self.current_regime = MarketRegime.TRENDING_DOWN
            else:
                self.current_regime = MarketRegime.NORMAL

        self.last_detection = datetime.now(UTC)

        # Publish regime change event
        await self.event_bus.publish(Event(
            event_type=EventType.MARKET_STATE_CHANGE,
            event_data={
                "regime": self.current_regime.value,
                "confidence": str(self.regime_confidence),
                "timestamp": self.last_detection.isoformat()
            }
        ))

        return self.current_regime

    def get_regime_characteristics(self, regime: MarketRegime) -> dict:
        """Get characteristics of a market regime."""
        characteristics = {
            MarketRegime.NORMAL: {
                "volatility": "moderate",
                "trend": "neutral",
                "risk_level": "normal",
                "suggested_strategies": ["mean_reversion", "arbitrage"]
            },
            MarketRegime.TRENDING_UP: {
                "volatility": "low",
                "trend": "bullish",
                "risk_level": "low",
                "suggested_strategies": ["trend_following", "momentum"]
            },
            MarketRegime.HIGH_VOLATILITY: {
                "volatility": "high",
                "trend": "uncertain",
                "risk_level": "high",
                "suggested_strategies": ["volatility_arbitrage", "options"]
            },
            MarketRegime.CRASH: {
                "volatility": "extreme",
                "trend": "bearish",
                "risk_level": "extreme",
                "suggested_strategies": ["defensive", "cash"]
            }
        }

        return characteristics.get(regime, characteristics[MarketRegime.NORMAL])
