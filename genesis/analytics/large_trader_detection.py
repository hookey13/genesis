"""Large Trader (Whale) Detection Algorithms."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics

import structlog
import numpy as np
import pandas as pd

from genesis.engine.event_bus import EventBus
from genesis.core.events import Event

logger = structlog.get_logger(__name__)


@dataclass
class WhaleActivity:
    """Detected whale trading activity."""
    
    symbol: str
    timestamp: datetime
    trade_size: Decimal
    price: Decimal
    side: str  # 'buy' or 'sell'
    percentile: Decimal  # Size percentile (0-100)
    vpin_score: Decimal  # Volume-synchronized PIN score
    cluster_id: Optional[str] = None  # ID if part of a cluster
    cumulative_volume: Decimal = Decimal("0")  # Total volume from this entity
    confidence: Decimal = Decimal("0.5")  # Detection confidence
    
    @property
    def notional(self) -> Decimal:
        """Calculate notional value."""
        return self.trade_size * self.price
    
    def is_significant(self) -> bool:
        """Check if whale activity is significant."""
        return self.percentile >= Decimal("95") and self.confidence >= Decimal("0.7")


@dataclass
class TradeCluster:
    """Cluster of related trades (possibly same entity)."""
    
    cluster_id: str
    symbol: str
    trades: List[WhaleActivity] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    @property
    def total_volume(self) -> Decimal:
        """Calculate total cluster volume."""
        return sum(t.trade_size for t in self.trades)
    
    @property
    def avg_trade_size(self) -> Decimal:
        """Calculate average trade size."""
        if not self.trades:
            return Decimal("0")
        return self.total_volume / len(self.trades)
    
    @property
    def dominant_side(self) -> str:
        """Determine dominant side (buy/sell)."""
        buy_volume = sum(t.trade_size for t in self.trades if t.side == 'buy')
        sell_volume = sum(t.trade_size for t in self.trades if t.side == 'sell')
        return 'buy' if buy_volume >= sell_volume else 'sell'
    
    def add_trade(self, trade: WhaleActivity) -> None:
        """Add trade to cluster."""
        self.trades.append(trade)
        trade.cluster_id = self.cluster_id
        if not self.end_time or trade.timestamp > self.end_time:
            self.end_time = trade.timestamp


@dataclass
class VPINData:
    """Volume-Synchronized Probability of Informed Trading data."""
    
    symbol: str
    timestamp: datetime
    vpin: Decimal  # VPIN score (0-1)
    buy_volume: Decimal
    sell_volume: Decimal
    bucket_size: Decimal  # Volume bucket size
    confidence: Decimal  # Confidence in classification
    
    def indicates_informed_trading(self) -> bool:
        """Check if VPIN indicates informed trading."""
        return self.vpin > Decimal("0.3")


class LargeTraderDetector:
    """Detects and tracks large trader (whale) activity."""
    
    def __init__(
        self,
        event_bus: EventBus,
        percentile_threshold: Decimal = Decimal("95"),
        cluster_time_window: int = 60,  # seconds
        vpin_bucket_size: Decimal = Decimal("100")  # Volume bucket size
    ):
        """Initialize large trader detector.
        
        Args:
            event_bus: Event bus for publishing signals
            percentile_threshold: Percentile threshold for whale detection
            cluster_time_window: Time window for clustering trades
            vpin_bucket_size: Volume bucket size for VPIN calculation
        """
        self.event_bus = event_bus
        self.percentile_threshold = percentile_threshold
        self.cluster_time_window = timedelta(seconds=cluster_time_window)
        self.vpin_bucket_size = vpin_bucket_size
        
        # Trade history for distribution analysis
        self.trade_history: Dict[str, deque] = {}
        self.history_size = 1000
        
        # Active clusters
        self.active_clusters: Dict[str, TradeCluster] = {}
        
        # VPIN calculation data
        self.vpin_buckets: Dict[str, List[Tuple[Decimal, Decimal]]] = {}
        self.vpin_history: Dict[str, deque] = {}
        
        # Cumulative volumes by suspected entities
        self.entity_volumes: Dict[str, Dict[str, Decimal]] = {}
    
    async def analyze_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        side: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[WhaleActivity]:
        """Analyze trade for large trader activity.
        
        Args:
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            side: Trade side ('buy' or 'sell')
            timestamp: Trade timestamp
            
        Returns:
            Whale activity if detected
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Update trade history
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.history_size)
        self.trade_history[symbol].append(quantity)
        
        # Calculate size percentile
        percentile = self._calculate_size_percentile(symbol, quantity)
        
        # Calculate VPIN score
        vpin_score = await self._update_vpin(symbol, quantity, side)
        
        # Check if this is whale activity
        if percentile >= self.percentile_threshold:
            whale_activity = WhaleActivity(
                symbol=symbol,
                timestamp=timestamp,
                trade_size=quantity,
                price=price,
                side=side,
                percentile=percentile,
                vpin_score=vpin_score,
                confidence=self._calculate_confidence(percentile, vpin_score)
            )
            
            # Check for clustering
            cluster = self._find_or_create_cluster(whale_activity)
            if cluster:
                cluster.add_trade(whale_activity)
                whale_activity.cumulative_volume = cluster.total_volume
            
            # Update entity tracking
            self._update_entity_tracking(whale_activity)
            
            # Publish whale detection event
            if whale_activity.is_significant():
                await self._publish_whale_signal(whale_activity)
            
            return whale_activity
        
        return None
    
    def _calculate_size_percentile(self, symbol: str, size: Decimal) -> Decimal:
        """Calculate size percentile in historical distribution.
        
        Args:
            symbol: Trading symbol
            size: Trade size
            
        Returns:
            Percentile (0-100)
        """
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 10:
            return Decimal("50")  # Default to median
        
        sizes = sorted(self.trade_history[symbol])
        n = len(sizes)
        
        # Find position in sorted list
        position = 0
        for s in sizes:
            if s < size:
                position += 1
            else:
                break
        
        percentile = Decimal(str(position * 100 / n))
        return percentile
    
    async def _update_vpin(self, symbol: str, volume: Decimal, side: str) -> Decimal:
        """Update and calculate VPIN score.
        
        Args:
            symbol: Trading symbol
            volume: Trade volume
            side: Trade side
            
        Returns:
            Updated VPIN score
        """
        if symbol not in self.vpin_buckets:
            self.vpin_buckets[symbol] = []
            self.vpin_history[symbol] = deque(maxlen=50)
        
        buckets = self.vpin_buckets[symbol]
        
        # Add to current bucket or create new one
        if not buckets or sum(b[0] + b[1] for b in [buckets[-1]]) >= self.vpin_bucket_size:
            # Start new bucket
            if side == 'buy':
                buckets.append((volume, Decimal("0")))
            else:
                buckets.append((Decimal("0"), volume))
        else:
            # Add to current bucket
            if side == 'buy':
                buckets[-1] = (buckets[-1][0] + volume, buckets[-1][1])
            else:
                buckets[-1] = (buckets[-1][0], buckets[-1][1] + volume)
        
        # Keep only recent buckets (last 50)
        if len(buckets) > 50:
            buckets.pop(0)
        
        # Calculate VPIN
        vpin = self._calculate_vpin(buckets)
        
        # Store VPIN data
        vpin_data = VPINData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            vpin=vpin,
            buy_volume=buckets[-1][0] if buckets else Decimal("0"),
            sell_volume=buckets[-1][1] if buckets else Decimal("0"),
            bucket_size=self.vpin_bucket_size,
            confidence=self._vpin_confidence(buckets)
        )
        
        self.vpin_history[symbol].append(vpin_data)
        
        # Check for informed trading
        if vpin_data.indicates_informed_trading():
            await self.event_bus.publish(Event(
                type="informed_trading_detected",
                data={
                    "symbol": symbol,
                    "vpin": float(vpin),
                    "confidence": float(vpin_data.confidence),
                    "timestamp": vpin_data.timestamp.isoformat()
                }
            ))
        
        return vpin
    
    def _calculate_vpin(self, buckets: List[Tuple[Decimal, Decimal]]) -> Decimal:
        """Calculate VPIN from volume buckets.
        
        Args:
            buckets: List of (buy_volume, sell_volume) tuples
            
        Returns:
            VPIN score (0-1)
        """
        if not buckets:
            return Decimal("0")
        
        # Calculate order imbalance for each bucket
        imbalances = []
        for buy_vol, sell_vol in buckets:
            total = buy_vol + sell_vol
            if total > 0:
                imbalance = abs(buy_vol - sell_vol) / total
                imbalances.append(imbalance)
        
        if not imbalances:
            return Decimal("0")
        
        # VPIN is average of imbalances
        vpin = sum(imbalances) / len(imbalances)
        return min(Decimal("1"), vpin)
    
    def _vpin_confidence(self, buckets: List[Tuple[Decimal, Decimal]]) -> Decimal:
        """Calculate confidence in VPIN score.
        
        Args:
            buckets: Volume buckets
            
        Returns:
            Confidence score (0-1)
        """
        if len(buckets) < 10:
            return Decimal("0.3")
        elif len(buckets) < 30:
            return Decimal("0.6")
        else:
            return Decimal("0.9")
    
    def _find_or_create_cluster(self, whale_activity: WhaleActivity) -> Optional[TradeCluster]:
        """Find existing cluster or create new one.
        
        Args:
            whale_activity: Whale activity to cluster
            
        Returns:
            Trade cluster or None
        """
        symbol = whale_activity.symbol
        current_time = whale_activity.timestamp
        
        # Clean up old clusters
        self._cleanup_old_clusters(current_time)
        
        # Look for active cluster
        for cluster_id, cluster in self.active_clusters.items():
            if cluster.symbol != symbol:
                continue
            
            # Check if within time window
            if cluster.end_time and current_time - cluster.end_time <= self.cluster_time_window:
                # Check if similar trade pattern
                if self._is_similar_pattern(whale_activity, cluster):
                    return cluster
        
        # Create new cluster
        cluster_id = f"{symbol}_{current_time.timestamp()}"
        cluster = TradeCluster(
            cluster_id=cluster_id,
            symbol=symbol,
            start_time=current_time
        )
        self.active_clusters[cluster_id] = cluster
        
        return cluster
    
    def _is_similar_pattern(self, activity: WhaleActivity, cluster: TradeCluster) -> bool:
        """Check if trade matches cluster pattern.
        
        Args:
            activity: Whale activity
            cluster: Trade cluster
            
        Returns:
            True if similar pattern
        """
        if not cluster.trades:
            return True
        
        # Check side consistency
        if cluster.dominant_side != activity.side:
            return False
        
        # Check size similarity (within 50% of average)
        avg_size = cluster.avg_trade_size
        if abs(activity.trade_size - avg_size) / avg_size > Decimal("0.5"):
            return False
        
        return True
    
    def _cleanup_old_clusters(self, current_time: datetime) -> None:
        """Remove old inactive clusters.
        
        Args:
            current_time: Current timestamp
        """
        to_remove = []
        for cluster_id, cluster in self.active_clusters.items():
            if cluster.end_time and current_time - cluster.end_time > self.cluster_time_window * 2:
                to_remove.append(cluster_id)
        
        for cluster_id in to_remove:
            del self.active_clusters[cluster_id]
    
    def _update_entity_tracking(self, whale_activity: WhaleActivity) -> None:
        """Update cumulative tracking of suspected entities.
        
        Args:
            whale_activity: Whale activity
        """
        symbol = whale_activity.symbol
        
        if symbol not in self.entity_volumes:
            self.entity_volumes[symbol] = {}
        
        # Use cluster ID as entity ID if available
        entity_id = whale_activity.cluster_id or f"whale_{whale_activity.timestamp.timestamp()}"
        
        if entity_id not in self.entity_volumes[symbol]:
            self.entity_volumes[symbol][entity_id] = Decimal("0")
        
        self.entity_volumes[symbol][entity_id] += whale_activity.trade_size
    
    def _calculate_confidence(self, percentile: Decimal, vpin: Decimal) -> Decimal:
        """Calculate detection confidence.
        
        Args:
            percentile: Size percentile
            vpin: VPIN score
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from percentile
        size_confidence = (percentile - Decimal("90")) / Decimal("10")
        size_confidence = max(Decimal("0"), min(Decimal("1"), size_confidence))
        
        # VPIN confidence
        vpin_confidence = vpin * Decimal("2")  # Scale VPIN contribution
        vpin_confidence = min(Decimal("1"), vpin_confidence)
        
        # Weighted average
        confidence = size_confidence * Decimal("0.6") + vpin_confidence * Decimal("0.4")
        
        return min(Decimal("1"), confidence)
    
    async def _publish_whale_signal(self, whale_activity: WhaleActivity) -> None:
        """Publish whale detection signal.
        
        Args:
            whale_activity: Detected whale activity
        """
        await self.event_bus.publish(Event(
            type="whale_activity_detected",
            data={
                "symbol": whale_activity.symbol,
                "trade_size": float(whale_activity.trade_size),
                "notional": float(whale_activity.notional),
                "side": whale_activity.side,
                "percentile": float(whale_activity.percentile),
                "vpin_score": float(whale_activity.vpin_score),
                "cluster_id": whale_activity.cluster_id,
                "cumulative_volume": float(whale_activity.cumulative_volume),
                "confidence": float(whale_activity.confidence),
                "timestamp": whale_activity.timestamp.isoformat()
            }
        ))
        
        logger.info(
            "whale_detected",
            symbol=whale_activity.symbol,
            size=float(whale_activity.trade_size),
            percentile=float(whale_activity.percentile),
            side=whale_activity.side
        )
    
    def get_active_whales(self, symbol: str) -> List[TradeCluster]:
        """Get currently active whale clusters.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of active clusters
        """
        return [
            cluster for cluster in self.active_clusters.values()
            if cluster.symbol == symbol
        ]
    
    def get_whale_statistics(self, symbol: str) -> Dict[str, any]:
        """Get whale activity statistics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Statistics dictionary
        """
        if symbol not in self.trade_history:
            return {}
        
        sizes = list(self.trade_history[symbol])
        if not sizes:
            return {}
        
        return {
            "mean_size": float(statistics.mean(sizes)),
            "median_size": float(statistics.median(sizes)),
            "std_dev": float(statistics.stdev(sizes)) if len(sizes) > 1 else 0,
            "p95_threshold": float(np.percentile([float(s) for s in sizes], 95)),
            "p99_threshold": float(np.percentile([float(s) for s in sizes], 99)),
            "active_clusters": len([c for c in self.active_clusters.values() if c.symbol == symbol]),
            "total_entities": len(self.entity_volumes.get(symbol, {}))
        }
    
    def get_vpin_trend(self, symbol: str, periods: int = 10) -> Optional[str]:
        """Get VPIN trend direction.
        
        Args:
            symbol: Trading symbol
            periods: Number of periods to analyze
            
        Returns:
            Trend direction ('increasing', 'decreasing', 'stable') or None
        """
        if symbol not in self.vpin_history:
            return None
        
        history = list(self.vpin_history[symbol])[-periods:]
        if len(history) < 3:
            return None
        
        vpins = [h.vpin for h in history]
        
        # Calculate trend
        increasing = sum(1 for i in range(1, len(vpins)) if vpins[i] > vpins[i-1])
        decreasing = sum(1 for i in range(1, len(vpins)) if vpins[i] < vpins[i-1])
        
        if increasing > len(vpins) * 0.6:
            return "increasing"
        elif decreasing > len(vpins) * 0.6:
            return "decreasing"
        else:
            return "stable"