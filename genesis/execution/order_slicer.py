"""Order slicing algorithms for VWAP execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from genesis.core.models import Order, OrderSide, OrderType
from genesis.execution.volume_curve import VolumeProfile

logger = structlog.get_logger(__name__)


class SlicingMethod(str, Enum):
    """Order slicing methods."""

    LINEAR = "LINEAR"  # Equal-sized slices
    VOLUME_WEIGHTED = "VOLUME_WEIGHTED"  # Based on volume curve
    TIME_WEIGHTED = "TIME_WEIGHTED"  # Based on time intervals
    ADAPTIVE = "ADAPTIVE"  # Dynamically adjusted
    ICEBERG = "ICEBERG"  # Hidden quantity with visible tip


@dataclass
class SliceConfig:
    """Configuration for order slicing."""

    method: SlicingMethod = SlicingMethod.VOLUME_WEIGHTED
    min_slice_size: Decimal = Decimal("0.001")
    max_slice_size: Decimal = Decimal("1.0")
    max_slices: int = 100
    iceberg_visible_ratio: Decimal = Decimal("0.1")  # 10% visible for iceberg
    adaptive_threshold: Decimal = Decimal("0.05")  # 5% market impact threshold
    randomize_sizes: bool = True  # Add randomness to avoid detection
    randomize_range: Decimal = Decimal("0.1")  # +/- 10% randomization


@dataclass
class OrderSlice:
    """Individual order slice."""

    slice_number: int
    total_slices: int
    quantity: Decimal
    visible_quantity: Optional[Decimal] = None  # For iceberg orders
    execution_time: Optional[datetime] = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderSlicer:
    """Slices large orders into smaller child orders."""

    def __init__(self, config: SliceConfig | None = None):
        """Initialize order slicer.

        Args:
            config: Slicing configuration.
        """
        self.config = config or SliceConfig()
        self.slice_cache: Dict[str, List[OrderSlice]] = {}

    def slice_order(
        self,
        total_quantity: Decimal,
        method: Optional[SlicingMethod] = None,
        volume_profile: Optional[VolumeProfile] = None,
        time_horizon: Optional[int] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[OrderSlice]:
        """Slice order into smaller pieces.

        Args:
            total_quantity: Total order quantity.
            method: Slicing method to use.
            volume_profile: Volume profile for weighted slicing.
            time_horizon: Time horizon in minutes.
            market_conditions: Current market conditions.

        Returns:
            List of order slices.
        """
        method = method or self.config.method

        if method == SlicingMethod.LINEAR:
            return self._linear_slicing(total_quantity)
        elif method == SlicingMethod.VOLUME_WEIGHTED:
            return self._volume_weighted_slicing(total_quantity, volume_profile)
        elif method == SlicingMethod.TIME_WEIGHTED:
            return self._time_weighted_slicing(total_quantity, time_horizon)
        elif method == SlicingMethod.ADAPTIVE:
            return self._adaptive_slicing(total_quantity, market_conditions)
        elif method == SlicingMethod.ICEBERG:
            return self._iceberg_slicing(total_quantity)
        else:
            return self._linear_slicing(total_quantity)

    def _linear_slicing(self, total_quantity: Decimal) -> List[OrderSlice]:
        """Create equal-sized slices.

        Args:
            total_quantity: Total order quantity.

        Returns:
            List of equal-sized slices.
        """
        # Calculate number of slices
        num_slices = self._calculate_slice_count(total_quantity)

        # Calculate base slice size
        base_size = total_quantity / num_slices

        slices = []
        remaining = total_quantity

        for i in range(num_slices):
            # Calculate slice size
            if i == num_slices - 1:
                # Last slice gets remaining quantity
                slice_size = remaining
            else:
                slice_size = min(base_size, remaining, self.config.max_slice_size)
                slice_size = max(slice_size, self.config.min_slice_size)

            # Add randomization if enabled
            if self.config.randomize_sizes and i < num_slices - 1:
                slice_size = self._randomize_size(slice_size)

            slices.append(
                OrderSlice(
                    slice_number=i + 1,
                    total_slices=num_slices,
                    quantity=slice_size,
                    priority=i,
                    metadata={"method": SlicingMethod.LINEAR.value},
                )
            )

            remaining -= slice_size

        return slices

    def _volume_weighted_slicing(
        self, total_quantity: Decimal, volume_profile: Optional[VolumeProfile] = None
    ) -> List[OrderSlice]:
        """Create slices weighted by volume profile.

        Args:
            total_quantity: Total order quantity.
            volume_profile: Volume distribution profile.

        Returns:
            List of volume-weighted slices.
        """
        if not volume_profile or not volume_profile.normalized_volumes:
            # Fall back to linear if no profile
            return self._linear_slicing(total_quantity)

        slices = []
        remaining = total_quantity

        # Use volume profile to determine slice sizes
        num_slices = min(len(volume_profile.normalized_volumes), self.config.max_slices)

        for i in range(num_slices):
            # Calculate slice size based on volume weight
            volume_weight = volume_profile.normalized_volumes[i]
            slice_size = total_quantity * volume_weight

            # Apply constraints
            slice_size = min(slice_size, remaining, self.config.max_slice_size)
            slice_size = max(slice_size, self.config.min_slice_size)

            # Add randomization
            if self.config.randomize_sizes and i < num_slices - 1:
                slice_size = self._randomize_size(slice_size)

            if slice_size > 0 and remaining > 0:
                actual_slice_size = min(slice_size, remaining)
                slices.append(
                    OrderSlice(
                        slice_number=i + 1,
                        total_slices=num_slices,
                        quantity=actual_slice_size,
                        execution_time=volume_profile.intervals[i]
                        if i < len(volume_profile.intervals)
                        else None,
                        priority=i,
                        metadata={
                            "method": SlicingMethod.VOLUME_WEIGHTED.value,
                            "volume_weight": float(volume_weight),
                        },
                    )
                )

                remaining -= actual_slice_size

                if remaining <= 0:
                    break

        # Handle any remaining quantity
        if remaining > self.config.min_slice_size:
            slices.append(
                OrderSlice(
                    slice_number=len(slices) + 1,
                    total_slices=len(slices) + 1,
                    quantity=remaining,
                    priority=len(slices),
                    metadata={
                        "method": SlicingMethod.VOLUME_WEIGHTED.value,
                        "residual": True,
                    },
                )
            )

        return slices

    def _time_weighted_slicing(
        self, total_quantity: Decimal, time_horizon: Optional[int] = None
    ) -> List[OrderSlice]:
        """Create slices weighted by time intervals.

        Args:
            total_quantity: Total order quantity.
            time_horizon: Time horizon in minutes.

        Returns:
            List of time-weighted slices.
        """
        if not time_horizon:
            time_horizon = 60  # Default 1 hour

        # Calculate number of slices based on time
        # Assume 1 slice per 5 minutes as baseline
        num_slices = min(time_horizon // 5, self.config.max_slices)
        num_slices = max(1, num_slices)

        slices = []
        remaining = total_quantity

        for i in range(num_slices):
            # Linear time weighting with front-loading
            time_weight = Decimal(str((num_slices - i) / sum(range(1, num_slices + 1))))
            slice_size = total_quantity * time_weight

            # Apply constraints
            slice_size = min(slice_size, remaining, self.config.max_slice_size)
            slice_size = max(slice_size, self.config.min_slice_size)

            # Add randomization
            if self.config.randomize_sizes and i < num_slices - 1:
                slice_size = self._randomize_size(slice_size)

            slices.append(
                OrderSlice(
                    slice_number=i + 1,
                    total_slices=num_slices,
                    quantity=min(slice_size, remaining),
                    priority=i,
                    metadata={
                        "method": SlicingMethod.TIME_WEIGHTED.value,
                        "time_weight": float(time_weight),
                        "minutes_from_start": i * 5,
                    },
                )
            )

            remaining -= slice_size

            if remaining <= 0:
                break

        return slices

    def _adaptive_slicing(
        self,
        total_quantity: Decimal,
        market_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[OrderSlice]:
        """Create adaptively sized slices based on market conditions.

        Args:
            total_quantity: Total order quantity.
            market_conditions: Current market conditions.

        Returns:
            List of adaptive slices.
        """
        if not market_conditions:
            # Fall back to linear if no market data
            return self._linear_slicing(total_quantity)

        # Extract market metrics
        liquidity = Decimal(str(market_conditions.get("liquidity", 1000000)))
        volatility = Decimal(str(market_conditions.get("volatility", 0.02)))
        spread = Decimal(str(market_conditions.get("spread", 0.001)))
        volume = Decimal(str(market_conditions.get("volume", 10000000)))

        # Adjust slice size based on conditions
        if liquidity > 0:
            # Base slice size on available liquidity
            max_impact_quantity = liquidity * self.config.adaptive_threshold
            optimal_slice_size = min(max_impact_quantity, self.config.max_slice_size)
        else:
            optimal_slice_size = self.config.max_slice_size

        # Adjust for volatility (smaller slices in high volatility)
        if volatility > Decimal("0.05"):  # High volatility
            optimal_slice_size *= Decimal("0.5")
        elif volatility > Decimal("0.03"):  # Medium volatility
            optimal_slice_size *= Decimal("0.75")

        # Adjust for spread (smaller slices for wide spreads)
        if spread > Decimal("0.005"):  # Wide spread
            optimal_slice_size *= Decimal("0.7")

        # Ensure within bounds
        optimal_slice_size = max(optimal_slice_size, self.config.min_slice_size)
        optimal_slice_size = min(optimal_slice_size, self.config.max_slice_size)

        # Calculate number of slices
        num_slices = int(total_quantity / optimal_slice_size) + 1
        num_slices = min(num_slices, self.config.max_slices)

        slices = []
        remaining = total_quantity

        for i in range(num_slices):
            slice_size = min(optimal_slice_size, remaining)

            # Add randomization
            if self.config.randomize_sizes and i < num_slices - 1:
                slice_size = self._randomize_size(slice_size)

            slices.append(
                OrderSlice(
                    slice_number=i + 1,
                    total_slices=num_slices,
                    quantity=slice_size,
                    priority=i,
                    metadata={
                        "method": SlicingMethod.ADAPTIVE.value,
                        "liquidity": float(liquidity),
                        "volatility": float(volatility),
                        "spread": float(spread),
                    },
                )
            )

            remaining -= slice_size

            if remaining <= 0:
                break

        return slices

    def _iceberg_slicing(self, total_quantity: Decimal) -> List[OrderSlice]:
        """Create iceberg order slices with hidden quantity.

        Args:
            total_quantity: Total order quantity.

        Returns:
            List of iceberg slices.
        """
        # Calculate visible and hidden quantities
        visible_size = total_quantity * self.config.iceberg_visible_ratio
        visible_size = max(visible_size, self.config.min_slice_size)
        visible_size = min(visible_size, self.config.max_slice_size)

        slices = []
        remaining = total_quantity
        slice_num = 0

        while remaining > 0:
            slice_num += 1

            # Determine slice size
            if remaining <= visible_size:
                # Last slice
                slice_size = remaining
                slice_visible = remaining
            else:
                # Regular iceberg slice
                slice_size = min(
                    visible_size / self.config.iceberg_visible_ratio, remaining
                )
                slice_visible = visible_size

            # Add randomization to visible quantity
            if self.config.randomize_sizes and remaining > visible_size:
                slice_visible = self._randomize_size(slice_visible)

            slices.append(
                OrderSlice(
                    slice_number=slice_num,
                    total_slices=0,  # Unknown for iceberg
                    quantity=slice_size,
                    visible_quantity=slice_visible,
                    priority=slice_num - 1,
                    metadata={
                        "method": SlicingMethod.ICEBERG.value,
                        "hidden_quantity": float(slice_size - slice_visible),
                    },
                )
            )

            remaining -= slice_size

            if slice_num >= self.config.max_slices:
                # Add remaining as final slice
                if remaining > 0:
                    slices.append(
                        OrderSlice(
                            slice_number=slice_num + 1,
                            total_slices=0,
                            quantity=remaining,
                            visible_quantity=remaining,
                            priority=slice_num,
                            metadata={
                                "method": SlicingMethod.ICEBERG.value,
                                "final": True,
                            },
                        )
                    )
                break

        # Update total slices count
        for slice in slices:
            slice.total_slices = len(slices)

        return slices

    def _calculate_slice_count(self, total_quantity: Decimal) -> int:
        """Calculate optimal number of slices.

        Args:
            total_quantity: Total order quantity.

        Returns:
            Number of slices.
        """
        if total_quantity <= self.config.min_slice_size:
            return 1

        # Calculate based on max slice size
        min_slices = int(total_quantity / self.config.max_slice_size) + 1

        # Calculate based on min slice size
        max_slices = int(total_quantity / self.config.min_slice_size)

        # Use geometric mean for balance
        import math

        optimal_slices = int(math.sqrt(min_slices * max_slices))

        # Apply maximum constraint
        return min(optimal_slices, self.config.max_slices)

    def _randomize_size(self, size: Decimal) -> Decimal:
        """Add randomization to slice size.

        Args:
            size: Base slice size.

        Returns:
            Randomized size.
        """
        import random

        # Generate random factor
        random_factor = Decimal(
            str(
                1
                + random.uniform(
                    -float(self.config.randomize_range),
                    float(self.config.randomize_range),
                )
            )
        )

        randomized = size * random_factor

        # Ensure within bounds
        randomized = max(randomized, self.config.min_slice_size)
        randomized = min(randomized, self.config.max_slice_size)

        return randomized

    def generate_child_orders(
        self,
        parent_order_id: str,
        symbol: str,
        side: OrderSide,
        slices: List[OrderSlice],
        order_type: OrderType = OrderType.LIMIT,
    ) -> List[Order]:
        """Generate child orders from slices.

        Args:
            parent_order_id: Parent order ID.
            symbol: Trading symbol.
            side: Order side.
            slices: List of order slices.
            order_type: Order type.

        Returns:
            List of child orders.
        """
        child_orders = []

        for slice in slices:
            child_order = Order(
                symbol=symbol,
                type=order_type,
                side=side,
                quantity=slice.quantity,
                slice_number=slice.slice_number,
                total_slices=slice.total_slices,
                metadata={
                    "parent_order_id": parent_order_id,
                    "slice_metadata": slice.metadata,
                },
            )

            child_orders.append(child_order)

        logger.info(
            "Generated child orders",
            parent_id=parent_order_id,
            num_children=len(child_orders),
            total_quantity=str(sum(o.quantity for o in child_orders)),
        )

        return child_orders
