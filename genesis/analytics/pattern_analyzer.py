"""
Win/Loss Pattern Analyzer for Project GENESIS.

This module analyzes trading patterns including win/loss streaks,
temporal patterns, position holding times, and close reason distributions.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.models import Position, Trade

logger = structlog.get_logger(__name__)


class TimeFrame(str, Enum):
    """Time frames for pattern analysis."""

    HOUR_OF_DAY = "hour_of_day"
    DAY_OF_WEEK = "day_of_week"
    DAY_OF_MONTH = "day_of_month"
    WEEK_OF_MONTH = "week_of_month"
    MONTH_OF_YEAR = "month_of_year"


@dataclass
class WinLossPattern:
    """Container for win/loss pattern analysis results."""

    # Basic statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Streak analysis
    current_streak: int  # Positive for wins, negative for losses
    max_win_streak: int
    max_loss_streak: int
    average_win_streak: Decimal
    average_loss_streak: Decimal

    # Size analysis
    average_win_size: Decimal
    average_loss_size: Decimal
    win_loss_ratio: Decimal  # Average win / Average loss
    expectancy: Decimal  # (Win rate * Avg win) - (Loss rate * Avg loss)

    # Holding time analysis
    average_win_duration_hours: Decimal
    average_loss_duration_hours: Decimal
    median_win_duration_hours: Decimal
    median_loss_duration_hours: Decimal

    # Temporal patterns
    win_rate_by_hour: dict[int, Decimal] = field(default_factory=dict)
    win_rate_by_day_of_week: dict[int, Decimal] = field(default_factory=dict)
    win_rate_by_day_of_month: dict[int, Decimal] = field(default_factory=dict)

    # Close reason analysis
    close_reason_distribution: dict[str, int] = field(default_factory=dict)
    win_close_reasons: dict[str, int] = field(default_factory=dict)
    loss_close_reasons: dict[str, int] = field(default_factory=dict)

    # Recovery patterns
    trades_recovering_from_drawdown: int = 0
    recovery_success_rate: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.win_rate),
            "current_streak": self.current_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "average_win_streak": str(self.average_win_streak),
            "average_loss_streak": str(self.average_loss_streak),
            "average_win_size": str(self.average_win_size),
            "average_loss_size": str(self.average_loss_size),
            "win_loss_ratio": str(self.win_loss_ratio),
            "expectancy": str(self.expectancy),
            "average_win_duration_hours": str(self.average_win_duration_hours),
            "average_loss_duration_hours": str(self.average_loss_duration_hours),
            "median_win_duration_hours": str(self.median_win_duration_hours),
            "median_loss_duration_hours": str(self.median_loss_duration_hours),
            "win_rate_by_hour": {
                str(k): str(v) for k, v in self.win_rate_by_hour.items()
            },
            "win_rate_by_day_of_week": {
                str(k): str(v) for k, v in self.win_rate_by_day_of_week.items()
            },
            "win_rate_by_day_of_month": {
                str(k): str(v) for k, v in self.win_rate_by_day_of_month.items()
            },
            "close_reason_distribution": self.close_reason_distribution,
            "win_close_reasons": self.win_close_reasons,
            "loss_close_reasons": self.loss_close_reasons,
            "trades_recovering_from_drawdown": self.trades_recovering_from_drawdown,
            "recovery_success_rate": str(self.recovery_success_rate),
        }


class PatternAnalyzer:
    """Analyzer for win/loss patterns and trading behavior."""

    def __init__(self):
        """Initialize the pattern analyzer."""
        self._pattern_cache: dict[str, WinLossPattern] = {}

    def analyze_patterns(
        self, trades: list[Trade], positions: list[Position] | None = None
    ) -> WinLossPattern:
        """
        Analyze win/loss patterns from trades and positions.

        Args:
            trades: List of completed trades
            positions: Optional list of positions for close reason analysis

        Returns:
            WinLossPattern with comprehensive analysis
        """
        if not trades:
            return self._empty_pattern()

        logger.info(f"Analyzing patterns for {len(trades)} trades")

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Separate wins and losses
        winning_trades = [t for t in sorted_trades if t.pnl_dollars > 0]
        losing_trades = [t for t in sorted_trades if t.pnl_dollars <= 0]

        # Calculate basic statistics
        total_trades = len(sorted_trades)
        win_rate = (
            Decimal(str(len(winning_trades))) / Decimal(str(total_trades))
            if total_trades > 0
            else Decimal("0")
        )

        # Analyze streaks
        streak_analysis = self._analyze_streaks(sorted_trades)

        # Analyze trade sizes
        size_analysis = self._analyze_sizes(winning_trades, losing_trades)

        # Analyze holding times
        duration_analysis = self._analyze_durations(
            winning_trades, losing_trades, positions
        )

        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(sorted_trades)

        # Analyze close reasons if positions provided
        close_reason_analysis = (
            self._analyze_close_reasons(positions) if positions else {}
        )

        # Analyze recovery patterns
        recovery_analysis = self._analyze_recovery_patterns(sorted_trades)

        # Create pattern result
        pattern = WinLossPattern(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            **streak_analysis,
            **size_analysis,
            **duration_analysis,
            **temporal_analysis,
            **close_reason_analysis,
            **recovery_analysis,
        )

        return pattern

    def _analyze_streaks(self, trades: list[Trade]) -> dict:
        """Analyze win/loss streaks."""
        if not trades:
            return {
                "current_streak": 0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "average_win_streak": Decimal("0"),
                "average_loss_streak": Decimal("0"),
            }

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0

        for trade in trades:
            if trade.pnl_dollars > 0:
                # Win
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
                current_win_streak += 1
                current_streak = current_win_streak
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                # Loss
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
                current_loss_streak += 1
                current_streak = -current_loss_streak
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        # Add final streak
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)

        # Calculate averages
        avg_win_streak = (
            sum(Decimal(str(s)) for s in win_streaks) / Decimal(str(len(win_streaks)))
            if win_streaks
            else Decimal("0")
        )
        avg_loss_streak = (
            sum(Decimal(str(s)) for s in loss_streaks) / Decimal(str(len(loss_streaks)))
            if loss_streaks
            else Decimal("0")
        )

        return {
            "current_streak": current_streak,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "average_win_streak": avg_win_streak.quantize(Decimal("0.01")),
            "average_loss_streak": avg_loss_streak.quantize(Decimal("0.01")),
        }

    def _analyze_sizes(
        self, winning_trades: list[Trade], losing_trades: list[Trade]
    ) -> dict:
        """Analyze win/loss sizes."""
        # Calculate average win size
        avg_win_size = (
            sum(t.pnl_dollars for t in winning_trades)
            / Decimal(str(len(winning_trades)))
            if winning_trades
            else Decimal("0")
        )

        # Calculate average loss size (as positive value)
        avg_loss_size = (
            abs(sum(t.pnl_dollars for t in losing_trades))
            / Decimal(str(len(losing_trades)))
            if losing_trades
            else Decimal("0")
        )

        # Calculate win/loss ratio
        win_loss_ratio = (
            avg_win_size / avg_loss_size if avg_loss_size > 0 else Decimal("999.99")
        )

        # Calculate expectancy
        total_trades = len(winning_trades) + len(losing_trades)
        if total_trades > 0:
            win_rate = Decimal(str(len(winning_trades))) / Decimal(str(total_trades))
            loss_rate = Decimal("1") - win_rate
            expectancy = (win_rate * avg_win_size) - (loss_rate * avg_loss_size)
        else:
            expectancy = Decimal("0")

        return {
            "average_win_size": avg_win_size.quantize(Decimal("0.01")),
            "average_loss_size": avg_loss_size.quantize(Decimal("0.01")),
            "win_loss_ratio": win_loss_ratio.quantize(Decimal("0.01")),
            "expectancy": expectancy.quantize(Decimal("0.01")),
        }

    def _analyze_durations(
        self,
        winning_trades: list[Trade],
        losing_trades: list[Trade],
        positions: list[Position] | None = None,
    ) -> dict:
        """Analyze position holding durations."""
        # For now, use a default duration since we need position data
        # In production, this would query position entry/exit times

        if positions:
            # Match trades to positions to get durations
            win_durations = []
            loss_durations = []

            position_map = {p.position_id: p for p in positions}

            for trade in winning_trades:
                if trade.position_id and trade.position_id in position_map:
                    position = position_map[trade.position_id]
                    if position.updated_at and position.created_at:
                        duration = (
                            position.updated_at - position.created_at
                        ).total_seconds() / 3600
                        win_durations.append(Decimal(str(duration)))

            for trade in losing_trades:
                if trade.position_id and trade.position_id in position_map:
                    position = position_map[trade.position_id]
                    if position.updated_at and position.created_at:
                        duration = (
                            position.updated_at - position.created_at
                        ).total_seconds() / 3600
                        loss_durations.append(Decimal(str(duration)))

            # Calculate averages and medians
            avg_win_duration = (
                sum(win_durations) / Decimal(str(len(win_durations)))
                if win_durations
                else Decimal("24")  # Default 24 hours
            )
            avg_loss_duration = (
                sum(loss_durations) / Decimal(str(len(loss_durations)))
                if loss_durations
                else Decimal("12")  # Default 12 hours
            )

            median_win_duration = (
                sorted(win_durations)[len(win_durations) // 2]
                if win_durations
                else Decimal("24")
            )
            median_loss_duration = (
                sorted(loss_durations)[len(loss_durations) // 2]
                if loss_durations
                else Decimal("12")
            )
        else:
            # Use defaults when no position data available
            avg_win_duration = Decimal("24")
            avg_loss_duration = Decimal("12")
            median_win_duration = Decimal("24")
            median_loss_duration = Decimal("12")

        return {
            "average_win_duration_hours": avg_win_duration.quantize(Decimal("0.1")),
            "average_loss_duration_hours": avg_loss_duration.quantize(Decimal("0.1")),
            "median_win_duration_hours": median_win_duration.quantize(Decimal("0.1")),
            "median_loss_duration_hours": median_loss_duration.quantize(Decimal("0.1")),
        }

    def _analyze_temporal_patterns(self, trades: list[Trade]) -> dict:
        """Analyze win rates by time periods."""
        # Group trades by time periods
        trades_by_hour = defaultdict(list)
        trades_by_day_of_week = defaultdict(list)
        trades_by_day_of_month = defaultdict(list)

        for trade in trades:
            hour = trade.timestamp.hour
            day_of_week = trade.timestamp.weekday()  # 0=Monday, 6=Sunday
            day_of_month = trade.timestamp.day

            trades_by_hour[hour].append(trade)
            trades_by_day_of_week[day_of_week].append(trade)
            trades_by_day_of_month[day_of_month].append(trade)

        # Calculate win rates for each period
        win_rate_by_hour = {}
        for hour, hour_trades in trades_by_hour.items():
            wins = sum(1 for t in hour_trades if t.pnl_dollars > 0)
            win_rate_by_hour[hour] = (
                Decimal(str(wins)) / Decimal(str(len(hour_trades)))
                if hour_trades
                else Decimal("0")
            )

        win_rate_by_day_of_week = {}
        for day, day_trades in trades_by_day_of_week.items():
            wins = sum(1 for t in day_trades if t.pnl_dollars > 0)
            win_rate_by_day_of_week[day] = (
                Decimal(str(wins)) / Decimal(str(len(day_trades)))
                if day_trades
                else Decimal("0")
            )

        win_rate_by_day_of_month = {}
        for day, day_trades in trades_by_day_of_month.items():
            wins = sum(1 for t in day_trades if t.pnl_dollars > 0)
            win_rate_by_day_of_month[day] = (
                Decimal(str(wins)) / Decimal(str(len(day_trades)))
                if day_trades
                else Decimal("0")
            )

        return {
            "win_rate_by_hour": win_rate_by_hour,
            "win_rate_by_day_of_week": win_rate_by_day_of_week,
            "win_rate_by_day_of_month": win_rate_by_day_of_month,
        }

    def _analyze_close_reasons(self, positions: list[Position]) -> dict:
        """Analyze distribution of position close reasons."""
        if not positions:
            return {
                "close_reason_distribution": {},
                "win_close_reasons": {},
                "loss_close_reasons": {},
            }

        # Count close reasons
        all_reasons = Counter()
        win_reasons = Counter()
        loss_reasons = Counter()

        for position in positions:
            reason = getattr(position, "close_reason", "unknown")
            if reason:
                all_reasons[reason] += 1

                if position.pnl_dollars > 0:
                    win_reasons[reason] += 1
                else:
                    loss_reasons[reason] += 1

        return {
            "close_reason_distribution": dict(all_reasons),
            "win_close_reasons": dict(win_reasons),
            "loss_close_reasons": dict(loss_reasons),
        }

    def _analyze_recovery_patterns(self, trades: list[Trade]) -> dict:
        """Analyze patterns of recovery from losses."""
        if len(trades) < 2:
            return {
                "trades_recovering_from_drawdown": 0,
                "recovery_success_rate": Decimal("0"),
            }

        recovery_attempts = 0
        successful_recoveries = 0

        for i in range(1, len(trades)):
            prev_trade = trades[i - 1]
            curr_trade = trades[i]

            # If previous trade was a loss
            if prev_trade.pnl_dollars < 0:
                recovery_attempts += 1
                # Check if current trade recovered the loss
                if curr_trade.pnl_dollars > abs(prev_trade.pnl_dollars):
                    successful_recoveries += 1

        recovery_rate = (
            Decimal(str(successful_recoveries)) / Decimal(str(recovery_attempts))
            if recovery_attempts > 0
            else Decimal("0")
        )

        return {
            "trades_recovering_from_drawdown": successful_recoveries,
            "recovery_success_rate": recovery_rate.quantize(Decimal("0.01")),
        }

    def identify_best_trading_times(
        self, pattern: WinLossPattern, min_trades: int = 10
    ) -> dict:
        """
        Identify the best times to trade based on win rates.

        Args:
            pattern: WinLossPattern analysis result
            min_trades: Minimum trades required for significance

        Returns:
            Dictionary with best hours and days
        """
        best_hours = []
        best_days = []

        # Find best hours (with sufficient data)
        for hour, win_rate in pattern.win_rate_by_hour.items():
            # Would need trade count to properly filter
            # For now, just take high win rate hours
            if win_rate > Decimal("0.6"):
                best_hours.append({"hour": hour, "win_rate": str(win_rate)})

        # Find best days of week
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for day, win_rate in pattern.win_rate_by_day_of_week.items():
            if win_rate > Decimal("0.6"):
                best_days.append({"day": day_names[day], "win_rate": str(win_rate)})

        # Sort by win rate
        best_hours.sort(key=lambda x: x["win_rate"], reverse=True)
        best_days.sort(key=lambda x: x["win_rate"], reverse=True)

        return {
            "best_trading_hours": best_hours[:3],  # Top 3 hours
            "best_trading_days": best_days[:2],  # Top 2 days
            "worst_trading_hours": sorted(
                [
                    {"hour": h, "win_rate": str(r)}
                    for h, r in pattern.win_rate_by_hour.items()
                    if r < Decimal("0.4")
                ],
                key=lambda x: x["win_rate"],
            )[:3],
            "worst_trading_days": sorted(
                [
                    {"day": day_names[d], "win_rate": str(r)}
                    for d, r in pattern.win_rate_by_day_of_week.items()
                    if r < Decimal("0.4")
                ],
                key=lambda x: x["win_rate"],
            )[:2],
        }

    def _empty_pattern(self) -> WinLossPattern:
        """Return empty pattern analysis."""
        return WinLossPattern(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=Decimal("0"),
            current_streak=0,
            max_win_streak=0,
            max_loss_streak=0,
            average_win_streak=Decimal("0"),
            average_loss_streak=Decimal("0"),
            average_win_size=Decimal("0"),
            average_loss_size=Decimal("0"),
            win_loss_ratio=Decimal("0"),
            expectancy=Decimal("0"),
            average_win_duration_hours=Decimal("0"),
            average_loss_duration_hours=Decimal("0"),
            median_win_duration_hours=Decimal("0"),
            median_loss_duration_hours=Decimal("0"),
        )
