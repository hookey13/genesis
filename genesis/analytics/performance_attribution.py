"""
Performance Attribution Engine for Project GENESIS.

This module provides comprehensive performance attribution by strategy, trading pair,
and time period, enabling detailed analysis of trading edge sources.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.models import Trade
from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


class AttributionPeriod(str, Enum):
    """Time periods for attribution analysis."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""

    period_start: datetime
    period_end: datetime
    attribution_type: str  # 'strategy', 'pair', 'time'
    attribution_key: str  # strategy_id, symbol, or time period
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: Decimal
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    max_consecutive_wins: int
    max_consecutive_losses: int
    largest_win: Decimal
    largest_loss: Decimal
    total_volume: Decimal
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "attribution_type": self.attribution_type,
            "attribution_key": self.attribution_key,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl),
            "win_rate": str(self.win_rate),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "profit_factor": str(self.profit_factor),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "total_volume": str(self.total_volume),
            "metadata": self.metadata,
        }


class PerformanceAttributionEngine:
    """Engine for detailed performance attribution analysis."""

    def __init__(self, repository: Repository):
        """
        Initialize the performance attribution engine.

        Args:
            repository: Data repository for accessing trade history
        """
        self.repository = repository
        self._attribution_cache: dict[str, AttributionResult] = {}

    async def attribute_by_strategy(
        self, start_date: datetime, end_date: datetime, strategy_id: str | None = None
    ) -> list[AttributionResult]:
        """
        Attribute performance by strategy.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            strategy_id: Optional specific strategy to analyze

        Returns:
            List of attribution results by strategy
        """
        logger.info(
            "Attributing performance by strategy",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            strategy_id=strategy_id,
        )

        # Query trades from events table
        trades = await self._query_trades(start_date, end_date, strategy_id=strategy_id)

        # Group trades by strategy
        strategy_trades: dict[str, list[Trade]] = {}
        for trade in trades:
            if trade.strategy_id not in strategy_trades:
                strategy_trades[trade.strategy_id] = []
            strategy_trades[trade.strategy_id].append(trade)

        # Calculate attribution for each strategy
        results = []
        for strategy_id, trades_list in strategy_trades.items():
            result = self._calculate_attribution(
                trades_list, start_date, end_date, "strategy", strategy_id
            )
            results.append(result)

            # Store in database
            await self._store_attribution_result(result)

        return results

    async def attribute_by_pair(
        self, start_date: datetime, end_date: datetime, symbol: str | None = None
    ) -> list[AttributionResult]:
        """
        Attribute performance by trading pair.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            symbol: Optional specific symbol to analyze

        Returns:
            List of attribution results by trading pair
        """
        logger.info(
            "Attributing performance by pair",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            symbol=symbol,
        )

        # Query trades from events table
        trades = await self._query_trades(start_date, end_date, symbol=symbol)

        # Group trades by symbol
        symbol_trades: dict[str, list[Trade]] = {}
        for trade in trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)

        # Calculate attribution for each symbol
        results = []
        for symbol, trades_list in symbol_trades.items():
            result = self._calculate_attribution(
                trades_list, start_date, end_date, "pair", symbol
            )
            results.append(result)

            # Store in database
            await self._store_attribution_result(result)

        return results

    async def attribute_by_time_period(
        self, start_date: datetime, end_date: datetime, period: AttributionPeriod
    ) -> list[AttributionResult]:
        """
        Attribute performance by time period.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            period: Time period for grouping (hourly, daily, weekly, etc.)

        Returns:
            List of attribution results by time period
        """
        logger.info(
            "Attributing performance by time period",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            period=period,
        )

        # Query trades from events table
        trades = await self._query_trades(start_date, end_date)

        # Group trades by time period
        period_trades: dict[str, list[Trade]] = {}
        for trade in trades:
            period_key = self._get_period_key(trade.timestamp, period)
            if period_key not in period_trades:
                period_trades[period_key] = []
            period_trades[period_key].append(trade)

        # Calculate attribution for each period
        results = []
        for period_key, trades_list in period_trades.items():
            # Calculate period bounds
            period_start, period_end = self._get_period_bounds(period_key, period)

            result = self._calculate_attribution(
                trades_list, period_start, period_end, "time", period_key
            )
            results.append(result)

            # Store in database
            await self._store_attribution_result(result)

        return results

    async def get_mae_analysis(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Analyze Maximum Adverse Excursion for positions.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            MAE analysis results
        """
        logger.info(
            "Analyzing MAE",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        # Query position events for MAE tracking
        positions = await self.repository.query_positions_with_mae(start_date, end_date)

        mae_stats = {
            "average_mae": Decimal("0"),
            "max_mae": Decimal("0"),
            "min_mae": Decimal("0"),
            "recovered_positions": 0,
            "total_positions": len(positions),
            "recovery_rate": Decimal("0"),
            "mae_by_strategy": {},
            "mae_by_pair": {},
        }

        if positions:
            total_mae = Decimal("0")
            min_mae = None

            for position in positions:
                mae = position.get("max_adverse_excursion", Decimal("0"))
                total_mae += mae

                if mae > mae_stats["max_mae"]:
                    mae_stats["max_mae"] = mae

                if min_mae is None or mae < min_mae:
                    min_mae = mae

                # Check if position recovered (use recovered_from_mae if available)
                if position.get("recovered_from_mae", False) or (position.get("pnl_dollars", Decimal("0")) > 0 and mae > Decimal(
                    "0"
                )):
                    mae_stats["recovered_positions"] += 1

                # Group by strategy
                strategy_id = position.get("strategy_id")
                if strategy_id:
                    if strategy_id not in mae_stats["mae_by_strategy"]:
                        mae_stats["mae_by_strategy"][strategy_id] = []
                    mae_stats["mae_by_strategy"][strategy_id].append(mae)

                # Group by pair
                symbol = position.get("symbol")
                if symbol:
                    if symbol not in mae_stats["mae_by_pair"]:
                        mae_stats["mae_by_pair"][symbol] = []
                    mae_stats["mae_by_pair"][symbol].append(mae)

            mae_stats["average_mae"] = total_mae / Decimal(str(len(positions)))
            mae_stats["min_mae"] = min_mae if min_mae is not None else Decimal("0")
            mae_stats["recovery_rate"] = (
                Decimal(str(mae_stats["recovered_positions"]))
                / Decimal(str(len(positions)))
                if len(positions) > 0
                else Decimal("0")
            )

        return mae_stats

    def _calculate_attribution(
        self,
        trades: list[Trade],
        start_date: datetime,
        end_date: datetime,
        attribution_type: str,
        attribution_key: str,
    ) -> AttributionResult:
        """Calculate attribution metrics for a group of trades."""
        if not trades:
            return AttributionResult(
                period_start=start_date,
                period_end=end_date,
                attribution_type=attribution_type,
                attribution_key=attribution_key,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=Decimal("0"),
                win_rate=Decimal("0"),
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                profit_factor=Decimal("0"),
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                total_volume=Decimal("0"),
            )

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl_dollars > 0]
        losing_trades = [t for t in trades if t.pnl_dollars <= 0]

        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        total_pnl = sum(t.pnl_dollars for t in trades)
        total_volume = sum(t.quantity * t.exit_price for t in trades)

        # Calculate averages
        average_win = (
            sum(t.pnl_dollars for t in winning_trades) / Decimal(str(winning_count))
            if winning_count > 0
            else Decimal("0")
        )
        average_loss = (
            abs(sum(t.pnl_dollars for t in losing_trades)) / Decimal(str(losing_count))
            if losing_count > 0
            else Decimal("0")
        )

        # Calculate win rate
        win_rate = (
            Decimal(str(winning_count)) / Decimal(str(total_trades))
            if total_trades > 0
            else Decimal("0")
        )

        # Calculate profit factor
        total_wins = sum(t.pnl_dollars for t in winning_trades)
        total_losses = abs(sum(t.pnl_dollars for t in losing_trades))
        profit_factor = (
            total_wins / total_losses if total_losses > 0 else Decimal("999.99")
        )

        # Calculate consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in sorted_trades:
            if trade.pnl_dollars > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        # Find largest win/loss
        largest_win = max((t.pnl_dollars for t in winning_trades), default=Decimal("0"))
        largest_loss = min((t.pnl_dollars for t in losing_trades), default=Decimal("0"))

        return AttributionResult(
            period_start=start_date,
            period_end=end_date,
            attribution_type=attribution_type,
            attribution_key=attribution_key,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            total_pnl=total_pnl,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_volume=total_volume,
        )

    async def _query_trades(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy_id: str | None = None,
        symbol: str | None = None,
    ) -> list[Trade]:
        """Query trades from the events table."""
        # Query events of type 'trade_completed' from repository
        events = await self.repository.query_events(
            event_type="trade_completed", start_date=start_date, end_date=end_date
        )

        trades = []
        for event in events:
            event_data = event.get("event_data", {})

            # Filter by strategy_id if provided
            if strategy_id and event_data.get("strategy_id") != strategy_id:
                continue

            # Filter by symbol if provided
            if symbol and event_data.get("symbol") != symbol:
                continue

            # Convert event to Trade object
            trade = Trade(
                trade_id=event_data.get("trade_id"),
                order_id=event_data.get("order_id"),
                position_id=event_data.get("position_id"),
                strategy_id=event_data.get("strategy_id"),
                symbol=event_data.get("symbol"),
                side=event_data.get("side"),
                entry_price=Decimal(str(event_data.get("entry_price", "0"))),
                exit_price=Decimal(str(event_data.get("exit_price", "0"))),
                quantity=Decimal(str(event_data.get("quantity", "0"))),
                pnl_dollars=Decimal(str(event_data.get("pnl_dollars", "0"))),
                pnl_percent=Decimal(str(event_data.get("pnl_percent", "0"))),
                timestamp=datetime.fromisoformat(event.get("created_at")),
            )
            trades.append(trade)

        return trades

    async def _store_attribution_result(self, result: AttributionResult) -> None:
        """Store attribution result in database."""
        await self.repository.store_attribution_result(result.to_dict())

    def _get_period_key(self, timestamp: datetime, period: AttributionPeriod) -> str:
        """Get period key for grouping trades."""
        if period == AttributionPeriod.HOURLY:
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif period == AttributionPeriod.DAILY:
            return timestamp.strftime("%Y-%m-%d")
        elif period == AttributionPeriod.WEEKLY:
            # Get start of week (Monday)
            week_start = timestamp - timedelta(days=timestamp.weekday())
            return week_start.strftime("%Y-W%V")
        elif period == AttributionPeriod.MONTHLY:
            return timestamp.strftime("%Y-%m")
        elif period == AttributionPeriod.QUARTERLY:
            quarter = (timestamp.month - 1) // 3 + 1
            return f"{timestamp.year}-Q{quarter}"
        elif period == AttributionPeriod.YEARLY:
            return str(timestamp.year)
        else:
            return timestamp.strftime("%Y-%m-%d")

    def _get_period_bounds(
        self, period_key: str, period: AttributionPeriod
    ) -> tuple[datetime, datetime]:
        """Get start and end datetime for a period key."""
        if period == AttributionPeriod.HOURLY:
            start = datetime.strptime(period_key, "%Y-%m-%d %H:00")
            end = start + timedelta(hours=1)
        elif period == AttributionPeriod.DAILY:
            start = datetime.strptime(period_key, "%Y-%m-%d")
            end = start + timedelta(days=1)
        elif period == AttributionPeriod.WEEKLY:
            year, week = period_key.split("-W")
            start = datetime.strptime(f"{year}-W{week}-1", "%Y-W%V-%w")
            end = start + timedelta(weeks=1)
        elif period == AttributionPeriod.MONTHLY:
            year, month = map(int, period_key.split("-"))
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
        elif period == AttributionPeriod.QUARTERLY:
            year, quarter = period_key.split("-Q")
            year = int(year)
            quarter = int(quarter)
            start_month = (quarter - 1) * 3 + 1
            start = datetime(year, start_month, 1)
            if quarter == 4:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, start_month + 3, 1)
        elif period == AttributionPeriod.YEARLY:
            year = int(period_key)
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
        else:
            start = datetime.strptime(period_key, "%Y-%m-%d")
            end = start + timedelta(days=1)

        # Ensure timezone aware
        start = start.replace(tzinfo=UTC)
        end = end.replace(tzinfo=UTC)

        return start, end
