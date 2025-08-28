"""
Performance Data Repository for Project GENESIS.

This module provides persistence and retrieval for all performance analytics data.
"""

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

import structlog

from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


class PerformanceRepository(Repository):
    """Repository for performance analytics data."""

    async def store_attribution_result(self, result: dict) -> None:
        """
        Store performance attribution result.

        Args:
            result: Attribution result dictionary
        """
        try:
            # Store in database with proper serialization
            await self.execute(
                """
                INSERT INTO attribution_results 
                (period_start, period_end, attribution_type, attribution_key, 
                 total_trades, winning_trades, losing_trades, total_pnl, 
                 win_rate, profit_factor, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("period_start"),
                    result.get("period_end"),
                    result.get("attribution_type"),
                    result.get("attribution_key"),
                    result.get("total_trades"),
                    result.get("winning_trades"),
                    result.get("losing_trades"),
                    str(result.get("total_pnl", "0")),
                    str(result.get("win_rate", "0")),
                    str(result.get("profit_factor", "0")),
                    json.dumps(result.get("metadata", {})),
                    datetime.now(UTC).isoformat(),
                ),
            )

            logger.debug(
                "Stored attribution result",
                type=result.get("attribution_type"),
                key=result.get("attribution_key"),
            )
        except Exception as e:
            logger.error(f"Failed to store attribution result: {e}")
            raise

    async def query_events(
        self,
        event_type: str,
        start_date: datetime,
        end_date: datetime,
        aggregate_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Query events from the event store.

        Args:
            event_type: Type of event to query
            start_date: Start of date range
            end_date: End of date range
            aggregate_id: Optional specific aggregate ID

        Returns:
            List of event records
        """
        query = """
            SELECT event_id, event_type, aggregate_id, aggregate_type, 
                   event_data, created_at
            FROM events
            WHERE event_type = ?
              AND created_at >= ?
              AND created_at <= ?
        """
        params = [event_type, start_date.isoformat(), end_date.isoformat()]

        if aggregate_id:
            query += " AND aggregate_id = ?"
            params.append(aggregate_id)

        query += " ORDER BY created_at ASC"

        try:
            rows = await self.fetch_all(query, params)

            events = []
            for row in rows:
                events.append(
                    {
                        "event_id": row[0],
                        "event_type": row[1],
                        "aggregate_id": row[2],
                        "aggregate_type": row[3],
                        "event_data": json.loads(row[4]) if row[4] else {},
                        "created_at": row[5],
                    }
                )

            return events
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return []

    async def query_positions_with_mae(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """
        Query positions with MAE data.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of position records with MAE
        """
        query = """
            SELECT position_id, symbol, strategy_id, pnl_dollars, 
                   max_adverse_excursion, created_at, closed_at
            FROM positions
            WHERE created_at >= ?
              AND created_at <= ?
              AND closed_at IS NOT NULL
            ORDER BY created_at ASC
        """

        try:
            rows = await self.fetch_all(
                query, (start_date.isoformat(), end_date.isoformat())
            )

            positions = []
            for row in rows:
                positions.append(
                    {
                        "position_id": row[0],
                        "symbol": row[1],
                        "strategy_id": row[2],
                        "pnl_dollars": Decimal(row[3]) if row[3] else Decimal("0"),
                        "max_adverse_excursion": (
                            Decimal(row[4]) if row[4] else Decimal("0")
                        ),
                        "created_at": row[5],
                        "closed_at": row[6],
                    }
                )

            return positions
        except Exception as e:
            logger.error(f"Failed to query positions with MAE: {e}")
            return []

    async def store_risk_metrics(
        self, metrics: dict, period_start: datetime, period_end: datetime
    ) -> None:
        """
        Store risk metrics calculation.

        Args:
            metrics: Risk metrics dictionary
            period_start: Start of calculation period
            period_end: End of calculation period
        """
        try:
            await self.execute(
                """
                INSERT INTO risk_metrics
                (period_start, period_end, sharpe_ratio, sortino_ratio,
                 calmar_ratio, max_drawdown, volatility, value_at_risk,
                 metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    period_start.isoformat(),
                    period_end.isoformat(),
                    str(metrics.get("sharpe_ratio", "0")),
                    str(metrics.get("sortino_ratio", "0")),
                    str(metrics.get("calmar_ratio", "0")),
                    str(metrics.get("max_drawdown", "0")),
                    str(metrics.get("volatility", "0")),
                    str(metrics.get("value_at_risk_95", "0")),
                    json.dumps(metrics),
                    datetime.now(UTC).isoformat(),
                ),
            )

            logger.debug("Stored risk metrics", period_start=period_start)
        except Exception as e:
            logger.error(f"Failed to store risk metrics: {e}")
            raise

    async def store_pattern_analysis(
        self, pattern: dict, analysis_date: datetime
    ) -> None:
        """
        Store win/loss pattern analysis.

        Args:
            pattern: Pattern analysis dictionary
            analysis_date: Date of analysis
        """
        try:
            await self.execute(
                """
                INSERT INTO pattern_analysis
                (analysis_date, total_trades, win_rate, max_win_streak,
                 max_loss_streak, avg_win_size, avg_loss_size,
                 win_loss_ratio, expectancy, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_date.isoformat(),
                    pattern.get("total_trades", 0),
                    str(pattern.get("win_rate", "0")),
                    pattern.get("max_win_streak", 0),
                    pattern.get("max_loss_streak", 0),
                    str(pattern.get("average_win_size", "0")),
                    str(pattern.get("average_loss_size", "0")),
                    str(pattern.get("win_loss_ratio", "0")),
                    str(pattern.get("expectancy", "0")),
                    json.dumps(pattern),
                    datetime.now(UTC).isoformat(),
                ),
            )

            logger.debug("Stored pattern analysis", date=analysis_date)
        except Exception as e:
            logger.error(f"Failed to store pattern analysis: {e}")
            raise

    async def get_latest_risk_metrics(self) -> Optional[dict]:
        """
        Get the most recent risk metrics.

        Returns:
            Latest risk metrics or None
        """
        query = """
            SELECT sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                   volatility, value_at_risk, metadata, created_at
            FROM risk_metrics
            ORDER BY created_at DESC
            LIMIT 1
        """

        try:
            row = await self.fetch_one(query)

            if row:
                return {
                    "sharpe_ratio": Decimal(row[0]) if row[0] else Decimal("0"),
                    "sortino_ratio": Decimal(row[1]) if row[1] else Decimal("0"),
                    "calmar_ratio": Decimal(row[2]) if row[2] else Decimal("0"),
                    "max_drawdown": Decimal(row[3]) if row[3] else Decimal("0"),
                    "volatility": Decimal(row[4]) if row[4] else Decimal("0"),
                    "value_at_risk": Decimal(row[5]) if row[5] else Decimal("0"),
                    "metadata": json.loads(row[6]) if row[6] else {},
                    "created_at": row[7],
                }

            return None
        except Exception as e:
            logger.error(f"Failed to get latest risk metrics: {e}")
            return None

    async def get_attribution_history(
        self, attribution_type: str, days: int = 30
    ) -> list[dict]:
        """
        Get historical attribution results.

        Args:
            attribution_type: Type of attribution (strategy, pair, time)
            days: Number of days of history

        Returns:
            List of attribution results
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        query = """
            SELECT period_start, period_end, attribution_key, total_trades,
                   winning_trades, losing_trades, total_pnl, win_rate,
                   profit_factor, metadata
            FROM attribution_results
            WHERE attribution_type = ?
              AND created_at >= ?
            ORDER BY period_start DESC
        """

        try:
            rows = await self.fetch_all(
                query, (attribution_type, cutoff_date.isoformat())
            )

            results = []
            for row in rows:
                results.append(
                    {
                        "period_start": row[0],
                        "period_end": row[1],
                        "attribution_key": row[2],
                        "total_trades": row[3],
                        "winning_trades": row[4],
                        "losing_trades": row[5],
                        "total_pnl": Decimal(row[6]) if row[6] else Decimal("0"),
                        "win_rate": Decimal(row[7]) if row[7] else Decimal("0"),
                        "profit_factor": Decimal(row[8]) if row[8] else Decimal("0"),
                        "metadata": json.loads(row[9]) if row[9] else {},
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Failed to get attribution history: {e}")
            return []

    async def cache_calculation(
        self, cache_key: str, result: Any, ttl_seconds: int = 3600
    ) -> None:
        """
        Cache expensive calculation results.

        Args:
            cache_key: Unique key for the calculation
            result: Result to cache
            ttl_seconds: Time to live in seconds
        """
        expiry = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

        try:
            await self.execute(
                """
                INSERT OR REPLACE INTO calculation_cache
                (cache_key, result, expiry, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    json.dumps(result, default=str),
                    expiry.isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )

            logger.debug(f"Cached calculation: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache calculation: {e}")

    async def get_cached_calculation(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached calculation if not expired.

        Args:
            cache_key: Unique key for the calculation

        Returns:
            Cached result or None if expired/not found
        """
        query = """
            SELECT result, expiry
            FROM calculation_cache
            WHERE cache_key = ?
        """

        try:
            row = await self.fetch_one(query, (cache_key,))

            if row:
                expiry = datetime.fromisoformat(row[1])
                if expiry > datetime.now(UTC):
                    return json.loads(row[0])
                else:
                    # Clean up expired cache
                    await self.execute(
                        "DELETE FROM calculation_cache WHERE cache_key = ?",
                        (cache_key,),
                    )

            return None
        except Exception as e:
            logger.error(f"Failed to get cached calculation: {e}")
            return None

    async def ensure_tables(self) -> None:
        """Ensure all performance-related tables exist."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS attribution_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                attribution_type TEXT NOT NULL,
                attribution_key TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl TEXT,
                win_rate TEXT,
                profit_factor TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                sharpe_ratio TEXT,
                sortino_ratio TEXT,
                calmar_ratio TEXT,
                max_drawdown TEXT,
                volatility TEXT,
                value_at_risk TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pattern_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT NOT NULL,
                total_trades INTEGER,
                win_rate TEXT,
                max_win_streak INTEGER,
                max_loss_streak INTEGER,
                avg_win_size TEXT,
                avg_loss_size TEXT,
                win_loss_ratio TEXT,
                expectancy TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS calculation_cache (
                cache_key TEXT PRIMARY KEY,
                result TEXT,
                expiry TEXT,
                created_at TEXT
            )
            """,
        ]

        for table_sql in tables:
            try:
                await self.execute(table_sql)
            except Exception as e:
                logger.error(f"Failed to create table: {e}")
