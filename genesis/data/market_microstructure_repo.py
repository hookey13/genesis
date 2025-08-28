"""Market Microstructure Data Repository."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class MarketMicrostructureRepository:
    """Repository for market microstructure data."""

    def __init__(self, db_path: str = ".genesis/data/genesis.db"):
        """Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None
        self._cache = {}
        self._cache_ttl = timedelta(seconds=30)

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        self.connection = await aiosqlite.connect(self.db_path)

        # Create tables if they don't exist
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                mid_price REAL,
                spread_bps INTEGER,
                imbalance_ratio REAL,
                bid_depth INTEGER,
                ask_depth INTEGER,
                total_bid_volume REAL,
                total_ask_volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS order_book_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                side TEXT NOT NULL,
                level INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                FOREIGN KEY (snapshot_id) REFERENCES order_book_snapshots(snapshot_id)
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS order_flow_metrics (
                metrics_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                ofi REAL NOT NULL,
                volume_ratio REAL NOT NULL,
                pressure_score REAL NOT NULL,
                net_flow REAL NOT NULL,
                trend TEXT,
                confidence REAL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS large_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                size REAL NOT NULL,
                price REAL NOT NULL,
                side TEXT NOT NULL,
                vpin_score REAL,
                cluster_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS manipulation_events (
                event_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS microstructure_regimes (
                regime_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL NOT NULL,
                metrics TEXT,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create missing tables referenced in the code
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS whale_activities (
                activity_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                trade_size REAL NOT NULL,
                price REAL NOT NULL,
                side TEXT NOT NULL,
                percentile REAL,
                vpin_score REAL,
                cluster_id TEXT,
                confidence REAL,
                notional REAL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS manipulation_patterns (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                manipulation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                cancellation_rate REAL,
                total_volume REAL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS microstructure_states (
                state_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                regime_confidence REAL NOT NULL,
                flow_imbalance REAL NOT NULL,
                whale_activity INTEGER NOT NULL,
                manipulation_detected INTEGER NOT NULL,
                toxicity REAL NOT NULL,
                execution_quality REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        await self.connection.commit()
        logger.info(
            "Market microstructure repository initialized", db_path=self.db_path
        )

    async def shutdown(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            logger.info("Market microstructure repository shutdown")

    async def store_order_book_snapshot(
        self,
        symbol: str,
        bids: list[dict[str, Any]],
        asks: list[dict[str, Any]],
        mid_price: Decimal | None,
        spread_bps: int | None,
        imbalance_ratio: Decimal | None,
        timestamp: datetime | None = None,
    ) -> str:
        """Store order book snapshot.

        Args:
            symbol: Trading symbol
            bids: List of bid levels
            asks: List of ask levels
            mid_price: Mid price
            spread_bps: Spread in basis points
            imbalance_ratio: Order book imbalance
            timestamp: Snapshot timestamp

        Returns:
            Snapshot ID
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        snapshot_id = str(uuid4())

        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        # Store snapshot metadata
        await self.connection.execute(
            """
            INSERT INTO order_book_snapshots
            (snapshot_id, symbol, timestamp, mid_price, spread_bps, imbalance_ratio,
             bid_depth, ask_depth, total_bid_volume, total_ask_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                symbol,
                timestamp,
                float(mid_price) if mid_price else None,
                spread_bps,
                float(imbalance_ratio) if imbalance_ratio else None,
                len(bids),
                len(asks),
                float(sum(Decimal(str(b["quantity"])) for b in bids)),
                float(sum(Decimal(str(a["quantity"])) for a in asks)),
            ),
        )

        # Store bid/ask levels
        for i, bid in enumerate(bids[:20]):  # Store top 20 levels
            await self.connection.execute(
                """
                INSERT INTO order_book_levels
                (snapshot_id, side, level, price, quantity)
                VALUES (?, ?, ?, ?, ?)
                """,
                (snapshot_id, "bid", i, float(bid["price"]), float(bid["quantity"])),
            )

        for i, ask in enumerate(asks[:20]):
            await self.connection.execute(
                """
                INSERT INTO order_book_levels
                (snapshot_id, side, level, price, quantity)
                VALUES (?, ?, ?, ?, ?)
                """,
                (snapshot_id, "ask", i, float(ask["price"]), float(ask["quantity"])),
            )

        await self.connection.commit()

        logger.debug(
            "order_book_snapshot_stored",
            symbol=symbol,
            snapshot_id=snapshot_id,
            bid_levels=len(bids),
            ask_levels=len(asks),
        )

        return snapshot_id

    async def store_order_flow_metrics(
        self,
        symbol: str,
        ofi: Decimal,
        volume_ratio: Decimal,
        pressure_score: Decimal,
        net_flow: Decimal,
        confidence: Decimal,
        timestamp: datetime | None = None,
    ) -> str:
        """Store order flow metrics.

        Args:
            symbol: Trading symbol
            ofi: Order flow imbalance
            volume_ratio: Buy/sell volume ratio
            pressure_score: Pressure score
            net_flow: Net flow
            confidence: Confidence score
            timestamp: Metrics timestamp

        Returns:
            Metrics ID
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        metrics_id = str(uuid4())

        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        await self.connection.execute(
            """
            INSERT INTO order_flow_metrics
            (metrics_id, symbol, timestamp, ofi, volume_ratio,
             pressure_score, net_flow, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics_id,
                symbol,
                timestamp,
                float(ofi),
                float(volume_ratio),
                float(pressure_score),
                float(net_flow),
                float(confidence),
            ),
        )

        await self.connection.commit()

        return metrics_id

    async def store_whale_activity(
        self,
        symbol: str,
        trade_size: Decimal,
        price: Decimal,
        side: str,
        percentile: Decimal,
        vpin_score: Decimal,
        cluster_id: str | None,
        confidence: Decimal,
        timestamp: datetime | None = None,
    ) -> str:
        """Store whale activity detection.

        Args:
            symbol: Trading symbol
            trade_size: Trade size
            price: Trade price
            side: Trade side
            percentile: Size percentile
            vpin_score: VPIN score
            cluster_id: Cluster ID if part of cluster
            confidence: Detection confidence
            timestamp: Activity timestamp

        Returns:
            Activity ID
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        activity_id = str(uuid4())

        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        await self.connection.execute(
            """
            INSERT INTO whale_activities
            (activity_id, symbol, timestamp, trade_size, price, side,
             percentile, vpin_score, cluster_id, confidence, notional)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                activity_id,
                symbol,
                timestamp,
                float(trade_size),
                float(price),
                side,
                float(percentile),
                float(vpin_score),
                cluster_id,
                float(confidence),
                float(trade_size * price),
            ),
        )

        await self.connection.commit()

        return activity_id

    async def store_manipulation_pattern(
        self,
        pattern_id: str,
        symbol: str,
        manipulation_type: str,
        confidence: Decimal,
        severity: str,
        cancellation_rate: Decimal,
        total_volume: Decimal,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> None:
        """Store detected manipulation pattern.

        Args:
            pattern_id: Pattern ID
            symbol: Trading symbol
            manipulation_type: Type of manipulation
            confidence: Detection confidence
            severity: Severity level
            cancellation_rate: Order cancellation rate
            total_volume: Total volume involved
            start_time: Pattern start time
            end_time: Pattern end time
        """
        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        await self.connection.execute(
            """
            INSERT INTO manipulation_patterns
            (pattern_id, symbol, manipulation_type, confidence, severity,
             cancellation_rate, total_volume, start_time, end_time, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern_id,
                symbol,
                manipulation_type,
                float(confidence),
                severity,
                float(cancellation_rate),
                float(total_volume),
                start_time,
                end_time,
                (end_time - start_time).total_seconds() if end_time else None,
            ),
        )

        await self.connection.commit()

    async def store_microstructure_state(
        self,
        symbol: str,
        regime: str,
        regime_confidence: Decimal,
        flow_imbalance: Decimal,
        whale_activity: bool,
        manipulation_detected: bool,
        toxicity: Decimal,
        execution_quality: Decimal,
        timestamp: datetime | None = None,
    ) -> str:
        """Store microstructure state.

        Args:
            symbol: Trading symbol
            regime: Market regime
            regime_confidence: Regime confidence
            flow_imbalance: Flow imbalance
            whale_activity: Whale activity detected
            manipulation_detected: Manipulation detected
            toxicity: Toxicity score
            execution_quality: Execution quality score
            timestamp: State timestamp

        Returns:
            State ID
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        state_id = str(uuid4())

        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        await self.connection.execute(
            """
            INSERT INTO microstructure_states
            (state_id, symbol, timestamp, regime, regime_confidence,
             flow_imbalance, whale_activity, manipulation_detected,
             toxicity, execution_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state_id,
                symbol,
                timestamp,
                regime,
                float(regime_confidence),
                float(flow_imbalance),
                whale_activity,
                manipulation_detected,
                float(toxicity),
                float(execution_quality),
            ),
        )

        await self.connection.commit()

        # Invalidate cache for this symbol
        self._invalidate_cache(symbol)

        return state_id

    async def get_recent_order_books(
        self, symbol: str, limit: int = 100, time_window: timedelta | None = None
    ) -> list[dict[str, Any]]:
        """Get recent order book snapshots.

        Args:
            symbol: Trading symbol
            limit: Maximum number of snapshots
            time_window: Time window for snapshots

        Returns:
            List of order book snapshots
        """
        cache_key = f"order_books_{symbol}_{limit}"

        # Check cache
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now(UTC) - cached_time < self._cache_ttl:
                return cached_data

        query = """
            SELECT * FROM order_book_snapshots
            WHERE symbol = ?
        """
        params = [symbol]

        if time_window:
            cutoff = datetime.now(UTC) - time_window
            query += " AND timestamp > ?"
            params.append(cutoff)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        result = await self.connection.execute(query, params)
        snapshots = result.fetchall()

        # Cache result
        self._cache[cache_key] = (snapshots, datetime.now(UTC))

        return snapshots

    async def get_order_flow_history(
        self, symbol: str, time_window: timedelta
    ) -> list[dict[str, Any]]:
        """Get order flow metrics history.

        Args:
            symbol: Trading symbol
            time_window: Time window

        Returns:
            List of order flow metrics
        """
        cutoff = datetime.now(UTC) - time_window

        result = await self.connection.execute(
            """
            SELECT * FROM order_flow_metrics
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """,
            (symbol, cutoff),
        )

        return result.fetchall()

    async def get_whale_activities(
        self,
        symbol: str,
        time_window: timedelta,
        min_percentile: Decimal = Decimal("95"),
    ) -> list[dict[str, Any]]:
        """Get whale activities.

        Args:
            symbol: Trading symbol
            time_window: Time window
            min_percentile: Minimum percentile threshold

        Returns:
            List of whale activities
        """
        cutoff = datetime.now(UTC) - time_window

        result = await self.connection.execute(
            """
            SELECT * FROM whale_activities
            WHERE symbol = ? AND timestamp > ? AND percentile >= ?
            ORDER BY timestamp DESC
            """,
            (symbol, cutoff, float(min_percentile)),
        )

        return result.fetchall()

    async def get_manipulation_patterns(
        self,
        symbol: str,
        time_window: timedelta,
        min_confidence: Decimal = Decimal("0.7"),
    ) -> list[dict[str, Any]]:
        """Get detected manipulation patterns.

        Args:
            symbol: Trading symbol
            time_window: Time window
            min_confidence: Minimum confidence threshold

        Returns:
            List of manipulation patterns
        """
        cutoff = datetime.now(UTC) - time_window

        result = await self.connection.execute(
            """
            SELECT * FROM manipulation_patterns
            WHERE symbol = ? AND start_time > ? AND confidence >= ?
            ORDER BY start_time DESC
            """,
            (symbol, cutoff, float(min_confidence)),
        )

        return result.fetchall()

    async def get_current_state(self, symbol: str) -> dict[str, Any] | None:
        """Get current microstructure state.

        Args:
            symbol: Trading symbol

        Returns:
            Current state or None
        """
        result = await self.connection.execute(
            """
            SELECT * FROM microstructure_states
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,),
        )

        return result.fetchone()

    async def get_regime_history(
        self, symbol: str, time_window: timedelta
    ) -> list[dict[str, Any]]:
        """Get regime transition history.

        Args:
            symbol: Trading symbol
            time_window: Time window

        Returns:
            List of regime states
        """
        cutoff = datetime.now(UTC) - time_window

        result = await self.connection.execute(
            """
            SELECT regime, regime_confidence, timestamp
            FROM microstructure_states
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp ASC
            """,
            (symbol, cutoff),
        )

        return result.fetchall()

    async def calculate_toxicity_statistics(
        self, symbol: str, time_window: timedelta
    ) -> dict[str, Any]:
        """Calculate toxicity statistics.

        Args:
            symbol: Trading symbol
            time_window: Time window

        Returns:
            Toxicity statistics
        """
        cutoff = datetime.now(UTC) - time_window

        # Get manipulation count
        manip_result = await self.connection.execute(
            """
            SELECT COUNT(*) as count
            FROM manipulation_patterns
            WHERE symbol = ? AND start_time > ?
            """,
            (symbol, cutoff),
        )
        manip_row = await manip_result.fetchone()
        manip_count = manip_row[0] if manip_row else 0

        # Get average toxicity from states
        toxicity_result = await self.connection.execute(
            """
            SELECT AVG(toxicity) as avg_toxicity,
                   MAX(toxicity) as max_toxicity,
                   MIN(toxicity) as min_toxicity
            FROM microstructure_states
            WHERE symbol = ? AND timestamp > ?
            """,
            (symbol, cutoff),
        )
        toxicity_row = await toxicity_result.fetchone()

        # Get whale activity frequency
        whale_result = await self.connection.execute(
            """
            SELECT COUNT(DISTINCT cluster_id) as cluster_count,
                   COUNT(*) as total_activities
            FROM whale_activities
            WHERE symbol = ? AND timestamp > ?
            """,
            (symbol, cutoff),
        )
        whale_row = await whale_result.fetchone()

        return {
            "symbol": symbol,
            "time_window_hours": time_window.total_seconds() / 3600,
            "manipulation_events": manip_count,
            "avg_toxicity": toxicity_row[0] if toxicity_row and toxicity_row[0] else 0,
            "max_toxicity": toxicity_row[1] if toxicity_row and toxicity_row[1] else 0,
            "min_toxicity": toxicity_row[2] if toxicity_row and toxicity_row[2] else 0,
            "whale_clusters": whale_row[0] if whale_row and whale_row[0] else 0,
            "whale_activities": whale_row[1] if whale_row and whale_row[1] else 0,
        }

    def _invalidate_cache(self, symbol: str) -> None:
        """Invalidate cache for a symbol.

        Args:
            symbol: Trading symbol
        """
        keys_to_remove = [k for k in self._cache if symbol in k]
        for key in keys_to_remove:
            del self._cache[key]

    async def cleanup_old_data(self, retention_days: int = 7) -> int:
        """Clean up old microstructure data.

        Args:
            retention_days: Days to retain data

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        total_deleted = 0

        if not self.connection:
            raise RuntimeError("Repository not initialized. Call initialize() first.")

        # Clean up order book snapshots and levels
        result = await self.connection.execute(
            "DELETE FROM order_book_levels WHERE snapshot_id IN "
            "(SELECT snapshot_id FROM order_book_snapshots WHERE timestamp < ?)",
            (cutoff,),
        )
        total_deleted += result.rowcount or 0

        result = await self.connection.execute(
            "DELETE FROM order_book_snapshots WHERE timestamp < ?", (cutoff,)
        )
        total_deleted += result.rowcount or 0

        # Clean up other tables
        for table in [
            "order_flow_metrics",
            "whale_activities",
            "manipulation_patterns",
            "microstructure_states",
        ]:
            result = await self.connection.execute(
                f"DELETE FROM {table} WHERE timestamp < ? OR start_time < ?",
                (cutoff, cutoff),
            )
            total_deleted += result.rowcount or 0

        await self.connection.commit()

        logger.info(
            "microstructure_data_cleanup",
            retention_days=retention_days,
            records_deleted=total_deleted,
        )

        return total_deleted
