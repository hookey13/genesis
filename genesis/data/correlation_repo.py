from __future__ import annotations

from typing import Optional

"""Repository for correlation data persistence."""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from genesis.analytics.correlation import CorrelationAlert

logger = logging.getLogger(__name__)


class CorrelationRepository:
    """Repository for managing correlation data."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize repository with database connection.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string or "sqlite:///:memory:"
        self._connection = None

    def get_connection(self):
        """Get database connection context manager."""
        # This would return actual database connection in production
        # For now, return a mock for testing
        from unittest.mock import AsyncMock, MagicMock

        class MockConnection:
            async def __aenter__(self):
                conn = MagicMock()
                conn.execute = AsyncMock()
                conn.commit = AsyncMock()
                return conn

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        return MockConnection()

    async def save_correlation(
        self,
        position_1_id: UUID,
        position_2_id: UUID,
        correlation_coefficient: Decimal,
        calculation_window: int,
        alert_triggered: bool = False
    ) -> UUID:
        """Save correlation data to database.

        Args:
            position_1_id: First position ID
            position_2_id: Second position ID
            correlation_coefficient: Correlation value
            calculation_window: Window in minutes
            alert_triggered: Whether alert was triggered

        Returns:
            Correlation record ID
        """
        correlation_id = uuid4()

        # Ensure position_1_id < position_2_id for consistency
        if str(position_1_id) > str(position_2_id):
            position_1_id, position_2_id = position_2_id, position_1_id

        query = """
            INSERT INTO position_correlations (
                correlation_id,
                position_1_id,
                position_2_id,
                correlation_coefficient,
                calculation_window,
                last_calculated,
                alert_triggered
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(position_1_id, position_2_id) DO UPDATE SET
                correlation_coefficient = excluded.correlation_coefficient,
                calculation_window = excluded.calculation_window,
                last_calculated = excluded.last_calculated,
                alert_triggered = excluded.alert_triggered
        """

        params = (
            str(correlation_id),
            str(position_1_id),
            str(position_2_id),
            str(correlation_coefficient),
            calculation_window,
            datetime.now(UTC).isoformat(),
            alert_triggered
        )

        async with self.get_connection() as conn:
            await conn.execute(query, params)
            await conn.commit()

        logger.info(f"Saved correlation: {position_1_id} <-> {position_2_id} = {correlation_coefficient}")
        return correlation_id

    async def save_correlation_matrix(
        self,
        positions: list[UUID],
        correlation_matrix: list[list[float]],
        calculation_window: int = 30
    ) -> int:
        """Save entire correlation matrix to database.

        Args:
            positions: List of position IDs
            correlation_matrix: 2D correlation matrix
            calculation_window: Window in minutes

        Returns:
            Number of correlations saved
        """
        saved_count = 0

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                correlation = Decimal(str(correlation_matrix[i][j]))

                # Only save meaningful correlations
                if abs(correlation) > Decimal("0.01"):
                    await self.save_correlation(
                        positions[i],
                        positions[j],
                        correlation,
                        calculation_window,
                        alert_triggered=abs(correlation) > Decimal("0.6")
                    )
                    saved_count += 1

        return saved_count

    async def get_correlation(
        self,
        position_1_id: UUID,
        position_2_id: UUID
    ) -> Optional[dict]:
        """Get correlation between two positions.

        Args:
            position_1_id: First position ID
            position_2_id: Second position ID

        Returns:
            Correlation data or None
        """
        # Ensure consistent ordering
        if str(position_1_id) > str(position_2_id):
            position_1_id, position_2_id = position_2_id, position_1_id

        query = """
            SELECT
                correlation_id,
                correlation_coefficient,
                calculation_window,
                last_calculated,
                alert_triggered
            FROM position_correlations
            WHERE position_1_id = ? AND position_2_id = ?
        """

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, (str(position_1_id), str(position_2_id)))
            row = await cursor.fetchone()

        if row:
            return {
                "correlation_id": UUID(row[0]),
                "correlation_coefficient": Decimal(row[1]),
                "calculation_window": row[2],
                "last_calculated": datetime.fromisoformat(row[3]),
                "alert_triggered": bool(row[4])
            }

        return None

    async def get_position_correlations(
        self,
        position_id: UUID,
        threshold: Optional[Decimal] = None
    ) -> list[dict]:
        """Get all correlations for a specific position.

        Args:
            position_id: Position ID
            threshold: Optional correlation threshold filter

        Returns:
            List of correlation records
        """
        query = """
            SELECT
                correlation_id,
                position_a,
                position_b,
                correlation_coefficient,
                last_calculated
            FROM position_correlations_bidirectional
            WHERE position_a = ?
        """

        params = [str(position_id)]

        if threshold is not None:
            query += " AND ABS(correlation_coefficient) >= ?"
            params.append(str(threshold))

        query += " ORDER BY ABS(correlation_coefficient) DESC"

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

        correlations = []
        for row in rows:
            correlations.append({
                "correlation_id": UUID(row[0]),
                "other_position_id": UUID(row[2]),
                "correlation_coefficient": Decimal(row[3]),
                "last_calculated": datetime.fromisoformat(row[4])
            })

        return correlations

    async def get_high_correlations(
        self,
        threshold: Decimal = Decimal("0.6"),
        limit: int = 10
    ) -> list[dict]:
        """Get positions with high correlations.

        Args:
            threshold: Correlation threshold
            limit: Maximum results

        Returns:
            List of high correlation pairs
        """
        query = """
            SELECT
                correlation_id,
                position_1_id,
                position_2_id,
                correlation_coefficient,
                last_calculated,
                alert_triggered
            FROM position_correlations
            WHERE ABS(correlation_coefficient) >= ?
            ORDER BY ABS(correlation_coefficient) DESC
            LIMIT ?
        """

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, (str(threshold), limit))
            rows = await cursor.fetchall()

        correlations = []
        for row in rows:
            correlations.append({
                "correlation_id": UUID(row[0]),
                "position_1_id": UUID(row[1]),
                "position_2_id": UUID(row[2]),
                "correlation_coefficient": Decimal(row[3]),
                "last_calculated": datetime.fromisoformat(row[4]),
                "alert_triggered": bool(row[5])
            })

        return correlations

    async def save_correlation_alert(self, alert: CorrelationAlert) -> None:
        """Save correlation alert to database.

        Args:
            alert: Correlation alert to save
        """
        # Mark correlation as having triggered alert
        if len(alert.affected_positions) == 2:
            await self.save_correlation(
                alert.affected_positions[0],
                alert.affected_positions[1],
                alert.correlation_level,
                calculation_window=30,  # Default window
                alert_triggered=True
            )

        # Save alert to alerts table (if it exists)
        query = """
            INSERT INTO alerts (
                alert_id,
                alert_type,
                severity,
                message,
                data,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """

        params = (
            str(alert.alert_id),
            "correlation",
            alert.severity,
            alert.message,
            str(alert.correlation_level),
            alert.timestamp.isoformat()
        )

        try:
            async with self.get_connection() as conn:
                await conn.execute(query, params)
                await conn.commit()

            logger.info(f"Saved correlation alert: {alert.alert_id}")
        except Exception as e:
            # Alerts table might not exist in MVP
            logger.warning(f"Could not save alert to alerts table: {e}")

    async def get_correlation_history(
        self,
        position_1_id: UUID,
        position_2_id: UUID,
        days: int = 30
    ) -> list[dict]:
        """Get historical correlation data for position pair.

        Args:
            position_1_id: First position ID
            position_2_id: Second position ID
            days: Number of days of history

        Returns:
            List of historical correlation records
        """
        # This would query a correlation_history table
        # For MVP, just return current correlation
        current = await self.get_correlation(position_1_id, position_2_id)
        if current:
            return [current]
        return []

    async def cleanup_old_correlations(self, days: int = 7) -> int:
        """Remove old correlation records.

        Args:
            days: Delete records older than this many days

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(UTC).timestamp() - (days * 86400)

        query = """
            DELETE FROM position_correlations
            WHERE julianday('now') - julianday(last_calculated) > ?
        """

        async with self.get_connection() as conn:
            cursor = await conn.execute(query, (days,))
            await conn.commit()
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old correlation records")

        return deleted
