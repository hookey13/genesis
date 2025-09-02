"""PostgreSQL repository implementation for Project GENESIS.

This module provides the PostgreSQL implementation of the repository pattern,
supporting high-performance concurrent operations with connection pooling.
"""

import asyncio
import csv
import json
import logging
import os
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
from asyncpg import Pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genesis.core.exceptions import DatabaseError, TransactionRollback
from genesis.data.repository import Repository
from genesis.core.models import (
    Account, Position, PositionCorrelation, TradingSession
)
from genesis.data.models_db import (
    AccountDB, PositionDB, OrderDB, TradingSessionDB,
    RiskMetricsDB, TiltProfileDB, BehavioralMetricsDB
)


class PostgresConfig:
    """PostgreSQL connection configuration."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "genesis_trading",
        user: str = "genesis",
        password: str = "",
        pool_size: int = 20,
        max_pool_size: int = 50,
        command_timeout: float = 60.0
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.command_timeout = command_timeout
    
    def get_dsn(self) -> str:
        """Get PostgreSQL DSN string.
        
        WARNING: This method returns a DSN with password included.
        Use only for debugging purposes and never log this value.
        """
        # Mask password in string representation
        import warnings
        warnings.warn("DSN contains sensitive credentials - do not log this value", UserWarning)
        return f"postgresql://{self.user}:{'*' * 8}@{self.host}:{self.port}/{self.database}"
    
    def _get_secure_dsn(self) -> str:
        """Get actual DSN for internal use only."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgresRepository(Repository):
    """PostgreSQL repository implementation with async support."""
    
    def __init__(self, config: PostgresConfig):
        """Initialize PostgreSQL repository.
        
        Args:
            config: PostgreSQL connection configuration
        """
        self.config = config
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        if self.pool is None:
            # Try to get credentials from Vault first
            username = self.config.user
            password = self.config.password
            
            try:
                from genesis.security.vault_integration import get_vault
                vault = await get_vault()
                username, password = await vault.credential_manager.get_database_credentials()
                self.logger.info("Using dynamic database credentials from Vault")
            except Exception as e:
                self.logger.warning(f"Failed to get Vault credentials, using fallback: {e}")
                # Fall back to config credentials if Vault is unavailable
                
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=username,
                password=password,
                min_size=self.config.pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'application_name': 'genesis_trading',
                    'jit': 'off'  # Disable JIT for consistent performance
                }
            )
            # Log connection without exposing credentials
            self.logger.info(f"PostgreSQL connection pool established: {self.config.host}:{self.config.port}/{self.config.database}")
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.logger.info("PostgreSQL connection pool closed")
    
    async def _ensure_connected(self) -> None:
        """Ensure connection pool is established."""
        if self.pool is None:
            async with self._lock:
                if self.pool is None:
                    await self.connect()
    
    # Account operations
    
    async def save_account(self, account: Dict[str, Any]) -> str:
        """Save or update account.
        
        Args:
            account: Account data dictionary
            
        Returns:
            Account ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.fetchrow("""
                    INSERT INTO accounts 
                    (account_id, balance_usdt, tier, locked_features, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (account_id) DO UPDATE SET
                        balance_usdt = EXCLUDED.balance_usdt,
                        tier = EXCLUDED.tier,
                        locked_features = EXCLUDED.locked_features,
                        updated_at = EXCLUDED.updated_at
                    RETURNING account_id
                """,
                account['account_id'],
                str(account['balance_usdt']),
                account.get('tier', 'SNIPER'),
                json.dumps(account.get('locked_features', [])),
                account.get('created_at', datetime.utcnow()),
                datetime.utcnow()
                )
                
                return result['account_id']
    
    async def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account by ID.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Account data or None
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM accounts WHERE account_id = $1
            """, account_id)
            
            if row:
                return dict(row)
            return None
    
    # Position operations
    
    async def save_position(self, position: Dict[str, Any]) -> str:
        """Save or update position.
        
        Args:
            position: Position data dictionary
            
        Returns:
            Position ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.fetchrow("""
                    INSERT INTO positions 
                    (position_id, account_id, symbol, side, entry_price, current_price,
                     quantity, dollar_value, pnl_dollars, pnl_percent, status, 
                     stop_loss, take_profit, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (position_id) DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        pnl_dollars = EXCLUDED.pnl_dollars,
                        pnl_percent = EXCLUDED.pnl_percent,
                        status = EXCLUDED.status,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        updated_at = EXCLUDED.updated_at
                    RETURNING position_id
                """,
                position['position_id'],
                position['account_id'],
                position['symbol'],
                position['side'],
                str(position['entry_price']),
                str(position.get('current_price', position['entry_price'])),
                str(position['quantity']),
                str(position['dollar_value']),
                str(position.get('pnl_dollars', '0')),
                str(position.get('pnl_percent', '0')),
                position.get('status', 'OPEN'),
                str(position['stop_loss']) if position.get('stop_loss') else None,
                str(position['take_profit']) if position.get('take_profit') else None,
                position.get('created_at', datetime.utcnow()),
                datetime.utcnow()
                )
                
                return result['position_id']
    
    async def get_positions(self, account_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get positions with optional filters.
        
        Args:
            account_id: Filter by account ID
            status: Filter by position status
            
        Returns:
            List of position dictionaries
        """
        await self._ensure_connected()
        
        query = "SELECT * FROM positions WHERE 1=1"
        params = []
        param_count = 0
        
        if account_id:
            param_count += 1
            query += f" AND account_id = ${param_count}"
            params.append(account_id)
        
        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    # Order operations
    
    async def save_order(self, order: Dict[str, Any]) -> str:
        """Save order to database.
        
        Args:
            order: Order data dictionary
            
        Returns:
            Order ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.fetchrow("""
                    INSERT INTO orders 
                    (order_id, position_id, account_id, client_order_id, exchange_order_id,
                     symbol, side, type, quantity, price, executed_price, executed_quantity,
                     status, slice_number, total_slices, latency_ms, slippage_percent,
                     created_at, executed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    ON CONFLICT (order_id) DO UPDATE SET
                        exchange_order_id = EXCLUDED.exchange_order_id,
                        executed_price = EXCLUDED.executed_price,
                        executed_quantity = EXCLUDED.executed_quantity,
                        status = EXCLUDED.status,
                        latency_ms = EXCLUDED.latency_ms,
                        slippage_percent = EXCLUDED.slippage_percent,
                        executed_at = EXCLUDED.executed_at
                    RETURNING order_id
                """,
                order['order_id'],
                order.get('position_id'),
                order['account_id'],
                order['client_order_id'],
                order.get('exchange_order_id'),
                order['symbol'],
                order['side'],
                order['type'],
                str(order['quantity']),
                str(order['price']) if order.get('price') else None,
                str(order['executed_price']) if order.get('executed_price') else None,
                str(order.get('executed_quantity', '0')),
                order.get('status', 'PENDING'),
                order.get('slice_number'),
                order.get('total_slices'),
                order.get('latency_ms'),
                str(order['slippage_percent']) if order.get('slippage_percent') else None,
                order.get('created_at', datetime.utcnow()),
                order.get('executed_at')
                )
                
                return result['order_id']
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order data or None
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM orders WHERE order_id = $1
            """, order_id)
            
            if row:
                return dict(row)
            return None
    
    # Trading session operations
    
    async def save_trading_session(self, session: Dict[str, Any]) -> str:
        """Save or update trading session.
        
        Args:
            session: Trading session data
            
        Returns:
            Session ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.fetchrow("""
                    INSERT INTO trading_sessions 
                    (session_id, account_id, session_date, starting_balance, current_balance,
                     ending_balance, realized_pnl, total_trades, winning_trades, losing_trades,
                     win_rate, average_r, max_drawdown, daily_loss_limit, tilt_events,
                     notes, is_active, created_at, updated_at, ended_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (session_id) DO UPDATE SET
                        current_balance = EXCLUDED.current_balance,
                        ending_balance = EXCLUDED.ending_balance,
                        realized_pnl = EXCLUDED.realized_pnl,
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        win_rate = EXCLUDED.win_rate,
                        average_r = EXCLUDED.average_r,
                        max_drawdown = EXCLUDED.max_drawdown,
                        tilt_events = EXCLUDED.tilt_events,
                        notes = EXCLUDED.notes,
                        is_active = EXCLUDED.is_active,
                        updated_at = EXCLUDED.updated_at,
                        ended_at = EXCLUDED.ended_at
                    RETURNING session_id
                """,
                session['session_id'],
                session['account_id'],
                session['session_date'],
                str(session['starting_balance']),
                str(session['current_balance']),
                str(session['ending_balance']) if session.get('ending_balance') else None,
                str(session.get('realized_pnl', '0')),
                session.get('total_trades', 0),
                session.get('winning_trades', 0),
                session.get('losing_trades', 0),
                str(session['win_rate']) if session.get('win_rate') else None,
                str(session['average_r']) if session.get('average_r') else None,
                str(session.get('max_drawdown', '0')),
                str(session['daily_loss_limit']),
                session.get('tilt_events', 0),
                session.get('notes'),
                session.get('is_active', True),
                session.get('created_at', datetime.utcnow()),
                datetime.utcnow(),
                session.get('ended_at')
                )
                
                return result['session_id']
    
    async def get_active_session(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get active trading session for account.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Active session data or None
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM trading_sessions 
                WHERE account_id = $1 AND is_active = true
                ORDER BY created_at DESC
                LIMIT 1
            """, account_id)
            
            if row:
                return dict(row)
            return None
    
    # Risk metrics operations
    
    async def save_risk_metrics(self, metrics: Dict[str, Any]) -> str:
        """Save risk metrics snapshot.
        
        Args:
            metrics: Risk metrics data
            
        Returns:
            Metric ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO risk_metrics 
                (metric_id, account_id, timestamp, total_exposure, position_count,
                 total_pnl_dollars, total_pnl_percent, max_position_size,
                 daily_pnl, risk_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING metric_id
            """,
            metrics['metric_id'],
            metrics['account_id'],
            metrics.get('timestamp', datetime.utcnow()),
            str(metrics['total_exposure']),
            metrics['position_count'],
            str(metrics['total_pnl_dollars']),
            str(metrics['total_pnl_percent']),
            str(metrics['max_position_size']),
            str(metrics['daily_pnl']),
            str(metrics['risk_score']) if metrics.get('risk_score') else None
            )
            
            return result['metric_id']
    
    # Behavioral metrics operations
    
    async def save_behavioral_metric(self, metric: Dict[str, Any]) -> str:
        """Save behavioral metric.
        
        Args:
            metric: Behavioral metric data
            
        Returns:
            Metric ID
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO behavioral_metrics 
                (metric_id, profile_id, session_id, metric_type, metric_value,
                 metrics_metadata, timestamp, session_context, time_of_day_bucket, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING metric_id
            """,
            metric['metric_id'],
            metric['profile_id'],
            metric.get('session_id'),
            metric['metric_type'],
            str(metric['metric_value']),
            json.dumps(metric['metrics_metadata']) if metric.get('metrics_metadata') else None,
            metric['timestamp'],
            metric.get('session_context'),
            metric.get('time_of_day_bucket'),
            metric.get('created_at', datetime.utcnow())
            )
            
            return result['metric_id']
    
    # Transaction management
    
    async def execute_transaction(self, operations: List[Tuple[str, List[Any]]]) -> None:
        """Execute multiple operations in a single transaction.
        
        Args:
            operations: List of (query, params) tuples
            
        Raises:
            TransactionRollback: If transaction fails
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            try:
                async with conn.transaction():
                    for query, params in operations:
                        await conn.execute(query, *params)
            except Exception as e:
                self.logger.error(f"Transaction failed: {str(e)}")
                raise TransactionRollback(
                    "Database transaction failed",
                    reason=str(e)
                )
    
    # Bulk operations
    
    async def bulk_insert(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Bulk insert records into table.
        
        Args:
            table: Table name
            records: List of record dictionaries
            
        Returns:
            Number of records inserted
        """
        if not records:
            return 0
        
        await self._ensure_connected()
        
        # Get column names from first record
        columns = list(records[0].keys())
        
        # Prepare values
        values = []
        for record in records:
            row = []
            for col in columns:
                value = record.get(col)
                # Convert Decimal to string for storage
                if isinstance(value, Decimal):
                    value = str(value)
                row.append(value)
            values.append(tuple(row))
        
        # Build INSERT query
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.executemany(query, values)
                # Parse result to get row count
                if result:
                    parts = result.split()
                    if len(parts) >= 2 and parts[0] == 'INSERT':
                        return int(parts[2])
                return len(values)
    
    # Query optimization methods
    
    async def execute_query(self, query: str, *params) -> List[Dict[str, Any]]:
        """Execute raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def execute_scalar(self, query: str, *params) -> Any:
        """Execute query and return single value.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Single value result
        """
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *params)
    
    # Health check
    
    async def health_check(self) -> bool:
        """Check database connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            await self._ensure_connected()
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    # Cleanup methods
    
    async def vacuum_analyze(self) -> None:
        """Run VACUUM ANALYZE for optimization."""
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            await conn.execute("VACUUM ANALYZE")
            self.logger.info("VACUUM ANALYZE completed")
    
    async def refresh_materialized_views(self) -> None:
        """Refresh all materialized views."""
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            # Refresh position correlation summary view
            await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY position_correlation_summary")
            self.logger.info("Materialized views refreshed")
    
    # Complete Repository interface implementation
    
    async def initialize(self) -> None:
        """Initialize the repository."""
        await self.connect()
    
    async def shutdown(self) -> None:
        """Shutdown the repository."""
        await self.disconnect()
    
    async def create_account(self, account: Account) -> str:
        """Create a new account."""
        account_dict = {
            'account_id': account.account_id,
            'balance_usdt': account.balance_usdt,
            'tier': account.tier,
            'locked_features': account.locked_features,
            'created_at': account.created_at
        }
        return await self.save_account(account_dict)
    
    async def update_account(self, account: Account) -> None:
        """Update existing account."""
        account_dict = {
            'account_id': account.account_id,
            'balance_usdt': account.balance_usdt,
            'tier': account.tier,
            'locked_features': account.locked_features
        }
        await self.save_account(account_dict)
    
    async def delete_account(self, account_id: str) -> None:
        """Delete account."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM accounts WHERE account_id = $1", account_id)
    
    async def list_accounts(self) -> list[Account]:
        """List all accounts."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM accounts")
            return [self._row_to_account(dict(row)) for row in rows]
    
    async def list_positions(self, account_id: str | None = None) -> list[Position]:
        """List positions with optional account filter."""
        positions = await self.get_positions(account_id)
        return [self._dict_to_position(p) for p in positions]
    
    async def create_position(self, position: Position) -> str:
        """Create a new position."""
        position_dict = self._position_to_dict(position)
        return await self.save_position(position_dict)
    
    async def get_position(self, position_id: str) -> Position | None:
        """Get position by ID."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM positions WHERE position_id = $1", position_id)
            if row:
                return self._dict_to_position(dict(row))
            return None
    
    async def get_positions_by_account(self, account_id: str, status: str | None = None) -> list[Position]:
        """Get all positions for an account."""
        positions = await self.get_positions(account_id, status)
        return [self._dict_to_position(p) for p in positions]
    
    async def update_position(self, position: Position) -> None:
        """Update existing position."""
        position_dict = self._position_to_dict(position)
        await self.save_position(position_dict)
    
    async def close_position(self, position_id: str, final_pnl: Decimal) -> None:
        """Close a position."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE positions 
                SET status = 'CLOSED', 
                    pnl_dollars = $1, 
                    closed_at = $2,
                    updated_at = $2
                WHERE position_id = $3
            """, str(final_pnl), datetime.utcnow(), position_id)
    
    async def create_session(self, session: TradingSession) -> str:
        """Create a new trading session."""
        session_dict = self._session_to_dict(session)
        return await self.save_trading_session(session_dict)
    
    async def get_session(self, session_id: str) -> TradingSession | None:
        """Get session by ID."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM trading_sessions WHERE session_id = $1", session_id)
            if row:
                return self._dict_to_session(dict(row))
            return None
    
    async def update_session(self, session: TradingSession) -> None:
        """Update existing session."""
        session_dict = self._session_to_dict(session)
        await self.save_trading_session(session_dict)
    
    async def end_session(self, session_id: str) -> None:
        """End a trading session."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE trading_sessions 
                SET is_active = false, 
                    ended_at = $1,
                    updated_at = $1
                WHERE session_id = $2
            """, datetime.utcnow(), session_id)
    
    async def save_correlation(self, correlation: PositionCorrelation) -> None:
        """Save position correlation."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO position_correlations 
                (position_a_id, position_b_id, correlation_coefficient, alert_triggered, calculated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (position_a_id, position_b_id) DO UPDATE SET
                    correlation_coefficient = EXCLUDED.correlation_coefficient,
                    alert_triggered = EXCLUDED.alert_triggered,
                    calculated_at = EXCLUDED.calculated_at
            """, 
            correlation.position_a_id, correlation.position_b_id,
            str(correlation.correlation_coefficient), correlation.alert_triggered,
            correlation.calculated_at)
    
    async def get_correlations(self, position_id: str) -> list[PositionCorrelation]:
        """Get correlations for a position."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM position_correlations 
                WHERE position_a_id = $1 OR position_b_id = $1
            """, position_id)
            return [self._dict_to_correlation(dict(row)) for row in rows]
    
    async def get_risk_metrics(self, account_id: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get risk metrics for time range."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM risk_metrics 
                WHERE account_id = $1 AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp
            """, account_id, start_time, end_time)
            return [dict(row) for row in rows]
    
    async def save_event(self, event_type: str, aggregate_id: str, event_data: dict[str, Any]) -> str:
        """Save an event to the event store."""
        await self._ensure_connected()
        event_id = str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            # Get next sequence number
            seq_num = await conn.fetchval("""
                SELECT COALESCE(MAX(sequence_number), 0) + 1 
                FROM events WHERE aggregate_id = $1
            """, aggregate_id)
            
            await conn.execute("""
                INSERT INTO events 
                (event_id, event_type, aggregate_id, event_data, sequence_number, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, event_id, event_type, aggregate_id, json.dumps(event_data), seq_num, datetime.utcnow())
            
            return event_id
    
    async def get_events(self, aggregate_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get events for an aggregate."""
        await self._ensure_connected()
        
        query = "SELECT * FROM events WHERE aggregate_id = $1"
        params = [aggregate_id]
        
        if event_type:
            query += " AND event_type = $2"
            params.append(event_type)
        
        query += " ORDER BY sequence_number"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def get_events_by_type(self, event_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get events by type and time range."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM events 
                WHERE event_type = $1 AND created_at >= $2 AND created_at <= $3
                ORDER BY created_at
            """, event_type, start_time, end_time)
            return [dict(row) for row in rows]
    
    async def get_events_by_aggregate(self, aggregate_id: str, start_time: datetime, end_time: datetime) -> list[Any]:
        """Get events for an aggregate within time range."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM events 
                WHERE aggregate_id = $1 AND created_at >= $2 AND created_at <= $3
                ORDER BY sequence_number
            """, aggregate_id, start_time, end_time)
            return [dict(row) for row in rows]
    
    async def get_trades_by_account(self, account_id: str, start_date: datetime, end_date: datetime) -> list[Any]:
        """Get trades for an account within date range."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM trades 
                WHERE account_id = $1 AND executed_at >= $2 AND executed_at <= $3
                ORDER BY executed_at
            """, account_id, start_date, end_date)
            return [dict(row) for row in rows]
    
    async def get_orders_by_account(self, account_id: str, start_date: datetime, end_date: datetime) -> list[Any]:
        """Get orders for an account within date range."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders 
                WHERE account_id = $1 AND created_at >= $2 AND created_at <= $3
                ORDER BY created_at
            """, account_id, start_date, end_date)
            return [dict(row) for row in rows]
    
    async def get_price_history(self, symbol: str, start_date: datetime, end_date: datetime) -> list[Any]:
        """Get historical price data for a symbol."""
        # This would typically query a price history table
        # For now, return empty list as price history might come from external source
        return []
    
    async def save_reconciliation_result(self, result: dict[str, Any]) -> None:
        """Save reconciliation result."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO reconciliation_results 
                (result_id, account_id, timestamp, status, details)
                VALUES ($1, $2, $3, $4, $5)
            """, 
            result.get('result_id', str(uuid.uuid4())),
            result['account_id'],
            result.get('timestamp', datetime.utcnow()),
            result['status'],
            json.dumps(result.get('details', {})))
    
    async def save_reconciliation_report(self, report: dict[str, Any]) -> None:
        """Save reconciliation report."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO reconciliation_reports 
                (report_id, account_id, timestamp, report_data)
                VALUES ($1, $2, $3, $4)
            """,
            report.get('report_id', str(uuid.uuid4())),
            report['account_id'],
            report.get('timestamp', datetime.utcnow()),
            json.dumps(report))
    
    async def get_orders_by_position(self, position_id: str) -> list[dict[str, Any]]:
        """Get all orders for a position."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders WHERE position_id = $1 ORDER BY created_at
            """, position_id)
            return [dict(row) for row in rows]
    
    async def update_order_status(self, order_id: str, status: str, executed_at: datetime | None = None) -> None:
        """Update order status."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE orders 
                SET status = $1, executed_at = $2
                WHERE order_id = $3
            """, status, executed_at, order_id)
    
    async def load_open_positions(self, account_id: str) -> list[Position]:
        """Load all open positions for recovery on startup."""
        positions = await self.get_positions_by_account(account_id, status='OPEN')
        return positions
    
    async def reconcile_positions(self, exchange_positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reconcile database positions with exchange state."""
        discrepancies = []
        
        for ex_pos in exchange_positions:
            db_pos = await self.get_position(ex_pos.get('position_id'))
            if db_pos:
                # Check for discrepancies
                if abs(float(db_pos.quantity) - float(ex_pos['quantity'])) > 0.00001:
                    discrepancies.append({
                        'position_id': ex_pos['position_id'],
                        'type': 'quantity_mismatch',
                        'db_value': db_pos.quantity,
                        'exchange_value': ex_pos['quantity']
                    })
            else:
                # Position exists on exchange but not in database
                discrepancies.append({
                    'position_id': ex_pos.get('position_id', 'unknown'),
                    'type': 'missing_in_db',
                    'exchange_data': ex_pos
                })
        
        return discrepancies
    
    async def backup(self, backup_path: Path | None = None) -> Path:
        """Create a backup of the database."""
        from pathlib import Path
        import subprocess
        
        if backup_path is None:
            backup_path = Path(f"backups/postgres_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql")
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use pg_dump for backup
        cmd = [
            'pg_dump',
            f'--host={self.config.host}',
            f'--port={self.config.port}',
            f'--username={self.config.user}',
            f'--dbname={self.config.database}',
            '--no-password',
            '--file', str(backup_path)
        ]
        
        # Use environment variable for password to avoid command line exposure
        env = os.environ.copy()
        env['PGPASSWORD'] = self.config.password
        
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Don't log stderr as it might contain sensitive info
            self.logger.error(f"Backup failed: Database backup operation failed")
            raise DatabaseError("Backup operation failed") from e
        
        return backup_path
    
    async def restore(self, backup_path: Path) -> None:
        """Restore database from backup."""
        import subprocess
        
        cmd = [
            'psql',
            f'--host={self.config.host}',
            f'--port={self.config.port}',
            f'--username={self.config.user}',
            f'--dbname={self.config.database}',
            '--no-password',
            '--file', str(backup_path)
        ]
        
        # Use environment variable for password to avoid command line exposure
        env = os.environ.copy()
        env['PGPASSWORD'] = self.config.password
        
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Don't log stderr as it might contain sensitive info
            self.logger.error(f"Restore failed: Database restore operation failed")
            raise DatabaseError("Restore operation failed") from e
    
    async def get_database_size(self) -> int:
        """Get current database size in bytes."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            size = await conn.fetchval("""
                SELECT pg_database_size($1)
            """, self.config.database)
            return size
    
    async def rotate_database(self) -> None:
        """Rotate database when size limit reached."""
        # PostgreSQL doesn't need rotation like SQLite
        # Instead, we can run maintenance operations
        await self.vacuum_analyze()
    
    async def export_trades_to_csv(self, account_id: str, start_date: date, end_date: date, output_path: Path) -> Path:
        """Export trades to CSV for tax reporting."""
        from pathlib import Path
        import csv
        
        trades = await self.get_trades_by_account(account_id, 
                                                  datetime.combine(start_date, datetime.min.time()),
                                                  datetime.combine(end_date, datetime.max.time()))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            if trades:
                writer = csv.DictWriter(csvfile, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
        
        return output_path
    
    async def export_performance_report(self, account_id: str, output_path: Path) -> Path:
        """Export performance metrics report."""
        from pathlib import Path
        
        metrics = await self.calculate_performance_metrics(account_id)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return output_path
    
    async def calculate_performance_metrics(self, account_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Calculate performance metrics."""
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            # Get session metrics
            if session_id:
                session = await conn.fetchrow("""
                    SELECT * FROM trading_sessions WHERE session_id = $1
                """, session_id)
            else:
                session = await conn.fetchrow("""
                    SELECT * FROM trading_sessions 
                    WHERE account_id = $1 AND is_active = true
                """, account_id)
            
            if not session:
                return {}
            
            # Calculate additional metrics
            total_pnl = await conn.fetchval("""
                SELECT SUM(CAST(pnl_dollars AS NUMERIC)) 
                FROM positions 
                WHERE account_id = $1 AND status = 'CLOSED'
            """, account_id) or 0
            
            return {
                'session_id': session['session_id'],
                'total_trades': session['total_trades'],
                'winning_trades': session['winning_trades'],
                'losing_trades': session['losing_trades'],
                'win_rate': session['win_rate'],
                'average_r': session['average_r'],
                'max_drawdown': session['max_drawdown'],
                'total_pnl': str(total_pnl),
                'realized_pnl': session['realized_pnl']
            }
    
    async def get_performance_report(self, account_id: str, start_date: date, end_date: date) -> dict[str, Any]:
        """Get comprehensive performance report."""
        from datetime import date
        
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            # Get all sessions in date range
            sessions = await conn.fetch("""
                SELECT * FROM trading_sessions 
                WHERE account_id = $1 
                AND session_date >= $2 
                AND session_date <= $3
                ORDER BY session_date
            """, account_id, 
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time()))
            
            # Aggregate metrics
            total_trades = sum(s['total_trades'] for s in sessions)
            winning_trades = sum(s['winning_trades'] for s in sessions)
            losing_trades = sum(s['losing_trades'] for s in sessions)
            
            return {
                'account_id': account_id,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_sessions': len(sessions),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'sessions': [dict(s) for s in sessions]
            }
    
    async def save_tilt_event(self, session_id: str, event_type: str, severity: str, 
                             indicator_values: dict[str, Any], intervention: str | None = None) -> str:
        """Save a tilt event."""
        await self._ensure_connected()
        event_id = str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            # Get profile_id from session
            profile_id = await conn.fetchval("""
                SELECT tp.profile_id 
                FROM tilt_profiles tp
                JOIN trading_sessions ts ON ts.account_id = tp.account_id
                WHERE ts.session_id = $1
            """, session_id)
            
            if profile_id:
                await conn.execute("""
                    INSERT INTO tilt_events 
                    (event_id, profile_id, event_type, tilt_indicators, 
                     tilt_score_before, tilt_score_after, intervention_message, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, event_id, profile_id, event_type, json.dumps(indicator_values),
                indicator_values.get('score_before', 0),
                indicator_values.get('score_after', 0),
                intervention, datetime.utcnow())
        
        return event_id
    
    async def get_tilt_events(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tilt events for a session."""
        await self._ensure_connected()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT te.* 
                FROM tilt_events te
                JOIN tilt_profiles tp ON te.profile_id = tp.profile_id
                JOIN trading_sessions ts ON ts.account_id = tp.account_id
                WHERE ts.session_id = $1
                ORDER BY te.timestamp
            """, session_id)
            
            return [dict(row) for row in rows]
    
    async def begin_transaction(self) -> None:
        """Begin a database transaction."""
        # Transactions are handled per-connection in asyncpg
        # This is a no-op for compatibility
        pass
    
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        # Transactions are handled per-connection in asyncpg
        # This is a no-op for compatibility
        pass
    
    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        # Transactions are handled per-connection in asyncpg
        # This is a no-op for compatibility
        pass
    
    async def set_database_info(self, key: str, value: str) -> None:
        """Set database metadata."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO database_info (key, value, updated_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
            """, key, value, datetime.utcnow())
    
    async def get_database_info(self, key: str) -> str | None:
        """Get database metadata."""
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            value = await conn.fetchval("""
                SELECT value FROM database_info WHERE key = $1
            """, key)
            return value
    
    # Helper methods for model conversion
    
    def _row_to_account(self, row: dict) -> Account:
        """Convert database row to Account model."""
        return Account(
            account_id=row['account_id'],
            balance_usdt=Decimal(row['balance_usdt']),
            tier=row['tier'],
            locked_features=json.loads(row['locked_features']) if row.get('locked_features') else [],
            created_at=row['created_at']
        )
    
    def _dict_to_position(self, data: dict) -> Position:
        """Convert dictionary to Position model."""
        return Position(
            position_id=data['position_id'],
            account_id=data['account_id'],
            symbol=data['symbol'],
            side=data['side'],
            entry_price=Decimal(data['entry_price']),
            current_price=Decimal(data.get('current_price', data['entry_price'])),
            quantity=Decimal(data['quantity']),
            dollar_value=Decimal(data['dollar_value']),
            pnl_dollars=Decimal(data.get('pnl_dollars', '0')),
            pnl_percent=Decimal(data.get('pnl_percent', '0')),
            status=data.get('status', 'OPEN'),
            stop_loss=Decimal(data['stop_loss']) if data.get('stop_loss') else None,
            take_profit=Decimal(data['take_profit']) if data.get('take_profit') else None,
            created_at=data.get('created_at', datetime.utcnow())
        )
    
    def _position_to_dict(self, position: Position) -> dict:
        """Convert Position model to dictionary."""
        return {
            'position_id': position.position_id,
            'account_id': position.account_id,
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'quantity': position.quantity,
            'dollar_value': position.dollar_value,
            'pnl_dollars': position.pnl_dollars,
            'pnl_percent': position.pnl_percent,
            'status': position.status,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'created_at': position.created_at
        }
    
    def _dict_to_session(self, data: dict) -> TradingSession:
        """Convert dictionary to TradingSession model."""
        return TradingSession(
            session_id=data['session_id'],
            account_id=data['account_id'],
            session_date=data['session_date'],
            starting_balance=Decimal(data['starting_balance']),
            current_balance=Decimal(data['current_balance']),
            ending_balance=Decimal(data['ending_balance']) if data.get('ending_balance') else None,
            realized_pnl=Decimal(data.get('realized_pnl', '0')),
            total_trades=data.get('total_trades', 0),
            winning_trades=data.get('winning_trades', 0),
            losing_trades=data.get('losing_trades', 0),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at', datetime.utcnow())
        )
    
    def _session_to_dict(self, session: TradingSession) -> dict:
        """Convert TradingSession model to dictionary."""
        return {
            'session_id': session.session_id,
            'account_id': session.account_id,
            'session_date': session.session_date,
            'starting_balance': session.starting_balance,
            'current_balance': session.current_balance,
            'ending_balance': session.ending_balance,
            'realized_pnl': session.realized_pnl,
            'total_trades': session.total_trades,
            'winning_trades': session.winning_trades,
            'losing_trades': session.losing_trades,
            'daily_loss_limit': getattr(session, 'daily_loss_limit', Decimal('1000')),
            'win_rate': str(session.winning_trades / session.total_trades) if session.total_trades > 0 else None,
            'is_active': session.is_active,
            'created_at': session.created_at
        }
    
    def _dict_to_correlation(self, data: dict) -> PositionCorrelation:
        """Convert dictionary to PositionCorrelation model."""
        return PositionCorrelation(
            position_a_id=data['position_a_id'],
            position_b_id=data['position_b_id'],
            correlation_coefficient=Decimal(data['correlation_coefficient']),
            alert_triggered=data['alert_triggered'],
            calculated_at=data['calculated_at']
        )