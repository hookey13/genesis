"""
Integration module for database components with the Genesis trading system.
Provides seamless connection management and query optimization.
"""

import logging
from typing import Any, Dict, List, Optional

from genesis.data.repository import Repository
from genesis.data.models_db import Base

from .postgres_manager import PostgresManager, DatabaseConfig
from .partition_manager import PartitionManager
from .pool_monitor import PoolMonitor

logger = logging.getLogger(__name__)


class DatabaseIntegration:
    """
    Integrates PostgreSQL with PgBouncer pooling and partitioning
    into the Genesis trading system.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database integration.
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        self.db_manager = PostgresManager(self.config)
        self.partition_manager = PartitionManager(self.db_manager)
        self.pool_monitor = PoolMonitor(self.db_manager)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize all database components."""
        if self._initialized:
            return
            
        try:
            # Connect to database
            await self.db_manager.connect()
            
            # Setup partitioned tables
            await self.partition_manager.setup_partitioned_tables()
            
            # Start maintenance tasks
            await self.partition_manager.start_maintenance_task(interval_hours=6)
            await self.pool_monitor.start_monitoring(interval_seconds=60)
            
            self._initialized = True
            logger.info("Database integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database integration: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Gracefully shutdown database components."""
        try:
            # Stop maintenance tasks
            await self.partition_manager.stop_maintenance_task()
            await self.pool_monitor.stop_monitoring()
            
            # Disconnect from database
            await self.db_manager.disconnect()
            
            self._initialized = False
            logger.info("Database integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
            
    async def execute_trading_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        use_partition_hint: bool = True
    ) -> List[Dict]:
        """
        Execute a trading-related query with optimizations.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            use_partition_hint: Whether to add partition hints
            
        Returns:
            List of result dictionaries
        """
        # Add partition hint for time-based queries
        if use_partition_hint and "WHERE" in query:
            # This is a simplified example - in production, use proper SQL parsing
            if "created_at" in query or "executed_at" in query or "timestamp" in query:
                # Query optimizer will use partition pruning automatically
                pass
                
        if params:
            rows = await self.db_manager.fetch(query, *params)
        else:
            rows = await self.db_manager.fetch(query)
            
        return [dict(row) for row in rows]
        
    async def insert_order(self, order_data: Dict) -> int:
        """
        Insert an order into the partitioned orders table.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            Order ID
        """
        query = """
        INSERT INTO orders (
            created_at, symbol, side, type, quantity, price,
            status, client_order_id, tier, strategy_name, metadata
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        ) RETURNING id
        """
        
        order_id = await self.db_manager.fetchval(
            query,
            order_data.get('created_at'),
            order_data.get('symbol'),
            order_data.get('side'),
            order_data.get('type'),
            order_data.get('quantity'),
            order_data.get('price'),
            order_data.get('status'),
            order_data.get('client_order_id'),
            order_data.get('tier'),
            order_data.get('strategy_name'),
            order_data.get('metadata', {})
        )
        
        return order_id
        
    async def insert_trade(self, trade_data: Dict) -> int:
        """
        Insert a trade into the partitioned trades table.
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Trade ID
        """
        query = """
        INSERT INTO trades (
            executed_at, order_id, symbol, side, quantity, price,
            commission, commission_asset, exchange_trade_id,
            is_maker, realized_pnl, tier, metadata
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
        ) RETURNING id
        """
        
        trade_id = await self.db_manager.fetchval(
            query,
            trade_data.get('executed_at'),
            trade_data.get('order_id'),
            trade_data.get('symbol'),
            trade_data.get('side'),
            trade_data.get('quantity'),
            trade_data.get('price'),
            trade_data.get('commission'),
            trade_data.get('commission_asset'),
            trade_data.get('exchange_trade_id'),
            trade_data.get('is_maker', False),
            trade_data.get('realized_pnl'),
            trade_data.get('tier'),
            trade_data.get('metadata', {})
        )
        
        return trade_id
        
    async def get_recent_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent orders with partition optimization.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders
            
        Returns:
            List of order dictionaries
        """
        conditions = ["created_at >= CURRENT_DATE - INTERVAL '7 days'"]
        params = []
        param_count = 0
        
        if symbol:
            param_count += 1
            conditions.append(f"symbol = ${param_count}")
            params.append(symbol)
            
        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)
            
        query = f"""
        SELECT * FROM orders
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        
        return await self.execute_trading_query(query, tuple(params) if params else None)
        
    async def get_trading_performance(
        self,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Get trading performance metrics for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Optional symbol filter
            
        Returns:
            Performance metrics dictionary
        """
        params = [start_date, end_date]
        symbol_condition = ""
        
        if symbol:
            symbol_condition = "AND symbol = $3"
            params.append(symbol)
            
        query = f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(quantity) as total_volume,
            SUM(realized_pnl) as total_pnl,
            AVG(realized_pnl) as avg_pnl,
            MAX(realized_pnl) as max_profit,
            MIN(realized_pnl) as max_loss,
            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades
        FROM trades
        WHERE executed_at >= $1::date
        AND executed_at < $2::date + INTERVAL '1 day'
        {symbol_condition}
        """
        
        result = await self.db_manager.fetchrow(query, *params)
        
        if result:
            metrics = dict(result)
            # Calculate win rate
            total = metrics['total_trades']
            if total > 0:
                metrics['win_rate'] = metrics['winning_trades'] / total * 100
            else:
                metrics['win_rate'] = 0
            return metrics
        
        return {}
        
    async def optimize_queries(self) -> None:
        """Run query optimization and update statistics."""
        # Update table statistics for query planner
        tables = ['orders', 'trades', 'market_data', 'tilt_events']
        for table in tables:
            await self.db_manager.execute(f"ANALYZE {table}")
            
        # Create any missing indexes
        await self._ensure_indexes()
        
        logger.info("Query optimization complete")
        
    async def _ensure_indexes(self) -> None:
        """Ensure all required indexes exist."""
        # Check and create indexes if missing
        index_queries = [
            # Orders table indexes
            """
            CREATE INDEX IF NOT EXISTS idx_orders_client_order_id
            ON orders (client_order_id)
            """,
            
            # Trades table indexes  
            """
            CREATE INDEX IF NOT EXISTS idx_trades_order_id
            ON trades (order_id)
            """,
            
            # Performance query optimization
            """
            CREATE INDEX IF NOT EXISTS idx_trades_performance
            ON trades (executed_at, symbol, realized_pnl)
            """
        ]
        
        for query in index_queries:
            await self.db_manager.execute(query)
            
    async def get_pool_health(self) -> Dict:
        """
        Get current pool health status.
        
        Returns:
            Dictionary with health status and metrics
        """
        is_healthy, issues = await self.pool_monitor.check_pool_health()
        metrics = await self.pool_monitor.get_pool_metrics()
        
        return {
            'is_healthy': is_healthy,
            'issues': issues,
            'metrics': metrics.to_dict(),
            'recommendations': await self.pool_monitor.recommend_pool_size()
        }
        
    async def get_partition_stats(self) -> Dict:
        """
        Get partition statistics for all tables.
        
        Returns:
            Dictionary with partition information
        """
        stats = {}
        
        for table_name in PartitionManager.PARTITIONED_TABLES:
            table_stats = await self.partition_manager.get_partition_statistics(table_name)
            stats[table_name] = {
                'partitions': table_stats,
                'total_partitions': len(table_stats),
                'total_rows': sum(p.get('row_count', 0) for p in table_stats)
            }
            
        return stats