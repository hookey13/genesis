"""
Automated partition management for time-series trading data.
Handles creation, maintenance, and pruning of PostgreSQL partitions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .postgres_manager import PostgresManager

logger = logging.getLogger(__name__)


class PartitionManager:
    """
    Manages table partitioning for time-series data.
    Creates monthly partitions and handles automated maintenance.
    """
    
    # Tables to partition with their configuration
    PARTITIONED_TABLES = {
        'orders': {
            'partition_key': 'created_at',
            'retention_months': 12,
            'indexes': [
                ('symbol, created_at DESC', 'btree'),
                ('status', 'btree', "WHERE status = 'pending'"),
                ('exchange_order_id', 'hash'),
            ]
        },
        'trades': {
            'partition_key': 'executed_at',
            'retention_months': 12,
            'indexes': [
                ('symbol, executed_at DESC', 'btree'),
                ('order_id', 'btree'),
            ]
        },
        'market_data': {
            'partition_key': 'timestamp',
            'retention_months': 6,
            'indexes': [
                ('symbol, timestamp DESC', 'btree'),
            ]
        },
        'tilt_events': {
            'partition_key': 'detected_at',
            'retention_months': 24,
            'indexes': [
                ('severity, detected_at DESC', 'btree'),
            ]
        }
    }
    
    def __init__(self, db_manager: PostgresManager):
        """
        Initialize partition manager.
        
        Args:
            db_manager: PostgreSQL connection manager
        """
        self.db = db_manager
        self._maintenance_task: Optional[asyncio.Task] = None
        
    async def setup_partitioned_tables(self) -> None:
        """Create partitioned table schemas if they don't exist."""
        await self._create_orders_table()
        await self._create_trades_table()
        await self._create_market_data_table()
        await self._create_tilt_events_table()
        await self._create_partition_functions()
        
        # Create initial partitions
        await self.create_future_partitions(months_ahead=3)
        
    async def _create_orders_table(self) -> None:
        """Create partitioned orders table."""
        query = """
        CREATE TABLE IF NOT EXISTS orders (
            id BIGSERIAL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
            type VARCHAR(10) NOT NULL CHECK (type IN ('market', 'limit', 'stop', 'stop_limit')),
            quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
            price DECIMAL(20, 8),
            stop_price DECIMAL(20, 8),
            status VARCHAR(20) NOT NULL CHECK (status IN (
                'pending', 'open', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired'
            )),
            exchange_order_id VARCHAR(100),
            client_order_id VARCHAR(100) UNIQUE NOT NULL,
            filled_quantity DECIMAL(20, 8) DEFAULT 0,
            average_fill_price DECIMAL(20, 8),
            commission DECIMAL(20, 8) DEFAULT 0,
            commission_asset VARCHAR(10),
            tier VARCHAR(20) NOT NULL,
            strategy_name VARCHAR(100),
            metadata JSONB DEFAULT '{}',
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at);
        
        -- Create update trigger for updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;
        CREATE TRIGGER update_orders_updated_at
        BEFORE UPDATE ON orders
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
        """
        
        await self.db.execute(query)
        logger.info("Created partitioned orders table")
        
    async def _create_trades_table(self) -> None:
        """Create partitioned trades table."""
        query = """
        CREATE TABLE IF NOT EXISTS trades (
            id BIGSERIAL,
            executed_at TIMESTAMPTZ NOT NULL,
            order_id BIGINT NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
            quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
            price DECIMAL(20, 8) NOT NULL CHECK (price > 0),
            commission DECIMAL(20, 8) DEFAULT 0,
            commission_asset VARCHAR(10),
            exchange_trade_id VARCHAR(100) UNIQUE,
            is_maker BOOLEAN DEFAULT FALSE,
            realized_pnl DECIMAL(20, 8),
            tier VARCHAR(20) NOT NULL,
            metadata JSONB DEFAULT '{}',
            PRIMARY KEY (id, executed_at)
        ) PARTITION BY RANGE (executed_at);
        """
        
        await self.db.execute(query)
        logger.info("Created partitioned trades table")
        
    async def _create_market_data_table(self) -> None:
        """Create partitioned market data table."""
        query = """
        CREATE TABLE IF NOT EXISTS market_data (
            id BIGSERIAL,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            bid_price DECIMAL(20, 8),
            bid_quantity DECIMAL(20, 8),
            ask_price DECIMAL(20, 8),
            ask_quantity DECIMAL(20, 8),
            last_price DECIMAL(20, 8),
            volume_24h DECIMAL(20, 8),
            spread DECIMAL(20, 8) GENERATED ALWAYS AS (ask_price - bid_price) STORED,
            metadata JSONB DEFAULT '{}',
            PRIMARY KEY (id, timestamp)
        ) PARTITION BY RANGE (timestamp);
        """
        
        await self.db.execute(query)
        logger.info("Created partitioned market_data table")
        
    async def _create_tilt_events_table(self) -> None:
        """Create partitioned tilt events table."""
        query = """
        CREATE TABLE IF NOT EXISTS tilt_events (
            id BIGSERIAL,
            detected_at TIMESTAMPTZ NOT NULL,
            severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
            indicator_type VARCHAR(50) NOT NULL,
            indicator_value DECIMAL(20, 8),
            threshold_value DECIMAL(20, 8),
            intervention_applied VARCHAR(100),
            tier_before VARCHAR(20),
            tier_after VARCHAR(20),
            metadata JSONB DEFAULT '{}',
            PRIMARY KEY (id, detected_at)
        ) PARTITION BY RANGE (detected_at);
        """
        
        await self.db.execute(query)
        logger.info("Created partitioned tilt_events table")
        
    async def _create_partition_functions(self) -> None:
        """Create helper functions for partition management."""
        query = """
        -- Function to create monthly partitions
        CREATE OR REPLACE FUNCTION create_monthly_partition(
            table_name text,
            start_date date
        )
        RETURNS void AS $$
        DECLARE
            partition_name text;
            end_date date;
            config record;
        BEGIN
            partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
            end_date := start_date + interval '1 month';
            
            -- Check if partition already exists
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = partition_name
                AND n.nspname = 'public'
            ) THEN
                -- Create partition
                EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
                               FOR VALUES FROM (%L) TO (%L)',
                               partition_name, table_name, start_date, end_date);
                
                RAISE NOTICE 'Created partition % for table %', partition_name, table_name;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to drop old partitions
        CREATE OR REPLACE FUNCTION drop_old_partition(
            table_name text,
            retention_months integer
        )
        RETURNS void AS $$
        DECLARE
            cutoff_date date;
            partition_record record;
        BEGIN
            cutoff_date := date_trunc('month', CURRENT_DATE - (retention_months || ' months')::interval);
            
            FOR partition_record IN
                SELECT c.relname as partition_name
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_inherits i ON i.inhrelid = c.oid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = table_name
                AND n.nspname = 'public'
                AND c.relname ~ (table_name || '_[0-9]{4}_[0-9]{2}$')
                AND to_date(right(c.relname, 7), 'YYYY_MM') < cutoff_date
            LOOP
                EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', partition_record.partition_name);
                RAISE NOTICE 'Dropped old partition %', partition_record.partition_name;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to get partition statistics
        CREATE OR REPLACE FUNCTION get_partition_stats(table_name text)
        RETURNS TABLE(
            partition_name text,
            row_count bigint,
            total_size text,
            index_size text
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                c.relname::text as partition_name,
                c.reltuples::bigint as row_count,
                pg_size_pretty(pg_relation_size(c.oid)) as total_size,
                pg_size_pretty(pg_indexes_size(c.oid)) as index_size
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_inherits i ON i.inhrelid = c.oid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = table_name
            AND n.nspname = 'public'
            ORDER BY c.relname;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        await self.db.execute(query)
        logger.info("Created partition management functions")
        
    async def create_monthly_partition(
        self,
        table_name: str,
        year: int,
        month: int
    ) -> None:
        """
        Create a monthly partition for a table.
        
        Args:
            table_name: Name of the partitioned table
            year: Year for the partition
            month: Month for the partition
        """
        partition_date = datetime(year, month, 1)
        partition_name = f"{table_name}_{year:04d}_{month:02d}"
        
        # Check if partition already exists
        exists = await self.db.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = $1 AND n.nspname = 'public'
            )
            """,
            partition_name
        )
        
        if exists:
            logger.debug(f"Partition {partition_name} already exists")
            return
            
        # Create partition
        await self.db.execute(
            f"SELECT create_monthly_partition($1, $2::date)",
            table_name,
            partition_date
        )
        
        # Create indexes for the partition
        if table_name in self.PARTITIONED_TABLES:
            await self._create_partition_indexes(partition_name, table_name)
            
        logger.info(f"Created partition {partition_name}")
        
    async def _create_partition_indexes(
        self,
        partition_name: str,
        table_name: str
    ) -> None:
        """
        Create indexes for a partition.
        
        Args:
            partition_name: Name of the partition
            table_name: Name of the parent table
        """
        config = self.PARTITIONED_TABLES.get(table_name, {})
        indexes = config.get('indexes', [])
        
        for index_def in indexes:
            if len(index_def) == 2:
                columns, method = index_def
                where_clause = ""
            else:
                columns, method, where_clause = index_def
                
            index_name = f"idx_{partition_name}_{columns.split(',')[0].strip().replace(' ', '_')}"
            
            query = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {partition_name} USING {method} ({columns})
            {where_clause}
            """
            
            await self.db.execute(query)
            
        logger.debug(f"Created indexes for partition {partition_name}")
        
    async def create_future_partitions(
        self,
        months_ahead: int = 3
    ) -> None:
        """
        Create partitions for future months.
        
        Args:
            months_ahead: Number of months to create partitions for
        """
        current_date = datetime.now()
        
        for table_name in self.PARTITIONED_TABLES:
            for month_offset in range(months_ahead + 1):
                target_date = current_date + timedelta(days=30 * month_offset)
                await self.create_monthly_partition(
                    table_name,
                    target_date.year,
                    target_date.month
                )
                
    async def prune_old_partitions(self) -> None:
        """Remove partitions older than retention period."""
        for table_name, config in self.PARTITIONED_TABLES.items():
            retention_months = config['retention_months']
            
            await self.db.execute(
                "SELECT drop_old_partition($1, $2)",
                table_name,
                retention_months
            )
            
            logger.info(
                f"Pruned old partitions for {table_name} "
                f"(retention: {retention_months} months)"
            )
            
    async def get_partition_statistics(
        self,
        table_name: str
    ) -> List[Dict[str, any]]:
        """
        Get statistics for all partitions of a table.
        
        Args:
            table_name: Name of the partitioned table
            
        Returns:
            List of partition statistics
        """
        rows = await self.db.fetch(
            "SELECT * FROM get_partition_stats($1)",
            table_name
        )
        
        return [dict(row) for row in rows]
        
    async def analyze_partitions(self) -> None:
        """Run ANALYZE on all partitions for query optimization."""
        for table_name in self.PARTITIONED_TABLES:
            await self.db.execute(f"ANALYZE {table_name}")
            logger.debug(f"Analyzed partitions for {table_name}")
            
    async def start_maintenance_task(
        self,
        interval_hours: int = 24
    ) -> None:
        """
        Start automated partition maintenance task.
        
        Args:
            interval_hours: Hours between maintenance runs
        """
        async def maintenance_loop():
            while True:
                try:
                    logger.info("Running partition maintenance")
                    await self.create_future_partitions()
                    await self.prune_old_partitions()
                    await self.analyze_partitions()
                    logger.info("Partition maintenance completed")
                except Exception as e:
                    logger.error(f"Partition maintenance failed: {e}")
                    
                await asyncio.sleep(interval_hours * 3600)
                
        self._maintenance_task = asyncio.create_task(maintenance_loop())
        logger.info(f"Started partition maintenance task (interval: {interval_hours}h)")
        
    async def stop_maintenance_task(self) -> None:
        """Stop the maintenance task."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
            logger.info("Stopped partition maintenance task")
            
    async def optimize_partition_queries(self) -> None:
        """Create optimized views for common partition queries."""
        queries = [
            # Recent orders view
            """
            CREATE OR REPLACE VIEW recent_orders AS
            SELECT * FROM orders
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            """,
            
            # Today's trades view
            """
            CREATE OR REPLACE VIEW todays_trades AS
            SELECT * FROM trades
            WHERE executed_at >= CURRENT_DATE
            """,
            
            # Pending orders view
            """
            CREATE OR REPLACE VIEW pending_orders AS
            SELECT * FROM orders
            WHERE status = 'pending'
            AND created_at >= CURRENT_DATE - INTERVAL '1 day'
            """,
            
            # Performance summary view
            """
            CREATE OR REPLACE VIEW performance_summary AS
            SELECT 
                DATE(executed_at) as trade_date,
                symbol,
                COUNT(*) as trade_count,
                SUM(quantity) as total_volume,
                SUM(realized_pnl) as total_pnl
            FROM trades
            WHERE executed_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(executed_at), symbol
            """
        ]
        
        for query in queries:
            await self.db.execute(query)
            
        logger.info("Created optimized partition query views")