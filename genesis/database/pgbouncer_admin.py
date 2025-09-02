"""
PgBouncer admin interface for statistics and management.
Provides direct access to PgBouncer's admin database for monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

import asyncpg

logger = logging.getLogger(__name__)


class PgBouncerAdmin:
    """
    Interface to PgBouncer admin database for statistics and management.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6432,
        user: str = "pgbouncer_admin",
        password: str = "",
        database: str = "pgbouncer"
    ):
        """
        Initialize PgBouncer admin connection.
        
        Args:
            host: PgBouncer host
            port: PgBouncer port
            user: Admin user (must be in admin_users)
            password: Admin password
            database: Should be 'pgbouncer' for admin access
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._conn: Optional[asyncpg.Connection] = None
        
    async def connect(self) -> None:
        """Connect to PgBouncer admin database."""
        try:
            self._conn = await asyncpg.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                timeout=10
            )
            logger.info("Connected to PgBouncer admin interface")
        except Exception as e:
            logger.error(f"Failed to connect to PgBouncer admin: {e}")
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from PgBouncer admin."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Disconnected from PgBouncer admin")
            
    async def get_stats(self) -> List[Dict[str, Any]]:
        """
        Get database statistics from PgBouncer.
        
        Returns:
            List of database statistics
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW STATS"
        rows = await self._conn.fetch(query)
        
        stats = []
        for row in rows:
            stats.append({
                'database': row['database'],
                'total_xact_count': row['total_xact_count'],
                'total_query_count': row['total_query_count'],
                'total_received': row['total_received'],
                'total_sent': row['total_sent'],
                'total_xact_time': row['total_xact_time'],
                'total_query_time': row['total_query_time'],
                'total_wait_time': row['total_wait_time'],
                'avg_xact_count': row['avg_xact_count'],
                'avg_query_count': row['avg_query_count'],
                'avg_recv': row['avg_recv'],
                'avg_sent': row['avg_sent'],
                'avg_xact_time': row['avg_xact_time'],
                'avg_query_time': row['avg_query_time'],
                'avg_wait_time': row['avg_wait_time'],
            })
            
        return stats
        
    async def get_pools(self) -> List[Dict[str, Any]]:
        """
        Get connection pool information.
        
        Returns:
            List of pool information
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW POOLS"
        rows = await self._conn.fetch(query)
        
        pools = []
        for row in rows:
            pools.append({
                'database': row['database'],
                'user': row['user'],
                'cl_active': row['cl_active'],  # Client connections active
                'cl_waiting': row['cl_waiting'],  # Client connections waiting
                'sv_active': row['sv_active'],  # Server connections active
                'sv_idle': row['sv_idle'],  # Server connections idle
                'sv_used': row['sv_used'],  # Server connections used
                'sv_tested': row['sv_tested'],  # Server connections tested
                'sv_login': row['sv_login'],  # Server connections logging in
                'maxwait': row['maxwait'],  # Max time in seconds a client waited
                'maxwait_us': row['maxwait_us'],  # Max wait time in microseconds
                'pool_mode': row['pool_mode'],
            })
            
        return pools
        
    async def get_clients(self) -> List[Dict[str, Any]]:
        """
        Get active client connections.
        
        Returns:
            List of client connections
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW CLIENTS"
        rows = await self._conn.fetch(query)
        
        clients = []
        for row in rows:
            clients.append({
                'type': row['type'],
                'user': row['user'],
                'database': row['database'],
                'state': row['state'],
                'addr': row['addr'],
                'port': row['port'],
                'local_addr': row['local_addr'],
                'local_port': row['local_port'],
                'connect_time': row['connect_time'],
                'request_time': row['request_time'],
                'wait': row['wait'],
                'wait_us': row['wait_us'],
                'close_needed': row['close_needed'],
                'ptr': row['ptr'],
                'link': row['link'],
                'remote_pid': row['remote_pid'],
                'tls': row['tls'],
            })
            
        return clients
        
    async def get_servers(self) -> List[Dict[str, Any]]:
        """
        Get server connections.
        
        Returns:
            List of server connections
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW SERVERS"
        rows = await self._conn.fetch(query)
        
        servers = []
        for row in rows:
            servers.append({
                'type': row['type'],
                'user': row['user'],
                'database': row['database'],
                'state': row['state'],
                'addr': row['addr'],
                'port': row['port'],
                'local_addr': row['local_addr'],
                'local_port': row['local_port'],
                'connect_time': row['connect_time'],
                'request_time': row['request_time'],
                'wait': row['wait'],
                'wait_us': row['wait_us'],
                'close_needed': row['close_needed'],
                'ptr': row['ptr'],
                'link': row['link'],
                'remote_pid': row['remote_pid'],
                'tls': row['tls'],
            })
            
        return servers
        
    async def get_databases(self) -> List[Dict[str, Any]]:
        """
        Get configured databases.
        
        Returns:
            List of database configurations
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW DATABASES"
        rows = await self._conn.fetch(query)
        
        databases = []
        for row in rows:
            databases.append({
                'name': row['name'],
                'host': row['host'],
                'port': row['port'],
                'database': row['database'],
                'force_user': row['force_user'],
                'pool_size': row['pool_size'],
                'min_pool_size': row['min_pool_size'],
                'reserve_pool': row['reserve_pool'],
                'pool_mode': row['pool_mode'],
                'max_connections': row['max_connections'],
                'current_connections': row['current_connections'],
                'paused': row['paused'],
                'disabled': row['disabled'],
            })
            
        return databases
        
    async def get_config(self) -> Dict[str, str]:
        """
        Get PgBouncer configuration.
        
        Returns:
            Dictionary of configuration parameters
        """
        if not self._conn:
            await self.connect()
            
        query = "SHOW CONFIG"
        rows = await self._conn.fetch(query)
        
        config = {}
        for row in rows:
            config[row['key']] = row['value']
            
        return config
        
    async def reload(self) -> None:
        """Reload PgBouncer configuration."""
        if not self._conn:
            await self.connect()
            
        await self._conn.execute("RELOAD")
        logger.info("PgBouncer configuration reloaded")
        
    async def pause(self, database: str) -> None:
        """
        Pause a database (stop accepting new queries).
        
        Args:
            database: Database name to pause
        """
        if not self._conn:
            await self.connect()
            
        await self._conn.execute(f"PAUSE {database}")
        logger.info(f"Database {database} paused")
        
    async def resume(self, database: str) -> None:
        """
        Resume a paused database.
        
        Args:
            database: Database name to resume
        """
        if not self._conn:
            await self.connect()
            
        await self._conn.execute(f"RESUME {database}")
        logger.info(f"Database {database} resumed")
        
    async def kill_database(self, database: str) -> None:
        """
        Kill all connections for a database.
        
        Args:
            database: Database name
        """
        if not self._conn:
            await self.connect()
            
        await self._conn.execute(f"KILL {database}")
        logger.warning(f"Killed all connections for database {database}")
        
    async def wait_close(self, database: str) -> None:
        """
        Wait for all clients to disconnect and close database.
        
        Args:
            database: Database name
        """
        if not self._conn:
            await self.connect()
            
        await self._conn.execute(f"DISABLE {database}")
        await self._conn.execute(f"WAIT_CLOSE {database}")
        logger.info(f"Database {database} closed after clients disconnected")
        
    async def get_pool_metrics(self, database: str = "genesis_trading") -> Dict[str, Any]:
        """
        Get detailed metrics for a specific database pool.
        
        Args:
            database: Database name
            
        Returns:
            Dictionary of pool metrics
        """
        stats = await self.get_stats()
        pools = await self.get_pools()
        
        # Find metrics for specified database
        db_stats = next((s for s in stats if s['database'] == database), {})
        db_pools = [p for p in pools if p['database'] == database]
        
        # Calculate aggregate pool metrics
        total_active = sum(p['cl_active'] for p in db_pools)
        total_waiting = sum(p['cl_waiting'] for p in db_pools)
        total_server_active = sum(p['sv_active'] for p in db_pools)
        total_server_idle = sum(p['sv_idle'] for p in db_pools)
        
        metrics = {
            'database': database,
            'client_connections_active': total_active,
            'client_connections_waiting': total_waiting,
            'server_connections_active': total_server_active,
            'server_connections_idle': total_server_idle,
            'total_queries': db_stats.get('total_query_count', 0),
            'avg_query_time_us': db_stats.get('avg_query_time', 0),
            'avg_wait_time_us': db_stats.get('avg_wait_time', 0),
            'total_received_bytes': db_stats.get('total_received', 0),
            'total_sent_bytes': db_stats.get('total_sent', 0),
        }
        
        # Calculate utilization
        if total_server_active + total_server_idle > 0:
            metrics['pool_utilization'] = (
                total_server_active / (total_server_active + total_server_idle) * 100
            )
        else:
            metrics['pool_utilization'] = 0
            
        return metrics
        
    async def monitor_pool_health(
        self,
        database: str = "genesis_trading",
        warning_threshold: int = 10,
        critical_threshold: int = 50
    ) -> Dict[str, Any]:
        """
        Monitor pool health and return status.
        
        Args:
            database: Database to monitor
            warning_threshold: Warning threshold for waiting connections
            critical_threshold: Critical threshold for waiting connections
            
        Returns:
            Health status dictionary
        """
        metrics = await self.get_pool_metrics(database)
        
        waiting = metrics['client_connections_waiting']
        
        if waiting >= critical_threshold:
            status = 'critical'
            message = f"Critical: {waiting} connections waiting in pool"
        elif waiting >= warning_threshold:
            status = 'warning'
            message = f"Warning: {waiting} connections waiting in pool"
        else:
            status = 'healthy'
            message = f"Healthy: {waiting} connections waiting in pool"
            
        return {
            'status': status,
            'message': message,
            'metrics': metrics,
            'thresholds': {
                'warning': warning_threshold,
                'critical': critical_threshold
            }
        }