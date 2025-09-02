"""
PostgreSQL Connection Manager with PgBouncer support.
Handles connection pooling, retries, and health checks for trading workloads.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from asyncpg import Connection, Pool, Record
from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration with PgBouncer support."""
    
    # Connection settings
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=6432, env="DB_PORT")  # PgBouncer port
    database: str = Field(default="genesis_trading", env="DB_NAME")
    user: str = Field(default="genesis", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    # Pool settings
    min_pool_size: int = Field(default=10, env="DB_MIN_POOL_SIZE")
    max_pool_size: int = Field(default=50, env="DB_MAX_POOL_SIZE")
    
    # Timeout settings (in seconds)
    connect_timeout: float = Field(default=10.0, env="DB_CONNECT_TIMEOUT")
    command_timeout: float = Field(default=30.0, env="DB_COMMAND_TIMEOUT")
    
    # Retry settings
    max_retries: int = Field(default=3, env="DB_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="DB_RETRY_DELAY")
    
    # SSL settings
    ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PostgresManager:
    """
    Manages PostgreSQL connections through PgBouncer.
    Provides connection pooling, health checks, and retry logic.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize PostgreSQL manager.
        
        Args:
            config: Database configuration. Uses defaults if not provided.
        """
        self.config = config or DatabaseConfig()
        self._pool: Optional[Pool] = None
        self._lock = asyncio.Lock()
        self._is_connected = False
        
    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL via PgBouncer."""
        async with self._lock:
            if self._is_connected:
                return
                
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    timeout=self.config.connect_timeout,
                    command_timeout=self.config.command_timeout,
                    # Connection init for Decimal support
                    init=self._init_connection,
                    # Server settings for PgBouncer compatibility
                    server_settings={
                        'application_name': 'genesis_trading',
                        'jit': 'off'  # Disable JIT for consistent performance
                    }
                )
                self._is_connected = True
                logger.info(
                    f"Connected to PostgreSQL via PgBouncer at {self.config.host}:{self.config.port}"
                )
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise
                
    async def disconnect(self) -> None:
        """Close connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                self._is_connected = False
                logger.info("Disconnected from PostgreSQL")
                
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.
        
        Yields:
            Connection object for executing queries.
        """
        if not self._is_connected:
            await self.connect()
            
        async with self._pool.acquire() as conn:
            yield conn
            
    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Command status string
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
            
    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Record]:
        """
        Execute a query and fetch all results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of records
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
            
    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Record]:
        """
        Execute a query and fetch a single row.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Single record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
            
    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a query and fetch a single value.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            column: Column index to fetch
            timeout: Query timeout in seconds
            
        Returns:
            Single value or None
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)
            
    async def execute_many(
        self,
        query: str,
        args_list: List[Tuple],
        timeout: Optional[float] = None
    ) -> None:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query to execute
            args_list: List of parameter tuples
            timeout: Query timeout in seconds
        """
        async with self.acquire() as conn:
            await conn.executemany(query, args_list, timeout=timeout)
            
    async def transaction(self):
        """
        Create a transaction context.
        
        Usage:
            async with db.transaction() as tx:
                await db.execute("INSERT INTO ...", ...)
                await db.execute("UPDATE ...", ...)
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
                
    async def health_check(self) -> bool:
        """
        Check database connectivity and health.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            result = await self.fetchval("SELECT 1", timeout=5.0)
            return result == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        if not self._pool:
            return {}
            
        return {
            "size": self._pool.get_size(),
            "free_size": self._pool.get_free_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "total_connections": self._pool.get_size(),
            "idle_connections": self._pool.get_idle_size() if hasattr(self._pool, 'get_idle_size') else None,
        }
        
    @staticmethod
    async def _init_connection(conn: Connection) -> None:
        """
        Initialize connection with custom type codecs.
        
        Args:
            conn: Connection to initialize
        """
        # Setup Decimal type handling for financial precision
        await conn.set_type_codec(
            'numeric',
            encoder=str,
            decoder=Decimal,
            schema='pg_catalog',
            format='text'
        )
        
        # Set statement timeout for safety
        await conn.execute("SET statement_timeout = '30s'")
        
        # Set timezone to UTC
        await conn.execute("SET timezone = 'UTC'")
        
    async def retry_operation(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await operation(*args, **kwargs)
            except (asyncpg.PostgresConnectionError, asyncpg.InterfaceError) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    # Try to reconnect if connection is lost
                    if not await self.health_check():
                        await self.disconnect()
                        await self.connect()
                else:
                    logger.error(f"Database operation failed after {self.config.max_retries} attempts")
                    
        raise last_error
        
    async def ensure_connected(self) -> None:
        """Ensure database connection is established."""
        if not self._is_connected:
            await self.connect()
        elif not await self.health_check():
            logger.warning("Connection unhealthy, reconnecting...")
            await self.disconnect()
            await self.connect()