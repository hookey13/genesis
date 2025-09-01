"""Connection pooling and request batching optimization for Project GENESIS."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp
import structlog
from aiohttp import ClientSession, TCPConnector
from websockets import WebSocketClientProtocol
from websockets.client import connect as ws_connect

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    IDLE = "idle"
    ACTIVE = "active"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    created_at: datetime
    last_used: datetime
    requests_handled: int
    errors: int
    total_latency: float
    state: ConnectionState


@dataclass
class BatchRequest:
    """Batched request container."""
    requests: List[Dict[str, Any]]
    futures: List[asyncio.Future]
    created_at: float
    batch_id: str


class HTTPConnectionPool:
    """HTTP connection pool with optimized settings for aiohttp."""
    
    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 30,
        keepalive_timeout: int = 30,
        force_close: bool = False,
        enable_cleanup_closed: bool = True
    ):
        """Initialize HTTP connection pool.
        
        Args:
            max_connections: Maximum total connections
            max_keepalive_connections: Maximum keepalive connections
            keepalive_timeout: Keepalive timeout in seconds
            force_close: Force close connections after response
            enable_cleanup_closed: Enable cleanup of closed connections
        """
        self.max_connections = max_connections
        
        # Create optimized connector
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=max_keepalive_connections,
            keepalive_timeout=keepalive_timeout,
            force_close=force_close,
            enable_cleanup_closed=enable_cleanup_closed,
            ttl_dns_cache=300  # DNS cache for 5 minutes
        )
        
        # Session management
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # Connection tracking
        self.active_connections = 0
        self.total_requests = 0
        self.total_errors = 0
        
        # Performance metrics
        self.latency_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        
        logger.info(
            "HTTPConnectionPool initialized",
            max_connections=max_connections,
            max_keepalive=max_keepalive_connections
        )
    
    async def get_session(self) -> ClientSession:
        """Get or create HTTP session.
        
        Returns:
            aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = ClientSession(
                        connector=self.connector,
                        connector_owner=False,
                        timeout=aiohttp.ClientTimeout(total=30)
                    )
                    logger.debug("Created new HTTP session")
        
        return self._session
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request through pool.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
        
        Returns:
            HTTP response
        """
        session = await self.get_session()
        start_time = time.perf_counter()
        
        try:
            self.active_connections += 1
            self.total_requests += 1
            
            async with session.request(method, url, **kwargs) as response:
                latency = time.perf_counter() - start_time
                self.latency_history.append(latency)
                
                # Log slow requests
                if latency > 1.0:
                    logger.warning(
                        "Slow HTTP request",
                        method=method,
                        url=url,
                        latency=latency
                    )
                
                return response
        
        except Exception as e:
            self.total_errors += 1
            self.error_history.append({
                "timestamp": datetime.now(),
                "error": str(e),
                "url": url
            })
            raise
        
        finally:
            self.active_connections -= 1
    
    async def close(self) -> None:
        """Close connection pool and session."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        await self.connector.close()
        logger.info("HTTPConnectionPool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        avg_latency = (
            sum(self.latency_history) / len(self.latency_history)
            if self.latency_history else 0
        )
        
        return {
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / self.total_requests if self.total_requests > 0 else 0,
            "avg_latency": avg_latency,
            "max_latency": max(self.latency_history) if self.latency_history else 0,
            "connector_stats": {
                "limit": self.connector.limit,
                "limit_per_host": self.connector.limit_per_host
            }
        }


class WebSocketConnectionPool:
    """WebSocket connection pool with automatic reconnection."""
    
    def __init__(
        self,
        max_connections: int = 10,
        reconnect_interval: int = 5,
        heartbeat_interval: int = 30
    ):
        """Initialize WebSocket connection pool.
        
        Args:
            max_connections: Maximum concurrent connections
            reconnect_interval: Reconnection interval in seconds
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.max_connections = max_connections
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        
        # Connection storage
        self.connections: Dict[str, WebSocketClientProtocol] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        self.reconnect_tasks: Dict[str, asyncio.Task] = {}
        
        # Message queues for reconnection
        self.pending_messages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info(
            "WebSocketConnectionPool initialized",
            max_connections=max_connections,
            reconnect_interval=reconnect_interval
        )
    
    async def connect(self, url: str, **kwargs) -> WebSocketClientProtocol:
        """Connect to WebSocket endpoint.
        
        Args:
            url: WebSocket URL
            **kwargs: Additional connection parameters
        
        Returns:
            WebSocket connection
        """
        if url in self.connections:
            conn = self.connections[url]
            if not conn.closed:
                return conn
        
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            # Find and close least recently used connection
            lru_url = min(
                self.connection_stats.keys(),
                key=lambda k: self.connection_stats[k].last_used
            )
            await self.disconnect(lru_url)
        
        # Create new connection
        try:
            conn = await ws_connect(url, **kwargs)
            self.connections[url] = conn
            self.connection_stats[url] = ConnectionStats(
                created_at=datetime.now(),
                last_used=datetime.now(),
                requests_handled=0,
                errors=0,
                total_latency=0,
                state=ConnectionState.ACTIVE
            )
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_task(url))
            
            # Send any pending messages
            await self._flush_pending_messages(url)
            
            logger.info("WebSocket connected", url=url)
            return conn
        
        except Exception as e:
            logger.error("WebSocket connection failed", url=url, error=str(e))
            
            # Schedule reconnection
            if url not in self.reconnect_tasks:
                self.reconnect_tasks[url] = asyncio.create_task(
                    self._reconnect_task(url, **kwargs)
                )
            
            raise
    
    async def disconnect(self, url: str) -> None:
        """Disconnect WebSocket connection.
        
        Args:
            url: WebSocket URL
        """
        if url in self.connections:
            conn = self.connections[url]
            if not conn.closed:
                await conn.close()
            
            del self.connections[url]
            
            # Cancel reconnect task if exists
            if url in self.reconnect_tasks:
                self.reconnect_tasks[url].cancel()
                del self.reconnect_tasks[url]
            
            logger.info("WebSocket disconnected", url=url)
    
    async def send(self, url: str, message: Any) -> None:
        """Send message through WebSocket connection.
        
        Args:
            url: WebSocket URL
            message: Message to send
        """
        if url in self.connections:
            conn = self.connections[url]
            if not conn.closed:
                await conn.send(message)
                self.connection_stats[url].requests_handled += 1
                self.connection_stats[url].last_used = datetime.now()
                return
        
        # Queue message for later delivery
        self.pending_messages[url].append(message)
        
        # Ensure reconnection is scheduled
        if url not in self.reconnect_tasks:
            self.reconnect_tasks[url] = asyncio.create_task(
                self._reconnect_task(url)
            )
    
    async def _reconnect_task(self, url: str, **kwargs) -> None:
        """Background task for reconnection.
        
        Args:
            url: WebSocket URL
            **kwargs: Connection parameters
        """
        while True:
            try:
                await asyncio.sleep(self.reconnect_interval)
                await self.connect(url, **kwargs)
                break
            except Exception as e:
                logger.debug("Reconnection attempt failed", url=url, error=str(e))
    
    async def _heartbeat_task(self, url: str) -> None:
        """Send periodic heartbeats to keep connection alive.
        
        Args:
            url: WebSocket URL
        """
        while url in self.connections:
            try:
                conn = self.connections[url]
                if not conn.closed:
                    await conn.ping()
                    await asyncio.sleep(self.heartbeat_interval)
                else:
                    break
            except Exception as e:
                logger.debug("Heartbeat failed", url=url, error=str(e))
                break
    
    async def _flush_pending_messages(self, url: str) -> None:
        """Send pending messages after reconnection.
        
        Args:
            url: WebSocket URL
        """
        if url in self.pending_messages:
            messages = list(self.pending_messages[url])
            self.pending_messages[url].clear()
            
            for message in messages:
                try:
                    await self.send(url, message)
                except Exception as e:
                    logger.error("Failed to send pending message", url=url, error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        active_connections = sum(
            1 for url, conn in self.connections.items()
            if not conn.closed
        )
        
        total_requests = sum(
            stats.requests_handled
            for stats in self.connection_stats.values()
        )
        
        total_errors = sum(
            stats.errors
            for stats in self.connection_stats.values()
        )
        
        return {
            "active_connections": active_connections,
            "total_connections": len(self.connections),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "reconnect_tasks": len(self.reconnect_tasks),
            "pending_messages": sum(len(q) for q in self.pending_messages.values())
        }


class RequestBatcher:
    """Batch multiple requests for efficient processing."""
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 0.1,
        max_batches: int = 100
    ):
        """Initialize request batcher.
        
        Args:
            batch_size: Maximum requests per batch
            batch_timeout: Maximum time to wait for batch to fill
            max_batches: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_batches = max_batches
        
        # Batch management
        self.pending_batches: Dict[str, BatchRequest] = {}
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        self.batch_counter = 0
        
        # Request coalescing
        self.request_cache: Dict[str, Any] = {}
        self.cache_ttl = 1.0  # 1 second cache
        
        # Backpressure handling
        self.queue_sizes: Dict[str, int] = defaultdict(int)
        self.max_queue_size = 1000
        
        logger.info(
            "RequestBatcher initialized",
            batch_size=batch_size,
            batch_timeout=batch_timeout
        )
    
    async def add_request(
        self,
        endpoint: str,
        request: Dict[str, Any],
        processor: Callable
    ) -> Any:
        """Add request to batch.
        
        Args:
            endpoint: Endpoint identifier
            request: Request data
            processor: Function to process batch
        
        Returns:
            Request result
        """
        # Check for backpressure
        if self.queue_sizes[endpoint] >= self.max_queue_size:
            raise RuntimeError(f"Queue full for endpoint: {endpoint}")
        
        # Check request cache for coalescing
        cache_key = f"{endpoint}:{hash(str(request))}"
        if cache_key in self.request_cache:
            cached_result, cached_time = self.request_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        # Create or get batch
        if endpoint not in self.pending_batches:
            batch_id = f"batch_{self.batch_counter}"
            self.batch_counter += 1
            
            self.pending_batches[endpoint] = BatchRequest(
                requests=[],
                futures=[],
                created_at=time.time(),
                batch_id=batch_id
            )
            
            # Schedule batch processing
            self.batch_tasks[endpoint] = asyncio.create_task(
                self._process_batch(endpoint, processor)
            )
        
        batch = self.pending_batches[endpoint]
        
        # Add request to batch
        future = asyncio.Future()
        batch.requests.append(request)
        batch.futures.append(future)
        self.queue_sizes[endpoint] += 1
        
        # Process immediately if batch is full
        if len(batch.requests) >= self.batch_size:
            if endpoint in self.batch_tasks:
                self.batch_tasks[endpoint].cancel()
            
            asyncio.create_task(self._process_batch(endpoint, processor))
        
        # Wait for result
        result = await future
        
        # Cache result for coalescing
        self.request_cache[cache_key] = (result, time.time())
        
        # Cleanup old cache entries periodically
        if len(self.request_cache) > 1000:
            self._cleanup_cache()
        
        return result
    
    async def _process_batch(
        self,
        endpoint: str,
        processor: Callable
    ) -> None:
        """Process a batch of requests.
        
        Args:
            endpoint: Endpoint identifier
            processor: Batch processing function
        """
        # Wait for timeout or batch to fill
        if endpoint in self.pending_batches:
            batch = self.pending_batches[endpoint]
            
            # Wait for timeout if batch not full
            if len(batch.requests) < self.batch_size:
                time_elapsed = time.time() - batch.created_at
                if time_elapsed < self.batch_timeout:
                    await asyncio.sleep(self.batch_timeout - time_elapsed)
            
            # Remove batch from pending
            if endpoint in self.pending_batches:
                batch = self.pending_batches.pop(endpoint)
                
                try:
                    # Process batch
                    results = await processor(batch.requests)
                    
                    # Distribute results
                    for future, result in zip(batch.futures, results):
                        if not future.done():
                            future.set_result(result)
                    
                    # Update queue size
                    self.queue_sizes[endpoint] -= len(batch.requests)
                    
                    logger.debug(
                        "Batch processed",
                        endpoint=endpoint,
                        batch_id=batch.batch_id,
                        size=len(batch.requests)
                    )
                
                except Exception as e:
                    # Set exception for all futures
                    for future in batch.futures:
                        if not future.done():
                            future.set_exception(e)
                    
                    logger.error(
                        "Batch processing failed",
                        endpoint=endpoint,
                        batch_id=batch.batch_id,
                        error=str(e)
                    )
                
                finally:
                    # Clean up task reference
                    if endpoint in self.batch_tasks:
                        del self.batch_tasks[endpoint]
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, cached_time) in self.request_cache.items()
            if current_time - cached_time > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.request_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics.
        
        Returns:
            Batcher statistics
        """
        return {
            "pending_batches": len(self.pending_batches),
            "active_tasks": len(self.batch_tasks),
            "total_batches": self.batch_counter,
            "cache_size": len(self.request_cache),
            "queue_sizes": dict(self.queue_sizes),
            "max_queue_size": self.max_queue_size
        }


class ConnectionPoolManager:
    """Central manager for all connection pools."""
    
    def __init__(self):
        """Initialize connection pool manager."""
        self.http_pool = HTTPConnectionPool()
        self.ws_pool = WebSocketConnectionPool()
        self.batcher = RequestBatcher()
        
        logger.info("ConnectionPoolManager initialized")
    
    async def optimize_pools(self) -> Dict[str, Any]:
        """Optimize connection pool settings based on usage.
        
        Returns:
            Optimization report
        """
        recommendations = []
        
        # Analyze HTTP pool
        http_stats = self.http_pool.get_stats()
        if http_stats["error_rate"] > 0.1:
            recommendations.append(
                "High HTTP error rate - consider increasing timeouts or connection limits"
            )
        
        if http_stats["avg_latency"] > 1.0:
            recommendations.append(
                "High average latency - consider connection pooling optimization"
            )
        
        # Analyze WebSocket pool
        ws_stats = self.ws_pool.get_stats()
        if ws_stats["reconnect_tasks"] > 5:
            recommendations.append(
                "Multiple WebSocket reconnections - check connection stability"
            )
        
        # Analyze batcher
        batch_stats = self.batcher.get_stats()
        max_queue = max(batch_stats["queue_sizes"].values()) if batch_stats["queue_sizes"] else 0
        if max_queue > self.batcher.max_queue_size * 0.8:
            recommendations.append(
                "Request queue near capacity - consider increasing batch size or processing rate"
            )
        
        return {
            "http_pool": http_stats,
            "websocket_pool": ws_stats,
            "batcher": batch_stats,
            "recommendations": recommendations
        }
    
    async def shutdown(self) -> None:
        """Shutdown all connection pools."""
        await self.http_pool.close()
        
        for url in list(self.ws_pool.connections.keys()):
            await self.ws_pool.disconnect(url)
        
        logger.info("ConnectionPoolManager shutdown complete")