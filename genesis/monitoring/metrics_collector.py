"""Metrics collector for trading components."""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import psutil
import structlog

from ..core.events import Event, EventType
from ..core.models import Order, OrderStatus, Position
from .prometheus_exporter import MetricsRegistry

logger = structlog.get_logger(__name__)

# Input validation patterns
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,20}$')
SIDE_PATTERN = re.compile(r'^(BUY|SELL|buy|sell)$')
EXCHANGE_PATTERN = re.compile(r'^[a-z0-9_]{2,50}$')
MAX_METRIC_VALUE = 1e12  # Maximum allowed metric value
MIN_METRIC_VALUE = -1e12  # Minimum allowed metric value


@dataclass
class TradingMetrics:
    """Container for trading metrics."""
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    orders_failed: int = 0
    trades_executed: int = 0

    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    positions_opened: int = 0
    positions_closed: int = 0
    current_positions: int = 0

    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    websocket_connected: bool = False
    websocket_latency_ms: float = 0.0
    rest_api_latency_ms: float = 0.0

    rate_limit_usage: float = 0.0
    rate_limit_remaining: int = 0

    tilt_score: float = 0.0
    tilt_indicators: dict[str, float] = field(default_factory=dict)
    
    # System health metrics
    cpu_usage: float = 0.0  # Percentage
    memory_usage: float = 0.0  # Bytes
    memory_percent: float = 0.0  # Percentage
    disk_usage_percent: float = 0.0  # Percentage
    disk_io_read_bytes: float = 0.0  # Bytes per second
    disk_io_write_bytes: float = 0.0  # Bytes per second
    network_bytes_sent: float = 0.0  # Bytes per second
    network_bytes_recv: float = 0.0  # Bytes per second
    network_packets_sent: float = 0.0  # Packets per second
    network_packets_recv: float = 0.0  # Packets per second
    connection_count: int = 0  # Number of network connections
    thread_count: int = 0  # Number of threads
    open_files: int = 0  # Number of open file descriptors
    system_uptime: float = 0.0  # Seconds
    process_uptime: float = 0.0  # Seconds
    
    # Health score (0-100)
    health_score: float = 100.0

    last_update: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and aggregates metrics from trading components."""

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self.metrics = TradingMetrics()
        self._start_time = time.time()
        self._peak_balance = Decimal("0")
        self._trades_history: list[dict[str, Any]] = []
        self._latency_samples: list[float] = []
        self._collection_interval = 10  # seconds
        self._collection_task: asyncio.Task | None = None

        # Register collector with registry
        self.registry.register_collector(self._update_metrics)

    async def start(self) -> None:
        """Start metrics collection."""
        if not self._collection_task:
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Started metrics collector")

    async def stop(self) -> None:
        """Stop metrics collection."""
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            logger.info("Stopped metrics collector")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while True:
            try:
                await self._collect_system_metrics()
                await self._calculate_derived_metrics()
                await self._update_prometheus_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection", error=str(e))
                await asyncio.sleep(self._collection_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system-level metrics with bounds checking."""
        try:
            # Process-level metrics
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            self.metrics.cpu_usage = max(0, min(100, cpu_percent))  # Bound to 0-100%
            
            # Memory metrics
            memory_info = process.memory_info()
            self.metrics.memory_usage = max(0, min(100 * 1024**3, memory_info.rss))  # Cap at 100GB
            
            # Calculate memory percentage
            total_memory = psutil.virtual_memory().total
            self.metrics.memory_percent = (memory_info.rss / total_memory * 100) if total_memory > 0 else 0
            
            # Thread and file descriptor count
            self.metrics.thread_count = process.num_threads()
            try:
                self.metrics.open_files = len(process.open_files())
            except (psutil.AccessDenied, AttributeError):
                self.metrics.open_files = 0
            
            # Network connections
            try:
                connections = process.connections()
                self.metrics.connection_count = min(len(connections), 10000)  # Cap at 10k
            except psutil.AccessDenied:
                self.metrics.connection_count = 0
            
            # Process uptime
            create_time = process.create_time()
            self.metrics.process_uptime = time.time() - create_time
            
            # System-wide metrics
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            self.metrics.disk_usage_percent = disk_usage.percent
            
            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                if hasattr(self, '_last_disk_io') and self._last_disk_io:
                    time_delta = time.time() - self._last_disk_io_time
                    if time_delta > 0:
                        self.metrics.disk_io_read_bytes = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta
                        self.metrics.disk_io_write_bytes = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta
                self._last_disk_io = disk_io
                self._last_disk_io_time = time.time()
            except (AttributeError, RuntimeError):
                # Disk I/O not available on all platforms
                pass
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_io') and self._last_net_io:
                    time_delta = time.time() - self._last_net_io_time
                    if time_delta > 0:
                        self.metrics.network_bytes_sent = (net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
                        self.metrics.network_bytes_recv = (net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta
                        self.metrics.network_packets_sent = (net_io.packets_sent - self._last_net_io.packets_sent) / time_delta
                        self.metrics.network_packets_recv = (net_io.packets_recv - self._last_net_io.packets_recv) / time_delta
                self._last_net_io = net_io
                self._last_net_io_time = time.time()
            except (AttributeError, RuntimeError):
                pass
            
            # System uptime
            boot_time = psutil.boot_time()
            self.metrics.system_uptime = time.time() - boot_time
            
            # Calculate health score
            self._calculate_health_score()

        except psutil.NoSuchProcess:
            logger.warning("Process not found for metrics collection")
        except psutil.AccessDenied:
            logger.warning("Access denied for system metrics")
        except Exception as e:
            logger.warning("Failed to collect system metrics", error=str(e))

    def _calculate_health_score(self) -> None:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # CPU penalty (high CPU usage reduces score)
        if self.metrics.cpu_usage > 90:
            score -= 30
        elif self.metrics.cpu_usage > 70:
            score -= 15
        elif self.metrics.cpu_usage > 50:
            score -= 5
            
        # Memory penalty
        if self.metrics.memory_percent > 90:
            score -= 30
        elif self.metrics.memory_percent > 70:
            score -= 15
        elif self.metrics.memory_percent > 50:
            score -= 5
            
        # Disk usage penalty
        if self.metrics.disk_usage_percent > 90:
            score -= 20
        elif self.metrics.disk_usage_percent > 80:
            score -= 10
        elif self.metrics.disk_usage_percent > 70:
            score -= 5
            
        # Connection count penalty (too many connections)
        if self.metrics.connection_count > 1000:
            score -= 10
        elif self.metrics.connection_count > 500:
            score -= 5
            
        # Thread count penalty
        if self.metrics.thread_count > 100:
            score -= 10
        elif self.metrics.thread_count > 50:
            score -= 5
            
        # Open files penalty
        if self.metrics.open_files > 1000:
            score -= 10
        elif self.metrics.open_files > 500:
            score -= 5
            
        # Rate limit penalty
        if self.metrics.rate_limit_usage > 90:
            score -= 20
        elif self.metrics.rate_limit_usage > 70:
            score -= 10
            
        # Tilt score penalty
        if self.metrics.tilt_score > 70:
            score -= 15
        elif self.metrics.tilt_score > 50:
            score -= 10
        elif self.metrics.tilt_score > 30:
            score -= 5
            
        # Ensure score stays within bounds
        self.metrics.health_score = max(0, min(100, score))
    
    async def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data."""
        # Calculate win rate
        if self.metrics.trades_executed > 0:
            wins = sum(1 for trade in self._trades_history if trade.get("pnl", 0) > 0)
            self.metrics.win_rate = wins / self.metrics.trades_executed

        # Calculate Sharpe ratio (simplified)
        if len(self._trades_history) > 1:
            returns = [trade.get("return", 0) for trade in self._trades_history]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                if std_return > 0:
                    self.metrics.sharpe_ratio = avg_return / std_return * (252 ** 0.5)  # Annualized

        # Calculate drawdown
        current_balance = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        if self._peak_balance > 0:
            self.metrics.current_drawdown = float(
                (self._peak_balance - current_balance) / self._peak_balance * 100
            )
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, self.metrics.current_drawdown)

    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics."""
        # Uptime
        await self.registry.set_gauge(
            "genesis_up",
            time.time() - self._start_time
        )

        # Trading metrics
        await self.registry.set_gauge(
            "genesis_position_count",
            float(self.metrics.current_positions)
        )

        await self.registry.set_gauge(
            "genesis_pnl_dollars",
            float(self.metrics.realized_pnl + self.metrics.unrealized_pnl)
        )

        await self.registry.set_gauge(
            "genesis_connection_status",
            1.0 if self.metrics.websocket_connected else 0.0
        )

        await self.registry.set_gauge(
            "genesis_rate_limit_usage_ratio",
            self.metrics.rate_limit_usage
        )

        await self.registry.set_gauge(
            "genesis_tilt_score",
            self.metrics.tilt_score
        )

        await self.registry.set_gauge(
            "genesis_drawdown_percent",
            self.metrics.current_drawdown
        )

        # System health metrics
        await self.registry.set_gauge(
            "genesis_cpu_usage_percent",
            float(self.metrics.cpu_usage),
            {"type": "process"}
        )
        
        await self.registry.set_gauge(
            "genesis_memory_usage_bytes",
            float(self.metrics.memory_usage),
            {"type": "rss"}
        )
        
        await self.registry.set_gauge(
            "genesis_memory_usage_percent",
            float(self.metrics.memory_percent)
        )
        
        await self.registry.set_gauge(
            "genesis_disk_usage_percent",
            float(self.metrics.disk_usage_percent)
        )
        
        await self.registry.set_gauge(
            "genesis_disk_io_read_bytes_per_second",
            float(self.metrics.disk_io_read_bytes)
        )
        
        await self.registry.set_gauge(
            "genesis_disk_io_write_bytes_per_second",
            float(self.metrics.disk_io_write_bytes)
        )
        
        await self.registry.set_gauge(
            "genesis_network_bytes_sent_per_second",
            float(self.metrics.network_bytes_sent)
        )
        
        await self.registry.set_gauge(
            "genesis_network_bytes_recv_per_second",
            float(self.metrics.network_bytes_recv)
        )
        
        await self.registry.set_gauge(
            "genesis_network_packets_sent_per_second",
            float(self.metrics.network_packets_sent)
        )
        
        await self.registry.set_gauge(
            "genesis_network_packets_recv_per_second",
            float(self.metrics.network_packets_recv)
        )
        
        await self.registry.set_gauge(
            "genesis_connection_count",
            float(self.metrics.connection_count)
        )
        
        await self.registry.set_gauge(
            "genesis_thread_count",
            float(self.metrics.thread_count)
        )
        
        await self.registry.set_gauge(
            "genesis_open_files_count",
            float(self.metrics.open_files)
        )
        
        await self.registry.set_gauge(
            "genesis_system_uptime_seconds",
            float(self.metrics.system_uptime)
        )
        
        await self.registry.set_gauge(
            "genesis_process_uptime_seconds",
            float(self.metrics.process_uptime)
        )
        
        await self.registry.set_gauge(
            "genesis_health_score",
            float(self.metrics.health_score),
            {"description": "Overall system health score (0-100)"}
        )

    async def _update_metrics(self) -> None:
        """Update metrics (called by registry during collection)."""
        await self._update_prometheus_metrics()

    async def record_order(self, order: Order) -> None:
        """Record order metrics with validation."""
        if not order:
            logger.warning("Attempted to record null order")
            return

        try:
            self.metrics.orders_placed += 1

            if order.status == OrderStatus.FILLED:
                self.metrics.orders_filled += 1
                self.metrics.trades_executed += 1
                await self.registry.increment_counter("genesis_trades_total")
            elif order.status == OrderStatus.CANCELLED:
                self.metrics.orders_cancelled += 1
            elif order.status in [OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                self.metrics.orders_failed += 1
                await self.registry.increment_counter("genesis_orders_failed_total")

            await self.registry.increment_counter("genesis_orders_total")

            logger.debug("Recorded order metrics",
                        order_id=order.client_order_id,
                        status=order.status.value)
        except AttributeError as e:
            logger.error("Invalid order object", error=str(e))
        except Exception as e:
            logger.error("Failed to record order metrics", error=str(e))

    async def record_execution_time(self, duration_seconds: float) -> None:
        """Record order execution time with validation."""
        try:
            # Validate duration is reasonable (0 to 60 seconds)
            if duration_seconds < 0:
                logger.warning("Negative execution time", duration=duration_seconds)
                return
            if duration_seconds > 60:
                logger.warning("Excessive execution time", duration=duration_seconds)
                duration_seconds = 60  # Cap at 60 seconds

            await self.registry.observe_histogram(
                "genesis_order_execution_time_seconds",
                duration_seconds
            )
        except Exception as e:
            logger.error("Failed to record execution time", error=str(e))

    async def record_websocket_latency(self, latency_ms: float) -> None:
        """Record WebSocket latency with validation."""
        try:
            # Validate latency is reasonable (0 to 10000ms)
            if latency_ms < 0:
                logger.warning("Negative latency", latency=latency_ms)
                return
            if latency_ms > 10000:
                logger.warning("Excessive latency", latency=latency_ms)
                latency_ms = 10000  # Cap at 10 seconds

            self.metrics.websocket_latency_ms = latency_ms
            await self.registry.observe_histogram(
                "genesis_websocket_latency_ms",
                latency_ms
            )

            # Keep sample for averaging
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > 100:
                self._latency_samples.pop(0)
        except Exception as e:
            logger.error("Failed to record websocket latency", error=str(e))

    async def update_position_metrics(self, positions: list[Position]) -> None:
        """Update position-related metrics."""
        self.metrics.current_positions = len(positions)

        # Calculate unrealized P&L
        unrealized = Decimal("0")
        for position in positions:
            if hasattr(position, 'unrealized_pnl'):
                unrealized += position.unrealized_pnl

        self.metrics.unrealized_pnl = unrealized

        logger.debug("Updated position metrics",
                    count=self.metrics.current_positions,
                    unrealized_pnl=float(unrealized))

    async def update_pnl(self, realized: Decimal, unrealized: Decimal) -> None:
        """Update P&L metrics with validation."""
        try:
            # Validate P&L values are reasonable
            if abs(float(realized)) > MAX_METRIC_VALUE:
                logger.warning("Excessive realized P&L", pnl=realized)
                return
            if abs(float(unrealized)) > MAX_METRIC_VALUE:
                logger.warning("Excessive unrealized P&L", pnl=unrealized)
                return

            self.metrics.realized_pnl = realized
            self.metrics.unrealized_pnl = unrealized

            # Calculate and update drawdown
            await self._calculate_derived_metrics()
        except (InvalidOperation, ValueError) as e:
            logger.error("Invalid P&L values", error=str(e))
        except Exception as e:
            logger.error("Failed to update P&L", error=str(e))

    async def update_connection_status(self, connected: bool, exchange: str = "binance") -> None:
        """Update connection status with validation."""
        try:
            # Validate exchange name
            if not EXCHANGE_PATTERN.match(exchange.lower()):
                logger.warning("Invalid exchange name", exchange=exchange)
                exchange = "unknown"

            self.metrics.websocket_connected = bool(connected)
            logger.info("Connection status updated",
                       exchange=exchange,
                       connected=connected)
        except Exception as e:
            logger.error("Failed to update connection status", error=str(e))

    async def update_rate_limits(self, used: int, limit: int) -> None:
        """Update rate limit metrics with validation."""
        try:
            # Validate inputs
            if used < 0 or limit <= 0:
                logger.warning("Invalid rate limit values", used=used, limit=limit)
                return

            if used > limit:
                logger.warning("Rate limit exceeded", used=used, limit=limit)
                used = limit  # Cap at limit

            self.metrics.rate_limit_usage = used / limit
            self.metrics.rate_limit_remaining = limit - used

            # Log warning if approaching limit
            if self.metrics.rate_limit_usage > 0.8:
                logger.warning("High rate limit usage",
                             usage_percent=self.metrics.rate_limit_usage * 100,
                             remaining=self.metrics.rate_limit_remaining)
        except Exception as e:
            logger.error("Failed to update rate limits", error=str(e))

    async def update_tilt_score(self, score: float, indicators: dict[str, float]) -> None:
        """Update tilt score and indicators with validation."""
        try:
            # Validate score is in range 0-100
            if score < 0 or score > 100:
                logger.warning("Invalid tilt score", score=score)
                score = max(0, min(100, score))  # Clamp to range

            # Validate indicators
            validated_indicators = {}
            for key, value in (indicators or {}).items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    # Sanitize key and bound value
                    safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', key)[:50]
                    safe_value = max(0, min(100, float(value)))
                    validated_indicators[safe_key] = safe_value

            self.metrics.tilt_score = score
            self.metrics.tilt_indicators = validated_indicators

            # Log warning if tilt score is high
            if score > 70:
                logger.warning("High tilt score detected",
                             score=score,
                             indicators=validated_indicators)
        except Exception as e:
            logger.error("Failed to update tilt score", error=str(e))

    async def record_trade(self, trade: dict[str, Any]) -> None:
        """Record trade for analysis with validation."""
        try:
            if not trade:
                logger.warning("Attempted to record null trade")
                return

            # Validate and sanitize trade data
            symbol = str(trade.get("symbol", ""))[:20]
            if symbol and not SYMBOL_PATTERN.match(symbol.upper()):
                logger.warning("Invalid trade symbol", symbol=symbol)
                return

            side = str(trade.get("side", "")).upper()
            if side and not SIDE_PATTERN.match(side):
                logger.warning("Invalid trade side", side=side)
                return

            # Validate numeric values
            try:
                quantity = float(trade.get("quantity", 0))
                price = float(trade.get("price", 0))
                pnl = float(trade.get("pnl", 0))
                ret = float(trade.get("return", 0))

                # Bounds checking
                if quantity < 0 or price < 0:
                    logger.warning("Negative trade values", quantity=quantity, price=price)
                    return

                if abs(pnl) > MAX_METRIC_VALUE or abs(ret) > 1000:
                    logger.warning("Excessive trade values", pnl=pnl, ret=ret)
                    return

            except (ValueError, TypeError) as e:
                logger.warning("Invalid trade numeric values", error=str(e))
                return

            trade_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "return": ret
            }

            self._trades_history.append(trade_record)

            # Keep only last 1000 trades
            if len(self._trades_history) > 1000:
                self._trades_history.pop(0)

            # Recalculate derived metrics
            await self._calculate_derived_metrics()
        except Exception as e:
            logger.error("Failed to record trade", error=str(e))

    async def handle_event(self, event: Event) -> None:
        """Handle events from the event bus."""
        try:
            if event.type == EventType.ORDER_PLACED:
                await self.record_order(event.data.get("order"))
            elif event.type == EventType.ORDER_FILLED:
                await self.record_order(event.data.get("order"))
                await self.record_trade(event.data)
            elif event.type == EventType.POSITION_OPENED:
                self.metrics.positions_opened += 1
            elif event.type == EventType.POSITION_CLOSED:
                self.metrics.positions_closed += 1
                await self.record_trade(event.data)
            elif event.type == EventType.CONNECTION_LOST:
                await self.update_connection_status(False)
            elif event.type == EventType.CONNECTION_RESTORED:
                await self.update_connection_status(True)
            elif event.type == EventType.TILT_WARNING:
                await self.update_tilt_score(
                    event.data.get("score", 0),
                    event.data.get("indicators", {})
                )
        except Exception as e:
            logger.error("Error handling event for metrics",
                        event_type=event.type,
                        error=str(e))

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current metrics summary."""
        return {
            "orders": {
                "placed": self.metrics.orders_placed,
                "filled": self.metrics.orders_filled,
                "cancelled": self.metrics.orders_cancelled,
                "failed": self.metrics.orders_failed
            },
            "positions": {
                "current": self.metrics.current_positions,
                "opened": self.metrics.positions_opened,
                "closed": self.metrics.positions_closed
            },
            "pnl": {
                "realized": float(self.metrics.realized_pnl),
                "unrealized": float(self.metrics.unrealized_pnl),
                "total": float(self.metrics.realized_pnl + self.metrics.unrealized_pnl)
            },
            "performance": {
                "win_rate": self.metrics.win_rate,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "current_drawdown": self.metrics.current_drawdown
            },
            "connection": {
                "websocket": self.metrics.websocket_connected,
                "latency_ms": self.metrics.websocket_latency_ms
            },
            "rate_limits": {
                "usage": self.metrics.rate_limit_usage,
                "remaining": self.metrics.rate_limit_remaining
            },
            "tilt": {
                "score": self.metrics.tilt_score,
                "indicators": self.metrics.tilt_indicators
            },
            "last_update": self.metrics.last_update.isoformat()
        }
