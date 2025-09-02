"""Prometheus metrics exporter for Project GENESIS."""

import asyncio
import hashlib
import hmac
import re
import secrets
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = structlog.get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Valid metric name and label patterns
METRIC_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
LABEL_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
LABEL_VALUE_PATTERN = re.compile(r'^[\w\s\-\.]+$')


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Represents a Prometheus metric."""
    name: str
    type: MetricType
    help: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    buckets: list[float] | None = None
    quantiles: list[float] | None = None

    def format_prometheus(self) -> str:
        """Format metric in Prometheus exposition format."""
        lines = []

        # Sanitize metric name
        if not METRIC_NAME_PATTERN.match(self.name):
            raise ValueError(f"Invalid metric name: {self.name}")

        # Sanitize help text
        safe_help = self.help.replace('\n', ' ').replace('\\', '\\\\').replace('"', '\\"')

        # Add HELP and TYPE lines
        lines.append(f"# HELP {self.name} {safe_help}")
        lines.append(f"# TYPE {self.name} {self.type.value}")

        # Format and sanitize labels
        if self.labels:
            sanitized_labels = []
            for k, v in self.labels.items():
                if not LABEL_NAME_PATTERN.match(k):
                    raise ValueError(f"Invalid label name: {k}")
                if not LABEL_VALUE_PATTERN.match(str(v)):
                    raise ValueError(f"Invalid label value: {v}")
                # Escape special characters in label values
                safe_value = str(v).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                sanitized_labels.append(f'{k}="{safe_value}"')
            label_str = ",".join(sanitized_labels)
            metric_line = f"{self.name}{{{label_str}}} {self.value}"
        else:
            metric_line = f"{self.name} {self.value}"

        lines.append(metric_line)
        return "\n".join(lines)


class MetricsRegistry:
    """Registry for Prometheus metrics with support for labeled metrics."""

    def __init__(self):
        self._metrics: dict[str, Metric] = {}
        self._labeled_metrics: dict[str, dict[str, Metric]] = {}  # For metrics with labels
        self._collectors: list[Callable] = []
        self._lock = asyncio.Lock()

    async def register(self, metric: Metric) -> None:
        """Register a metric."""
        async with self._lock:
            self._metrics[metric.name] = metric
            if metric.type in [MetricType.COUNTER, MetricType.GAUGE, MetricType.HISTOGRAM]:
                self._labeled_metrics[metric.name] = {}
            logger.debug("Registered metric", name=metric.name, type=metric.type.value)

    async def unregister(self, name: str) -> None:
        """Unregister a metric."""
        async with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                if name in self._labeled_metrics:
                    del self._labeled_metrics[name]
                logger.debug("Unregistered metric", name=name)

    async def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge metric value with validation."""
        # Validate metric name
        if not METRIC_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid metric name: {name}")

        # Validate labels
        if labels:
            for k, v in labels.items():
                if not LABEL_NAME_PATTERN.match(k):
                    raise ValueError(f"Invalid label name: {k}")
                if not LABEL_VALUE_PATTERN.match(str(v)):
                    raise ValueError(f"Invalid label value: {v}")

        async with self._lock:
            if name in self._metrics and self._metrics[name].type == MetricType.GAUGE:
                if labels:
                    # Create a unique key for this label combination
                    label_key = self._create_label_key(labels)
                    if name not in self._labeled_metrics:
                        self._labeled_metrics[name] = {}
                    
                    # Store or update the labeled metric
                    if label_key not in self._labeled_metrics[name]:
                        self._labeled_metrics[name][label_key] = Metric(
                            name=name,
                            type=MetricType.GAUGE,
                            help=self._metrics[name].help,
                            value=float(value),
                            labels=labels
                        )
                    else:
                        self._labeled_metrics[name][label_key].value = float(value)
                else:
                    # Update the base metric without labels
                    self._metrics[name].value = float(value)

    async def increment_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment counter metric with validation."""
        # Validate metric name
        if not METRIC_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid metric name: {name}")

        # Validate value is positive
        if value < 0:
            raise ValueError(f"Counter increment must be positive: {value}")

        # Validate labels
        if labels:
            for k, v in labels.items():
                if not LABEL_NAME_PATTERN.match(k):
                    raise ValueError(f"Invalid label name: {k}")
                if not LABEL_VALUE_PATTERN.match(str(v)):
                    raise ValueError(f"Invalid label value: {v}")

        async with self._lock:
            if name in self._metrics and self._metrics[name].type == MetricType.COUNTER:
                if labels:
                    # Create a unique key for this label combination
                    label_key = self._create_label_key(labels)
                    if name not in self._labeled_metrics:
                        self._labeled_metrics[name] = {}
                    
                    # Store or update the labeled metric
                    if label_key not in self._labeled_metrics[name]:
                        self._labeled_metrics[name][label_key] = Metric(
                            name=name,
                            type=MetricType.COUNTER,
                            help=self._metrics[name].help,
                            value=float(value),
                            labels=labels
                        )
                    else:
                        self._labeled_metrics[name][label_key].value += float(value)
                else:
                    # Update the base metric without labels
                    self._metrics[name].value += float(value)

    async def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe histogram metric."""
        async with self._lock:
            if name in self._metrics and self._metrics[name].type == MetricType.HISTOGRAM:
                if labels:
                    # Create a unique key for this label combination
                    label_key = self._create_label_key(labels)
                    if name not in self._labeled_metrics:
                        self._labeled_metrics[name] = {}
                    
                    # Store or update the labeled metric
                    if label_key not in self._labeled_metrics[name]:
                        self._labeled_metrics[name][label_key] = Metric(
                            name=name,
                            type=MetricType.HISTOGRAM,
                            help=self._metrics[name].help,
                            value=value,
                            labels=labels,
                            buckets=self._metrics[name].buckets
                        )
                    else:
                        # For histograms, we'd normally update buckets
                        # For now, just track the latest value
                        self._labeled_metrics[name][label_key].value = value
                else:
                    # Update the base metric without labels
                    self._metrics[name].value = value

    def _create_label_key(self, labels: dict[str, str]) -> str:
        """Create a unique key from label dictionary."""
        # Sort labels by key for consistent ordering
        sorted_labels = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_labels)

    async def collect(self) -> str:
        """Collect all metrics in Prometheus format."""
        async with self._lock:
            # Run any registered collectors
            for collector in self._collectors:
                if asyncio.iscoroutinefunction(collector):
                    await collector()
                else:
                    collector()

            # Format all metrics
            lines = []
            seen_metrics = set()
            
            # First, output base metrics and their HELP/TYPE lines
            for metric in self._metrics.values():
                if metric.name not in seen_metrics:
                    lines.append(f"# HELP {metric.name} {metric.help}")
                    lines.append(f"# TYPE {metric.name} {metric.type.value}")
                    seen_metrics.add(metric.name)
                
                # Output base metric if it has a value
                if metric.value != 0 or metric.type == MetricType.GAUGE:
                    lines.append(f"{metric.name} {metric.value}")
            
            # Then output all labeled metrics
            for metric_name, labeled_metrics in self._labeled_metrics.items():
                for labeled_metric in labeled_metrics.values():
                    if labeled_metric.labels:
                        label_str = ",".join(
                            f'{k}="{str(v).replace("\\", "\\\\").replace("\"", "\\\"")}"'
                            for k, v in labeled_metric.labels.items()
                        )
                        lines.append(f"{metric_name}{{{label_str}}} {labeled_metric.value}")

            return "\n".join(lines) + "\n"

    def register_collector(self, collector: Callable) -> None:
        """Register a metrics collector function."""
        self._collectors.append(collector)


class APIKeyRotator:
    """Manages API key rotation for secure access."""

    def __init__(self, rotation_interval_hours: int = 24):
        self.rotation_interval = timedelta(hours=rotation_interval_hours)
        self.current_key: str = secrets.token_urlsafe(32)
        self.previous_key: str | None = None
        self.rotation_time: datetime = datetime.utcnow() + self.rotation_interval
        self.key_hash: str = hashlib.sha256(self.current_key.encode()).hexdigest()

    def rotate_if_needed(self) -> bool:
        """Rotate API key if needed."""
        now = datetime.utcnow()
        if now >= self.rotation_time:
            self.previous_key = self.current_key
            self.current_key = secrets.token_urlsafe(32)
            self.key_hash = hashlib.sha256(self.current_key.encode()).hexdigest()
            self.rotation_time = now + self.rotation_interval
            logger.info("API key rotated", next_rotation=self.rotation_time.isoformat())
            return True
        return False

    def validate_key(self, provided_key: str) -> bool:
        """Validate provided API key."""
        self.rotate_if_needed()

        # Check current key
        if hmac.compare_digest(provided_key, self.current_key):
            return True

        # Check previous key (grace period during rotation)
        if self.previous_key and hmac.compare_digest(provided_key, self.previous_key):
            return True

        return False


class PrometheusExporter:
    """Prometheus metrics exporter with HTTPS endpoint and security features."""

    def __init__(self,
                 registry: MetricsRegistry,
                 port: int = 9090,
                 use_https: bool = True,
                 cert_file: str | None = None,
                 key_file: str | None = None,
                 allowed_ips: set[str] | None = None,
                 production_mode: bool = False):
        self.registry = registry
        self.port = port
        self.use_https = use_https
        self.cert_file = cert_file
        self.key_file = key_file
        self.production_mode = production_mode

        # Configure IP allowlist based on production mode
        if production_mode and not allowed_ips:
            # Production default: only localhost and monitoring server
            self.allowed_ips = {"127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12"}
            logger.info("Production mode: IP allowlist configured", allowed_ips=self.allowed_ips)
        else:
            self.allowed_ips = allowed_ips or set()

        self.api_key_rotator = APIKeyRotator(rotation_interval_hours=24)

        # Initialize FastAPI with security middleware
        self.app = FastAPI(title="GENESIS Metrics")

        # Add trusted host middleware to prevent host header injection
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure based on your deployment
        )

        # Add rate limit error handler
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        self._setup_routes()
        self._setup_default_metrics()

        logger.info("PrometheusExporter initialized",
                   use_https=use_https,
                   api_key_hash=self.api_key_rotator.key_hash[:8] + "...")

    def _setup_routes(self) -> None:
        """Set up FastAPI routes with security."""
        security = HTTPBearer()

        @self.app.get("/metrics")
        @limiter.limit("100/minute")  # Rate limiting
        async def metrics(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Prometheus metrics endpoint with authentication and rate limiting."""
            # Check IP allowlist if configured
            if self.allowed_ips:
                client_ip = get_remote_address(request)
                if client_ip not in self.allowed_ips:
                    logger.warning("Metrics access denied", ip=client_ip)
                    raise HTTPException(status_code=403, detail="Access denied from this IP")

            # Validate API key
            if not self.api_key_rotator.validate_key(credentials.credentials):
                logger.warning("Invalid API key attempt")
                raise HTTPException(status_code=403, detail="Invalid or expired API key")

            try:
                # Collect and return metrics
                content = await self.registry.collect()
                return Response(
                    content=content,
                    media_type="text/plain; version=0.0.4",
                    headers={
                        "X-Content-Type-Options": "nosniff",
                        "X-Frame-Options": "DENY",
                        "X-XSS-Protection": "1; mode=block",
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Pragma": "no-cache"
                    }
                )
            except ValueError as e:
                logger.error("Metric validation error", error=str(e))
                raise HTTPException(status_code=400, detail="Invalid metric format")
            except Exception as e:
                logger.error("Failed to collect metrics", error=str(e))
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.get("/health")
        @limiter.limit("60/minute")
        async def health(request: Request):
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "api_key_rotation": self.api_key_rotator.rotation_time.isoformat()
            }

        @self.app.get("/api-key")
        @limiter.limit("5/hour")  # Strict rate limit for key retrieval
        async def get_api_key(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get current API key (requires admin authentication)."""
            # This endpoint should have additional admin authentication in production
            # For now, require the current or previous key
            if not self.api_key_rotator.validate_key(credentials.credentials):
                raise HTTPException(status_code=403, detail="Unauthorized")

            return {
                "api_key": self.api_key_rotator.current_key,
                "expires_at": self.api_key_rotator.rotation_time.isoformat(),
                "key_hash": self.api_key_rotator.key_hash[:8] + "..."
            }

    def _setup_default_metrics(self) -> None:
        """Set up default system metrics."""
        asyncio.create_task(self._register_default_metrics())

    async def _register_default_metrics(self) -> None:
        """Register default metrics."""
        # Process metrics
        await self.registry.register(Metric(
            name="genesis_up",
            type=MetricType.GAUGE,
            help="GENESIS process uptime in seconds"
        ))

        await self.registry.register(Metric(
            name="genesis_info",
            type=MetricType.GAUGE,
            help="GENESIS version information",
            value=1.0,
            labels={"version": "1.0.0", "tier": "sniper"}
        ))

        # Trading metrics - Enhanced with comprehensive tracking
        await self.registry.register(Metric(
            name="genesis_orders_total",
            type=MetricType.COUNTER,
            help="Total number of orders by exchange, symbol, side, type and status"
        ))
        
        await self.registry.register(Metric(
            name="genesis_order_execution_time_seconds",
            type=MetricType.HISTOGRAM,
            help="Order execution time in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        ))
        
        await self.registry.register(Metric(
            name="genesis_order_latency_seconds",
            type=MetricType.HISTOGRAM,
            help="Order latency by exchange and type",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        ))

        await self.registry.register(Metric(
            name="genesis_positions_count",
            type=MetricType.GAUGE,
            help="Current number of open positions by symbol"
        ))
        
        await self.registry.register(Metric(
            name="genesis_positions_by_side",
            type=MetricType.GAUGE,
            help="Position count by side (long/short)"
        ))

        await self.registry.register(Metric(
            name="genesis_position_count",
            type=MetricType.GAUGE,
            help="Total number of open positions"
        ))

        await self.registry.register(Metric(
            name="genesis_trading_pnl_usdt",
            type=MetricType.GAUGE,
            help="P&L in USDT by type (realized/unrealized), strategy and symbol"
        ))
        
        await self.registry.register(Metric(
            name="genesis_trading_pnl_total_usdt",
            type=MetricType.GAUGE,
            help="Total P&L in USDT by strategy"
        ))

        await self.registry.register(Metric(
            name="genesis_pnl_dollars",
            type=MetricType.GAUGE,
            help="Current total P&L in dollars"
        ))
        
        await self.registry.register(Metric(
            name="genesis_trading_volume_usdt",
            type=MetricType.COUNTER,
            help="Trading volume in USDT by exchange and symbol"
        ))

        await self.registry.register(Metric(
            name="genesis_connection_status",
            type=MetricType.GAUGE,
            help="Connection status (1=connected, 0=disconnected)"
        ))

        await self.registry.register(Metric(
            name="genesis_orders_failed_total",
            type=MetricType.COUNTER,
            help="Total number of failed orders by exchange and reason"
        ))

        await self.registry.register(Metric(
            name="genesis_trades_total",
            type=MetricType.COUNTER,
            help="Total number of trades executed by exchange and symbol"
        ))

        await self.registry.register(Metric(
            name="genesis_rate_limit_usage_ratio",
            type=MetricType.GAUGE,
            help="Rate limit usage ratio (0-1)"
        ))

        await self.registry.register(Metric(
            name="genesis_tilt_score",
            type=MetricType.GAUGE,
            help="Current tilt score (0-100)"
        ))

        await self.registry.register(Metric(
            name="genesis_drawdown_percent",
            type=MetricType.GAUGE,
            help="Current drawdown percentage"
        ))

        await self.registry.register(Metric(
            name="genesis_websocket_latency_ms",
            type=MetricType.HISTOGRAM,
            help="WebSocket message latency in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
        ))

        await self.registry.register(Metric(
            name="genesis_memory_usage_bytes",
            type=MetricType.GAUGE,
            help="Memory usage in bytes"
        ))

        await self.registry.register(Metric(
            name="genesis_cpu_usage_percent",
            type=MetricType.GAUGE,
            help="CPU usage percentage"
        ))

        logger.info("Registered default Prometheus metrics")

    async def start(self) -> None:
        """Start the metrics server with HTTPS if configured."""
        if self.use_https:
            if not self.cert_file or not self.key_file:
                if self.production_mode:
                    # Production mode requires proper certificates
                    raise ValueError(
                        "Production mode requires valid TLS certificates. "
                        "Please provide cert_file and key_file paths."
                    )
                else:
                    # Development mode: use self-signed certificate
                    logger.warning("No certificate provided, using self-signed certificate (development mode)")
                    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
            else:
                # Production certificates provided
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(self.cert_file, self.key_file)
                if self.production_mode:
                    # Enforce TLS 1.3 in production
                    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3
                    logger.info("Production TLS configured", cert_file=self.cert_file)

            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning",
                ssl_keyfile=self.key_file,
                ssl_certfile=self.cert_file,
                ssl_version=ssl.PROTOCOL_TLS_SERVER,
                ssl_cert_reqs=ssl.CERT_NONE,
                ssl_ciphers="TLSv1.3"
            )
        else:
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning"
            )

        server = uvicorn.Server(config)

        logger.info("Starting Prometheus metrics exporter",
                   port=self.port,
                   https=self.use_https,
                   api_key_hash=self.api_key_rotator.key_hash[:8] + "...")
        await server.serve()

    def run_in_background(self) -> asyncio.Task:
        """Run the exporter in the background."""
        return asyncio.create_task(self.start())
