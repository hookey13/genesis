"""Alert manager for monitoring and notifications with security enhancements."""

import asyncio
import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import aiohttp
import structlog
from aiohttp import TCPConnector

logger = structlog.get_logger(__name__)

# Input validation patterns
ALERT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
LABEL_KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
LABEL_VALUE_PATTERN = re.compile(r'^[\w\s\-\.]{1,200}$')

# Circuit breaker for external services
class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "closed"  # closed, open, half-open

    def call_succeeded(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = "closed"

    def call_failed(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning("Circuit breaker opened", failures=self.failure_count)

    def can_attempt_call(self) -> bool:
        """Check if call can be attempted."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure > self.recovery_timeout:
                    self.state = "half-open"
                    return True
            return False

        return True  # half-open state


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert state."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Defines an alert rule with validation."""
    name: str
    description: str
    condition: Callable[[], bool]  # Function that returns True when alert should fire
    severity: AlertSeverity
    threshold: float
    duration: timedelta = timedelta(seconds=0)  # How long condition must be true
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    cooldown: timedelta = timedelta(minutes=5)  # Minimum time between alerts

    def __post_init__(self):
        """Validate alert rule fields."""
        if not ALERT_NAME_PATTERN.match(self.name):
            raise ValueError(f"Invalid alert name: {self.name}")

        if not self.description or len(self.description) > 500:
            raise ValueError(f"Invalid description length: {len(self.description)}")

        if self.threshold < -1e6 or self.threshold > 1e6:
            raise ValueError(f"Threshold out of bounds: {self.threshold}")

        # Validate labels
        for key, value in self.labels.items():
            if not LABEL_KEY_PATTERN.match(key):
                raise ValueError(f"Invalid label key: {key}")
            if not LABEL_VALUE_PATTERN.match(str(value)):
                raise ValueError(f"Invalid label value: {value}")

    def __hash__(self):
        return hash(self.name)


@dataclass
class Alert:
    """Represents a fired alert."""
    rule: AlertRule
    state: AlertState
    fired_at: datetime
    resolved_at: datetime | None = None
    value: float | None = None
    message: str | None = None
    notification_sent: bool = False

    def format_message(self) -> str:
        """Format alert message for notification with sanitization."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        emoji = severity_emoji.get(self.rule.severity, "📢")
        state_str = "FIRING" if self.state == AlertState.FIRING else "RESOLVED"

        # Sanitize string fields to prevent injection
        safe_name = html.escape(self.rule.name)
        safe_description = html.escape(self.rule.description)

        message = f"{emoji} **[{state_str}]** {safe_name}\n"
        message += f"**Severity:** {self.rule.severity.value}\n"
        message += f"**Description:** {safe_description}\n"

        if self.value is not None:
            message += f"**Value:** {self.value:.2f} (threshold: {self.rule.threshold:.2f})\n"

        if self.message:
            safe_message = html.escape(self.message[:500])  # Limit length
            message += f"**Details:** {safe_message}\n"

        message += f"**Time:** {self.fired_at.isoformat()}\n"

        if self.rule.labels:
            safe_labels = [f'{html.escape(k)}={html.escape(str(v))}'
                          for k, v in self.rule.labels.items()]
            message += f"**Labels:** {', '.join(safe_labels)}\n"

        return message


class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Returns True if successful."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel (placeholder)."""

    def __init__(self, smtp_config: dict[str, Any]):
        self.smtp_config = smtp_config

    async def send(self, alert: Alert) -> bool:
        """Send email notification."""
        # In production, implement SMTP sending
        logger.info("Would send email notification",
                   alert=alert.rule.name,
                   severity=alert.rule.severity.value)
        return True


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel with connection pooling and circuit breaker."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.circuit_breaker = CircuitBreaker()
        self.connector = None
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if not self.session or self.session.closed:
            self.connector = TCPConnector(
                limit=10,  # Connection pool limit
                limit_per_host=5,
                ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    async def send(self, alert: Alert) -> bool:
        """Send Slack notification with circuit breaker."""
        if not self.circuit_breaker.can_attempt_call():
            logger.warning("Circuit breaker open for Slack notifications")
            return False

        try:
            # Validate webhook URL
            if not self.webhook_url.startswith(('https://hooks.slack.com/', 'https://discord.com/api/webhooks/')):
                logger.error("Invalid webhook URL")
                return False

            payload = {
                "text": alert.format_message(),
                "attachments": [{
                    "color": self._get_color(alert.rule.severity),
                    "fields": [
                        {"title": "Alert", "value": html.escape(alert.rule.name), "short": True},
                        {"title": "Severity", "value": alert.rule.severity.value, "short": True},
                        {"title": "State", "value": alert.state.value, "short": True},
                        {"title": "Time", "value": alert.fired_at.isoformat(), "short": True}
                    ]
                }]
            }

            session = await self._get_session()
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status == 200:
                    self.circuit_breaker.call_succeeded()
                    return True
                else:
                    self.circuit_breaker.call_failed()
                    logger.warning("Slack notification failed", status=response.status)
                    return False
        except TimeoutError:
            self.circuit_breaker.call_failed()
            logger.error("Slack notification timeout")
            return False
        except Exception as e:
            self.circuit_breaker.call_failed()
            logger.error("Failed to send Slack notification", error=str(e))
            return False

    def _get_color(self, severity: AlertSeverity) -> str:
        """Get color for severity level."""
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000"
        }
        return colors.get(severity, "#808080")


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel with connection pooling and circuit breaker."""

    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
        self.circuit_breaker = CircuitBreaker()
        self.connector = None
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if not self.session or self.session.closed:
            self.connector = TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification with circuit breaker."""
        if not self.circuit_breaker.can_attempt_call():
            logger.warning("Circuit breaker open for PagerDuty notifications")
            return False

        try:
            # Only send critical alerts to PagerDuty
            if alert.rule.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                return True

            # Validate integration key format
            if not re.match(r'^[a-f0-9]{32}$', self.integration_key):
                logger.error("Invalid PagerDuty integration key format")
                return False

            event_action = "trigger" if alert.state == AlertState.FIRING else "resolve"

            # Sanitize payload data
            payload = {
                "routing_key": self.integration_key,
                "event_action": event_action,
                "dedup_key": f"genesis-{alert.rule.name}"[:255],  # PagerDuty limit
                "payload": {
                    "summary": html.escape(alert.rule.description)[:1024],
                    "severity": self._get_pd_severity(alert.rule.severity),
                    "source": "genesis-monitoring",
                    "custom_details": {
                        "alert_name": html.escape(alert.rule.name),
                        "value": float(alert.value) if alert.value else 0,
                        "threshold": float(alert.rule.threshold),
                        "message": html.escape(str(alert.message or ""))[:500]
                    }
                }
            }

            session = await self._get_session()
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 202:
                    self.circuit_breaker.call_succeeded()
                    return True
                else:
                    self.circuit_breaker.call_failed()
                    logger.warning("PagerDuty notification failed", status=response.status)
                    return False
        except TimeoutError:
            self.circuit_breaker.call_failed()
            logger.error("PagerDuty notification timeout")
            return False
        except Exception as e:
            self.circuit_breaker.call_failed()
            logger.error("Failed to send PagerDuty notification", error=str(e))
            return False

    def _get_pd_severity(self, severity: AlertSeverity) -> str:
        """Map severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }
        return mapping.get(severity, "error")


class AlertManager:
    """Manages alert rules and notifications."""

    def __init__(self, metrics_collector=None, risk_engine=None, tilt_detector=None):
        self.rules: set[AlertRule] = set()
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.notification_channels: list[NotificationChannel] = []
        self._evaluation_interval = 10  # seconds
        self._evaluation_task: asyncio.Task | None = None
        self._condition_states: dict[str, datetime] = {}  # Track when conditions became true

        # Integration with real components
        self.metrics_collector = metrics_collector
        self.risk_engine = risk_engine
        self.tilt_detector = tilt_detector

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule with validation."""
        try:
            # Validate rule (validation happens in __post_init__)
            if len(self.rules) >= 1000:
                logger.warning("Maximum alert rules reached")
                return

            self.rules.add(rule)
            logger.info("Added alert rule",
                       name=rule.name,
                       severity=rule.severity.value)
        except ValueError as e:
            logger.error("Invalid alert rule", error=str(e))

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        self.rules = {r for r in self.rules if r.name != name}
        if name in self.active_alerts:
            del self.active_alerts[name]
        logger.info("Removed alert rule", name=name)

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.notification_channels.append(channel)
        logger.info("Added notification channel",
                   channel_type=type(channel).__name__)

    async def start(self) -> None:
        """Start alert evaluation."""
        if not self._evaluation_task:
            self._evaluation_task = asyncio.create_task(self._evaluation_loop())
            self._setup_default_rules()
            logger.info("Started alert manager")

    async def stop(self) -> None:
        """Stop alert evaluation and cleanup resources."""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
            self._evaluation_task = None

        # Cleanup notification channels
        for channel in self.notification_channels:
            if hasattr(channel, 'close'):
                await channel.close()

        logger.info("Stopped alert manager")

    def _setup_default_rules(self) -> None:
        """Set up default alert rules with real metric checks."""

        # Drawdown alerts
        self.add_rule(AlertRule(
            name="high_drawdown_1h",
            description="Drawdown exceeds 5% in 1 hour",
            condition=lambda: self._check_drawdown(threshold=5.0, hours=1),
            severity=AlertSeverity.WARNING,
            threshold=5.0,
            duration=timedelta(minutes=1),
            labels={"category": "risk", "timeframe": "1h"}
        ))

        self.add_rule(AlertRule(
            name="high_drawdown_24h",
            description="Drawdown exceeds 10% in 24 hours",
            condition=lambda: self._check_drawdown(threshold=10.0, hours=24),
            severity=AlertSeverity.ERROR,
            threshold=10.0,
            duration=timedelta(minutes=5),
            labels={"category": "risk", "timeframe": "24h"}
        ))

        # Connection alerts
        self.add_rule(AlertRule(
            name="websocket_disconnected",
            description="WebSocket disconnected for more than 60 seconds",
            condition=lambda: self._check_connection_status(),
            severity=AlertSeverity.ERROR,
            threshold=60.0,
            duration=timedelta(seconds=60),
            labels={"category": "connection", "service": "websocket"}
        ))

        # Rate limit alerts
        self.add_rule(AlertRule(
            name="rate_limit_warning",
            description="Rate limit usage exceeds 80%",
            condition=lambda: self._check_rate_limit(threshold=0.8),
            severity=AlertSeverity.WARNING,
            threshold=0.8,
            duration=timedelta(seconds=30),
            labels={"category": "limits", "type": "rate_limit"}
        ))

        # Order failure alerts
        self.add_rule(AlertRule(
            name="order_failures",
            description="More than 3 order failures in 5 minutes",
            condition=lambda: self._check_order_failures(threshold=3, minutes=5),
            severity=AlertSeverity.ERROR,
            threshold=3.0,
            duration=timedelta(seconds=10),
            labels={"category": "execution", "type": "order_failure"}
        ))

        # Tilt detection
        self.add_rule(AlertRule(
            name="tilt_warning",
            description="Tilt score exceeds warning threshold",
            condition=lambda: self._check_tilt_score(threshold=70.0),
            severity=AlertSeverity.WARNING,
            threshold=70.0,
            duration=timedelta(minutes=2),
            labels={"category": "behavior", "type": "tilt"}
        ))

        self.add_rule(AlertRule(
            name="tilt_critical",
            description="Tilt score exceeds critical threshold",
            condition=lambda: self._check_tilt_score(threshold=90.0),
            severity=AlertSeverity.CRITICAL,
            threshold=90.0,
            duration=timedelta(seconds=30),
            labels={"category": "behavior", "type": "tilt"}
        ))

    def _check_drawdown(self, threshold: float, hours: int) -> bool:
        """Check if drawdown exceeds threshold."""
        if not self.metrics_collector:
            return False

        try:
            # Get current drawdown from metrics
            current_drawdown = self.metrics_collector.metrics.current_drawdown
            return current_drawdown > threshold
        except Exception as e:
            logger.error("Failed to check drawdown", error=str(e))
            return False

    def _check_connection_status(self) -> bool:
        """Check if WebSocket is disconnected."""
        if not self.metrics_collector:
            return False

        try:
            # Check if WebSocket is connected
            return not self.metrics_collector.metrics.websocket_connected
        except Exception as e:
            logger.error("Failed to check connection status", error=str(e))
            return False

    def _check_rate_limit(self, threshold: float) -> bool:
        """Check if rate limit usage exceeds threshold."""
        if not self.metrics_collector:
            return False

        try:
            # Check rate limit usage
            usage = self.metrics_collector.metrics.rate_limit_usage
            return usage > threshold
        except Exception as e:
            logger.error("Failed to check rate limit", error=str(e))
            return False

    def _check_order_failures(self, threshold: int, minutes: int) -> bool:
        """Check if order failures exceed threshold."""
        if not self.metrics_collector:
            return False

        try:
            # Check recent order failures
            # In a real implementation, we'd check failures within the time window
            failures = self.metrics_collector.metrics.orders_failed
            return failures > threshold
        except Exception as e:
            logger.error("Failed to check order failures", error=str(e))
            return False

    def _check_tilt_score(self, threshold: float) -> bool:
        """Check if tilt score exceeds threshold."""
        if self.tilt_detector:
            try:
                # Get tilt score from detector
                tilt_score = self.tilt_detector.get_current_score()
                return tilt_score > threshold
            except Exception as e:
                logger.error("Failed to check tilt score from detector", error=str(e))

        # Fallback to metrics collector
        if self.metrics_collector:
            try:
                tilt_score = self.metrics_collector.metrics.tilt_score
                return tilt_score > threshold
            except Exception as e:
                logger.error("Failed to check tilt score from metrics", error=str(e))

        return False

    async def _evaluation_loop(self) -> None:
        """Main alert evaluation loop."""
        while True:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(self._evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert evaluation", error=str(e))
                await asyncio.sleep(self._evaluation_interval)

    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules."""
        current_time = datetime.now()

        for rule in self.rules:
            try:
                # Check if condition is met
                condition_met = rule.condition()

                if condition_met:
                    # Track when condition first became true
                    if rule.name not in self._condition_states:
                        self._condition_states[rule.name] = current_time

                    # Check if duration requirement is met
                    condition_duration = current_time - self._condition_states[rule.name]

                    if condition_duration >= rule.duration:
                        # Fire or update alert
                        await self._fire_alert(rule)
                else:
                    # Condition not met, clear state
                    if rule.name in self._condition_states:
                        del self._condition_states[rule.name]

                    # Resolve alert if active
                    if rule.name in self.active_alerts:
                        await self._resolve_alert(rule.name)

            except Exception as e:
                logger.error("Error evaluating alert rule",
                           rule=rule.name,
                           error=str(e))

    async def _fire_alert(self, rule: AlertRule) -> None:
        """Fire an alert."""
        if rule.name in self.active_alerts:
            # Alert already active
            return

        # Check cooldown
        for alert in reversed(self.alert_history):
            if alert.rule.name == rule.name:
                time_since_last = datetime.now() - alert.fired_at
                if time_since_last < rule.cooldown:
                    return
                break

        # Create alert
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )

        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)

        # Keep history limited
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        logger.warning("Alert fired",
                      name=rule.name,
                      severity=rule.severity.value,
                      description=rule.description)

        # Send notifications
        await self._send_notifications(alert)

    async def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an alert."""
        if rule_name not in self.active_alerts:
            return

        alert = self.active_alerts[rule_name]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now()

        del self.active_alerts[rule_name]

        logger.info("Alert resolved",
                   name=rule_name,
                   duration=(alert.resolved_at - alert.fired_at).total_seconds())

        # Send resolution notification
        await self._send_notifications(alert)

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        if alert.notification_sent and alert.state == AlertState.FIRING:
            return

        for channel in self.notification_channels:
            try:
                success = await channel.send(alert)
                if success:
                    alert.notification_sent = True
                    logger.debug("Sent notification",
                               channel=type(channel).__name__,
                               alert=alert.rule.name)
            except Exception as e:
                logger.error("Failed to send notification",
                           channel=type(channel).__name__,
                           alert=alert.rule.name,
                           error=str(e))

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    async def test_alert(self, rule_name: str) -> None:
        """Manually trigger an alert for testing."""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        if rule:
            await self._fire_alert(rule)
            logger.info("Test alert fired", name=rule_name)
        else:
            logger.warning("Alert rule not found", name=rule_name)
