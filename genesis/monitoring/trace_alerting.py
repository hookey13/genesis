"""Trace-based alerting for Genesis distributed tracing."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(Enum):
    """Alert state."""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    description: str
    condition: Callable[[dict[str, Any]], bool]
    severity: AlertSeverity
    threshold_value: float
    evaluation_period_seconds: int = 60
    cooldown_seconds: int = 300
    annotations: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    details: dict[str, Any]
    fired_at: datetime
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    suppressed_until: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


class TraceAlertManager:
    """Manage alerts based on trace analysis."""

    def __init__(
        self,
        alert_retention_hours: int = 24,
        max_alerts_per_rule: int = 10,
        suppression_duration_minutes: int = 30,
    ):
        """Initialize trace alert manager.
        
        Args:
            alert_retention_hours: How long to retain resolved alerts
            max_alerts_per_rule: Maximum alerts per rule before suppression
            suppression_duration_minutes: Duration to suppress alerts after max reached
        """
        self.alert_retention_hours = alert_retention_hours
        self.max_alerts_per_rule = max_alerts_per_rule
        self.suppression_duration_minutes = suppression_duration_minutes

        # Alert storage
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.rule_fire_counts: dict[str, int] = {}
        self.rule_last_fired: dict[str, datetime] = {}

        # Alert callbacks
        self.alert_callbacks: list[Callable] = []

        # Initialize default alert rules
        self._initialize_default_rules()

        logger.info(
            "Trace alert manager initialized",
            retention_hours=alert_retention_hours,
            max_alerts_per_rule=max_alerts_per_rule,
        )

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules for trace-based monitoring."""

        # High latency alert
        self.add_rule(
            AlertRule(
                name="high_latency",
                description="Operation latency exceeds critical threshold",
                condition=lambda metrics: metrics.get("avg_duration_ms", 0) > 100,
                severity=AlertSeverity.CRITICAL,
                threshold_value=100.0,
                evaluation_period_seconds=60,
                annotations={
                    "summary": "High latency detected in {{ operation_name }}",
                    "description": "Average latency {{ current_value }}ms exceeds threshold {{ threshold_value }}ms",
                },
                labels={"category": "performance"},
            )
        )

        # High error rate alert
        self.add_rule(
            AlertRule(
                name="high_error_rate",
                description="Operation error rate exceeds threshold",
                condition=lambda metrics: metrics.get("error_rate", 0) > 0.05,
                severity=AlertSeverity.HIGH,
                threshold_value=5.0,
                evaluation_period_seconds=120,
                annotations={
                    "summary": "High error rate in {{ operation_name }}",
                    "description": "Error rate {{ current_value }}% exceeds threshold {{ threshold_value }}%",
                },
                labels={"category": "reliability"},
            )
        )

        # Slow operation rate alert
        self.add_rule(
            AlertRule(
                name="high_slow_rate",
                description="Too many operations are slow",
                condition=lambda metrics: metrics.get("slow_rate", 0) > 0.2,
                severity=AlertSeverity.MEDIUM,
                threshold_value=20.0,
                evaluation_period_seconds=180,
                annotations={
                    "summary": "High rate of slow operations in {{ operation_name }}",
                    "description": "{{ current_value }}% of operations are slow (threshold: {{ threshold_value }}%)",
                },
                labels={"category": "performance"},
            )
        )

        # Service dependency failure alert
        self.add_rule(
            AlertRule(
                name="service_dependency_failure",
                description="Service dependency experiencing failures",
                condition=lambda dep: dep.get("error_rate", 0) > 0.1,
                severity=AlertSeverity.HIGH,
                threshold_value=10.0,
                evaluation_period_seconds=60,
                annotations={
                    "summary": "Service dependency {{ target_service }} failing",
                    "description": "Error rate {{ current_value }}% for calls to {{ target_service }}",
                },
                labels={"category": "dependency"},
            )
        )

        # Trace collection failure alert
        self.add_rule(
            AlertRule(
                name="trace_collection_failure",
                description="Trace collection is not working properly",
                condition=lambda metrics: metrics.get("traces_per_minute", 1) < 1,
                severity=AlertSeverity.HIGH,
                threshold_value=1.0,
                evaluation_period_seconds=300,
                annotations={
                    "summary": "Trace collection appears to be failing",
                    "description": "Receiving {{ current_value }} traces per minute (expected > {{ threshold_value }})",
                },
                labels={"category": "observability"},
            )
        )

        # Performance degradation alert
        self.add_rule(
            AlertRule(
                name="performance_degradation",
                description="Overall system performance is degrading",
                condition=lambda metrics: (
                    metrics.get("p99_latency_ms", 0) >
                    metrics.get("baseline_p99_latency_ms", 50) * 1.5
                ),
                severity=AlertSeverity.MEDIUM,
                threshold_value=150.0,  # 150% of baseline
                evaluation_period_seconds=300,
                annotations={
                    "summary": "System performance degradation detected",
                    "description": "P99 latency is {{ current_value }}% of baseline",
                },
                labels={"category": "performance"},
            )
        )

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.name] = rule
        self.rule_fire_counts[rule.name] = 0
        logger.info(
            "Alert rule added",
            rule_name=rule.name,
            severity=rule.severity.value,
        )

    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            if rule_name in self.rule_fire_counts:
                del self.rule_fire_counts[rule_name]
            if rule_name in self.rule_last_fired:
                del self.rule_last_fired[rule_name]
            logger.info("Alert rule removed", rule_name=rule_name)

    def evaluate_metrics(self, metrics: dict[str, Any]) -> list[Alert]:
        """Evaluate metrics against alert rules.
        
        Args:
            metrics: Metrics dictionary to evaluate
        
        Returns:
            List of new alerts that were triggered
        """
        new_alerts = []
        current_time = datetime.now(UTC)

        for rule_name, rule in self.alert_rules.items():
            # Check if rule is in cooldown
            last_fired = self.rule_last_fired.get(rule_name)
            if last_fired:
                cooldown_end = last_fired + timedelta(seconds=rule.cooldown_seconds)
                if current_time < cooldown_end:
                    continue

            # Check if rule is suppressed
            if self._is_rule_suppressed(rule_name):
                continue

            # Evaluate condition
            try:
                if rule.condition(metrics):
                    alert = self._create_alert(rule, metrics)

                    # Check if similar alert already exists
                    existing_alert = self._find_existing_alert(rule_name, metrics)
                    if not existing_alert:
                        # New alert
                        self.active_alerts[alert.id] = alert
                        new_alerts.append(alert)
                        self.rule_fire_counts[rule_name] += 1
                        self.rule_last_fired[rule_name] = current_time

                        # Check if we need to suppress future alerts
                        if self.rule_fire_counts[rule_name] >= self.max_alerts_per_rule:
                            self._suppress_rule(rule_name)

                        # Trigger callbacks
                        self._trigger_callbacks(alert)

                        logger.warning(
                            "Alert triggered",
                            alert_id=alert.id,
                            rule_name=rule_name,
                            severity=rule.severity.value,
                            message=alert.message,
                        )
                else:
                    # Condition not met, check if we should resolve existing alerts
                    self._check_for_resolution(rule_name, metrics)

            except Exception as e:
                logger.error(
                    "Error evaluating alert rule",
                    rule_name=rule_name,
                    error=str(e),
                )

        return new_alerts

    def _create_alert(self, rule: AlertRule, metrics: dict[str, Any]) -> Alert:
        """Create an alert from a rule and metrics.
        
        Args:
            rule: Alert rule that triggered
            metrics: Metrics that triggered the alert
        
        Returns:
            Created alert
        """
        import uuid

        # Format message with template variables
        message = rule.annotations.get("summary", rule.description)
        for key, value in metrics.items():
            message = message.replace(f"{{{{ {key} }}}}", str(value))
        message = message.replace("{{ threshold_value }}", str(rule.threshold_value))

        return Alert(
            id=str(uuid.uuid4()),
            rule_name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            message=message,
            details=metrics.copy(),
            fired_at=datetime.now(UTC),
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
        )

    def _find_existing_alert(
        self,
        rule_name: str,
        metrics: dict[str, Any]
    ) -> Alert | None:
        """Find existing alert for the same rule and context.
        
        Args:
            rule_name: Name of the alert rule
            metrics: Current metrics
        
        Returns:
            Existing alert if found
        """
        for alert in self.active_alerts.values():
            if alert.rule_name == rule_name and alert.state == AlertState.FIRING:
                # Check if it's for the same operation/context
                if metrics.get("operation_name") == alert.details.get("operation_name"):
                    return alert
        return None

    def _check_for_resolution(self, rule_name: str, metrics: dict[str, Any]) -> None:
        """Check if existing alerts should be resolved.
        
        Args:
            rule_name: Name of the alert rule
            metrics: Current metrics
        """
        current_time = datetime.now(UTC)

        for alert_id, alert in list(self.active_alerts.items()):
            if alert.rule_name == rule_name and alert.state == AlertState.FIRING:
                # Check if it's for the same operation/context
                if metrics.get("operation_name") == alert.details.get("operation_name"):
                    # Resolve the alert
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = current_time

                    # Move to history
                    self.alert_history.append(alert)
                    del self.active_alerts[alert_id]

                    logger.info(
                        "Alert resolved",
                        alert_id=alert_id,
                        rule_name=rule_name,
                        duration_seconds=(current_time - alert.fired_at).total_seconds(),
                    )

    def _is_rule_suppressed(self, rule_name: str) -> bool:
        """Check if a rule is currently suppressed.
        
        Args:
            rule_name: Name of the rule
        
        Returns:
            True if rule is suppressed
        """
        for alert in self.active_alerts.values():
            if (alert.rule_name == rule_name and
                alert.state == AlertState.SUPPRESSED and
                alert.suppressed_until and
                alert.suppressed_until > datetime.now(UTC)):
                return True
        return False

    def _suppress_rule(self, rule_name: str) -> None:
        """Suppress a rule for the configured duration.
        
        Args:
            rule_name: Name of the rule to suppress
        """
        import uuid

        suppression_alert = Alert(
            id=str(uuid.uuid4()),
            rule_name=rule_name,
            severity=AlertSeverity.INFO,
            state=AlertState.SUPPRESSED,
            message=f"Alert rule '{rule_name}' suppressed due to excessive firing",
            details={"suppressed_for_minutes": self.suppression_duration_minutes},
            fired_at=datetime.now(UTC),
            suppressed_until=datetime.now(UTC) + timedelta(minutes=self.suppression_duration_minutes),
        )

        self.active_alerts[suppression_alert.id] = suppression_alert
        self.rule_fire_counts[rule_name] = 0  # Reset counter

        logger.info(
            "Alert rule suppressed",
            rule_name=rule_name,
            duration_minutes=self.suppression_duration_minutes,
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "operator"
    ) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who acknowledged the alert
        
        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.now(UTC)
            alert.acknowledged_by = acknowledged_by

            logger.info(
                "Alert acknowledged",
                alert_id=alert_id,
                acknowledged_by=acknowledged_by,
            )
            return True
        return False

    def get_active_alerts(
        self,
        severity_filter: AlertSeverity | None = None,
        unacknowledged_only: bool = False
    ) -> list[Alert]:
        """Get active alerts.
        
        Args:
            severity_filter: Filter by severity
            unacknowledged_only: Only return unacknowledged alerts
        
        Returns:
            List of active alerts
        """
        alerts = []

        for alert in self.active_alerts.values():
            if alert.state != AlertState.FIRING:
                continue

            if severity_filter and alert.severity != severity_filter:
                continue

            if unacknowledged_only and alert.acknowledged_at:
                continue

            alerts.append(alert)

        # Sort by severity and time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }

        return sorted(
            alerts,
            key=lambda a: (severity_order[a.severity], a.fired_at)
        )

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics.
        
        Returns:
            Alert summary dictionary
        """
        active_alerts = self.get_active_alerts()

        summary = {
            "total_active": len(active_alerts),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
            "unacknowledged": 0,
            "suppressed_rules": [],
        }

        for alert in active_alerts:
            summary[alert.severity.value] += 1
            if not alert.acknowledged_at:
                summary["unacknowledged"] += 1

        # Check for suppressed rules
        for alert in self.active_alerts.values():
            if alert.state == AlertState.SUPPRESSED and alert.suppressed_until:
                if alert.suppressed_until > datetime.now(UTC):
                    summary["suppressed_rules"].append({
                        "rule_name": alert.rule_name,
                        "suppressed_until": alert.suppressed_until.isoformat(),
                        "remaining_minutes": (
                            alert.suppressed_until - datetime.now(UTC)
                        ).total_seconds() / 60,
                    })

        return summary

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert callback function.
        
        Args:
            callback: Function to call when alert triggers
        """
        self.alert_callbacks.append(callback)

    def _trigger_callbacks(self, alert: Alert) -> None:
        """Trigger all registered callbacks for an alert.
        
        Args:
            alert: Alert that triggered
        """
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(
                    "Error in alert callback",
                    error=str(e),
                    alert_id=alert.id,
                )

    def cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts from history."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=self.alert_retention_hours)

        # Remove old alerts from history
        while self.alert_history:
            oldest = self.alert_history[0]
            if oldest.resolved_at and oldest.resolved_at < cutoff_time:
                self.alert_history.popleft()
            else:
                break

        # Remove expired suppressions
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.state == AlertState.SUPPRESSED:
                if alert.suppressed_until and alert.suppressed_until < datetime.now(UTC):
                    del self.active_alerts[alert_id]


# Global alert manager instance
_alert_manager: TraceAlertManager | None = None


def get_trace_alert_manager() -> TraceAlertManager:
    """Get or create the global trace alert manager instance.
    
    Returns:
        TraceAlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = TraceAlertManager()
    return _alert_manager
