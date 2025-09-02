"""Trace analysis and performance bottleneck detection for Genesis."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SpanMetrics:
    """Metrics for a span type."""
    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    error_count: int = 0
    slow_count: int = 0  # Count of spans exceeding threshold
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        return self.total_duration_ms / self.count if self.count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / self.count if self.count > 0 else 0.0

    @property
    def slow_rate(self) -> float:
        """Calculate slow operation rate."""
        return self.slow_count / self.count if self.count > 0 else 0.0


@dataclass
class ServiceDependency:
    """Service dependency information."""
    source_service: str
    target_service: str
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / self.call_count if self.call_count > 0 else 0.0


@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck."""
    operation_name: str
    bottleneck_type: str  # "latency", "error_rate", "throughput"
    severity: str  # "critical", "high", "medium", "low"
    current_value: float
    threshold_value: float
    impact_score: float  # 0-100
    recommendations: list[str]
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class TraceAnalyzer:
    """Analyze traces for performance bottlenecks and insights."""

    def __init__(
        self,
        slow_threshold_ms: float = 5.0,
        critical_threshold_ms: float = 100.0,
        error_rate_threshold: float = 0.01,
        sample_window_minutes: int = 5,
    ):
        """Initialize trace analyzer.
        
        Args:
            slow_threshold_ms: Threshold for slow operations
            critical_threshold_ms: Threshold for critical latency
            error_rate_threshold: Threshold for error rate alerts
            sample_window_minutes: Time window for analysis
        """
        self.slow_threshold_ms = slow_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.sample_window_minutes = sample_window_minutes

        # Metrics storage
        self.span_metrics: dict[str, SpanMetrics] = defaultdict(SpanMetrics)
        self.service_dependencies: dict[tuple, ServiceDependency] = {}
        self.bottlenecks: list[PerformanceBottleneck] = []

        # Critical operations to monitor closely
        self.critical_operations = {
            "order_execution",
            "risk_check",
            "tilt_detection",
            "market_data_processing",
            "exchange_api",
        }

        logger.info(
            "Trace analyzer initialized",
            slow_threshold_ms=slow_threshold_ms,
            critical_threshold_ms=critical_threshold_ms,
            error_rate_threshold=error_rate_threshold,
        )

    def process_span(self, span_data: dict[str, Any]) -> None:
        """Process a span and update metrics.
        
        Args:
            span_data: Span data dictionary
        """
        operation_name = span_data.get("name", "unknown")
        duration_ms = span_data.get("duration_ms", 0)
        has_error = span_data.get("error", False)
        attributes = span_data.get("attributes", {})

        # Update span metrics
        metrics = self.span_metrics[operation_name]
        metrics.count += 1
        metrics.total_duration_ms += duration_ms
        metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
        metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)

        if has_error:
            metrics.error_count += 1

        if duration_ms > self.slow_threshold_ms:
            metrics.slow_count += 1

        metrics.last_updated = datetime.now(UTC)

        # Process service dependencies if this is a client span
        if span_data.get("kind") == "CLIENT":
            self._process_service_dependency(span_data)

        # Check for bottlenecks in critical operations
        if any(op in operation_name.lower() for op in self.critical_operations):
            self._check_for_bottlenecks(operation_name, metrics)

    def _process_service_dependency(self, span_data: dict[str, Any]) -> None:
        """Process service dependency from client span.
        
        Args:
            span_data: Span data dictionary
        """
        source_service = span_data.get("service_name", "unknown")
        target_service = span_data.get("attributes", {}).get("peer.service", "external")
        duration_ms = span_data.get("duration_ms", 0)
        has_error = span_data.get("error", False)

        dep_key = (source_service, target_service)

        if dep_key not in self.service_dependencies:
            self.service_dependencies[dep_key] = ServiceDependency(
                source_service=source_service,
                target_service=target_service,
            )

        dep = self.service_dependencies[dep_key]
        dep.call_count += 1
        dep.total_latency_ms += duration_ms

        if has_error:
            dep.error_count += 1

    def _check_for_bottlenecks(
        self,
        operation_name: str,
        metrics: SpanMetrics
    ) -> None:
        """Check for performance bottlenecks.
        
        Args:
            operation_name: Name of the operation
            metrics: Span metrics
        """
        bottlenecks_found = []

        # Check for high latency
        if metrics.avg_duration_ms > self.critical_threshold_ms:
            bottlenecks_found.append(
                PerformanceBottleneck(
                    operation_name=operation_name,
                    bottleneck_type="latency",
                    severity="critical",
                    current_value=metrics.avg_duration_ms,
                    threshold_value=self.critical_threshold_ms,
                    impact_score=min(100, (metrics.avg_duration_ms / self.critical_threshold_ms) * 50),
                    recommendations=[
                        f"Optimize {operation_name} to reduce latency",
                        "Consider caching frequently accessed data",
                        "Review database queries for optimization",
                        "Check for blocking I/O operations",
                    ],
                )
            )
        elif metrics.avg_duration_ms > self.slow_threshold_ms:
            bottlenecks_found.append(
                PerformanceBottleneck(
                    operation_name=operation_name,
                    bottleneck_type="latency",
                    severity="high" if metrics.avg_duration_ms > self.slow_threshold_ms * 5 else "medium",
                    current_value=metrics.avg_duration_ms,
                    threshold_value=self.slow_threshold_ms,
                    impact_score=min(80, (metrics.avg_duration_ms / self.slow_threshold_ms) * 30),
                    recommendations=[
                        f"Review {operation_name} for optimization opportunities",
                        "Profile the operation to identify slow components",
                        "Consider async processing for non-critical paths",
                    ],
                )
            )

        # Check for high error rate
        if metrics.error_rate > self.error_rate_threshold:
            severity = "critical" if metrics.error_rate > 0.1 else "high" if metrics.error_rate > 0.05 else "medium"
            bottlenecks_found.append(
                PerformanceBottleneck(
                    operation_name=operation_name,
                    bottleneck_type="error_rate",
                    severity=severity,
                    current_value=metrics.error_rate * 100,
                    threshold_value=self.error_rate_threshold * 100,
                    impact_score=min(100, metrics.error_rate * 1000),
                    recommendations=[
                        f"Investigate error patterns in {operation_name}",
                        "Add retry logic with exponential backoff",
                        "Improve error handling and recovery",
                        "Check for resource exhaustion or rate limits",
                    ],
                )
            )

        # Check for high slow operation rate
        if metrics.slow_rate > 0.2:  # More than 20% of operations are slow
            bottlenecks_found.append(
                PerformanceBottleneck(
                    operation_name=operation_name,
                    bottleneck_type="throughput",
                    severity="high" if metrics.slow_rate > 0.5 else "medium",
                    current_value=metrics.slow_rate * 100,
                    threshold_value=20.0,
                    impact_score=min(90, metrics.slow_rate * 150),
                    recommendations=[
                        f"Investigate variability in {operation_name} performance",
                        "Check for resource contention or locking issues",
                        "Consider load balancing or horizontal scaling",
                        "Review batch processing opportunities",
                    ],
                )
            )

        # Add new bottlenecks to the list
        for bottleneck in bottlenecks_found:
            # Check if similar bottleneck already exists
            existing = next(
                (b for b in self.bottlenecks
                 if b.operation_name == bottleneck.operation_name
                 and b.bottleneck_type == bottleneck.bottleneck_type),
                None
            )

            if existing:
                # Update existing bottleneck
                existing.current_value = bottleneck.current_value
                existing.impact_score = bottleneck.impact_score
                existing.detected_at = bottleneck.detected_at
                existing.severity = bottleneck.severity
            else:
                # Add new bottleneck
                self.bottlenecks.append(bottleneck)

                logger.warning(
                    "Performance bottleneck detected",
                    operation=operation_name,
                    type=bottleneck.bottleneck_type,
                    severity=bottleneck.severity,
                    current_value=bottleneck.current_value,
                    threshold=bottleneck.threshold_value,
                )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all tracked operations.
        
        Returns:
            Performance summary dictionary
        """
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "analysis_window_minutes": self.sample_window_minutes,
            "total_operations": len(self.span_metrics),
            "critical_bottlenecks": 0,
            "high_bottlenecks": 0,
            "operations": {},
            "top_slow_operations": [],
            "top_error_operations": [],
        }

        # Count bottlenecks by severity
        for bottleneck in self.bottlenecks:
            if bottleneck.severity == "critical":
                summary["critical_bottlenecks"] += 1
            elif bottleneck.severity == "high":
                summary["high_bottlenecks"] += 1

        # Compile operation metrics
        for op_name, metrics in self.span_metrics.items():
            summary["operations"][op_name] = {
                "count": metrics.count,
                "avg_duration_ms": round(metrics.avg_duration_ms, 2),
                "min_duration_ms": round(metrics.min_duration_ms, 2),
                "max_duration_ms": round(metrics.max_duration_ms, 2),
                "error_rate": round(metrics.error_rate * 100, 2),
                "slow_rate": round(metrics.slow_rate * 100, 2),
            }

        # Find top slow operations
        slow_ops = sorted(
            self.span_metrics.items(),
            key=lambda x: x[1].avg_duration_ms,
            reverse=True
        )[:5]

        summary["top_slow_operations"] = [
            {
                "operation": op_name,
                "avg_duration_ms": round(metrics.avg_duration_ms, 2),
                "count": metrics.count,
            }
            for op_name, metrics in slow_ops
        ]

        # Find top error operations
        error_ops = sorted(
            [(op, m) for op, m in self.span_metrics.items() if m.error_count > 0],
            key=lambda x: x[1].error_rate,
            reverse=True
        )[:5]

        summary["top_error_operations"] = [
            {
                "operation": op_name,
                "error_rate": round(metrics.error_rate * 100, 2),
                "error_count": metrics.error_count,
                "total_count": metrics.count,
            }
            for op_name, metrics in error_ops
        ]

        return summary

    def get_bottlenecks(
        self,
        severity_filter: str | None = None
    ) -> list[PerformanceBottleneck]:
        """Get current performance bottlenecks.
        
        Args:
            severity_filter: Filter by severity level
        
        Returns:
            List of performance bottlenecks
        """
        if severity_filter:
            return [b for b in self.bottlenecks if b.severity == severity_filter]

        # Sort by impact score
        return sorted(self.bottlenecks, key=lambda b: b.impact_score, reverse=True)

    def get_service_dependencies(self) -> list[dict[str, Any]]:
        """Get service dependency information.
        
        Returns:
            List of service dependencies
        """
        dependencies = []

        for dep in self.service_dependencies.values():
            dependencies.append({
                "source": dep.source_service,
                "target": dep.target_service,
                "call_count": dep.call_count,
                "avg_latency_ms": round(dep.avg_latency_ms, 2),
                "error_rate": round(dep.error_rate * 100, 2),
                "error_count": dep.error_count,
            })

        # Sort by call count
        return sorted(dependencies, key=lambda d: d["call_count"], reverse=True)

    def generate_performance_report(self) -> str:
        """Generate a detailed performance report.
        
        Returns:
            Performance report as string
        """
        report = []
        report.append("=" * 60)
        report.append("GENESIS TRACE ANALYSIS PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now(UTC).isoformat()}")
        report.append(f"Analysis Window: {self.sample_window_minutes} minutes")
        report.append("")

        # Summary
        summary = self.get_performance_summary()
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Operations Tracked: {summary['total_operations']}")
        report.append(f"Critical Bottlenecks: {summary['critical_bottlenecks']}")
        report.append(f"High Priority Bottlenecks: {summary['high_bottlenecks']}")
        report.append("")

        # Top slow operations
        if summary["top_slow_operations"]:
            report.append("TOP SLOW OPERATIONS")
            report.append("-" * 40)
            for op in summary["top_slow_operations"]:
                report.append(
                    f"  {op['operation']}: {op['avg_duration_ms']}ms avg "
                    f"({op['count']} calls)"
                )
            report.append("")

        # Top error operations
        if summary["top_error_operations"]:
            report.append("TOP ERROR-PRONE OPERATIONS")
            report.append("-" * 40)
            for op in summary["top_error_operations"]:
                report.append(
                    f"  {op['operation']}: {op['error_rate']}% error rate "
                    f"({op['error_count']}/{op['total_count']} calls)"
                )
            report.append("")

        # Critical bottlenecks
        critical_bottlenecks = self.get_bottlenecks("critical")
        if critical_bottlenecks:
            report.append("CRITICAL BOTTLENECKS")
            report.append("-" * 40)
            for b in critical_bottlenecks:
                report.append(f"  Operation: {b.operation_name}")
                report.append(f"  Type: {b.bottleneck_type}")
                report.append(f"  Current: {b.current_value:.2f}")
                report.append(f"  Threshold: {b.threshold_value:.2f}")
                report.append(f"  Impact Score: {b.impact_score:.1f}/100")
                report.append("  Recommendations:")
                for rec in b.recommendations[:2]:  # Show top 2 recommendations
                    report.append(f"    - {rec}")
                report.append("")

        # Service dependencies
        deps = self.get_service_dependencies()
        if deps:
            report.append("SERVICE DEPENDENCIES")
            report.append("-" * 40)
            for dep in deps[:5]:  # Show top 5 dependencies
                report.append(
                    f"  {dep['source']} -> {dep['target']}: "
                    f"{dep['call_count']} calls, "
                    f"{dep['avg_latency_ms']}ms avg, "
                    f"{dep['error_rate']}% errors"
                )
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def clear_old_data(self, retention_minutes: int = 60) -> None:
        """Clear old data to prevent memory growth.
        
        Args:
            retention_minutes: Data retention period in minutes
        """
        cutoff_time = datetime.now(UTC) - timedelta(minutes=retention_minutes)

        # Clear old span metrics
        old_operations = [
            op for op, metrics in self.span_metrics.items()
            if metrics.last_updated < cutoff_time
        ]

        for op in old_operations:
            del self.span_metrics[op]

        # Clear old bottlenecks
        self.bottlenecks = [
            b for b in self.bottlenecks
            if b.detected_at >= cutoff_time
        ]

        if old_operations:
            logger.info(
                "Cleared old trace data",
                operations_removed=len(old_operations),
                retention_minutes=retention_minutes,
            )


# Global trace analyzer instance
_trace_analyzer: TraceAnalyzer | None = None


def get_trace_analyzer() -> TraceAnalyzer:
    """Get or create the global trace analyzer instance.
    
    Returns:
        TraceAnalyzer instance
    """
    global _trace_analyzer
    if _trace_analyzer is None:
        _trace_analyzer = TraceAnalyzer()
    return _trace_analyzer
