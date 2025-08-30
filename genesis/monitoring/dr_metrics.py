"""DR metrics integration for Prometheus monitoring."""

from datetime import datetime
from typing import Any

import structlog

from genesis.monitoring.metrics_collector import MetricsCollector
from genesis.monitoring.prometheus_exporter import Metric, MetricType

logger = structlog.get_logger(__name__)


class DRMetricsCollector:
    """Collects and exports DR-related metrics."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        dr_orchestrator,
        backup_manager,
        recovery_engine,
        failover_coordinator
    ):
        """Initialize DR metrics collector.
        
        Args:
            metrics_collector: Main metrics collector
            dr_orchestrator: DR orchestrator instance
            backup_manager: Backup manager instance
            recovery_engine: Recovery engine instance
            failover_coordinator: Failover coordinator instance
        """
        self.metrics_collector = metrics_collector
        self.dr_orchestrator = dr_orchestrator
        self.backup_manager = backup_manager
        self.recovery_engine = recovery_engine
        self.failover_coordinator = failover_coordinator

        # Register DR metrics
        self._register_metrics()

    def _register_metrics(self) -> None:
        """Register DR metrics with Prometheus."""
        # Backup metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_backup_total",
                type=MetricType.COUNTER,
                help="Total number of backups created"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_backup_size_bytes",
                type=MetricType.GAUGE,
                help="Size of last backup in bytes"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_backup_duration_seconds",
                type=MetricType.HISTOGRAM,
                help="Backup duration in seconds"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_backup_age_hours",
                type=MetricType.GAUGE,
                help="Age of last backup in hours"
            )
        )

        # Replication metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_replication_lag_seconds",
                type=MetricType.GAUGE,
                help="Replication lag in seconds"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_replication_queue_size",
                type=MetricType.GAUGE,
                help="Number of items in replication queue"
            )
        )

        # Recovery metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_recovery_total",
                type=MetricType.COUNTER,
                help="Total number of recovery operations"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_recovery_duration_seconds",
                type=MetricType.HISTOGRAM,
                help="Recovery duration in seconds"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_rpo_minutes",
                type=MetricType.GAUGE,
                help="Current Recovery Point Objective in minutes"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_rto_minutes",
                type=MetricType.GAUGE,
                help="Current Recovery Time Objective in minutes"
            )
        )

        # Failover metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_failover_total",
                type=MetricType.COUNTER,
                help="Total number of failovers"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_failover_duration_seconds",
                type=MetricType.HISTOGRAM,
                help="Failover duration in seconds"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_health_check_failures",
                type=MetricType.COUNTER,
                help="Total health check failures",
                labels={"check_name": ""}
            )
        )

        # DR readiness metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_dr_readiness_score",
                type=MetricType.GAUGE,
                help="DR readiness score (0-1)"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_dr_component_score",
                type=MetricType.GAUGE,
                help="DR component readiness score",
                labels={"component": ""}
            )
        )

        # Emergency closure metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_emergency_closure_total",
                type=MetricType.COUNTER,
                help="Total emergency closures"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_emergency_positions_closed",
                type=MetricType.COUNTER,
                help="Total positions closed in emergencies"
            )
        )

        # DR test metrics
        self.metrics_collector.register_metric(
            Metric(
                name="genesis_dr_test_total",
                type=MetricType.COUNTER,
                help="Total DR tests run"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_dr_test_success_rate",
                type=MetricType.GAUGE,
                help="DR test success rate"
            )
        )

        self.metrics_collector.register_metric(
            Metric(
                name="genesis_dr_test_last_run_timestamp",
                type=MetricType.GAUGE,
                help="Timestamp of last DR test"
            )
        )

    async def collect_metrics(self) -> None:
        """Collect all DR metrics."""
        await self._collect_backup_metrics()
        await self._collect_replication_metrics()
        await self._collect_recovery_metrics()
        await self._collect_failover_metrics()
        await self._collect_readiness_metrics()
        await self._collect_test_metrics()

    async def _collect_backup_metrics(self) -> None:
        """Collect backup-related metrics."""
        try:
            # Get backup status
            status = self.backup_manager.get_backup_status()

            # Update backup count
            if hasattr(self.backup_manager, 'backup_history'):
                self.metrics_collector.increment_counter(
                    "genesis_backup_total",
                    value=len(self.backup_manager.backup_history)
                )

            # Update backup age
            if status["last_full_backup"]:
                last_backup = datetime.fromisoformat(status["last_full_backup"])
                age_hours = (datetime.utcnow() - last_backup).total_seconds() / 3600

                self.metrics_collector.set_gauge(
                    "genesis_backup_age_hours",
                    age_hours
                )

            # Get latest backup size
            if self.backup_manager.backup_history:
                latest = self.backup_manager.backup_history[-1]
                self.metrics_collector.set_gauge(
                    "genesis_backup_size_bytes",
                    latest.size_bytes
                )

        except Exception as e:
            logger.error("Failed to collect backup metrics", error=str(e))

    async def _collect_replication_metrics(self) -> None:
        """Collect replication metrics."""
        try:
            # Get replication status
            status = self.backup_manager.replication_manager.get_replication_status()

            # Update replication lag
            self.metrics_collector.set_gauge(
                "genesis_replication_lag_seconds",
                status["replication_lag_seconds"]
            )

            # Update queue size
            self.metrics_collector.set_gauge(
                "genesis_replication_queue_size",
                status["queue_size"]
            )

        except Exception as e:
            logger.error("Failed to collect replication metrics", error=str(e))

    async def _collect_recovery_metrics(self) -> None:
        """Collect recovery metrics."""
        try:
            # Get recovery status
            status = self.recovery_engine.get_recovery_status()

            # Update recovery metrics
            if status["recovery_metrics"]:
                metrics = status["recovery_metrics"]

                if "recovery_time_seconds" in metrics:
                    self.metrics_collector.observe_histogram(
                        "genesis_recovery_duration_seconds",
                        metrics["recovery_time_seconds"]
                    )

            # Calculate RPO
            rpo_result = await self.dr_orchestrator._verify_rpo()
            self.metrics_collector.set_gauge(
                "genesis_rpo_minutes",
                rpo_result["data_loss_minutes"]
            )

            # Set RTO target
            self.metrics_collector.set_gauge(
                "genesis_rto_minutes",
                self.dr_orchestrator.RTO_TARGET_MINUTES
            )

        except Exception as e:
            logger.error("Failed to collect recovery metrics", error=str(e))

    async def _collect_failover_metrics(self) -> None:
        """Collect failover metrics."""
        try:
            # Get failover status
            status = self.failover_coordinator.get_status()

            # Update failover count
            self.metrics_collector.set_gauge(
                "genesis_failover_total",
                status["failover_count"]
            )

            # Update health check failures
            if status["health_status"]:
                for check_name, check_status in status["health_status"]["checks"].items():
                    if not check_status["is_healthy"]:
                        self.metrics_collector.increment_counter(
                            "genesis_health_check_failures",
                            labels={"check_name": check_name}
                        )

        except Exception as e:
            logger.error("Failed to collect failover metrics", error=str(e))

    async def _collect_readiness_metrics(self) -> None:
        """Collect DR readiness metrics."""
        try:
            # Calculate readiness score
            readiness = await self.dr_orchestrator.calculate_readiness_score()

            # Update overall score
            self.metrics_collector.set_gauge(
                "genesis_dr_readiness_score",
                readiness["overall_score"]
            )

            # Update component scores
            for component, score in readiness["components"].items():
                self.metrics_collector.set_gauge(
                    "genesis_dr_component_score",
                    score,
                    labels={"component": component}
                )

        except Exception as e:
            logger.error("Failed to collect readiness metrics", error=str(e))

    async def _collect_test_metrics(self) -> None:
        """Collect DR test metrics."""
        try:
            if hasattr(self.dr_orchestrator, 'dr_test_runner'):
                test_runner = self.dr_orchestrator.dr_test_runner

                # Get test metrics
                metrics = test_runner.get_performance_metrics()

                # Update test count
                self.metrics_collector.set_gauge(
                    "genesis_dr_test_total",
                    metrics["total_tests"]
                )

                # Update success rate
                self.metrics_collector.set_gauge(
                    "genesis_dr_test_success_rate",
                    metrics["success_rate"]
                )

                # Update last test timestamp
                if test_runner.last_test_time:
                    timestamp = test_runner.last_test_time.timestamp()
                    self.metrics_collector.set_gauge(
                        "genesis_dr_test_last_run_timestamp",
                        timestamp
                    )

        except Exception as e:
            logger.error("Failed to collect test metrics", error=str(e))

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get metrics for dashboard display.
        
        Returns:
            Dashboard metrics dictionary
        """
        metrics = {}

        try:
            # Backup metrics
            backup_status = self.backup_manager.get_backup_status()
            metrics["backup"] = {
                "last_backup": backup_status["last_full_backup"],
                "backup_count": backup_status["backup_count"],
                "scheduler_running": backup_status["scheduler_running"]
            }

            # Replication metrics
            replication_status = self.backup_manager.replication_manager.get_replication_status()
            metrics["replication"] = {
                "lag_seconds": replication_status["replication_lag_seconds"],
                "queue_size": replication_status["queue_size"],
                "total_replicated": replication_status["total_replicated"]
            }

            # Recovery metrics
            recovery_status = self.recovery_engine.get_recovery_status()
            metrics["recovery"] = {
                "in_progress": recovery_status["recovery_in_progress"],
                "last_recovery": recovery_status["last_recovery_timestamp"]
            }

            # Failover metrics
            failover_status = self.failover_coordinator.get_status()
            metrics["failover"] = {
                "state": failover_status["state"],
                "monitoring": failover_status["monitoring"],
                "failover_count": failover_status["failover_count"]
            }

            # DR readiness
            dr_status = self.dr_orchestrator.get_status()
            metrics["readiness"] = {
                "score": dr_status["readiness_score"],
                "dr_active": dr_status["dr_active"]
            }

        except Exception as e:
            logger.error("Failed to get dashboard metrics", error=str(e))

        return metrics

    def create_grafana_dashboard(self) -> dict[str, Any]:
        """Create Grafana dashboard configuration for DR metrics.
        
        Returns:
            Grafana dashboard JSON
        """
        return {
            "dashboard": {
                "title": "Disaster Recovery Dashboard",
                "panels": [
                    {
                        "title": "DR Readiness Score",
                        "type": "gauge",
                        "targets": [{
                            "expr": "genesis_dr_readiness_score"
                        }],
                        "thresholds": {
                            "steps": [
                                {"value": 0, "color": "red"},
                                {"value": 0.6, "color": "yellow"},
                                {"value": 0.8, "color": "green"}
                            ]
                        }
                    },
                    {
                        "title": "Backup Age",
                        "type": "graph",
                        "targets": [{
                            "expr": "genesis_backup_age_hours"
                        }],
                        "alert": {
                            "conditions": [{
                                "evaluator": {
                                    "params": [8],
                                    "type": "gt"
                                }
                            }]
                        }
                    },
                    {
                        "title": "Replication Lag",
                        "type": "graph",
                        "targets": [{
                            "expr": "genesis_replication_lag_seconds"
                        }],
                        "alert": {
                            "conditions": [{
                                "evaluator": {
                                    "params": [300],
                                    "type": "gt"
                                }
                            }]
                        }
                    },
                    {
                        "title": "RTO/RPO Compliance",
                        "type": "stat",
                        "targets": [
                            {"expr": "genesis_rto_minutes", "legendFormat": "RTO"},
                            {"expr": "genesis_rpo_minutes", "legendFormat": "RPO"}
                        ]
                    },
                    {
                        "title": "Health Check Status",
                        "type": "table",
                        "targets": [{
                            "expr": "rate(genesis_health_check_failures[5m])"
                        }]
                    },
                    {
                        "title": "DR Test Success Rate",
                        "type": "gauge",
                        "targets": [{
                            "expr": "genesis_dr_test_success_rate"
                        }],
                        "thresholds": {
                            "steps": [
                                {"value": 0, "color": "red"},
                                {"value": 0.9, "color": "yellow"},
                                {"value": 0.95, "color": "green"}
                            ]
                        }
                    }
                ]
            }
        }

    def get_alert_rules(self) -> list[dict[str, Any]]:
        """Get Prometheus alert rules for DR.
        
        Returns:
            List of alert rules
        """
        return [
            {
                "alert": "BackupTooOld",
                "expr": "genesis_backup_age_hours > 8",
                "for": "5m",
                "labels": {
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "Backup is older than 8 hours",
                    "description": "Last backup was {{ $value }} hours ago"
                }
            },
            {
                "alert": "HighReplicationLag",
                "expr": "genesis_replication_lag_seconds > 300",
                "for": "5m",
                "labels": {
                    "severity": "critical"
                },
                "annotations": {
                    "summary": "Replication lag exceeds 5 minutes",
                    "description": "Current lag: {{ $value }} seconds"
                }
            },
            {
                "alert": "LowDRReadiness",
                "expr": "genesis_dr_readiness_score < 0.7",
                "for": "30m",
                "labels": {
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "DR readiness score below threshold",
                    "description": "Score: {{ $value }}"
                }
            },
            {
                "alert": "DRTestOverdue",
                "expr": "time() - genesis_dr_test_last_run_timestamp > 2592000",
                "for": "1h",
                "labels": {
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "DR test not run in 30 days",
                    "description": "Last test was {{ $value }} seconds ago"
                }
            },
            {
                "alert": "FailoverActive",
                "expr": "genesis_failover_total > 0",
                "for": "1m",
                "labels": {
                    "severity": "critical"
                },
                "annotations": {
                    "summary": "Failover has been triggered",
                    "description": "System is running on backup infrastructure"
                }
            }
        ]
