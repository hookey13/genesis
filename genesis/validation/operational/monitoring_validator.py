"""Monitoring and observability validation module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import structlog
import yaml

from genesis.validation.base import (
    CheckStatus,
    ValidationCheck,
    ValidationContext,
    ValidationEvidence,
    ValidationMetadata,
    ValidationResult,
    Validator,
)

logger = structlog.get_logger(__name__)


class MonitoringValidator(Validator):
    """Validates monitoring and observability setup."""

    REQUIRED_METRICS = [
        "genesis_order_execution_latency_seconds",
        "genesis_api_calls_total",
        "genesis_active_positions",
        "genesis_memory_usage_bytes",
        "genesis_total_pnl_usdt",
        "genesis_error_rate_per_minute",
        "genesis_system_health_score",
    ]

    REQUIRED_DASHBOARDS = [
        "trading-overview",
        "system-health",
        "performance-metrics",
        "error-tracking",
        "api-usage",
    ]

    REQUIRED_ALERTS = [
        "high_error_rate",
        "low_system_health",
        "high_latency",
        "position_limit_exceeded",
        "api_rate_limit_warning",
    ]

    def __init__(self, genesis_root: Path | None = None):
        """Initialize monitoring validator.

        Args:
            genesis_root: Root directory of Genesis project
        """
        super().__init__(
            validator_id="OPS-001",
            name="MonitoringValidator",
            description="Validates monitoring and observability setup including Prometheus metrics, Grafana dashboards, and alert rules",
        )
        self.genesis_root = genesis_root or Path.cwd()
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        self.set_timeout(120)  # 2 minutes for monitoring checks
        self.set_retry_policy(retry_count=2, retry_delay_seconds=5)

    async def run_validation(self, context: ValidationContext) -> ValidationResult:
        """Run monitoring validation checks.

        Args:
            context: Validation context with configuration

        Returns:
            ValidationResult with checks and evidence
        """
        logger.info("Starting monitoring validation")
        start_time = datetime.utcnow()

        # Update genesis_root from context if available
        if context.genesis_root:
            self.genesis_root = Path(context.genesis_root)

        checks = []
        evidence = ValidationEvidence()

        # Check Prometheus metrics
        metrics_check = await self._create_prometheus_check()
        checks.append(metrics_check)

        # Check Grafana dashboards
        dashboard_check = await self._create_grafana_check()
        checks.append(dashboard_check)

        # Check alert rules
        alerts_check = await self._create_alert_rules_check()
        checks.append(alerts_check)

        # Check monitoring configuration files
        config_check = await self._create_monitoring_configs_check()
        checks.append(config_check)

        # Determine overall status
        failed_checks = [c for c in checks if c.status == CheckStatus.FAILED]
        warning_checks = [c for c in checks if c.status == CheckStatus.WARNING]

        if not failed_checks and not warning_checks:
            overall_status = CheckStatus.PASSED
            message = "Monitoring fully configured and operational"
        elif failed_checks:
            overall_status = CheckStatus.FAILED
            message = f"Monitoring gaps detected - {len(failed_checks)} checks failed"
        else:
            overall_status = CheckStatus.WARNING
            message = (
                f"Monitoring partially configured - {len(warning_checks)} warnings"
            )

        # Create metadata
        metadata = ValidationMetadata(
            version="1.0.0",
            environment=context.environment,
            run_id=context.metadata.run_id if context.metadata else "local-run",
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            machine_info={},
            additional_info={"genesis_root": str(self.genesis_root)},
        )

        # Create result
        result = ValidationResult(
            validator_id=self.validator_id,
            validator_name=self.name,
            status=overall_status,
            message=message,
            checks=checks,
            evidence=evidence,
            metadata=metadata,
        )

        # Update counts and score
        result.update_counts()

        return result

    async def _create_prometheus_check(self) -> ValidationCheck:
        """Create validation check for Prometheus metrics.

        Returns:
            ValidationCheck for Prometheus metrics
        """
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()

        metrics_result = await self._validate_prometheus_metrics()

        if metrics_result["passed"]:
            status = CheckStatus.PASSED
            severity = "low"
        else:
            status = CheckStatus.FAILED
            severity = "high"

        # Add evidence
        evidence.metrics["metrics_found"] = metrics_result.get("metrics_found", [])
        evidence.metrics["metrics_missing"] = metrics_result.get("metrics_missing", [])

        return ValidationCheck(
            id="MON-001",
            name="Prometheus Metrics",
            description="Verify required Prometheus metrics are configured",
            category="monitoring",
            status=status,
            details=metrics_result["message"],
            is_blocking=True,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity=severity,
            remediation=(
                "Configure missing metrics in Prometheus exporters"
                if not metrics_result["passed"]
                else None
            ),
            tags=["prometheus", "metrics", "observability"],
        )

    async def _create_grafana_check(self) -> ValidationCheck:
        """Create validation check for Grafana dashboards.

        Returns:
            ValidationCheck for Grafana dashboards
        """
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()

        dashboard_result = await self._validate_grafana_dashboards()

        if dashboard_result["passed"]:
            status = CheckStatus.PASSED
            severity = "low"
        else:
            status = CheckStatus.WARNING  # Dashboards are important but not blocking
            severity = "medium"

        # Add evidence
        evidence.metrics["dashboards_found"] = dashboard_result.get(
            "dashboards_found", []
        )
        evidence.metrics["dashboards_missing"] = dashboard_result.get(
            "dashboards_missing", []
        )

        return ValidationCheck(
            id="MON-002",
            name="Grafana Dashboards",
            description="Verify required Grafana dashboards exist",
            category="monitoring",
            status=status,
            details=dashboard_result["message"],
            is_blocking=False,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity=severity,
            remediation=(
                "Create missing dashboards in config/grafana/"
                if not dashboard_result["passed"]
                else None
            ),
            tags=["grafana", "dashboards", "visualization"],
        )

    async def _create_alert_rules_check(self) -> ValidationCheck:
        """Create validation check for alert rules.

        Returns:
            ValidationCheck for alert rules
        """
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()

        alerts_result = await self._validate_alert_rules()

        if alerts_result["passed"]:
            status = CheckStatus.PASSED
            severity = "low"
        else:
            status = CheckStatus.FAILED
            severity = "high"

        # Add evidence
        evidence.metrics["alerts_found"] = alerts_result.get("alerts_found", [])
        evidence.metrics["alerts_missing"] = alerts_result.get("alerts_missing", [])

        return ValidationCheck(
            id="MON-003",
            name="Alert Rules",
            description="Verify critical alert rules are configured",
            category="monitoring",
            status=status,
            details=alerts_result["message"],
            is_blocking=True,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity=severity,
            remediation=(
                "Add missing alerts to config/prometheus/alerts.yml"
                if not alerts_result["passed"]
                else None
            ),
            tags=["alerts", "prometheus", "monitoring"],
        )

    async def _create_monitoring_configs_check(self) -> ValidationCheck:
        """Create validation check for monitoring configuration files.

        Returns:
            ValidationCheck for monitoring configs
        """
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()

        config_result = await self._validate_monitoring_configs()

        if config_result["passed"]:
            status = CheckStatus.PASSED
            severity = "low"
        else:
            status = CheckStatus.FAILED
            severity = "critical"

        # Add evidence
        evidence.metrics["configs_found"] = config_result.get("configs_found", [])
        evidence.metrics["configs_missing"] = config_result.get("configs_missing", [])
        evidence.metrics["config_errors"] = config_result.get("config_errors", [])

        return ValidationCheck(
            id="MON-004",
            name="Monitoring Configuration",
            description="Verify monitoring configuration files exist and are valid",
            category="monitoring",
            status=status,
            details=config_result["message"],
            is_blocking=True,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity=severity,
            remediation=(
                "Create missing configuration files and fix YAML errors"
                if not config_result["passed"]
                else None
            ),
            tags=["configuration", "prometheus", "grafana"],
        )

    async def _validate_prometheus_metrics(self) -> dict[str, Any]:
        """Validate that required Prometheus metrics are configured.

        Returns:
            Validation result for Prometheus metrics
        """
        result = {
            "passed": False,
            "message": "",
            "metrics_found": [],
            "metrics_missing": [],
        }

        try:
            # Check if Prometheus is accessible
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.prometheus_url}/api/v1/label/__name__/values"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        available_metrics = data.get("data", [])

                        # Check for required metrics
                        for metric in self.REQUIRED_METRICS:
                            if metric in available_metrics:
                                result["metrics_found"].append(metric)
                            else:
                                result["metrics_missing"].append(metric)

                        if not result["metrics_missing"]:
                            result["passed"] = True
                            result["message"] = (
                                f"All {len(self.REQUIRED_METRICS)} required metrics configured"
                            )
                        else:
                            result["message"] = (
                                f"Missing {len(result['metrics_missing'])} required metrics"
                            )
                    else:
                        result["message"] = (
                            f"Prometheus API returned status {resp.status}"
                        )
        except aiohttp.ClientError as e:
            result["message"] = f"Failed to connect to Prometheus: {e!s}"
            result["details"] = "Ensure Prometheus is running and accessible"
        except Exception as e:
            result["message"] = f"Unexpected error: {e!s}"

        return result

    async def _validate_grafana_dashboards(self) -> dict[str, Any]:
        """Validate that required Grafana dashboards exist.

        Returns:
            Validation result for Grafana dashboards
        """
        result = {
            "passed": False,
            "message": "",
            "dashboards_found": [],
            "dashboards_missing": [],
        }

        # Check local Grafana configuration files
        grafana_dir = self.genesis_root / "config" / "grafana"
        if grafana_dir.exists():
            dashboard_files = list(grafana_dir.glob("*.json"))
            found_dashboards = []

            for dashboard_file in dashboard_files:
                try:
                    with open(dashboard_file) as f:
                        dashboard_data = json.load(f)
                        dashboard_uid = dashboard_data.get("uid", "")
                        dashboard_title = dashboard_data.get("title", "")

                        # Check if this matches a required dashboard
                        for required in self.REQUIRED_DASHBOARDS:
                            if (
                                required in dashboard_uid.lower()
                                or required in dashboard_title.lower()
                            ):
                                found_dashboards.append(required)
                                break
                except Exception as e:
                    logger.warning(f"Failed to parse dashboard {dashboard_file}: {e}")

            result["dashboards_found"] = list(set(found_dashboards))
            result["dashboards_missing"] = [
                d
                for d in self.REQUIRED_DASHBOARDS
                if d not in result["dashboards_found"]
            ]

            if not result["dashboards_missing"]:
                result["passed"] = True
                result["message"] = (
                    f"All {len(self.REQUIRED_DASHBOARDS)} required dashboards configured"
                )
            else:
                result["message"] = (
                    f"Missing {len(result['dashboards_missing'])} required dashboards"
                )
        else:
            result["message"] = "Grafana configuration directory not found"
            result["dashboards_missing"] = self.REQUIRED_DASHBOARDS

        return result

    async def _validate_alert_rules(self) -> dict[str, Any]:
        """Validate that required alert rules are configured.

        Returns:
            Validation result for alert rules
        """
        result = {
            "passed": False,
            "message": "",
            "alerts_found": [],
            "alerts_missing": [],
        }

        # Check for Prometheus alert rules configuration
        prometheus_rules_file = (
            self.genesis_root / "config" / "prometheus" / "alerts.yml"
        )
        if prometheus_rules_file.exists():
            try:
                with open(prometheus_rules_file) as f:
                    alert_config = yaml.safe_load(f)

                    configured_alerts = []
                    if alert_config and "groups" in alert_config:
                        for group in alert_config["groups"]:
                            for rule in group.get("rules", []):
                                if rule.get("alert"):
                                    configured_alerts.append(rule["alert"])

                    # Check for required alerts
                    for alert in self.REQUIRED_ALERTS:
                        if alert in configured_alerts:
                            result["alerts_found"].append(alert)
                        else:
                            result["alerts_missing"].append(alert)

                    if not result["alerts_missing"]:
                        result["passed"] = True
                        result["message"] = (
                            f"All {len(self.REQUIRED_ALERTS)} required alerts configured"
                        )
                    else:
                        result["message"] = (
                            f"Missing {len(result['alerts_missing'])} required alerts"
                        )
            except Exception as e:
                result["message"] = f"Failed to parse alert rules: {e!s}"
        else:
            result["message"] = "Alert rules configuration file not found"
            result["alerts_missing"] = self.REQUIRED_ALERTS

        return result

    async def _validate_monitoring_configs(self) -> dict[str, Any]:
        """Validate monitoring configuration files exist and are valid.

        Returns:
            Validation result for configuration files
        """
        result = {
            "passed": False,
            "message": "",
            "configs_found": [],
            "configs_missing": [],
            "config_errors": [],
        }

        required_configs = {
            "prometheus.yml": self.genesis_root
            / "config"
            / "prometheus"
            / "prometheus.yml",
            "grafana.ini": self.genesis_root / "config" / "grafana" / "grafana.ini",
            "alerts.yml": self.genesis_root / "config" / "prometheus" / "alerts.yml",
        }

        for config_name, config_path in required_configs.items():
            if config_path.exists():
                result["configs_found"].append(config_name)

                # Validate YAML files
                if config_name.endswith(".yml"):
                    try:
                        with open(config_path) as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        result["config_errors"].append(
                            f"{config_name}: Invalid YAML - {e!s}"
                        )
            else:
                result["configs_missing"].append(config_name)

        if not result["configs_missing"] and not result["config_errors"]:
            result["passed"] = True
            result["message"] = "All monitoring configuration files present and valid"
        else:
            issues = []
            if result["configs_missing"]:
                issues.append(f"{len(result['configs_missing'])} missing configs")
            if result["config_errors"]:
                issues.append(f"{len(result['config_errors'])} config errors")
            result["message"] = f"Configuration issues: {', '.join(issues)}"

        return result

    def generate_report(self, result: ValidationResult | None = None) -> str:
        """Generate a detailed monitoring validation report.

        Args:
            result: ValidationResult to generate report from

        Returns:
            Formatted report string
        """
        if not result:
            return "No validation results available. Run run_validation() first."

        report = []
        report.append("=" * 80)
        report.append("MONITORING VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validator: {result.validator_name} ({result.validator_id})")
        report.append(f"Status: {result.status.value.upper()}")
        report.append(f"Score: {result.score}%")
        report.append(f"Summary: {result.message}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)

        for check in result.checks:
            status_icon = (
                "✓"
                if check.status == CheckStatus.PASSED
                else "✗" if check.status == CheckStatus.FAILED else "⚠"
            )
            report.append(f"{status_icon} [{check.id}] {check.name}: {check.details}")

            # Add details for non-passing checks
            if check.status != CheckStatus.PASSED:
                if check.remediation:
                    report.append(f"  Remediation: {check.remediation}")
                if check.evidence.metrics.get("metrics_missing"):
                    report.append(
                        f"  Missing metrics: {', '.join(check.evidence.metrics['metrics_missing'])}"
                    )
                if check.evidence.metrics.get("dashboards_missing"):
                    report.append(
                        f"  Missing dashboards: {', '.join(check.evidence.metrics['dashboards_missing'])}"
                    )
                if check.evidence.metrics.get("alerts_missing"):
                    report.append(
                        f"  Missing alerts: {', '.join(check.evidence.metrics['alerts_missing'])}"
                    )
                if check.evidence.metrics.get("configs_missing"):
                    report.append(
                        f"  Missing configs: {', '.join(check.evidence.metrics['configs_missing'])}"
                    )
                if check.evidence.metrics.get("config_errors"):
                    for error in check.evidence.metrics["config_errors"]:
                        report.append(f"  Config error: {error}")

        report.append("")
        report.append(f"Total Checks: {len(result.checks)}")
        report.append(
            f"Passed: {result.passed_checks}, Failed: {result.failed_checks}, Warnings: {result.warning_checks}"
        )
        if result.metadata:
            report.append(
                f"Execution Time: {result.metadata.duration_ms / 1000:.2f} seconds"
            )
        report.append("=" * 80)

        return "\n".join(report)
