"""Monitoring and observability validation module."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import structlog
import yaml

logger = structlog.get_logger(__name__)


class MonitoringValidator:
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
        self.genesis_root = genesis_root or Path.cwd()
        self.results: Dict[str, Any] = {}
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"

    async def validate(self) -> Dict[str, Any]:
        """Run monitoring validation checks.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting monitoring validation")
        start_time = datetime.utcnow()

        self.results = {
            "validator": "monitoring",
            "timestamp": start_time.isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": [],
        }

        # Check Prometheus metrics
        metrics_result = await self._validate_prometheus_metrics()
        self.results["checks"]["prometheus_metrics"] = metrics_result

        # Check Grafana dashboards
        dashboard_result = await self._validate_grafana_dashboards()
        self.results["checks"]["grafana_dashboards"] = dashboard_result

        # Check alert rules
        alerts_result = await self._validate_alert_rules()
        self.results["checks"]["alert_rules"] = alerts_result

        # Check monitoring configuration files
        config_result = await self._validate_monitoring_configs()
        self.results["checks"]["monitoring_configs"] = config_result

        # Calculate overall score
        total_checks = len(self.results["checks"])
        passed_checks = sum(
            1 for check in self.results["checks"].values() if check.get("passed", False)
        )
        self.results["score"] = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0

        # Determine overall status
        if all(check.get("passed", False) for check in self.results["checks"].values()):
            self.results["passed"] = True
            self.results["summary"] = "Monitoring fully configured and operational"
        else:
            self.results["passed"] = False
            self.results["summary"] = "Monitoring gaps detected - review failed checks"

        # Add execution time
        self.results["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

        return self.results

    async def _validate_prometheus_metrics(self) -> Dict[str, Any]:
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
                async with session.get(f"{self.prometheus_url}/api/v1/label/__name__/values") as resp:
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
                            result["message"] = f"All {len(self.REQUIRED_METRICS)} required metrics configured"
                        else:
                            result["message"] = f"Missing {len(result['metrics_missing'])} required metrics"
                    else:
                        result["message"] = f"Prometheus API returned status {resp.status}"
        except aiohttp.ClientError as e:
            result["message"] = f"Failed to connect to Prometheus: {str(e)}"
            result["details"] = "Ensure Prometheus is running and accessible"
        except Exception as e:
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    async def _validate_grafana_dashboards(self) -> Dict[str, Any]:
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
                    with open(dashboard_file, "r") as f:
                        dashboard_data = json.load(f)
                        dashboard_uid = dashboard_data.get("uid", "")
                        dashboard_title = dashboard_data.get("title", "")
                        
                        # Check if this matches a required dashboard
                        for required in self.REQUIRED_DASHBOARDS:
                            if required in dashboard_uid.lower() or required in dashboard_title.lower():
                                found_dashboards.append(required)
                                break
                except Exception as e:
                    logger.warning(f"Failed to parse dashboard {dashboard_file}: {e}")
            
            result["dashboards_found"] = list(set(found_dashboards))
            result["dashboards_missing"] = [
                d for d in self.REQUIRED_DASHBOARDS if d not in result["dashboards_found"]
            ]
            
            if not result["dashboards_missing"]:
                result["passed"] = True
                result["message"] = f"All {len(self.REQUIRED_DASHBOARDS)} required dashboards configured"
            else:
                result["message"] = f"Missing {len(result['dashboards_missing'])} required dashboards"
        else:
            result["message"] = "Grafana configuration directory not found"
            result["dashboards_missing"] = self.REQUIRED_DASHBOARDS

        return result

    async def _validate_alert_rules(self) -> Dict[str, Any]:
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
        prometheus_rules_file = self.genesis_root / "config" / "prometheus" / "alerts.yml"
        if prometheus_rules_file.exists():
            try:
                with open(prometheus_rules_file, "r") as f:
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
                        result["message"] = f"All {len(self.REQUIRED_ALERTS)} required alerts configured"
                    else:
                        result["message"] = f"Missing {len(result['alerts_missing'])} required alerts"
            except Exception as e:
                result["message"] = f"Failed to parse alert rules: {str(e)}"
        else:
            result["message"] = "Alert rules configuration file not found"
            result["alerts_missing"] = self.REQUIRED_ALERTS

        return result

    async def _validate_monitoring_configs(self) -> Dict[str, Any]:
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
            "prometheus.yml": self.genesis_root / "config" / "prometheus" / "prometheus.yml",
            "grafana.ini": self.genesis_root / "config" / "grafana" / "grafana.ini",
            "alerts.yml": self.genesis_root / "config" / "prometheus" / "alerts.yml",
        }

        for config_name, config_path in required_configs.items():
            if config_path.exists():
                result["configs_found"].append(config_name)
                
                # Validate YAML files
                if config_name.endswith(".yml"):
                    try:
                        with open(config_path, "r") as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        result["config_errors"].append(f"{config_name}: Invalid YAML - {str(e)}")
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

    def generate_report(self) -> str:
        """Generate a detailed monitoring validation report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run validate() first."

        report = []
        report.append("=" * 80)
        report.append("MONITORING VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Status: {'PASSED' if self.results['passed'] else 'FAILED'}")
        report.append(f"Score: {self.results['score']}%")
        report.append(f"Summary: {self.results['summary']}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)
        
        for check_name, check_result in self.results["checks"].items():
            status = "✓" if check_result.get("passed", False) else "✗"
            report.append(f"{status} {check_name}: {check_result.get('message', '')}")
            
            # Add details for failed checks
            if not check_result.get("passed", False):
                if check_result.get("metrics_missing"):
                    report.append(f"  Missing metrics: {', '.join(check_result['metrics_missing'])}")
                if check_result.get("dashboards_missing"):
                    report.append(f"  Missing dashboards: {', '.join(check_result['dashboards_missing'])}")
                if check_result.get("alerts_missing"):
                    report.append(f"  Missing alerts: {', '.join(check_result['alerts_missing'])}")
                if check_result.get("configs_missing"):
                    report.append(f"  Missing configs: {', '.join(check_result['configs_missing'])}")
                if check_result.get("config_errors"):
                    for error in check_result["config_errors"]:
                        report.append(f"  Config error: {error}")

        report.append("")
        report.append(f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds")
        report.append("=" * 80)

        return "\n".join(report)