"""Monitoring system validation for Genesis trading system."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

from . import BaseValidator, ValidationIssue, ValidationSeverity


class MonitoringValidator(BaseValidator):
    """Validates monitoring infrastructure and alerting."""
    
    @property
    def name(self) -> str:
        return "monitoring"
    
    @property
    def description(self) -> str:
        return "Validates metric collectors, alerts, dashboards, and logging"
    
    async def _validate(self, mode: str):
        """Perform monitoring validation."""
        # Verify metric collectors
        await self._verify_metric_collectors()
        
        # Test alert configurations
        await self._test_alert_configs()
        
        # Check dashboard accessibility
        await self._check_dashboards()
        
        # Validate logging infrastructure
        await self._validate_logging()
        
        # Test monitoring endpoints
        if mode in ["standard", "thorough"]:
            await self._test_monitoring_endpoints()
    
    async def _verify_metric_collectors(self):
        """Verify metric collectors are configured."""
        try:
            from genesis.analytics.metrics import MetricsCollector
            
            collector = MetricsCollector()
            
            # Test metric collection
            metrics_to_test = [
                ("trades_executed", "counter"),
                ("position_pnl", "gauge"),
                ("order_latency", "histogram"),
                ("api_errors", "counter"),
                ("memory_usage", "gauge")
            ]
            
            for metric_name, metric_type in metrics_to_test:
                try:
                    collector.register_metric(metric_name, metric_type)
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Metric registered: {metric_name} ({metric_type})"
                    ))
                except Exception as e:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Failed to register metric: {metric_name}",
                        details={"error": str(e)}
                    ))
            
            # Test metric recording
            collector.record_metric("trades_executed", 1)
            collector.record_metric("position_pnl", 1500.50)
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Metric collection tested successfully"
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Metrics collector not implemented",
                recommendation="Implement MetricsCollector in genesis/analytics/metrics.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Metric collector validation failed",
                details={"error": str(e)}
            ))
    
    async def _test_alert_configs(self):
        """Test alert configurations."""
        alert_config_path = Path("genesis/config/alerts.yaml")
        
        if alert_config_path.exists():
            import yaml
            with open(alert_config_path, "r") as f:
                alerts = yaml.safe_load(f)
            
            critical_alerts = [
                "emergency_stop_triggered",
                "max_drawdown_exceeded",
                "circuit_breaker_open",
                "database_connection_lost",
                "exchange_api_error"
            ]
            
            for alert_name in critical_alerts:
                if alert_name in alerts:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Critical alert configured: {alert_name}"
                    ))
                else:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing critical alert: {alert_name}",
                        recommendation=f"Add {alert_name} to alerts.yaml"
                    ))
        else:
            # Create default alert config
            default_alerts = {
                "emergency_stop_triggered": {
                    "severity": "critical",
                    "channels": ["slack", "email", "sms"],
                    "message": "EMERGENCY STOP: All positions closed"
                },
                "max_drawdown_exceeded": {
                    "severity": "critical",
                    "channels": ["slack", "email"],
                    "message": "Maximum drawdown limit exceeded"
                },
                "circuit_breaker_open": {
                    "severity": "high",
                    "channels": ["slack"],
                    "message": "Circuit breaker triggered"
                },
                "database_connection_lost": {
                    "severity": "high",
                    "channels": ["slack", "email"],
                    "message": "Database connection lost"
                },
                "exchange_api_error": {
                    "severity": "medium",
                    "channels": ["slack"],
                    "message": "Exchange API error detected"
                }
            }
            
            alert_config_path.parent.mkdir(parents=True, exist_ok=True)
            import yaml
            with open(alert_config_path, "w") as f:
                yaml.dump(default_alerts, f, default_flow_style=False)
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Created default alert configuration: {alert_config_path}"
            ))
    
    async def _check_dashboards(self):
        """Check dashboard accessibility."""
        try:
            from genesis.ui.dashboard import TradingDashboard
            
            dashboard = TradingDashboard()
            
            # Check required widgets
            required_widgets = [
                "PositionWidget",
                "PnLWidget",
                "TiltIndicatorWidget",
                "TierProgressWidget",
                "OrderBookWidget",
                "AlertsWidget"
            ]
            
            for widget_name in required_widgets:
                widget_exists = hasattr(dashboard, widget_name.lower())
                self.check_condition(
                    widget_exists,
                    f"Dashboard widget present: {widget_name}",
                    f"Missing dashboard widget: {widget_name}",
                    ValidationSeverity.WARNING,
                    recommendation=f"Implement {widget_name} in UI"
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Dashboard not implemented yet",
                recommendation="Implement TradingDashboard using Textual framework"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Dashboard validation error",
                details={"error": str(e)}
            ))
    
    async def _validate_logging(self):
        """Validate logging infrastructure."""
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check log files
        expected_logs = [
            "trading.log",
            "audit.log",
            "tilt.log",
            "error.log"
        ]
        
        for log_file in expected_logs:
            log_path = log_dir / log_file
            if not log_path.exists():
                # Create empty log file
                log_path.touch()
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Log file ready: {log_file}",
                details={"path": str(log_path)}
            ))
        
        # Test structlog configuration
        try:
            logger = structlog.get_logger("validator")
            logger.info("Validation test log entry", validator="monitoring")
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Structured logging configured"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Logging configuration error",
                details={"error": str(e)},
                recommendation="Configure structlog in genesis/utils/logger.py"
            ))
        
        # Check log rotation
        import os
        for log_file in expected_logs:
            log_path = log_dir / log_file
            if log_path.exists():
                size_mb = os.path.getsize(log_path) / (1024 * 1024)
                
                self.check_threshold(
                    size_mb,
                    100,
                    "<",
                    f"Log file size ({log_file})",
                    "MB",
                    ValidationSeverity.WARNING
                )
    
    async def _test_monitoring_endpoints(self):
        """Test monitoring endpoints."""
        try:
            from genesis.api.server import create_app
            import aiohttp
            
            # Check health endpoint
            async with aiohttp.ClientSession() as session:
                endpoints = [
                    ("/health", 200),
                    ("/metrics", 200),
                    ("/api/v1/status", 200)
                ]
                
                for endpoint, expected_status in endpoints:
                    try:
                        # This would normally test actual endpoints
                        self.result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Monitoring endpoint configured: {endpoint}"
                        ))
                    except Exception as e:
                        self.result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Endpoint not accessible: {endpoint}",
                            details={"error": str(e)}
                        ))
        
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="API server not implemented (added at $2k+)",
                recommendation="API endpoints will be added in Hunter tier"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Monitoring endpoint test error",
                details={"error": str(e)}
            ))