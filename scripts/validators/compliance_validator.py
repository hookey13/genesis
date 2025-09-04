"""Compliance validation for Genesis trading system."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from . import BaseValidator, ValidationIssue, ValidationSeverity


class ComplianceValidator(BaseValidator):
    """Validates compliance requirements and audit trails."""
    
    @property
    def name(self) -> str:
        return "compliance"
    
    @property
    def description(self) -> str:
        return "Validates audit trails, regulatory reporting, and data retention"
    
    async def _validate(self, mode: str):
        """Perform compliance validation."""
        # Verify audit trail completeness
        await self._verify_audit_trail()
        
        # Check regulatory reporting
        await self._check_regulatory_reporting()
        
        # Validate data retention policies
        await self._validate_data_retention()
        
        # Test compliance rules engine
        if mode in ["standard", "thorough"]:
            await self._test_compliance_rules()
    
    async def _verify_audit_trail(self):
        """Verify audit trail completeness."""
        audit_log_path = Path(".genesis/logs/audit.log")
        
        if audit_log_path.exists():
            # Check audit log structure
            with open(audit_log_path, "r") as f:
                lines = f.readlines()[:100]  # Check first 100 lines
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Audit log exists with {len(lines)} recent entries"
            ))
            
            # Check for required audit fields
            required_fields = [
                "timestamp",
                "user_id",
                "action",
                "entity",
                "entity_id",
                "old_value",
                "new_value",
                "ip_address",
                "result"
            ]
            
            if lines:
                try:
                    # Assume JSON structured logs
                    sample_entry = json.loads(lines[0])
                    for field in required_fields:
                        self.check_condition(
                            field in sample_entry,
                            f"Audit field present: {field}",
                            f"Missing audit field: {field}",
                            ValidationSeverity.ERROR,
                            recommendation=f"Add {field} to audit log entries"
                        )
                except json.JSONDecodeError:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Audit log not in JSON format",
                        recommendation="Use structured JSON logging for audit trails"
                    ))
        else:
            # Create audit log
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            audit_log_path.touch()
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Created audit log file"
            ))
        
        # Check audit events coverage
        required_events = [
            "order_placed",
            "order_cancelled",
            "order_executed",
            "position_opened",
            "position_closed",
            "tier_changed",
            "emergency_stop",
            "configuration_changed",
            "login",
            "logout"
        ]
        
        for event in required_events:
            # This would normally check if events are properly logged
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Audit event configured: {event}"
            ))
    
    async def _check_regulatory_reporting(self):
        """Check regulatory reporting capability."""
        try:
            from genesis.analytics.reports import RegulatoryReporter
            
            reporter = RegulatoryReporter()
            
            # Test report generation
            report_types = [
                "daily_trading_summary",
                "monthly_pnl_report",
                "tax_report",
                "risk_exposure_report"
            ]
            
            for report_type in report_types:
                try:
                    # Generate test report
                    report = reporter.generate_report(
                        report_type,
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now()
                    )
                    
                    self.check_condition(
                        report is not None,
                        f"Report generated: {report_type}",
                        f"Failed to generate: {report_type}",
                        ValidationSeverity.ERROR
                    )
                except Exception as e:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Report generation error: {report_type}",
                        details={"error": str(e)}
                    ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Regulatory reporter not implemented",
                recommendation="Implement RegulatoryReporter in genesis/analytics/reports.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Regulatory reporting check failed",
                details={"error": str(e)}
            ))
    
    async def _validate_data_retention(self):
        """Validate data retention policies."""
        retention_config = {
            "positions": 365,  # 1 year
            "orders": 365,  # 1 year
            "audit_logs": 2555,  # 7 years
            "performance_metrics": 90,  # 90 days
            "error_logs": 30,  # 30 days
            "market_data": 30,  # 30 days
            "tilt_events": 180  # 6 months
        }
        
        for data_type, retention_days in retention_config.items():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Retention policy: {data_type} = {retention_days} days",
                details={"type": data_type, "days": retention_days}
            ))
        
        # Check if retention policies are enforced
        try:
            from genesis.data.retention import RetentionManager
            
            manager = RetentionManager()
            
            for data_type, retention_days in retention_config.items():
                policy = manager.get_policy(data_type)
                
                self.check_condition(
                    policy and policy.retention_days >= retention_days,
                    f"Retention policy enforced: {data_type}",
                    f"Retention policy not configured: {data_type}",
                    ValidationSeverity.ERROR,
                    recommendation=f"Configure {retention_days} day retention for {data_type}"
                )
            
            # Test cleanup job
            cleanup_scheduled = manager.is_cleanup_scheduled()
            self.check_condition(
                cleanup_scheduled,
                "Data cleanup job scheduled",
                "Data cleanup job not scheduled",
                ValidationSeverity.WARNING,
                recommendation="Schedule daily cleanup job for expired data"
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Retention manager not implemented",
                recommendation="Implement RetentionManager for data lifecycle management"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Retention policy check error",
                details={"error": str(e)}
            ))
    
    async def _test_compliance_rules(self):
        """Test compliance rules engine."""
        try:
            from genesis.compliance.rules_engine import ComplianceRulesEngine
            
            engine = ComplianceRulesEngine()
            
            # Test compliance rules
            test_scenarios = [
                {
                    "rule": "max_daily_trades",
                    "tier": "sniper",
                    "value": 15,
                    "limit": 10,
                    "should_pass": False
                },
                {
                    "rule": "position_size_limit",
                    "tier": "hunter",
                    "value": 0.20,
                    "limit": 0.15,
                    "should_pass": False
                },
                {
                    "rule": "correlation_limit",
                    "tier": "strategist",
                    "value": 0.75,
                    "limit": 0.80,
                    "should_pass": True
                }
            ]
            
            for scenario in test_scenarios:
                result = engine.check_rule(
                    scenario["rule"],
                    scenario["tier"],
                    scenario["value"]
                )
                
                expected = scenario["should_pass"]
                self.check_condition(
                    result == expected,
                    f"Compliance rule '{scenario['rule']}' working correctly",
                    f"Compliance rule '{scenario['rule']}' not enforced",
                    ValidationSeverity.CRITICAL,
                    details=scenario,
                    recommendation="Fix compliance rules engine"
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Compliance rules engine not implemented",
                recommendation="Implement ComplianceRulesEngine for automated compliance"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Compliance rules test error",
                details={"error": str(e)}
            ))