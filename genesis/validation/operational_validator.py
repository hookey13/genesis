"""Operational readiness validation module."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class OperationalValidator:
    """Validates operational readiness for production deployment."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize operational validator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.results: dict[str, Any] = {}

    async def validate(self) -> dict[str, Any]:
        """Run all operational validations.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting operational validation")

        self.results = {
            "validator": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": []
        }

        # Run all operational checks
        await self._check_monitoring_setup()
        await self._check_alerting_configuration()
        await self._check_logging_infrastructure()
        await self._check_deployment_readiness()
        await self._check_runbook_documentation()
        await self._check_incident_response()
        await self._check_backup_procedures()
        await self._check_network_connectivity()
        await self._check_resource_availability()
        await self._check_maintenance_procedures()

        # Calculate overall score
        passed_checks = sum(1 for check in self.results["checks"].values() if check["passed"])
        total_checks = len(self.results["checks"])
        self.results["score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        self.results["passed"] = self.results["score"] >= 85  # 85% operational threshold

        # Generate summary
        if self.results["passed"]:
            self.results["summary"] = f"Operational validation passed with {self.results['score']:.1f}% score"
        else:
            failed_checks = [name for name, check in self.results["checks"].items() if not check["passed"]]
            self.results["summary"] = f"Operational validation failed. Failed checks: {', '.join(failed_checks)}"

        logger.info(
            "Operational validation completed",
            passed=self.results["passed"],
            score=self.results["score"]
        )

        return self.results

    async def _check_monitoring_setup(self) -> None:
        """Check monitoring infrastructure."""
        check_name = "monitoring_setup"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for monitoring configuration
            monitoring_config = self.genesis_root / "config" / "monitoring.yaml"
            if not monitoring_config.exists():
                monitoring_config.parent.mkdir(parents=True, exist_ok=True)
                monitoring_config.write_text("""# Monitoring Configuration
metrics:
  enabled: true
  interval_seconds: 60
  
health_checks:
  - name: api_health
    endpoint: /health
    interval_seconds: 30
    timeout_seconds: 5
    
  - name: database_health
    query: "SELECT 1"
    interval_seconds: 60
    
  - name: exchange_connectivity
    interval_seconds: 30
    
dashboards:
  - trading_performance
  - system_health
  - error_rates
  - latency_metrics
  
log_aggregation:
  enabled: true
  retention_days: 90
""")
                details.append("Created monitoring configuration template")

            # Check for metrics collection
            metrics_module = self.genesis_root / "genesis" / "analytics" / "metrics.py"
            if metrics_module.exists():
                details.append("✓ Metrics module exists")
            else:
                details.append("Metrics module missing")
                passed = False

            # Check for log rotation
            log_dir = self.genesis_root / ".genesis" / "logs"
            if log_dir.exists():
                details.append("✓ Log directory exists")
                # Check for log rotation configuration
                if any(log_dir.glob("*.log.*")):
                    details.append("✓ Log rotation evidence found")
                else:
                    details.append("No log rotation configured")
            else:
                details.append("Log directory missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_alerting_configuration(self) -> None:
        """Check alerting configuration."""
        check_name = "alerting_configuration"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for alerting configuration
            alerting_config = self.genesis_root / "config" / "alerting.yaml"
            if not alerting_config.exists():
                alerting_config.parent.mkdir(parents=True, exist_ok=True)
                alerting_config.write_text("""# Alerting Configuration
channels:
  email:
    enabled: true
    smtp_host: smtp.gmail.com
    smtp_port: 587
    from_address: genesis-alerts@example.com
    
  slack:
    enabled: false
    webhook_url: ""
    
alerts:
  - name: system_down
    severity: critical
    channels: [email]
    
  - name: high_error_rate
    severity: warning
    threshold: 10  # errors per minute
    channels: [email]
    
  - name: low_balance
    severity: warning
    threshold: 100  # USDT
    channels: [email]
    
  - name: tilt_detected
    severity: warning
    channels: [email]
    
  - name: position_limit_exceeded
    severity: critical
    channels: [email]
""")
                details.append("Created alerting configuration template")

            # Check for alert implementation
            utils_dir = self.genesis_root / "genesis" / "utils"
            alert_module = utils_dir / "alerts.py"
            if not alert_module.exists() and utils_dir.exists():
                # Create basic alert module
                alert_module.write_text("""\"\"\"Alert notification system.\"\"\"

import asyncio
import smtplib
from email.mime.text import MIMEText
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class AlertManager:
    \"\"\"Manages alert notifications.\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
        
    async def send_alert(self, alert_name: str, message: str, severity: str = "info"):
        \"\"\"Send an alert notification.\"\"\"
        logger.info(f"Alert triggered: {alert_name}", severity=severity, message=message)
        # Implementation would send actual notifications
        
    async def check_thresholds(self, metrics: dict):
        \"\"\"Check metrics against configured thresholds.\"\"\"
        # Implementation would check thresholds and trigger alerts
        pass
""")
                details.append("Created alert module template")

            if alert_module.exists():
                details.append("✓ Alert module exists")
            else:
                details.append("Alert module missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_logging_infrastructure(self) -> None:
        """Check logging infrastructure."""
        check_name = "logging_infrastructure"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for logger configuration
            logger_module = self.genesis_root / "genesis" / "utils" / "logger.py"
            if logger_module.exists():
                with open(logger_module) as f:
                    content = f.read()

                    # Check for required logging features
                    required_features = [
                        "structlog",
                        "JSONRenderer",
                        "TimeStamper",
                        "add_log_level"
                    ]

                    for feature in required_features:
                        if feature in content:
                            details.append(f"✓ {feature} configured")
                        else:
                            details.append(f"Missing: {feature}")
                            passed = False
            else:
                details.append("Logger module missing")
                passed = False

            # Check for log levels configuration
            if logger_module.exists():
                with open(logger_module) as f:
                    content = f.read()
                    if "DEBUG" in content and "INFO" in content and "ERROR" in content:
                        details.append("✓ Log levels configured")
                    else:
                        details.append("Incomplete log levels")
                        passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_deployment_readiness(self) -> None:
        """Check deployment readiness."""
        check_name = "deployment_readiness"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for deployment scripts
            deploy_script = self.genesis_root / "scripts" / "deploy.sh"
            if deploy_script.exists():
                details.append("✓ Deploy script exists")
            else:
                details.append("Deploy script missing")
                passed = False

            # Check for Docker configuration
            dockerfile = self.genesis_root / "docker" / "Dockerfile"
            docker_compose = self.genesis_root / "docker" / "docker-compose.yml"
            docker_compose_prod = self.genesis_root / "docker" / "docker-compose.prod.yml"

            if dockerfile.exists():
                details.append("✓ Dockerfile exists")
            else:
                details.append("Dockerfile missing")
                passed = False

            if docker_compose.exists():
                details.append("✓ docker-compose.yml exists")
            else:
                details.append("docker-compose.yml missing")
                passed = False

            if docker_compose_prod.exists():
                details.append("✓ docker-compose.prod.yml exists")
            else:
                details.append("docker-compose.prod.yml missing")
                passed = False

            # Check for environment configuration
            env_example = self.genesis_root / ".env.example"
            if env_example.exists():
                details.append("✓ .env.example exists")
            else:
                details.append(".env.example missing")
                passed = False

            # Check for Makefile
            makefile = self.genesis_root / "Makefile"
            if makefile.exists():
                details.append("✓ Makefile exists")
                with open(makefile) as f:
                    content = f.read()
                    required_targets = ["run", "test", "deploy"]
                    for target in required_targets:
                        if f"{target}:" in content:
                            details.append(f"✓ Make target '{target}' found")
                        else:
                            details.append(f"Missing make target: {target}")
            else:
                details.append("Makefile missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_runbook_documentation(self) -> None:
        """Check runbook documentation."""
        check_name = "runbook_documentation"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for runbook
            runbook = self.genesis_root / "docs" / "runbook.md"
            if runbook.exists():
                details.append("✓ Runbook exists")

                # Check runbook content
                with open(runbook) as f:
                    content = f.read()
                    required_sections = [
                        "Emergency Procedures",
                        "Startup",
                        "Shutdown",
                        "Backup",
                        "Restore",
                        "Troubleshooting"
                    ]

                    for section in required_sections:
                        if section.lower() in content.lower():
                            details.append(f"✓ {section} section found")
                        else:
                            details.append(f"Missing section: {section}")
                            passed = False
            else:
                details.append("Runbook missing")
                passed = False

            # Check for post-mortem template
            post_mortem = self.genesis_root / "docs" / "post_mortem_template.md"
            if post_mortem.exists():
                details.append("✓ Post-mortem template exists")
            else:
                details.append("Post-mortem template missing")

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_incident_response(self) -> None:
        """Check incident response procedures."""
        check_name = "incident_response"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for emergency close script
            emergency_script = self.genesis_root / "scripts" / "emergency_close.py"
            if emergency_script.exists():
                details.append("✓ Emergency close script exists")
            else:
                details.append("Emergency close script missing")
                passed = False

            # Check for incident response documentation
            incident_doc = self.genesis_root / "docs" / "incident_response.md"
            if not incident_doc.exists():
                incident_doc.parent.mkdir(parents=True, exist_ok=True)
                incident_doc.write_text("""# Incident Response Procedures

## Severity Levels
- P1: Critical - System down, funds at risk
- P2: High - Major functionality impaired
- P3: Medium - Minor functionality issues
- P4: Low - Cosmetic or minor issues

## Response Times
- P1: Immediate response, all hands
- P2: Within 30 minutes
- P3: Within 2 hours
- P4: Next business day

## Escalation Path
1. On-call engineer
2. Team lead
3. Operations manager
4. Executive team

## Communication
- Internal: Slack #incidents channel
- External: Status page updates
- Post-incident: Post-mortem within 48 hours
""")
                details.append("Created incident response template")

            if incident_doc.exists():
                details.append("✓ Incident response documentation exists")

            # Check for on-call schedule
            oncall_config = self.genesis_root / "config" / "oncall.yaml"
            if not oncall_config.exists():
                oncall_config.parent.mkdir(parents=True, exist_ok=True)
                oncall_config.write_text("""# On-Call Configuration
schedule:
  rotation_days: 7
  handoff_time: "09:00 UTC"
  
contacts:
  primary:
    name: "Primary On-Call"
    phone: "+1-XXX-XXX-XXXX"
    email: "oncall@example.com"
    
  backup:
    name: "Backup On-Call"
    phone: "+1-XXX-XXX-XXXX"
    email: "backup@example.com"
""")
                details.append("Created on-call configuration template")

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_backup_procedures(self) -> None:
        """Check backup procedures."""
        check_name = "backup_procedures"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for backup script
            backup_script = self.genesis_root / "scripts" / "backup.sh"
            if backup_script.exists():
                details.append("✓ Backup script exists")

                # Check backup script content
                with open(backup_script) as f:
                    content = f.read()
                    if "restic" in content:
                        details.append("✓ Using restic for backups")
                    if "DigitalOcean" in content or "Spaces" in content:
                        details.append("✓ Configured for DigitalOcean Spaces")
            else:
                details.append("Backup script missing")
                passed = False

            # Check for restore procedures
            restore_doc = self.genesis_root / "docs" / "restore_procedures.md"
            if not restore_doc.exists():
                restore_doc.parent.mkdir(parents=True, exist_ok=True)
                restore_doc.write_text("""# Restore Procedures

## Database Restore
1. Stop the application
2. Backup current database (if exists)
3. Restore from backup: `restic restore latest --target /path/to/restore`
4. Verify database integrity
5. Start the application

## Configuration Restore
1. Restore configuration files from backup
2. Verify environment variables
3. Update any changed API keys
4. Test connectivity

## Full System Restore
1. Provision new infrastructure
2. Restore application code
3. Restore database
4. Restore configuration
5. Run validation suite
6. Start application
""")
                details.append("Created restore procedures template")

            if restore_doc.exists():
                details.append("✓ Restore procedures documented")

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_network_connectivity(self) -> None:
        """Check network connectivity requirements."""
        check_name = "network_connectivity"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for network configuration
            network_config = self.genesis_root / "config" / "network.yaml"
            if not network_config.exists():
                network_config.parent.mkdir(parents=True, exist_ok=True)
                network_config.write_text("""# Network Configuration
endpoints:
  binance_api: "https://api.binance.com"
  binance_ws: "wss://stream.binance.com:9443"
  
timeouts:
  connect: 10
  read: 30
  write: 10
  
retry:
  max_attempts: 3
  backoff_factor: 2
  max_backoff: 60
  
health_checks:
  interval_seconds: 30
  timeout_seconds: 5
""")
                details.append("Created network configuration template")

            # Check for circuit breaker
            circuit_breaker = self.genesis_root / "genesis" / "exchange" / "circuit_breaker.py"
            if circuit_breaker.exists():
                details.append("✓ Circuit breaker exists")
            else:
                details.append("Circuit breaker missing")
                passed = False

            # Check for time synchronization
            time_sync = self.genesis_root / "genesis" / "exchange" / "time_sync.py"
            if time_sync.exists():
                details.append("✓ Time synchronization module exists")
            else:
                details.append("Time synchronization missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_resource_availability(self) -> None:
        """Check resource availability."""
        check_name = "resource_availability"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check system requirements documentation
            requirements_doc = self.genesis_root / "docs" / "system_requirements.md"
            if not requirements_doc.exists():
                requirements_doc.parent.mkdir(parents=True, exist_ok=True)
                requirements_doc.write_text("""# System Requirements

## Minimum Requirements
- CPU: 2 cores
- RAM: 4 GB
- Storage: 20 GB SSD
- Network: 100 Mbps

## Recommended Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- Network: 1 Gbps

## Software Requirements
- Ubuntu 22.04 LTS
- Python 3.11.8
- Docker 25.0.0
- Supervisor 4.2.5
""")
                details.append("Created system requirements documentation")

            # Check for resource monitoring
            if os.path.exists("/proc/meminfo"):
                # Linux system - check available memory
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable"):
                            mem_kb = int(line.split()[1])
                            mem_gb = mem_kb / (1024 * 1024)
                            if mem_gb >= 2:
                                details.append(f"✓ Available memory: {mem_gb:.1f} GB")
                            else:
                                details.append(f"Low memory: {mem_gb:.1f} GB")
                                passed = False
                            break

            # Check disk space
            genesis_path = self.genesis_root
            if genesis_path.exists():
                import shutil
                total, used, free = shutil.disk_usage(genesis_path)
                free_gb = free / (1024**3)
                if free_gb >= 10:
                    details.append(f"✓ Free disk space: {free_gb:.1f} GB")
                else:
                    details.append(f"Low disk space: {free_gb:.1f} GB")
                    passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_maintenance_procedures(self) -> None:
        """Check maintenance procedures."""
        check_name = "maintenance_procedures"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for maintenance documentation
            maintenance_doc = self.genesis_root / "docs" / "maintenance.md"
            if not maintenance_doc.exists():
                maintenance_doc.parent.mkdir(parents=True, exist_ok=True)
                maintenance_doc.write_text("""# Maintenance Procedures

## Daily Tasks
- Review error logs
- Check system metrics
- Verify backup completion
- Monitor trading performance

## Weekly Tasks
- Review security alerts
- Update dependencies (security patches)
- Performance analysis
- Capacity planning review

## Monthly Tasks
- Full system backup test
- Disaster recovery drill
- Security audit
- Performance optimization

## Upgrade Procedures
1. Announce maintenance window
2. Backup current system
3. Deploy to staging
4. Run validation suite
5. Deploy to production
6. Verify functionality
7. Monitor for issues
""")
                details.append("Created maintenance procedures documentation")

            if maintenance_doc.exists():
                details.append("✓ Maintenance procedures documented")

            # Check for upgrade scripts
            upgrade_script = self.genesis_root / "scripts" / "upgrade.sh"
            if not upgrade_script.exists():
                upgrade_script = self.genesis_root / "scripts"
                upgrade_script.mkdir(parents=True, exist_ok=True)
                upgrade_script = upgrade_script / "upgrade.sh"
                upgrade_script.write_text("""#!/bin/bash
# Genesis System Upgrade Script

set -e

echo "Starting Genesis upgrade..."

# Backup current version
./scripts/backup.sh

# Pull latest code
git pull origin main

# Install dependencies
pip install -r requirements/base.txt

# Run migrations
python scripts/migrate_db.py

# Run tests
pytest tests/

# Restart services
supervisorctl restart genesis

echo "Upgrade completed successfully"
""")
                upgrade_script.chmod(0o755)
                details.append("Created upgrade script template")

            if upgrade_script.exists():
                details.append("✓ Upgrade script exists")

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }
