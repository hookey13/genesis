"""Go-live decision engine with automated deployment triggers."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from genesis.validation.orchestrator import ValidationCheck, ValidationReport

logger = structlog.get_logger(__name__)


class DeploymentTarget(Enum):
    """Deployment target environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class Override:
    """Manual override for go-live decision."""
    reason: str
    authorized_by: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    authorization_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reason": self.reason,
            "authorized_by": self.authorized_by,
            "timestamp": self.timestamp.isoformat(),
            "authorization_code": self.authorization_code
        }


@dataclass
class GoLiveDecision:
    """Go-live readiness decision."""
    ready: bool
    score: float
    pipeline_name: str
    blocking_issues: list[ValidationCheck]
    warnings: list[ValidationCheck]
    override: Override | None = None
    deployment_target: DeploymentTarget | None = None
    deployment_allowed: bool = False
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ready": self.ready,
            "score": self.score,
            "pipeline_name": self.pipeline_name,
            "blocking_issues": [
                {
                    "name": issue.name,
                    "message": issue.message,
                    "severity": issue.severity
                }
                for issue in self.blocking_issues
            ],
            "warnings": [
                {
                    "name": warning.name,
                    "message": warning.message,
                    "severity": warning.severity
                }
                for warning in self.warnings
            ],
            "override": self.override.to_dict() if self.override else None,
            "deployment_target": self.deployment_target.value if self.deployment_target else None,
            "deployment_allowed": self.deployment_allowed,
            "decision_timestamp": self.decision_timestamp.isoformat()
        }


class DecisionEngine:
    """Makes go/no-go decisions based on validation results."""

    # Minimum scores required for different environments
    MIN_SCORES = {
        DeploymentTarget.DEVELOPMENT: 70.0,
        DeploymentTarget.STAGING: 85.0,
        DeploymentTarget.PRODUCTION: 95.0
    }

    # Authorized users for overrides (in production, this would be from a secure config)
    AUTHORIZED_USERS = {
        "admin": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",  # SHA256 of "12345"
        "lead_dev": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",  # SHA256 of "123"
    }

    def __init__(self, genesis_root: Path | None = None):
        """Initialize decision engine.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.audit_log_path = self.genesis_root / ".genesis" / "audit.log"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def make_decision(
        self,
        report: ValidationReport,
        target: DeploymentTarget = DeploymentTarget.PRODUCTION
    ) -> GoLiveDecision:
        """Make go/no-go decision based on validation report.
        
        Args:
            report: Validation report
            target: Deployment target environment
            
        Returns:
            Go-live decision
        """
        score = report.overall_score
        blocking = self._extract_blocking_issues(report)
        warnings = self._extract_warnings(report)

        # Check minimum score for target
        min_score = self.MIN_SCORES.get(target, 95.0)
        ready = score >= min_score and len(blocking) == 0

        # Determine if deployment is allowed
        deployment_allowed = ready and report.overall_passed

        decision = GoLiveDecision(
            ready=ready,
            score=score,
            pipeline_name=report.pipeline_name,
            blocking_issues=blocking,
            warnings=warnings,
            deployment_target=target,
            deployment_allowed=deployment_allowed
        )

        # Audit log the decision
        self._audit_decision(decision)

        logger.info(
            "Go-live decision made",
            ready=ready,
            score=score,
            target=target.value,
            blocking_count=len(blocking),
            deployment_allowed=deployment_allowed
        )

        return decision

    def apply_override(
        self,
        decision: GoLiveDecision,
        override_reason: str,
        username: str,
        password: str
    ) -> GoLiveDecision:
        """Apply manual override to decision.
        
        Args:
            decision: Original decision
            override_reason: Reason for override
            username: Username requesting override
            password: Password for authorization
            
        Returns:
            Updated decision with override
            
        Raises:
            UnauthorizedError: If user is not authorized
        """
        # Validate authorization
        if not self._validate_authorization(username, password):
            logger.warning(
                "Unauthorized override attempt",
                username=username,
                decision_id=id(decision)
            )
            raise UnauthorizedError(f"User {username} not authorized for override")

        # Create override
        override = Override(
            reason=override_reason,
            authorized_by=username,
            authorization_code=self._generate_auth_code(username, override_reason)
        )

        # Apply override
        decision.override = override
        decision.ready = True
        decision.deployment_allowed = True

        # Audit log the override
        self._audit_override(decision, override)

        logger.info(
            "Override applied to decision",
            username=username,
            reason=override_reason,
            decision_id=id(decision)
        )

        return decision

    def trigger_deployment(
        self,
        decision: GoLiveDecision,
        dry_run: bool = True
    ) -> dict[str, Any]:
        """Trigger deployment based on decision.
        
        Args:
            decision: Go-live decision
            dry_run: If True, simulate deployment without executing
            
        Returns:
            Deployment result
        """
        if not decision.deployment_allowed and not decision.override:
            logger.warning("Deployment not allowed", decision_id=id(decision))
            return {
                "success": False,
                "error": "Deployment not allowed - validation failed",
                "decision": decision.to_dict()
            }

        target = decision.deployment_target or DeploymentTarget.DEVELOPMENT

        logger.info(
            "Triggering deployment",
            target=target.value,
            dry_run=dry_run,
            override=bool(decision.override)
        )

        if dry_run:
            # Simulate deployment
            result = {
                "success": True,
                "dry_run": True,
                "target": target.value,
                "timestamp": datetime.utcnow().isoformat(),
                "decision": decision.to_dict(),
                "message": f"Dry run deployment to {target.value} would succeed"
            }
        else:
            # Execute actual deployment
            result = self._execute_deployment(decision, target)

        # Audit log deployment
        self._audit_deployment(decision, result)

        return result

    def _execute_deployment(
        self,
        decision: GoLiveDecision,
        target: DeploymentTarget
    ) -> dict[str, Any]:
        """Execute actual deployment.
        
        Args:
            decision: Go-live decision
            target: Deployment target
            
        Returns:
            Deployment result
        """
        deployment_scripts = {
            DeploymentTarget.DEVELOPMENT: "scripts/deploy_dev.sh",
            DeploymentTarget.STAGING: "scripts/deploy_staging.sh",
            DeploymentTarget.PRODUCTION: "scripts/deploy.sh"
        }

        script_path = self.genesis_root / deployment_scripts.get(target, "scripts/deploy.sh")

        if not script_path.exists():
            logger.error("Deployment script not found", script=str(script_path))
            return {
                "success": False,
                "error": f"Deployment script not found: {script_path}",
                "target": target.value
            }

        try:
            # Execute deployment script
            result = subprocess.run(
                [str(script_path)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=self.genesis_root
            )

            success = result.returncode == 0

            return {
                "success": success,
                "target": target.value,
                "timestamp": datetime.utcnow().isoformat(),
                "decision": decision.to_dict(),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error("Deployment timed out", target=target.value)
            return {
                "success": False,
                "error": "Deployment timed out after 10 minutes",
                "target": target.value
            }
        except Exception as e:
            logger.error("Deployment failed", error=str(e), target=target.value)
            return {
                "success": False,
                "error": str(e),
                "target": target.value
            }

    def perform_safety_checks(self, decision: GoLiveDecision) -> list[dict[str, Any]]:
        """Perform additional safety checks before deployment.
        
        Args:
            decision: Go-live decision
            
        Returns:
            List of safety check results
        """
        checks = []

        # Check 1: Verify no uncommitted changes
        check_result = self._check_git_status()
        checks.append(check_result)

        # Check 2: Verify database backup exists
        check_result = self._check_backup_status()
        checks.append(check_result)

        # Check 3: Verify monitoring is active
        check_result = self._check_monitoring_status()
        checks.append(check_result)

        # Check 4: Verify rollback plan exists
        check_result = self._check_rollback_plan()
        checks.append(check_result)

        # Check 5: Verify deployment window
        check_result = self._check_deployment_window()
        checks.append(check_result)

        return checks

    def _check_git_status(self) -> dict[str, Any]:
        """Check git repository status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.genesis_root
            )

            has_changes = bool(result.stdout.strip())

            return {
                "name": "git_status",
                "passed": not has_changes,
                "message": "Repository is clean" if not has_changes else "Uncommitted changes detected",
                "details": result.stdout if has_changes else None
            }
        except Exception as e:
            return {
                "name": "git_status",
                "passed": False,
                "message": f"Failed to check git status: {e!s}"
            }

    def _check_backup_status(self) -> dict[str, Any]:
        """Check if recent backup exists."""
        backup_dir = self.genesis_root / ".genesis" / "backups"

        if not backup_dir.exists():
            return {
                "name": "backup_status",
                "passed": False,
                "message": "Backup directory does not exist"
            }

        # Check for recent backup (within last 24 hours)
        import time
        current_time = time.time()
        recent_backup_found = False

        for backup_file in backup_dir.glob("*.db"):
            file_time = backup_file.stat().st_mtime
            age_hours = (current_time - file_time) / 3600

            if age_hours < 24:
                recent_backup_found = True
                break

        return {
            "name": "backup_status",
            "passed": recent_backup_found,
            "message": "Recent backup found" if recent_backup_found else "No recent backup (< 24h) found"
        }

    def _check_monitoring_status(self) -> dict[str, Any]:
        """Check if monitoring is active."""
        # This would check actual monitoring service in production
        # For now, check if metrics collector exists
        metrics_path = self.genesis_root / "genesis" / "monitoring" / "metrics_collector.py"

        return {
            "name": "monitoring_status",
            "passed": metrics_path.exists(),
            "message": "Monitoring system configured" if metrics_path.exists() else "Monitoring not configured"
        }

    def _check_rollback_plan(self) -> dict[str, Any]:
        """Check if rollback plan exists."""
        rollback_script = self.genesis_root / "scripts" / "rollback.sh"
        rollback_doc = self.genesis_root / "docs" / "runbooks" / "rollback-procedure.md"

        has_script = rollback_script.exists()
        has_doc = rollback_doc.exists()

        return {
            "name": "rollback_plan",
            "passed": has_script or has_doc,
            "message": "Rollback plan available" if (has_script or has_doc) else "No rollback plan found"
        }

    def _check_deployment_window(self) -> dict[str, Any]:
        """Check if current time is within deployment window."""
        # In production, this would check against configured deployment windows
        # For now, check if it's during business hours (9 AM - 5 PM)
        current_hour = datetime.utcnow().hour

        is_business_hours = 9 <= current_hour < 17

        return {
            "name": "deployment_window",
            "passed": is_business_hours,
            "message": f"Current hour ({current_hour}:00 UTC) is {'within' if is_business_hours else 'outside'} deployment window (9:00-17:00 UTC)"
        }

    def _extract_blocking_issues(self, report: ValidationReport) -> list[ValidationCheck]:
        """Extract blocking issues from report.
        
        Args:
            report: Validation report
            
        Returns:
            List of blocking issues
        """
        blocking = []

        # Get from report's blocking issues
        blocking.extend(report.blocking_issues)

        # Also extract critical issues from individual results
        for result in report.results:
            for check in result.checks:
                if not check.passed and check.severity in ["critical", "error"]:
                    # Avoid duplicates
                    if not any(b.name == check.name for b in blocking):
                        blocking.append(check)

        return blocking

    def _extract_warnings(self, report: ValidationReport) -> list[ValidationCheck]:
        """Extract warnings from report.
        
        Args:
            report: Validation report
            
        Returns:
            List of warnings
        """
        warnings = []

        for result in report.results:
            for check in result.checks:
                if check.severity == "warning":
                    warnings.append(check)

        return warnings

    def _validate_authorization(self, username: str, password: str) -> bool:
        """Validate user authorization for override.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authorized
        """
        if username not in self.AUTHORIZED_USERS:
            return False

        # Hash the password and compare
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return self.AUTHORIZED_USERS[username] == password_hash

    def _generate_auth_code(self, username: str, reason: str) -> str:
        """Generate authorization code for audit trail.
        
        Args:
            username: Username
            reason: Override reason
            
        Returns:
            Authorization code
        """
        timestamp = datetime.utcnow().isoformat()
        data = f"{username}:{reason}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _audit_decision(self, decision: GoLiveDecision):
        """Audit log a decision.
        
        Args:
            decision: Decision to audit
        """
        self._write_audit_log({
            "type": "decision",
            "timestamp": decision.decision_timestamp.isoformat(),
            "ready": decision.ready,
            "score": decision.score,
            "pipeline": decision.pipeline_name,
            "blocking_issues": len(decision.blocking_issues),
            "warnings": len(decision.warnings),
            "deployment_target": decision.deployment_target.value if decision.deployment_target else None,
            "deployment_allowed": decision.deployment_allowed
        })

    def _audit_override(self, decision: GoLiveDecision, override: Override):
        """Audit log an override.
        
        Args:
            decision: Decision being overridden
            override: Override details
        """
        self._write_audit_log({
            "type": "override",
            "timestamp": override.timestamp.isoformat(),
            "authorized_by": override.authorized_by,
            "reason": override.reason,
            "authorization_code": override.authorization_code,
            "original_ready": not decision.ready or decision.override is None,
            "original_score": decision.score,
            "deployment_target": decision.deployment_target.value if decision.deployment_target else None
        })

    def _audit_deployment(self, decision: GoLiveDecision, result: dict[str, Any]):
        """Audit log a deployment.
        
        Args:
            decision: Deployment decision
            result: Deployment result
        """
        self._write_audit_log({
            "type": "deployment",
            "timestamp": datetime.utcnow().isoformat(),
            "success": result.get("success", False),
            "target": result.get("target"),
            "dry_run": result.get("dry_run", False),
            "override_used": decision.override is not None,
            "score": decision.score,
            "error": result.get("error")
        })

    def _write_audit_log(self, entry: dict[str, Any]):
        """Write entry to audit log.
        
        Args:
            entry: Audit log entry
        """
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to write audit log", error=str(e))


class UnauthorizedError(Exception):
    """Raised when authorization fails."""
    pass
