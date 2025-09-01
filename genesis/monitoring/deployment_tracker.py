"""Deployment tracking and rollback capability for Genesis."""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentType(Enum):
    """Type of deployment."""
    RELEASE = "release"
    HOTFIX = "hotfix"
    ROLLBACK = "rollback"
    CONFIG_CHANGE = "config_change"
    EMERGENCY = "emergency"


@dataclass
class DeploymentInfo:
    """Information about a deployment."""
    id: str
    version: str
    type: DeploymentType
    status: DeploymentStatus
    timestamp: datetime
    deployed_by: str
    git_commit: str
    git_branch: str
    environment: str
    
    # Optional fields
    description: Optional[str] = None
    rollback_version: Optional[str] = None
    duration: Optional[timedelta] = None
    error_message: Optional[str] = None
    
    # Metrics
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    
    # Health checks
    health_checks_passed: bool = False
    test_results: Dict[str, bool] = field(default_factory=dict)
    
    # Rollback info
    can_rollback: bool = True
    rollback_command: Optional[str] = None


@dataclass
class DeploymentMetrics:
    """Deployment metrics and statistics."""
    total_deployments: int = 0
    successful_deployments: int = 0
    failed_deployments: int = 0
    rolled_back_deployments: int = 0
    
    success_rate: float = 0.0
    mean_deployment_time: timedelta = timedelta(0)
    last_deployment: Optional[datetime] = None
    last_successful_deployment: Optional[datetime] = None
    
    deployments_today: int = 0
    deployments_this_week: int = 0
    deployments_this_month: int = 0
    
    hotfix_count: int = 0
    emergency_count: int = 0


class DeploymentTracker:
    """Tracks deployments and provides rollback capability."""
    
    def __init__(self, storage_path: str = ".genesis/deployments"):
        """Initialize deployment tracker.
        
        Args:
            storage_path: Path to store deployment history
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.current_deployment: Optional[DeploymentInfo] = None
        self.deployment_history: List[DeploymentInfo] = []
        self.metrics = DeploymentMetrics()
        
        # Load history
        self._load_history()
    
    def start_deployment(
        self,
        version: str,
        deployment_type: DeploymentType,
        deployed_by: str,
        description: Optional[str] = None,
        environment: str = "production",
    ) -> DeploymentInfo:
        """Start tracking a new deployment.
        
        Args:
            version: Version being deployed
            deployment_type: Type of deployment
            deployed_by: Who initiated the deployment
            description: Optional description
            environment: Target environment
            
        Returns:
            DeploymentInfo object
        """
        # Get git information
        git_commit = self._get_git_commit()
        git_branch = self._get_git_branch()
        
        # Generate deployment ID
        deployment_id = self._generate_deployment_id(version, git_commit)
        
        # Create deployment info
        deployment = DeploymentInfo(
            id=deployment_id,
            version=version,
            type=deployment_type,
            status=DeploymentStatus.IN_PROGRESS,
            timestamp=datetime.now(UTC),
            deployed_by=deployed_by,
            git_commit=git_commit,
            git_branch=git_branch,
            environment=environment,
            description=description,
        )
        
        # Get git statistics
        stats = self._get_git_stats()
        deployment.files_changed = stats.get("files_changed", 0)
        deployment.lines_added = stats.get("lines_added", 0)
        deployment.lines_removed = stats.get("lines_removed", 0)
        
        # Set as current deployment
        self.current_deployment = deployment
        
        # Log deployment start
        logger.info(
            "Deployment started",
            deployment_id=deployment_id,
            version=version,
            type=deployment_type.value,
        )
        
        return deployment
    
    def complete_deployment(
        self,
        success: bool,
        error_message: Optional[str] = None,
        health_checks_passed: bool = True,
        test_results: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Complete the current deployment.
        
        Args:
            success: Whether deployment was successful
            error_message: Error message if failed
            health_checks_passed: Whether health checks passed
            test_results: Test results dictionary
        """
        if not self.current_deployment:
            logger.warning("No deployment in progress")
            return
        
        # Update deployment status
        self.current_deployment.status = (
            DeploymentStatus.SUCCESS if success else DeploymentStatus.FAILED
        )
        self.current_deployment.error_message = error_message
        self.current_deployment.health_checks_passed = health_checks_passed
        self.current_deployment.test_results = test_results or {}
        
        # Calculate duration
        self.current_deployment.duration = (
            datetime.now(UTC) - self.current_deployment.timestamp
        )
        
        # Add to history
        self.deployment_history.insert(0, self.current_deployment)
        
        # Update metrics
        self._update_metrics()
        
        # Save to disk
        self._save_deployment(self.current_deployment)
        
        # Log completion
        logger.info(
            "Deployment completed",
            deployment_id=self.current_deployment.id,
            status=self.current_deployment.status.value,
            duration=str(self.current_deployment.duration),
        )
        
        # Clear current deployment
        self.current_deployment = None
    
    def rollback(
        self,
        target_version: Optional[str] = None,
        reason: str = "Manual rollback",
    ) -> bool:
        """Rollback to a previous deployment.
        
        Args:
            target_version: Version to rollback to (latest successful if None)
            reason: Reason for rollback
            
        Returns:
            True if rollback successful
        """
        # Find target deployment
        if target_version:
            target = self._find_deployment_by_version(target_version)
        else:
            target = self._find_last_successful_deployment()
        
        if not target:
            logger.error("No valid deployment found for rollback")
            return False
        
        if not target.can_rollback:
            logger.error("Target deployment cannot be rolled back")
            return False
        
        # Create rollback deployment
        rollback = DeploymentInfo(
            id=self._generate_deployment_id(f"rollback_{target.version}", ""),
            version=f"rollback_to_{target.version}",
            type=DeploymentType.ROLLBACK,
            status=DeploymentStatus.IN_PROGRESS,
            timestamp=datetime.now(UTC),
            deployed_by="system",
            git_commit=target.git_commit,
            git_branch=target.git_branch,
            environment=target.environment,
            description=f"Rollback: {reason}",
            rollback_version=target.version,
        )
        
        self.current_deployment = rollback
        
        # Execute rollback
        success = self._execute_rollback(target)
        
        # Complete rollback deployment
        self.complete_deployment(
            success=success,
            error_message=None if success else "Rollback failed",
        )
        
        if success:
            # Mark original as rolled back
            for deployment in self.deployment_history:
                if deployment.version == target.version:
                    deployment.status = DeploymentStatus.ROLLED_BACK
                    break
        
        return success
    
    def get_deployment_history(
        self,
        limit: int = 20,
        status_filter: Optional[DeploymentStatus] = None,
    ) -> List[DeploymentInfo]:
        """Get deployment history.
        
        Args:
            limit: Maximum number of deployments to return
            status_filter: Optional status filter
            
        Returns:
            List of deployments
        """
        history = self.deployment_history
        
        if status_filter:
            history = [d for d in history if d.status == status_filter]
        
        return history[:limit]
    
    def get_metrics(self) -> DeploymentMetrics:
        """Get deployment metrics.
        
        Returns:
            DeploymentMetrics object
        """
        self._update_metrics()
        return self.metrics
    
    def can_deploy(self) -> tuple[bool, str]:
        """Check if deployment is allowed.
        
        Returns:
            Tuple of (can_deploy, reason)
        """
        # Check if deployment in progress
        if self.current_deployment:
            return False, "Deployment already in progress"
        
        # Check recent failures
        recent_failures = sum(
            1 for d in self.deployment_history[:5]
            if d.status == DeploymentStatus.FAILED
        )
        if recent_failures >= 3:
            return False, "Too many recent failures"
        
        # Check deployment frequency
        if self.metrics.deployments_today >= 10:
            return False, "Daily deployment limit reached"
        
        return True, "OK"
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()[:8]
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _get_git_stats(self) -> Dict[str, int]:
        """Get git statistics for current changes."""
        try:
            # Get diff stats
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                capture_output=True,
                text=True,
                check=True,
            )
            
            lines = result.stdout.strip().split("\n")
            if lines:
                # Parse the summary line
                summary = lines[-1]
                stats = {}
                
                # Extract files changed
                if "file" in summary:
                    files_match = summary.split(",")[0]
                    stats["files_changed"] = int(files_match.split()[0])
                
                # Extract insertions/deletions
                if "insertion" in summary:
                    for part in summary.split(","):
                        if "insertion" in part:
                            stats["lines_added"] = int(part.split()[0])
                        elif "deletion" in part:
                            stats["lines_removed"] = int(part.split()[0])
                
                return stats
        except:
            return {}
    
    def _generate_deployment_id(self, version: str, commit: str) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.now(UTC).isoformat()
        content = f"{version}_{commit}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _find_deployment_by_version(self, version: str) -> Optional[DeploymentInfo]:
        """Find deployment by version."""
        for deployment in self.deployment_history:
            if deployment.version == version:
                return deployment
        return None
    
    def _find_last_successful_deployment(self) -> Optional[DeploymentInfo]:
        """Find last successful deployment."""
        for deployment in self.deployment_history:
            if deployment.status == DeploymentStatus.SUCCESS:
                return deployment
        return None
    
    def _execute_rollback(self, target: DeploymentInfo) -> bool:
        """Execute rollback to target deployment."""
        try:
            # Checkout target commit
            subprocess.run(
                ["git", "checkout", target.git_commit],
                check=True,
                capture_output=True,
            )
            
            # Run rollback command if specified
            if target.rollback_command:
                subprocess.run(
                    target.rollback_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                )
            
            logger.info(
                "Rollback executed successfully",
                target_version=target.version,
                commit=target.git_commit,
            )
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(
                "Rollback failed",
                error=str(e),
                target_version=target.version,
            )
            return False
    
    def _update_metrics(self) -> None:
        """Update deployment metrics."""
        if not self.deployment_history:
            return
        
        now = datetime.now(UTC)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=now.weekday())
        month_start = now.replace(day=1)
        
        # Count deployments
        self.metrics.total_deployments = len(self.deployment_history)
        self.metrics.successful_deployments = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.SUCCESS
        )
        self.metrics.failed_deployments = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.FAILED
        )
        self.metrics.rolled_back_deployments = sum(
            1 for d in self.deployment_history
            if d.status == DeploymentStatus.ROLLED_BACK
        )
        
        # Calculate success rate
        if self.metrics.total_deployments > 0:
            self.metrics.success_rate = (
                self.metrics.successful_deployments / self.metrics.total_deployments
            )
        
        # Time-based counts
        self.metrics.deployments_today = sum(
            1 for d in self.deployment_history
            if d.timestamp >= today_start
        )
        self.metrics.deployments_this_week = sum(
            1 for d in self.deployment_history
            if d.timestamp >= week_start
        )
        self.metrics.deployments_this_month = sum(
            1 for d in self.deployment_history
            if d.timestamp >= month_start
        )
        
        # Type counts
        self.metrics.hotfix_count = sum(
            1 for d in self.deployment_history
            if d.type == DeploymentType.HOTFIX
        )
        self.metrics.emergency_count = sum(
            1 for d in self.deployment_history
            if d.type == DeploymentType.EMERGENCY
        )
        
        # Mean deployment time
        durations = [
            d.duration for d in self.deployment_history
            if d.duration and d.status == DeploymentStatus.SUCCESS
        ]
        if durations:
            total_duration = sum(durations, timedelta())
            self.metrics.mean_deployment_time = total_duration / len(durations)
        
        # Last deployment times
        if self.deployment_history:
            self.metrics.last_deployment = self.deployment_history[0].timestamp
            
            for deployment in self.deployment_history:
                if deployment.status == DeploymentStatus.SUCCESS:
                    self.metrics.last_successful_deployment = deployment.timestamp
                    break
    
    def _save_deployment(self, deployment: DeploymentInfo) -> None:
        """Save deployment to disk."""
        filename = f"{deployment.id}.json"
        filepath = self.storage_path / filename
        
        # Convert to dict
        data = asdict(deployment)
        
        # Convert datetime objects to strings
        for key in ["timestamp", "resolved_at", "acknowledged_at"]:
            if key in data and data[key]:
                data[key] = data[key].isoformat()
        
        # Convert timedelta to seconds
        if data.get("duration"):
            data["duration"] = data["duration"].total_seconds()
        
        # Convert enums to strings
        data["type"] = data["type"].value
        data["status"] = data["status"].value
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_history(self) -> None:
        """Load deployment history from disk."""
        if not self.storage_path.exists():
            return
        
        deployments = []
        
        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                # Convert strings back to datetime
                for key in ["timestamp", "resolved_at", "acknowledged_at"]:
                    if key in data and data[key]:
                        data[key] = datetime.fromisoformat(data[key])
                
                # Convert seconds back to timedelta
                if data.get("duration"):
                    data["duration"] = timedelta(seconds=data["duration"])
                
                # Convert strings back to enums
                data["type"] = DeploymentType(data["type"])
                data["status"] = DeploymentStatus(data["status"])
                
                deployment = DeploymentInfo(**data)
                deployments.append(deployment)
                
            except Exception as e:
                logger.warning(f"Failed to load deployment {filepath}: {e}")
        
        # Sort by timestamp (newest first)
        self.deployment_history = sorted(
            deployments,
            key=lambda d: d.timestamp,
            reverse=True,
        )