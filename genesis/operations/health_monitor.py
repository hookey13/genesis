"""Comprehensive Health Check and Self-Healing System."""

import asyncio
import os
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from pydantic import BaseModel, Field

from genesis.utils.logger import get_logger, LoggerType


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class HealthCheck(BaseModel):
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = Field(default_factory=dict)
    remediation_attempted: bool = False


class HealthConfig(BaseModel):
    """Health monitoring configuration."""
    
    check_interval_seconds: int = Field(60, description="Seconds between health checks")
    enable_auto_remediation: bool = Field(True, description="Enable automatic remediation")
    escalation_threshold: int = Field(3, description="Failures before escalation")
    disk_usage_threshold: float = Field(0.9, description="Disk usage threshold")
    memory_usage_threshold: float = Field(0.85, description="Memory usage threshold")
    connection_pool_threshold: int = Field(90, description="Connection pool usage %")


class HealthMonitor:
    """Monitors system health and performs self-healing."""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
        self.health_history: List[HealthCheck] = []
        self.failure_counts: Dict[str, int] = {}
        self.remediations: Dict[str, Callable] = self._setup_remediations()
    
    def _setup_remediations(self) -> Dict[str, Callable]:
        """Setup remediation functions."""
        return {
            "database_connection_pool": self._remediate_connection_pool,
            "memory_leak": self._remediate_memory_leak,
            "disk_space": self._remediate_disk_space,
            "api_degradation": self._remediate_api_degradation
        }
    
    async def check_database_health(self) -> HealthCheck:
        """Check database connection pool health."""
        try:
            # Check connection pool usage
            # This would integrate with actual database pool
            pool_usage = 50  # Mock value
            
            if pool_usage > self.config.connection_pool_threshold:
                return HealthCheck(
                    name="database_connection_pool",
                    status=HealthStatus.CRITICAL,
                    message=f"Connection pool exhausted: {pool_usage}%",
                    timestamp=datetime.utcnow(),
                    metrics={"pool_usage": pool_usage}
                )
            elif pool_usage > 70:
                return HealthCheck(
                    name="database_connection_pool",
                    status=HealthStatus.DEGRADED,
                    message=f"High connection pool usage: {pool_usage}%",
                    timestamp=datetime.utcnow(),
                    metrics={"pool_usage": pool_usage}
                )
            else:
                return HealthCheck(
                    name="database_connection_pool",
                    status=HealthStatus.HEALTHY,
                    message="Database connections healthy",
                    timestamp=datetime.utcnow(),
                    metrics={"pool_usage": pool_usage}
                )
                
        except Exception as e:
            return HealthCheck(
                name="database_connection_pool",
                status=HealthStatus.FAILED,
                message=str(e),
                timestamp=datetime.utcnow()
            )
    
    async def check_memory_health(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100
            
            if usage_percent > self.config.memory_usage_threshold:
                return HealthCheck(
                    name="memory_leak",
                    status=HealthStatus.CRITICAL,
                    message=f"High memory usage: {usage_percent:.1%}",
                    timestamp=datetime.utcnow(),
                    metrics={
                        "usage_percent": usage_percent,
                        "available_mb": memory.available / 1024 / 1024
                    }
                )
            else:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory usage normal: {usage_percent:.1%}",
                    timestamp=datetime.utcnow(),
                    metrics={"usage_percent": usage_percent}
                )
                
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.FAILED,
                message=str(e),
                timestamp=datetime.utcnow()
            )
    
    async def check_disk_health(self) -> HealthCheck:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent / 100
            
            if usage_percent > self.config.disk_usage_threshold:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.CRITICAL,
                    message=f"Low disk space: {usage_percent:.1%} used",
                    timestamp=datetime.utcnow(),
                    metrics={
                        "usage_percent": usage_percent,
                        "free_gb": disk.free / 1024 / 1024 / 1024
                    }
                )
            else:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk usage normal: {usage_percent:.1%}",
                    timestamp=datetime.utcnow(),
                    metrics={"usage_percent": usage_percent}
                )
                
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.FAILED,
                message=str(e),
                timestamp=datetime.utcnow()
            )
    
    async def check_api_health(self) -> HealthCheck:
        """Check exchange API health."""
        try:
            # Mock API check - would ping actual exchange
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.binance.com/api/v3/ping",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return HealthCheck(
                            name="exchange_api",
                            status=HealthStatus.HEALTHY,
                            message="Exchange API responsive",
                            timestamp=datetime.utcnow()
                        )
                    else:
                        return HealthCheck(
                            name="api_degradation",
                            status=HealthStatus.DEGRADED,
                            message=f"API returned {response.status}",
                            timestamp=datetime.utcnow()
                        )
                        
        except asyncio.TimeoutError:
            return HealthCheck(
                name="api_degradation",
                status=HealthStatus.CRITICAL,
                message="API timeout",
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            return HealthCheck(
                name="exchange_api",
                status=HealthStatus.FAILED,
                message=str(e),
                timestamp=datetime.utcnow()
            )
    
    async def _remediate_connection_pool(self) -> bool:
        """Remediate database connection pool issues."""
        try:
            self.logger.info("remediating_connection_pool")
            # Reset connection pool
            # This would integrate with actual database
            await asyncio.sleep(1)  # Mock remediation
            return True
        except Exception as e:
            self.logger.error("remediation_failed", error=str(e))
            return False
    
    async def _remediate_memory_leak(self) -> bool:
        """Remediate memory issues."""
        try:
            self.logger.info("remediating_memory_leak")
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # If still high, schedule controlled restart
            memory = psutil.virtual_memory()
            if memory.percent / 100 > self.config.memory_usage_threshold:
                self.logger.warning("scheduling_controlled_restart")
                # This would trigger graceful restart
            
            return True
        except Exception as e:
            self.logger.error("remediation_failed", error=str(e))
            return False
    
    async def _remediate_disk_space(self) -> bool:
        """Remediate disk space issues."""
        try:
            self.logger.info("remediating_disk_space")
            
            # Clean up old logs
            log_dir = Path(".genesis/logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log.*"):
                    if log_file.stat().st_mtime < (datetime.now() - timedelta(days=7)).timestamp():
                        log_file.unlink()
                        self.logger.info("deleted_old_log", file=str(log_file))
            
            # Clean up old backups
            backup_dir = Path(".genesis/backups")
            if backup_dir.exists():
                backups = sorted(
                    backup_dir.glob("*.db"),
                    key=lambda p: p.stat().st_mtime
                )
                for backup in backups[:-5]:  # Keep last 5
                    backup.unlink()
                    self.logger.info("deleted_old_backup", file=str(backup))
            
            return True
        except Exception as e:
            self.logger.error("remediation_failed", error=str(e))
            return False
    
    async def _remediate_api_degradation(self) -> bool:
        """Remediate API issues."""
        try:
            self.logger.info("remediating_api_degradation")
            # Activate circuit breaker
            # This would integrate with exchange gateway
            return True
        except Exception as e:
            self.logger.error("remediation_failed", error=str(e))
            return False
    
    async def perform_health_checks(self) -> Dict[str, HealthCheck]:
        """Perform all health checks."""
        checks = await asyncio.gather(
            self.check_database_health(),
            self.check_memory_health(),
            self.check_disk_health(),
            self.check_api_health(),
            return_exceptions=True
        )
        
        results = {}
        for check in checks:
            if isinstance(check, HealthCheck):
                results[check.name] = check
                self.health_history.append(check)
        
        return results
    
    async def attempt_remediation(self, check: HealthCheck) -> bool:
        """Attempt to remediate a failed health check."""
        if not self.config.enable_auto_remediation:
            return False
        
        if check.name in self.remediations:
            self.logger.info(
                "attempting_remediation",
                check=check.name,
                status=check.status
            )
            
            success = await self.remediations[check.name]()
            check.remediation_attempted = True
            
            if success:
                self.failure_counts[check.name] = 0
                self.logger.info("remediation_successful", check=check.name)
            else:
                self.failure_counts[check.name] = self.failure_counts.get(check.name, 0) + 1
                
                if self.failure_counts[check.name] >= self.config.escalation_threshold:
                    await self.escalate_to_operator(check)
            
            return success
        
        return False
    
    async def escalate_to_operator(self, check: HealthCheck) -> None:
        """Escalate unresolved issues to human operator."""
        self.logger.critical(
            "escalation_required",
            check=check.name,
            status=check.status,
            message=check.message,
            failure_count=self.failure_counts.get(check.name, 0)
        )
        
        # This would send alert via email/SMS/Slack
        # Implementation depends on notification service
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        recent_checks = self.health_history[-100:] if self.health_history else []
        
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.FAILED: 0
        }
        
        for check in recent_checks:
            status_counts[check.status] += 1
        
        overall_status = HealthStatus.HEALTHY
        if status_counts[HealthStatus.FAILED] > 0:
            overall_status = HealthStatus.FAILED
        elif status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "overall_status": overall_status,
            "status_counts": status_counts,
            "recent_remediations": sum(
                1 for c in recent_checks if c.remediation_attempted
            ),
            "active_issues": [
                c.name for c in recent_checks[-10:]
                if c.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]
            ]
        }
    
    async def health_monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while True:
            try:
                # Perform checks
                checks = await self.perform_health_checks()
                
                # Attempt remediation for critical issues
                for check in checks.values():
                    if check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                        await self.attempt_remediation(check)
                
                # Log summary
                summary = await self.get_health_summary()
                self.logger.info(
                    "health_check_completed",
                    overall_status=summary["overall_status"],
                    active_issues=summary["active_issues"]
                )
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                self.logger.error("health_monitoring_error", error=str(e))
                await asyncio.sleep(self.config.check_interval_seconds)


from pathlib import Path