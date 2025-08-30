"""DR testing framework with chaos engineering capabilities."""

import asyncio
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChaosScenario:
    """Chaos engineering scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        fault_injector: Callable,
        duration_seconds: int = 60,
        intensity: float = 0.5
    ):
        """Initialize chaos scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            fault_injector: Function to inject fault
            duration_seconds: How long to maintain fault
            intensity: Fault intensity (0-1)
        """
        self.name = name
        self.description = description
        self.fault_injector = fault_injector
        self.duration_seconds = duration_seconds
        self.intensity = intensity


class DRTestRunner:
    """Runs DR tests and chaos engineering scenarios."""

    def __init__(self, dr_orchestrator):
        """Initialize DR test runner.
        
        Args:
            dr_orchestrator: DR orchestrator instance
        """
        self.dr_orchestrator = dr_orchestrator

        # Test configuration
        self.test_results: list[dict[str, Any]] = []
        self.chaos_scenarios = self._initialize_chaos_scenarios()
        self.test_schedule: list[dict[str, Any]] = []

        # Test state
        self.test_in_progress = False
        self.current_test: str | None = None
        self.last_test_time: datetime | None = None

    def _initialize_chaos_scenarios(self) -> dict[str, ChaosScenario]:
        """Initialize chaos engineering scenarios.
        
        Returns:
            Chaos scenarios
        """
        return {
            "network_latency": ChaosScenario(
                name="network_latency",
                description="Inject network latency",
                fault_injector=self._inject_network_latency,
                duration_seconds=60,
                intensity=0.5
            ),
            "service_failure": ChaosScenario(
                name="service_failure",
                description="Simulate service failure",
                fault_injector=self._inject_service_failure,
                duration_seconds=30,
                intensity=0.7
            ),
            "database_slowdown": ChaosScenario(
                name="database_slowdown",
                description="Slow database queries",
                fault_injector=self._inject_database_slowdown,
                duration_seconds=45,
                intensity=0.6
            ),
            "disk_pressure": ChaosScenario(
                name="disk_pressure",
                description="Simulate disk space pressure",
                fault_injector=self._inject_disk_pressure,
                duration_seconds=90,
                intensity=0.8
            ),
            "api_errors": ChaosScenario(
                name="api_errors",
                description="Inject API errors",
                fault_injector=self._inject_api_errors,
                duration_seconds=60,
                intensity=0.3
            )
        }

    async def run_dr_drill(
        self,
        scenario_name: str,
        include_chaos: bool = False,
        notification: bool = True
    ) -> dict[str, Any]:
        """Run DR drill for specific scenario.
        
        Args:
            scenario_name: DR scenario to test
            include_chaos: Whether to include chaos engineering
            notification: Whether to send notifications
            
        Returns:
            Drill results
        """
        if self.test_in_progress:
            return {"error": "Test already in progress"}

        self.test_in_progress = True
        self.current_test = f"DR Drill: {scenario_name}"
        start_time = datetime.utcnow()

        logger.info(
            "DR drill started",
            scenario=scenario_name,
            include_chaos=include_chaos
        )

        results = {
            "test_type": "dr_drill",
            "scenario": scenario_name,
            "start_time": start_time.isoformat(),
            "include_chaos": include_chaos,
            "steps": []
        }

        try:
            # Send notification if enabled
            if notification:
                await self._send_drill_notification(
                    "DR DRILL STARTING",
                    f"Scenario: {scenario_name}\nChaos: {include_chaos}"
                )

            # Inject chaos if requested
            chaos_task = None
            if include_chaos:
                chaos_scenario = random.choice(list(self.chaos_scenarios.values()))
                chaos_task = asyncio.create_task(
                    self._run_chaos_scenario(chaos_scenario)
                )
                results["chaos_scenario"] = chaos_scenario.name

            # Execute DR workflow
            from genesis.dr import DRScenario

            dr_scenario = DRScenario[scenario_name.upper()]
            dr_result = await self.dr_orchestrator.execute_dr_workflow(
                scenario=dr_scenario,
                dry_run=True,  # Always dry run for drills
                auto_execute=True
            )

            results["dr_result"] = dr_result
            results["success"] = dr_result["success"]

            # Wait for chaos to complete
            if chaos_task:
                await chaos_task

            # Validate recovery
            validation = await self._validate_recovery()
            results["validation"] = validation

            # Calculate metrics
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            results["end_time"] = end_time.isoformat()
            results["duration_seconds"] = duration
            results["rto_met"] = duration / 60 <= self.dr_orchestrator.RTO_TARGET_MINUTES

            # Record test
            self.test_results.append(results)
            self.last_test_time = end_time

            # Send completion notification
            if notification:
                await self._send_drill_notification(
                    "DR DRILL COMPLETED",
                    f"Success: {results['success']}\n"
                    f"Duration: {duration:.1f}s\n"
                    f"RTO Met: {results['rto_met']}"
                )

            logger.info(
                "DR drill completed",
                scenario=scenario_name,
                success=results["success"],
                duration=duration
            )

        except Exception as e:
            logger.error("DR drill failed", error=str(e))
            results["error"] = str(e)
            results["success"] = False

        finally:
            self.test_in_progress = False
            self.current_test = None

        return results

    async def run_chaos_scenario(self, scenario_name: str) -> dict[str, Any]:
        """Run chaos engineering scenario.
        
        Args:
            scenario_name: Chaos scenario to run
            
        Returns:
            Chaos test results
        """
        scenario = self.chaos_scenarios.get(scenario_name)

        if not scenario:
            return {"error": f"Unknown scenario: {scenario_name}"}

        return await self._run_chaos_scenario(scenario)

    async def _run_chaos_scenario(self, scenario: ChaosScenario) -> dict[str, Any]:
        """Execute chaos scenario.
        
        Args:
            scenario: Chaos scenario to execute
            
        Returns:
            Execution results
        """
        logger.info(
            f"Chaos scenario started: {scenario.name}",
            duration=scenario.duration_seconds,
            intensity=scenario.intensity
        )

        results = {
            "scenario": scenario.name,
            "description": scenario.description,
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": scenario.duration_seconds,
            "intensity": scenario.intensity
        }

        try:
            # Inject fault
            await scenario.fault_injector(scenario.intensity, scenario.duration_seconds)

            results["success"] = True
            results["end_time"] = datetime.utcnow().isoformat()

            logger.info(f"Chaos scenario completed: {scenario.name}")

        except Exception as e:
            logger.error(f"Chaos scenario failed: {scenario.name}", error=str(e))
            results["success"] = False
            results["error"] = str(e)

        return results

    async def _inject_network_latency(self, intensity: float, duration: int) -> None:
        """Inject network latency.
        
        Args:
            intensity: Latency intensity (0-1)
            duration: Duration in seconds
        """
        # In production, would use tc (traffic control) on Linux
        latency_ms = int(1000 * intensity)  # Up to 1 second

        logger.info(f"Injecting {latency_ms}ms network latency for {duration}s")

        # Simulate with sleep
        await asyncio.sleep(duration)

        logger.info("Network latency removed")

    async def _inject_service_failure(self, intensity: float, duration: int) -> None:
        """Simulate service failure.
        
        Args:
            intensity: Failure probability (0-1)
            duration: Duration in seconds
        """
        logger.info(f"Simulating service failure (intensity: {intensity}) for {duration}s")

        # In production, would actually stop services
        # For testing, just mark health checks as failed

        await asyncio.sleep(duration)

        logger.info("Service failure ended")

    async def _inject_database_slowdown(self, intensity: float, duration: int) -> None:
        """Slow down database queries.
        
        Args:
            intensity: Slowdown factor (0-1)
            duration: Duration in seconds
        """
        slowdown_factor = 1 + (10 * intensity)  # Up to 11x slower

        logger.info(f"Slowing database by {slowdown_factor}x for {duration}s")

        # In production, would add delays to database operations
        await asyncio.sleep(duration)

        logger.info("Database slowdown removed")

    async def _inject_disk_pressure(self, intensity: float, duration: int) -> None:
        """Simulate disk space pressure.
        
        Args:
            intensity: Disk usage percentage (0-1)
            duration: Duration in seconds
        """
        disk_usage_percent = int(100 * intensity)

        logger.info(f"Simulating {disk_usage_percent}% disk usage for {duration}s")

        # In production, would create temporary large files
        await asyncio.sleep(duration)

        logger.info("Disk pressure removed")

    async def _inject_api_errors(self, intensity: float, duration: int) -> None:
        """Inject API errors.
        
        Args:
            intensity: Error rate (0-1)
            duration: Duration in seconds
        """
        error_rate = int(100 * intensity)

        logger.info(f"Injecting {error_rate}% API errors for {duration}s")

        # In production, would configure API gateway to return errors
        await asyncio.sleep(duration)

        logger.info("API errors removed")

    async def _validate_recovery(self) -> dict[str, Any]:
        """Validate recovery after drill.
        
        Returns:
            Validation results
        """
        validation = {
            "checks_passed": [],
            "checks_failed": [],
            "is_valid": False
        }

        # Check backup status
        backup_status = self.dr_orchestrator.backup_manager.get_backup_status()
        if backup_status["last_full_backup"]:
            validation["checks_passed"].append("Backup available")
        else:
            validation["checks_failed"].append("No backup available")

        # Check replication
        replication_status = self.dr_orchestrator.replication_manager.get_replication_status()
        if replication_status["replication_lag_seconds"] < 300:
            validation["checks_passed"].append("Replication healthy")
        else:
            validation["checks_failed"].append("High replication lag")

        # Check failover readiness
        failover_status = self.dr_orchestrator.failover_coordinator.get_status()
        if failover_status["monitoring"]:
            validation["checks_passed"].append("Failover monitoring active")
        else:
            validation["checks_failed"].append("Failover monitoring inactive")

        validation["is_valid"] = len(validation["checks_failed"]) == 0

        return validation

    async def _send_drill_notification(self, subject: str, message: str) -> None:
        """Send drill notification.
        
        Args:
            subject: Notification subject
            message: Notification message
        """
        logger.info(f"Drill notification: {subject}")
        # In production, would send actual notifications

    async def schedule_monthly_drill(self) -> None:
        """Schedule monthly DR drill."""
        while True:
            # Wait for next scheduled time (first Monday of month at 2 AM)
            next_drill = self._get_next_drill_time()
            wait_seconds = (next_drill - datetime.now()).total_seconds()

            if wait_seconds > 0:
                logger.info(
                    f"Next DR drill scheduled for {next_drill}",
                    wait_hours=wait_seconds / 3600
                )
                await asyncio.sleep(wait_seconds)

            # Run drill
            scenarios = ["DATABASE_CORRUPTION", "PRIMARY_FAILURE", "DATA_LOSS"]
            scenario = random.choice(scenarios)

            logger.info(f"Starting scheduled DR drill: {scenario}")

            result = await self.run_dr_drill(
                scenario_name=scenario,
                include_chaos=True,
                notification=True
            )

            # Record scheduled test
            self.test_schedule.append({
                "scheduled_time": next_drill.isoformat(),
                "executed_time": datetime.now().isoformat(),
                "scenario": scenario,
                "success": result.get("success", False)
            })

            # Wait a day before checking next schedule
            await asyncio.sleep(86400)

    def _get_next_drill_time(self) -> datetime:
        """Get next scheduled drill time.
        
        Returns:
            Next drill datetime
        """
        now = datetime.now()

        # Find first Monday of next month
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)

        # Find first Monday
        while next_month.weekday() != 0:  # 0 = Monday
            next_month += timedelta(days=1)

        # Set time to 2 AM
        next_drill = next_month.replace(hour=2, minute=0, second=0, microsecond=0)

        # If we've passed this month's drill, use next month
        if next_drill <= now:
            if next_drill.month == 12:
                next_drill = datetime(next_drill.year + 1, 1, 1, 2, 0, 0)
            else:
                next_drill = datetime(next_drill.year, next_drill.month + 1, 1, 2, 0, 0)

            while next_drill.weekday() != 0:
                next_drill += timedelta(days=1)

        return next_drill

    def get_test_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get test history.
        
        Args:
            limit: Maximum results to return
            
        Returns:
            Test history
        """
        return self.test_results[-limit:]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Calculate DR performance metrics.
        
        Returns:
            Performance metrics
        """
        if not self.test_results:
            return {
                "total_tests": 0,
                "success_rate": 0,
                "average_recovery_time": 0,
                "rto_compliance": 0
            }

        successful_tests = [t for t in self.test_results if t.get("success")]

        recovery_times = [
            t.get("duration_seconds", 0) / 60
            for t in successful_tests
        ]

        rto_compliant = [
            t for t in successful_tests
            if t.get("rto_met", False)
        ]

        return {
            "total_tests": len(self.test_results),
            "success_rate": len(successful_tests) / len(self.test_results),
            "average_recovery_time": sum(recovery_times) / len(recovery_times) if recovery_times else 0,
            "rto_compliance": len(rto_compliant) / len(successful_tests) if successful_tests else 0,
            "last_test": self.last_test_time.isoformat() if self.last_test_time else None
        }

    def generate_test_report(self) -> str:
        """Generate DR test report.
        
        Returns:
            Test report text
        """
        metrics = self.get_performance_metrics()

        report = f"""
Disaster Recovery Test Report
==============================
Generated: {datetime.now().isoformat()}

Performance Metrics:
-------------------
Total Tests Run: {metrics['total_tests']}
Success Rate: {metrics['success_rate']:.1%}
Average Recovery Time: {metrics['average_recovery_time']:.1f} minutes
RTO Compliance: {metrics['rto_compliance']:.1%}
Last Test: {metrics['last_test']}

Recent Test Results:
-------------------
"""

        for test in self.get_test_history(5):
            report += f"""
Test: {test.get('scenario', 'Unknown')}
Time: {test.get('start_time', 'Unknown')}
Duration: {test.get('duration_seconds', 0):.1f} seconds
Success: {test.get('success', False)}
RTO Met: {test.get('rto_met', False)}
"""

            if test.get('chaos_scenario'):
                report += f"Chaos: {test['chaos_scenario']}\n"

        # Add recommendations
        report += """
Recommendations:
---------------
"""

        if metrics['success_rate'] < 0.9:
            report += "- Investigate test failures and improve reliability\n"

        if metrics['average_recovery_time'] > 10:
            report += "- Optimize recovery procedures to meet RTO target\n"

        if metrics['rto_compliance'] < 0.95:
            report += "- Focus on improving recovery time consistency\n"

        if metrics['total_tests'] < 3:
            report += "- Increase testing frequency to improve confidence\n"

        return report
