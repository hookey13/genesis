"""System stability validation over extended operation."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import structlog

logger = structlog.get_logger(__name__)


class ValidationResult:
    """Standardized validation result."""
    
    def __init__(
        self,
        check_id: str,
        status: str,
        message: str,
        evidence: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.check_id = check_id
        self.status = status
        self.message = message
        self.evidence = evidence
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "evidence": self.evidence,
            "metadata": self.metadata
        }


class CheckStatus:
    """Validation check status constants."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class StabilityValidator:
    """Validates system stability over extended operation."""
    
    TEST_DURATION_HOURS = 48
    MAX_MEMORY_GROWTH_PERCENT = 10  # Max 10% memory growth allowed
    MAX_ERROR_RATE = 0.001  # 0.1% error rate
    MAX_LATENCY_DEGRADATION = 1.2  # Max 20% latency increase
    MAX_CPU_USAGE = 80  # Max 80% CPU usage sustained
    MIN_UPTIME_PERCENT = 99.9  # 99.9% uptime requirement
    
    def __init__(self):
        """Initialize stability validator."""
        self.test_results_file = Path(".genesis/tests/stability_test_results.json")
        self.monitoring_log = Path(".genesis/logs/monitoring.log")
        
    async def validate(self) -> Dict[str, Any]:
        """Validate system stability over extended operation."""
        try:
            # Check if stability test has been run
            test_results = await self._load_stability_test_results()
            
            if not test_results:
                # Check if test should be run or just report missing
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.FAILED,
                    message=f"Stability test not completed. Run {self.TEST_DURATION_HOURS}-hour test required",
                    evidence={"test_run": False},
                    metadata={"required_duration": self.TEST_DURATION_HOURS}
                ).to_dict()
            
            # Validate test duration
            if test_results.get("duration_hours", 0) < self.TEST_DURATION_HOURS:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.FAILED,
                    message=f"Stability test only ran for {test_results['duration_hours']:.1f} hours",
                    evidence=test_results,
                    metadata={"required": self.TEST_DURATION_HOURS}
                ).to_dict()
            
            # Check for memory leaks
            memory_growth = self._calculate_memory_growth(test_results)
            if memory_growth > self.MAX_MEMORY_GROWTH_PERCENT / 100:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.FAILED,
                    message=f"Memory leak detected: {memory_growth:.1%} growth",
                    evidence={
                        "memory_growth": memory_growth,
                        "initial_memory_mb": test_results.get("initial_memory", 0) / 1024 / 1024,
                        "final_memory_mb": test_results.get("final_memory", 0) / 1024 / 1024
                    },
                    metadata={"max_allowed": f"{self.MAX_MEMORY_GROWTH_PERCENT}%"}
                ).to_dict()
            
            # Check error rate
            error_rate = test_results.get("error_rate", 0)
            if error_rate > self.MAX_ERROR_RATE:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.FAILED,
                    message=f"Error rate {error_rate:.2%} exceeds threshold",
                    evidence={
                        "error_rate": error_rate,
                        "total_errors": test_results.get("total_errors", 0),
                        "total_operations": test_results.get("total_operations", 0)
                    },
                    metadata={"max_allowed": f"{self.MAX_ERROR_RATE:.1%}"}
                ).to_dict()
            
            # Check performance degradation
            latency_degradation = self._calculate_latency_degradation(test_results)
            if latency_degradation > self.MAX_LATENCY_DEGRADATION:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.WARNING,
                    message=f"Performance degradation detected: {latency_degradation:.1f}x slower",
                    evidence={
                        "latency_degradation": latency_degradation,
                        "initial_latency_ms": test_results.get("initial_latency", 0),
                        "final_latency_ms": test_results.get("final_latency", 0)
                    },
                    metadata={"max_allowed": f"{self.MAX_LATENCY_DEGRADATION}x"}
                ).to_dict()
            
            # Check CPU usage
            avg_cpu = test_results.get("avg_cpu_percent", 0)
            if avg_cpu > self.MAX_CPU_USAGE:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.WARNING,
                    message=f"High CPU usage: {avg_cpu:.1f}%",
                    evidence={
                        "avg_cpu_percent": avg_cpu,
                        "peak_cpu_percent": test_results.get("peak_cpu_percent", 0)
                    },
                    metadata={"max_recommended": f"{self.MAX_CPU_USAGE}%"}
                ).to_dict()
            
            # Check uptime
            uptime_percent = test_results.get("uptime_percent", 0)
            if uptime_percent < self.MIN_UPTIME_PERCENT:
                return ValidationResult(
                    check_id="TECH-003",
                    status=CheckStatus.FAILED,
                    message=f"Uptime {uptime_percent:.2f}% below requirement",
                    evidence={
                        "uptime_percent": uptime_percent,
                        "total_downtime_seconds": test_results.get("total_downtime", 0)
                    },
                    metadata={"required": f"{self.MIN_UPTIME_PERCENT}%"}
                ).to_dict()
            
            # Generate stability report
            report = await self._generate_stability_report(test_results)
            
            return ValidationResult(
                check_id="TECH-003",
                status=CheckStatus.PASSED,
                message=f"System stable for {test_results['duration_hours']:.1f} hours",
                evidence={
                    "duration_hours": test_results["duration_hours"],
                    "memory_growth": f"{memory_growth:.1%}",
                    "error_rate": f"{error_rate:.3%}",
                    "uptime": f"{uptime_percent:.2f}%",
                    "avg_cpu": f"{avg_cpu:.1f}%"
                },
                metadata={"report": report}
            ).to_dict()
            
        except Exception as e:
            logger.error("Stability validation failed", error=str(e))
            return ValidationResult(
                check_id="TECH-003",
                status=CheckStatus.FAILED,
                message=f"Validation error: {str(e)}",
                evidence={"error": str(e)},
                metadata={}
            ).to_dict()
    
    async def _load_stability_test_results(self) -> Optional[Dict[str, Any]]:
        """Load stability test results from file."""
        if not self.test_results_file.exists():
            return None
        
        try:
            with open(self.test_results_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load stability test results", error=str(e))
            return None
    
    async def _run_stability_test(self) -> Dict[str, Any]:
        """Run extended stability test (abbreviated for development)."""
        logger.info("Starting stability test", duration_hours=self.TEST_DURATION_HOURS)
        
        # Initialize metrics
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        initial_latency = await self._measure_latency()
        
        errors = 0
        operations = 0
        cpu_samples = []
        memory_samples = []
        latency_samples = []
        downtime = 0
        
        # For development, run abbreviated test
        test_duration = min(60, self.TEST_DURATION_HOURS * 3600)  # 1 minute for dev
        
        while time.time() - start_time < test_duration:
            try:
                # Simulate operations
                operations += 1
                
                # Sample metrics
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                memory_samples.append(psutil.Process().memory_info().rss)
                latency_samples.append(await self._measure_latency())
                
                # Simulate random errors (for testing)
                import random
                if random.random() < 0.0001:  # 0.01% error rate
                    errors += 1
                
                await asyncio.sleep(1)
                
            except Exception as e:
                errors += 1
                logger.error("Error during stability test", error=str(e))
        
        # Calculate final metrics
        duration_hours = (time.time() - start_time) / 3600
        final_memory = psutil.Process().memory_info().rss
        final_latency = await self._measure_latency()
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "duration_hours": duration_hours,
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "initial_latency": initial_latency,
            "final_latency": final_latency,
            "total_errors": errors,
            "total_operations": operations,
            "error_rate": errors / max(1, operations),
            "avg_cpu_percent": sum(cpu_samples) / max(1, len(cpu_samples)),
            "peak_cpu_percent": max(cpu_samples) if cpu_samples else 0,
            "avg_memory": sum(memory_samples) / max(1, len(memory_samples)),
            "peak_memory": max(memory_samples) if memory_samples else 0,
            "avg_latency": sum(latency_samples) / max(1, len(latency_samples)),
            "uptime_percent": 100 * (1 - downtime / (time.time() - start_time)),
            "total_downtime": downtime
        }
        
        # Save results
        self.test_results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    async def _measure_latency(self) -> float:
        """Measure system latency in milliseconds."""
        start = time.perf_counter()
        
        # Simulate some work
        await asyncio.sleep(0.001)
        
        # Simple computation
        _ = sum(range(1000))
        
        return (time.perf_counter() - start) * 1000
    
    def _calculate_memory_growth(self, test_results: Dict[str, Any]) -> float:
        """Calculate memory growth percentage."""
        initial = test_results.get("initial_memory", 1)
        final = test_results.get("final_memory", initial)
        
        if initial <= 0:
            return 0
        
        return (final - initial) / initial
    
    def _calculate_latency_degradation(self, test_results: Dict[str, Any]) -> float:
        """Calculate latency degradation factor."""
        initial = test_results.get("initial_latency", 1)
        final = test_results.get("final_latency", initial)
        
        if initial <= 0:
            return 1
        
        return final / initial
    
    async def _generate_stability_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed stability report."""
        return {
            "test_duration": f"{test_results['duration_hours']:.1f} hours",
            "memory": {
                "initial_mb": test_results.get("initial_memory", 0) / 1024 / 1024,
                "final_mb": test_results.get("final_memory", 0) / 1024 / 1024,
                "peak_mb": test_results.get("peak_memory", 0) / 1024 / 1024,
                "avg_mb": test_results.get("avg_memory", 0) / 1024 / 1024,
                "growth_percent": self._calculate_memory_growth(test_results) * 100
            },
            "cpu": {
                "average_percent": test_results.get("avg_cpu_percent", 0),
                "peak_percent": test_results.get("peak_cpu_percent", 0)
            },
            "latency": {
                "initial_ms": test_results.get("initial_latency", 0),
                "final_ms": test_results.get("final_latency", 0),
                "average_ms": test_results.get("avg_latency", 0),
                "degradation_factor": self._calculate_latency_degradation(test_results)
            },
            "reliability": {
                "uptime_percent": test_results.get("uptime_percent", 0),
                "total_operations": test_results.get("total_operations", 0),
                "total_errors": test_results.get("total_errors", 0),
                "error_rate_percent": test_results.get("error_rate", 0) * 100
            },
            "test_completed": datetime.fromisoformat(test_results.get("timestamp", datetime.utcnow().isoformat()))
        }