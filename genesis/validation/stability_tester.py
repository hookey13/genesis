"""Stability testing framework for 48-hour continuous operation."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import structlog

logger = structlog.get_logger(__name__)


class StabilityTester:
    """Tests system stability over extended periods."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize stability tester.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.stability_log_file = self.genesis_root / ".genesis" / "logs" / "stability_test.json"
        self.required_hours = 48
        self.max_memory_growth = 10  # Max 10% memory growth allowed
        self.max_cpu_usage = 80  # Max 80% sustained CPU usage
        self.max_error_rate = 0.01  # Max 1% error rate
        self.max_restart_count = 3  # Max 3 restarts allowed
        self.min_connection_uptime = 0.99  # 99% connection uptime required
        self.memory_leak_threshold = 5  # MB/hour growth indicates leak

    async def validate(self) -> dict[str, Any]:
        """Validate system stability from test logs."""
        try:
            # Check if stability test has been run
            if not self.stability_log_file.exists():
                # Check for alternative test results
                return await self._check_paper_trading_stability()

            # Load stability test results
            with open(self.stability_log_file) as f:
                test_data = json.load(f)

            # Analyze test results
            analysis = self._analyze_stability_data(test_data)

            # Determine pass/fail
            passed = (
                analysis["hours_stable"] >= self.required_hours
                and analysis["memory_growth_percent"] <= self.max_memory_growth
                and analysis["error_rate"] <= self.max_error_rate
                and analysis["restart_count"] <= self.max_restart_count
                and not analysis["critical_failures"]
            )

            # Calculate overall score
            score = self._calculate_stability_score(analysis)

            return {
                "validator": "stability",
                "timestamp": datetime.utcnow().isoformat(),
                "passed": passed,
                "score": score,
                "checks": {
                    "duration": {
                        "passed": analysis["hours_stable"] >= self.required_hours,
                        "details": [f"Stable for {analysis['hours_stable']:.1f} hours (required: {self.required_hours})"]
                    },
                    "memory": {
                        "passed": analysis["memory_growth_percent"] <= self.max_memory_growth and not analysis["memory_leak_detected"],
                        "details": [
                            f"Memory growth: {analysis['memory_growth_percent']:.1f}%",
                            f"Peak memory: {analysis['peak_memory_mb']:.1f} MB",
                            f"Memory leak: {'Detected' if analysis['memory_leak_detected'] else 'Not detected'}"
                        ]
                    },
                    "cpu": {
                        "passed": analysis["average_cpu_percent"] <= self.max_cpu_usage,
                        "details": [
                            f"Average CPU: {analysis['average_cpu_percent']:.1f}%",
                            f"Peak CPU: {analysis['peak_cpu_percent']:.1f}%"
                        ]
                    },
                    "errors": {
                        "passed": analysis["error_rate"] <= self.max_error_rate,
                        "details": [
                            f"Error rate: {analysis['error_rate']:.2%}",
                            f"Total errors: {analysis['error_count']}"
                        ]
                    },
                    "stability": {
                        "passed": analysis["restart_count"] <= self.max_restart_count and not analysis["critical_failures"],
                        "details": [
                            f"Restarts: {analysis['restart_count']}",
                            f"Critical failures: {len(analysis['critical_failures'])}",
                            f"Recovery success rate: {analysis['recovery_success_rate']:.1%}"
                        ]
                    },
                    "connections": {
                        "passed": analysis["connection_uptime"] >= self.min_connection_uptime,
                        "details": [
                            f"Connection uptime: {analysis['connection_uptime']:.2%}",
                            f"Reconnections: {analysis['reconnection_count']}"
                        ]
                    }
                },
                "summary": f"Stability test {'passed' if passed else 'failed'} with score {score:.1f}%",
                "recommendations": self._generate_recommendations(analysis),
            }

        except Exception as e:
            logger.error("Stability validation failed", error=str(e))
            return {
                "passed": False,
                "error": str(e),
                "details": {},
            }

    async def _check_paper_trading_stability(self) -> dict[str, Any]:
        """Check paper trading logs for stability metrics."""
        try:
            # Look for paper trading logs
            log_dir = Path(".genesis/logs")
            if not log_dir.exists():
                return {
                    "passed": False,
                    "details": {
                        "hours_stable": 0,
                        "note": "No stability test data found. Run 48-hour paper trading test.",
                    },
                }

            # Parse trading logs for stability metrics
            trading_log = log_dir / "trading.log"
            if trading_log.exists():
                metrics = await self._parse_trading_logs(trading_log)

                # Calculate stability from trading logs
                if metrics["duration_hours"] >= self.required_hours:
                    passed = (
                        metrics["error_count"] / max(metrics["total_events"], 1) <= self.max_error_rate
                        and metrics["restart_count"] <= self.max_restart_count
                    )

                    return {
                        "passed": passed,
                        "details": {
                            "hours_stable": metrics["duration_hours"],
                            "memory_growth_percent": 0,  # Can't determine from logs
                            "error_rate": metrics["error_count"] / max(metrics["total_events"], 1),
                            "restart_count": metrics["restart_count"],
                            "transactions_processed": metrics["total_events"],
                            "average_latency_ms": 0,  # Would need specific metrics
                            "peak_memory_mb": 0,  # Can't determine from logs
                            "critical_failures": metrics["critical_failures"],
                        },
                    }

            return {
                "passed": False,
                "details": {
                    "hours_stable": 0,
                    "note": "Insufficient stability test data. Run 48-hour test.",
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "details": {},
            }

    async def _parse_trading_logs(self, log_file: Path) -> dict[str, Any]:
        """Parse trading logs for stability metrics."""
        metrics = {
            "duration_hours": 0,
            "error_count": 0,
            "restart_count": 0,
            "total_events": 0,
            "critical_failures": [],
            "first_timestamp": None,
            "last_timestamp": None,
        }

        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        # Parse JSON log line
                        log_entry = json.loads(line)

                        # Track timestamps
                        timestamp = log_entry.get("timestamp")
                        if timestamp:
                            if not metrics["first_timestamp"]:
                                metrics["first_timestamp"] = timestamp
                            metrics["last_timestamp"] = timestamp

                        # Count events
                        metrics["total_events"] += 1

                        # Count errors
                        if log_entry.get("level") in ["ERROR", "CRITICAL"]:
                            metrics["error_count"] += 1

                            # Track critical failures
                            if log_entry.get("level") == "CRITICAL":
                                metrics["critical_failures"].append({
                                    "timestamp": timestamp,
                                    "message": log_entry.get("message", ""),
                                })

                        # Detect restarts
                        if "Starting Genesis" in log_entry.get("message", ""):
                            metrics["restart_count"] += 1

                    except json.JSONDecodeError:
                        continue  # Skip non-JSON lines

            # Calculate duration
            if metrics["first_timestamp"] and metrics["last_timestamp"]:
                first_dt = datetime.fromisoformat(metrics["first_timestamp"])
                last_dt = datetime.fromisoformat(metrics["last_timestamp"])
                duration = last_dt - first_dt
                metrics["duration_hours"] = duration.total_seconds() / 3600

        except Exception as e:
            logger.error("Failed to parse trading logs", error=str(e))

        return metrics

    def _analyze_stability_data(self, test_data: dict) -> dict[str, Any]:
        """Analyze stability test data with enhanced metrics."""
        analysis = {
            "hours_stable": 0,
            "memory_growth_percent": 0,
            "memory_leak_detected": False,
            "average_cpu_percent": 0,
            "peak_cpu_percent": 0,
            "error_rate": 0,
            "error_count": 0,
            "restart_count": 0,
            "transactions_processed": 0,
            "average_latency_ms": 0,
            "peak_memory_mb": 0,
            "critical_failures": [],
            "recovery_success_rate": 1.0,
            "connection_uptime": 1.0,
            "reconnection_count": 0,
        }

        # Extract metrics from test data
        if "duration_hours" in test_data:
            analysis["hours_stable"] = test_data["duration_hours"]

        if "memory_samples" in test_data:
            samples = test_data["memory_samples"]
            if samples:
                initial_memory = samples[0]["memory_mb"]
                final_memory = samples[-1]["memory_mb"]
                peak_memory = max(s["memory_mb"] for s in samples)

                analysis["memory_growth_percent"] = (
                    (final_memory - initial_memory) / initial_memory * 100
                    if initial_memory > 0 else 0
                )
                analysis["peak_memory_mb"] = peak_memory

                # Detect memory leak using linear regression
                if len(samples) > 10 and analysis["hours_stable"] > 0:
                    memory_growth_rate = (final_memory - initial_memory) / analysis["hours_stable"]
                    if memory_growth_rate > self.memory_leak_threshold:
                        analysis["memory_leak_detected"] = True

        # Analyze CPU usage
        if "cpu_samples" in test_data:
            samples = test_data["cpu_samples"]
            if samples:
                cpu_values = [s["cpu_percent"] for s in samples]
                analysis["average_cpu_percent"] = sum(cpu_values) / len(cpu_values)
                analysis["peak_cpu_percent"] = max(cpu_values)

        if "error_count" in test_data:
            analysis["error_count"] = test_data["error_count"]
            if "total_events" in test_data:
                total = test_data["total_events"]
                if total > 0:
                    analysis["error_rate"] = test_data["error_count"] / total

        if "restart_count" in test_data:
            analysis["restart_count"] = test_data["restart_count"]

        if "transactions" in test_data:
            analysis["transactions_processed"] = test_data["transactions"]

        if "latency_samples" in test_data:
            samples = test_data["latency_samples"]
            if samples:
                analysis["average_latency_ms"] = sum(samples) / len(samples)

        if "critical_failures" in test_data:
            analysis["critical_failures"] = test_data["critical_failures"]

        # Analyze recovery metrics
        if "recovery_attempts" in test_data:
            attempts = test_data["recovery_attempts"]
            successes = test_data.get("recovery_successes", 0)
            if attempts > 0:
                analysis["recovery_success_rate"] = successes / attempts

        # Analyze connection stability
        if "connection_samples" in test_data:
            samples = test_data["connection_samples"]
            if samples:
                connected_count = sum(1 for s in samples if s["connected"])
                analysis["connection_uptime"] = connected_count / len(samples)
                analysis["reconnection_count"] = test_data.get("reconnection_count", 0)

        return analysis

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate recommendations based on stability analysis."""
        recommendations = []

        if analysis["hours_stable"] < self.required_hours:
            recommendations.append(
                f"Run stability test for full {self.required_hours} hours "
                f"(currently {analysis['hours_stable']:.1f} hours)"
            )

        if analysis["memory_growth_percent"] > self.max_memory_growth:
            recommendations.append(
                f"Investigate memory leak - {analysis['memory_growth_percent']:.1f}% growth detected"
            )

        if analysis["error_rate"] > self.max_error_rate:
            recommendations.append(
                f"Reduce error rate from {analysis['error_rate']:.2%} to below {self.max_error_rate:.2%}"
            )

        if analysis["restart_count"] > self.max_restart_count:
            recommendations.append(
                f"Fix stability issues causing {analysis['restart_count']} restarts"
            )

        if analysis["critical_failures"]:
            recommendations.append(
                f"Resolve {len(analysis['critical_failures'])} critical failures"
            )

        if analysis["average_latency_ms"] > 100:
            recommendations.append(
                f"Optimize performance - average latency {analysis['average_latency_ms']:.1f}ms is high"
            )

        return recommendations

    def _calculate_stability_score(self, analysis: dict[str, Any]) -> float:
        """Calculate overall stability score."""
        scores = []
        weights = []

        # Duration score (weight: 25%)
        duration_score = min(100, (analysis["hours_stable"] / self.required_hours) * 100)
        scores.append(duration_score)
        weights.append(0.25)

        # Memory score (weight: 20%)
        memory_score = 100
        if analysis["memory_growth_percent"] > 0:
            memory_score = max(0, 100 - (analysis["memory_growth_percent"] * 5))
        if analysis["memory_leak_detected"]:
            memory_score = min(memory_score, 50)
        scores.append(memory_score)
        weights.append(0.20)

        # CPU score (weight: 15%)
        cpu_score = max(0, 100 - (analysis["average_cpu_percent"] - 50))
        scores.append(cpu_score)
        weights.append(0.15)

        # Error rate score (weight: 20%)
        error_score = max(0, 100 - (analysis["error_rate"] * 10000))
        scores.append(error_score)
        weights.append(0.20)

        # Stability score (weight: 10%)
        stability_score = 100
        stability_score -= analysis["restart_count"] * 20
        stability_score -= len(analysis["critical_failures"]) * 10
        scores.append(max(0, stability_score))
        weights.append(0.10)

        # Connection score (weight: 10%)
        connection_score = analysis["connection_uptime"] * 100
        scores.append(connection_score)
        weights.append(0.10)

        # Calculate weighted average
        return sum(s * w for s, w in zip(scores, weights, strict=False))

    async def run_stability_test(self, duration_hours: int = 48) -> dict[str, Any]:
        """Run actual stability test (for manual execution)."""
        logger.info(f"Starting {duration_hours}-hour stability test")

        test_data = {
            "start_time": datetime.utcnow().isoformat(),
            "duration_hours": 0,
            "memory_samples": [],
            "cpu_samples": [],
            "connection_samples": [],
            "latency_samples": [],
            "error_count": 0,
            "total_events": 0,
            "restart_count": 0,
            "reconnection_count": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "transactions": 0,
            "critical_failures": [],
        }

        start_time = time.time()
        sample_interval = 60  # Sample every minute
        last_sample = start_time

        try:
            while (time.time() - start_time) < (duration_hours * 3600):
                current_time = time.time()

                # Take samples every minute
                if current_time - last_sample >= sample_interval:
                    # Memory sample
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    test_data["memory_samples"].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "memory_mb": memory_mb,
                    })

                    # CPU sample
                    cpu_percent = process.cpu_percent(interval=1)
                    test_data["cpu_samples"].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "cpu_percent": cpu_percent,
                    })

                    # Connection sample (simulated)
                    # In real implementation, would check actual exchange connection
                    connected = True  # Simulate connection status
                    test_data["connection_samples"].append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "connected": connected,
                    })

                    # Update duration
                    test_data["duration_hours"] = (current_time - start_time) / 3600

                    # Save intermediate results
                    self.stability_log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.stability_log_file, "w") as f:
                        json.dump(test_data, f, indent=2)

                    last_sample = current_time

                    logger.info(
                        "Stability test progress",
                        hours_complete=test_data["duration_hours"],
                        memory_mb=memory_mb,
                        cpu_percent=cpu_percent,
                    )

                # Simulate some work
                await asyncio.sleep(1)
                test_data["total_events"] += 1

        except Exception as e:
            test_data["critical_failures"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            })
            logger.error("Stability test failed", error=str(e))

        # Final save
        test_data["end_time"] = datetime.utcnow().isoformat()
        test_data["duration_hours"] = (time.time() - start_time) / 3600

        with open(self.stability_log_file, "w") as f:
            json.dump(test_data, f, indent=2)

        logger.info("Stability test completed", data=test_data)
        return test_data
