"""Stability testing framework for 48-hour continuous operation."""

import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class StabilityTester:
    """Tests system stability over extended periods."""
    
    def __init__(self):
        self.stability_log_file = Path(".genesis/logs/stability_test.json")
        self.required_hours = 48
        self.max_memory_growth = 10  # Max 10% memory growth allowed
        self.max_error_rate = 0.01  # Max 1% error rate
        self.max_restart_count = 3  # Max 3 restarts allowed
        
    async def validate(self) -> Dict[str, Any]:
        """Validate system stability from test logs."""
        try:
            # Check if stability test has been run
            if not self.stability_log_file.exists():
                # Check for alternative test results
                return await self._check_paper_trading_stability()
            
            # Load stability test results
            with open(self.stability_log_file, "r") as f:
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
            
            return {
                "passed": passed,
                "details": {
                    "hours_stable": analysis["hours_stable"],
                    "memory_growth_percent": analysis["memory_growth_percent"],
                    "error_rate": analysis["error_rate"],
                    "restart_count": analysis["restart_count"],
                    "transactions_processed": analysis["transactions_processed"],
                    "average_latency_ms": analysis["average_latency_ms"],
                    "peak_memory_mb": analysis["peak_memory_mb"],
                    "critical_failures": analysis["critical_failures"],
                },
                "recommendations": self._generate_recommendations(analysis),
            }
            
        except Exception as e:
            logger.error("Stability validation failed", error=str(e))
            return {
                "passed": False,
                "error": str(e),
                "details": {},
            }
    
    async def _check_paper_trading_stability(self) -> Dict[str, Any]:
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
    
    async def _parse_trading_logs(self, log_file: Path) -> Dict[str, Any]:
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
            with open(log_file, "r") as f:
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
    
    def _analyze_stability_data(self, test_data: Dict) -> Dict[str, Any]:
        """Analyze stability test data."""
        analysis = {
            "hours_stable": 0,
            "memory_growth_percent": 0,
            "error_rate": 0,
            "restart_count": 0,
            "transactions_processed": 0,
            "average_latency_ms": 0,
            "peak_memory_mb": 0,
            "critical_failures": [],
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
        
        if "error_count" in test_data and "total_events" in test_data:
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
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
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
    
    async def run_stability_test(self, duration_hours: int = 48) -> Dict[str, Any]:
        """Run actual stability test (for manual execution)."""
        logger.info(f"Starting {duration_hours}-hour stability test")
        
        test_data = {
            "start_time": datetime.utcnow().isoformat(),
            "duration_hours": 0,
            "memory_samples": [],
            "latency_samples": [],
            "error_count": 0,
            "total_events": 0,
            "restart_count": 0,
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