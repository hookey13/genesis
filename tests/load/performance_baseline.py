"""
Performance baseline tracking for Project GENESIS.
Establishes and monitors performance benchmarks over time.
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from decimal import Decimal
import structlog

logger = structlog.get_logger()


class PerformanceBaseline:
    """Track and compare performance metrics against baselines."""
    
    def __init__(self, baseline_file: str = ".genesis/baselines/performance.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_metrics = {}
        self.baselines = self.load_baselines()
    
    def load_baselines(self) -> Dict[str, Any]:
        """Load existing performance baselines."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baselines(self):
        """Save current baselines to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2, default=str)
    
    async def measure_latency(self, operation: str, func, *args, **kwargs):
        """Measure operation latency."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if operation not in self.current_metrics:
            self.current_metrics[operation] = []
        self.current_metrics[operation].append(latency_ms)
        
        return result, latency_ms
    
    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "p50": sorted_values[int(n * 0.5)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1],
            "mean": sum(sorted_values) / n,
            "min": sorted_values[0],
            "max": sorted_values[-1]
        }
    
    def compare_with_baseline(self, metric: str, current_value: float) -> Dict[str, Any]:
        """Compare current metric with baseline."""
        if metric not in self.baselines:
            return {
                "status": "no_baseline",
                "current": current_value,
                "message": f"No baseline for {metric}"
            }
        
        baseline = self.baselines[metric]["value"]
        threshold = self.baselines[metric].get("threshold", 1.1)  # 10% tolerance
        
        if current_value <= baseline * threshold:
            status = "pass"
            message = f"{metric}: {current_value:.2f}ms (baseline: {baseline:.2f}ms)"
        else:
            status = "fail"
            degradation = ((current_value - baseline) / baseline) * 100
            message = f"{metric}: {current_value:.2f}ms degraded {degradation:.1f}% from baseline {baseline:.2f}ms"
        
        return {
            "status": status,
            "current": current_value,
            "baseline": baseline,
            "threshold": threshold,
            "message": message
        }
    
    def update_baseline(self, metric: str, value: float, threshold: float = 1.1):
        """Update or create a baseline."""
        self.baselines[metric] = {
            "value": value,
            "threshold": threshold,
            "updated": datetime.utcnow().isoformat(),
            "samples": self.current_metrics.get(metric, [])[:100]  # Keep last 100 samples
        }
        self.save_baselines()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance comparison report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "summary": {
                "total_metrics": 0,
                "passed": 0,
                "failed": 0,
                "no_baseline": 0
            }
        }
        
        for operation, latencies in self.current_metrics.items():
            percentiles = self.calculate_percentiles(latencies)
            comparison = self.compare_with_baseline(f"{operation}_p95", percentiles.get("p95", 0))
            
            report["metrics"][operation] = {
                "percentiles": percentiles,
                "comparison": comparison,
                "samples": len(latencies)
            }
            
            report["summary"]["total_metrics"] += 1
            report["summary"][comparison["status"]] = report["summary"].get(comparison["status"], 0) + 1
        
        return report


class PerformanceBenchmarks:
    """Standard performance benchmarks for GENESIS components."""
    
    # Latency requirements from Epic 10 specs
    REQUIREMENTS = {
        "market_analysis_latency_ms": 10,      # <10ms per pair
        "order_execution_latency_ms": 50,      # <50ms
        "strategy_calculation_ms": 100,        # <100ms per cycle
        "websocket_processing_ms": 5,          # <5ms
        "database_query_ms": 10,              # <10ms
        "state_recovery_seconds": 5,           # <5s restart
    }
    
    # Memory and CPU requirements
    RESOURCE_LIMITS = {
        "memory_usage_mb": 2048,              # <2GB normal load
        "cpu_usage_percent": 50,              # <50% with 10 strategies
        "open_connections": 100,              # Max concurrent connections
    }
    
    # Throughput requirements
    THROUGHPUT = {
        "orders_per_second": 1000,            # 1000+ orders/sec
        "market_updates_per_second": 10000,   # 10k+ updates/sec
        "strategies_concurrent": 10,          # 10 concurrent strategies
    }
    
    @classmethod
    def validate_latency(cls, operation: str, latency_ms: float) -> bool:
        """Validate latency against requirements."""
        requirement = cls.REQUIREMENTS.get(operation)
        if requirement:
            return latency_ms <= requirement
        return True
    
    @classmethod
    def validate_throughput(cls, metric: str, value: float) -> bool:
        """Validate throughput against requirements."""
        requirement = cls.THROUGHPUT.get(metric)
        if requirement:
            return value >= requirement
        return True
    
    @classmethod
    def validate_resources(cls, metric: str, value: float) -> bool:
        """Validate resource usage against limits."""
        limit = cls.RESOURCE_LIMITS.get(metric)
        if limit:
            return value <= limit
        return True


async def run_baseline_tests():
    """Run standard baseline performance tests."""
    baseline = PerformanceBaseline()
    benchmarks = PerformanceBenchmarks()
    
    # Simulate various operations
    async def mock_market_analysis():
        await asyncio.sleep(0.008)  # 8ms
        return {"opportunity": "found"}
    
    async def mock_order_execution():
        await asyncio.sleep(0.04)  # 40ms
        return {"order_id": "12345"}
    
    async def mock_strategy_calc():
        await asyncio.sleep(0.08)  # 80ms
        return {"signal": "buy"}
    
    async def mock_db_query():
        await asyncio.sleep(0.007)  # 7ms
        return {"data": "result"}
    
    # Run performance measurements
    logger.info("Running baseline performance tests...")
    
    # Market analysis (100 samples)
    for _ in range(100):
        _, latency = await baseline.measure_latency(
            "market_analysis", mock_market_analysis
        )
    
    # Order execution (100 samples)
    for _ in range(100):
        _, latency = await baseline.measure_latency(
            "order_execution", mock_order_execution
        )
    
    # Strategy calculation (50 samples)
    for _ in range(50):
        _, latency = await baseline.measure_latency(
            "strategy_calculation", mock_strategy_calc
        )
    
    # Database queries (200 samples)
    for _ in range(200):
        _, latency = await baseline.measure_latency(
            "database_query", mock_db_query
        )
    
    # Generate report
    report = baseline.generate_report()
    
    # Validate against requirements
    results = {
        "passed": [],
        "failed": []
    }
    
    for operation, data in report["metrics"].items():
        p95_latency = data["percentiles"]["p95"]
        requirement_key = f"{operation}_latency_ms"
        
        if benchmarks.validate_latency(requirement_key, p95_latency):
            results["passed"].append(f"{operation}: {p95_latency:.2f}ms")
        else:
            requirement = benchmarks.REQUIREMENTS.get(requirement_key, "N/A")
            results["failed"].append(
                f"{operation}: {p95_latency:.2f}ms (requirement: <{requirement}ms)"
            )
    
    # Update baselines if all tests pass
    if not results["failed"]:
        logger.info("All performance tests passed. Updating baselines...")
        for operation, data in report["metrics"].items():
            baseline.update_baseline(
                f"{operation}_p95",
                data["percentiles"]["p95"],
                threshold=1.1  # 10% tolerance
            )
    
    # Save detailed report
    report_file = Path(".genesis/reports/performance_baseline.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump({
            "report": report,
            "validation": results,
            "requirements": {
                "latency": benchmarks.REQUIREMENTS,
                "throughput": benchmarks.THROUGHPUT,
                "resources": benchmarks.RESOURCE_LIMITS
            }
        }, f, indent=2, default=str)
    
    logger.info("Performance baseline report saved", 
                report_file=str(report_file),
                passed=len(results["passed"]),
                failed=len(results["failed"]))
    
    return results


if __name__ == "__main__":
    asyncio.run(run_baseline_tests())