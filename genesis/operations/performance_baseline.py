"""Performance Baseline Recalculation System."""

import asyncio
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Deque, Optional, Any

from pydantic import BaseModel, Field

from genesis.utils.logger import get_logger, LoggerType


class PerformanceMetric(BaseModel):
    """Performance metric with statistical analysis."""
    
    operation: str
    samples: List[float]
    p50: float
    p95: float
    p99: float
    mean: float
    std_dev: float
    last_updated: datetime


class PerformanceBaseline:
    """Manages performance baseline calculations."""
    
    def __init__(self, window_size: int = 1000, recalc_interval_hours: int = 168):
        self.window_size = window_size
        self.recalc_interval_hours = recalc_interval_hours
        self.metrics: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines: Dict[str, PerformanceMetric] = {}
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
    
    def record_metric(self, operation: str, latency_ms: float) -> None:
        """Record a performance metric."""
        self.metrics[operation].append(latency_ms)
        
        # Check for degradation
        if operation in self.baselines:
            baseline = self.baselines[operation]
            if latency_ms > baseline.mean + (2 * baseline.std_dev):
                self.logger.warning(
                    "performance_degradation",
                    operation=operation,
                    latency_ms=latency_ms,
                    threshold_ms=baseline.mean + (2 * baseline.std_dev)
                )
    
    async def recalculate_baselines(self) -> Dict[str, PerformanceMetric]:
        """Recalculate all performance baselines."""
        new_baselines = {}
        
        for operation, samples in self.metrics.items():
            if len(samples) >= 10:  # Minimum samples
                sorted_samples = sorted(samples)
                
                metric = PerformanceMetric(
                    operation=operation,
                    samples=list(samples),
                    p50=sorted_samples[len(sorted_samples) // 2],
                    p95=sorted_samples[int(len(sorted_samples) * 0.95)],
                    p99=sorted_samples[int(len(sorted_samples) * 0.99)],
                    mean=statistics.mean(samples),
                    std_dev=statistics.stdev(samples) if len(samples) > 1 else 0,
                    last_updated=datetime.utcnow()
                )
                
                new_baselines[operation] = metric
        
        self.baselines = new_baselines
        self.logger.info(
            "baselines_recalculated",
            operations=len(new_baselines)
        )
        
        return new_baselines
    
    async def export_to_prometheus(self) -> None:
        """Export baselines to Prometheus."""
        # Implementation would export metrics to Prometheus
        pass
    
    async def start_automatic_recalculation(self) -> None:
        """Start automatic baseline recalculation."""
        while True:
            await asyncio.sleep(self.recalc_interval_hours * 3600)
            await self.recalculate_baselines()
            await self.export_to_prometheus()