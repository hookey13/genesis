"""
Memory profiler for detecting memory leaks and monitoring resource usage.
Provides continuous memory monitoring, leak detection, and growth analysis.
"""

import asyncio
import gc
import tracemalloc
import time
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
import structlog
from collections import deque
import statistics

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a point in time."""
    timestamp: datetime
    rss_bytes: int  # Resident Set Size
    vms_bytes: int  # Virtual Memory Size
    available_bytes: int
    percent_used: float
    gc_stats: Dict[str, int]
    top_allocations: List[Tuple[str, int]]
    tracemalloc_snapshot: Optional[Any] = None


@dataclass
class MemoryTrend:
    """Memory usage trend analysis."""
    growth_rate_per_hour: float
    average_usage_bytes: int
    peak_usage_bytes: int
    leak_detected: bool
    leak_confidence: float  # 0.0 to 1.0
    estimated_time_to_oom: Optional[float] = None  # hours


@dataclass
class LeakDetectionResult:
    """Result of memory leak detection analysis."""
    has_leak: bool
    confidence: float
    growth_rate: float
    suspicious_allocations: List[Tuple[str, int, float]]  # (location, size, growth_rate)
    recommendation: str


class MemoryProfiler:
    """
    Advanced memory profiler for detecting leaks and monitoring usage patterns.
    """
    
    def __init__(
        self,
        growth_threshold: float = 0.05,  # 5% growth threshold
        snapshot_interval: int = 60,  # seconds between snapshots
        history_size: int = 1440,  # 24 hours of minute snapshots
        enable_tracemalloc: bool = True
    ):
        self.growth_threshold = growth_threshold
        self.snapshot_interval = snapshot_interval
        self.history_size = history_size
        self.enable_tracemalloc = enable_tracemalloc
        
        # Memory tracking
        self.baseline_memory: Optional[int] = None
        self.snapshots: deque = deque(maxlen=history_size)
        self.tracemalloc_snapshots: deque = deque(maxlen=10)  # Keep last 10 for comparison
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.start_time: Optional[datetime] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'memory_percent': 80.0,
            'growth_rate_per_hour': 0.05,  # 5% per hour
            'leak_confidence': 0.7,
        }
        
        # Process handle
        self.process = psutil.Process()
        
        logger.info("MemoryProfiler initialized", 
                   growth_threshold=growth_threshold,
                   snapshot_interval=snapshot_interval)
    
    async def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        if self.is_monitoring:
            logger.warning("Memory monitoring already active")
            return
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Store top 10 frames
            logger.info("Tracemalloc started")
        
        # Take baseline snapshot
        self.baseline_memory = self.process.memory_info().rss
        await self.take_snapshot()
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory monitoring started", baseline_memory_mb=self.baseline_memory / 1024 / 1024)
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.snapshot_interval)
                
                snapshot = await self.take_snapshot()
                
                # Check for alerts
                if snapshot.percent_used > self.alert_thresholds['memory_percent']:
                    logger.warning("High memory usage alert",
                                 percent_used=snapshot.percent_used,
                                 threshold=self.alert_thresholds['memory_percent'])
                
                # Detect leaks periodically (every 10 snapshots)
                if len(self.snapshots) % 10 == 0 and len(self.snapshots) > 20:
                    leak_result = self.detect_leaks()
                    if leak_result.has_leak:
                        logger.error("Memory leak detected",
                                   confidence=leak_result.confidence,
                                   growth_rate=leak_result.growth_rate,
                                   recommendation=leak_result.recommendation)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
    
    async def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        memory_info = self.process.memory_info()
        memory_stats = psutil.virtual_memory()
        
        # Get GC stats
        gc_stats = {
            f"generation_{i}": gc.get_count()[i] 
            for i in range(gc.get_count().__len__())
        }
        gc_stats['objects'] = len(gc.get_objects())
        
        # Get top allocations if tracemalloc is enabled
        top_allocations = []
        tracemalloc_snapshot = None
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc_snapshot = tracemalloc.take_snapshot()
            self.tracemalloc_snapshots.append(tracemalloc_snapshot)
            
            top_stats = tracemalloc_snapshot.statistics('lineno')[:10]
            for stat in top_stats:
                top_allocations.append((str(stat.traceback), stat.size))
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_bytes=memory_info.rss,
            vms_bytes=memory_info.vms,
            available_bytes=memory_stats.available,
            percent_used=memory_stats.percent,
            gc_stats=gc_stats,
            top_allocations=top_allocations,
            tracemalloc_snapshot=tracemalloc_snapshot
        )
        
        self.snapshots.append(snapshot)
        
        logger.debug("Memory snapshot taken",
                    rss_mb=memory_info.rss / 1024 / 1024,
                    percent_used=memory_stats.percent)
        
        return snapshot
    
    def detect_leaks(self) -> LeakDetectionResult:
        """
        Analyze snapshots for memory leaks using multiple detection methods.
        """
        if len(self.snapshots) < 10:
            return LeakDetectionResult(
                has_leak=False,
                confidence=0.0,
                growth_rate=0.0,
                suspicious_allocations=[],
                recommendation="Insufficient data for leak detection"
            )
        
        # Method 1: Linear regression on memory usage
        growth_rate = self._calculate_growth_rate()
        
        # Method 2: Check for monotonic increase
        monotonic_score = self._check_monotonic_increase()
        
        # Method 3: Analyze allocation patterns
        suspicious_allocations = self._find_suspicious_allocations()
        
        # Calculate overall confidence
        confidence = 0.0
        if growth_rate > self.growth_threshold:
            confidence += 0.4
        confidence += monotonic_score * 0.3
        if suspicious_allocations:
            confidence += 0.3
        
        has_leak = confidence >= self.alert_thresholds['leak_confidence']
        
        # Generate recommendation
        if has_leak:
            if suspicious_allocations:
                recommendation = f"Memory leak detected with {confidence:.0%} confidence. " \
                               f"Check allocations at: {suspicious_allocations[0][0]}"
            else:
                recommendation = f"Memory leak suspected ({confidence:.0%} confidence). " \
                               f"Growth rate: {growth_rate:.2%} per hour"
        else:
            recommendation = "No memory leak detected. Memory usage appears stable."
        
        return LeakDetectionResult(
            has_leak=has_leak,
            confidence=confidence,
            growth_rate=growth_rate,
            suspicious_allocations=suspicious_allocations,
            recommendation=recommendation
        )
    
    def _calculate_growth_rate(self) -> float:
        """Calculate memory growth rate using linear regression."""
        if len(self.snapshots) < 2:
            return 0.0
        
        # Get memory values and timestamps
        timestamps = []
        memory_values = []
        
        start_time = self.snapshots[0].timestamp
        for snapshot in self.snapshots:
            elapsed_hours = (snapshot.timestamp - start_time).total_seconds() / 3600
            timestamps.append(elapsed_hours)
            memory_values.append(snapshot.rss_bytes)
        
        if not timestamps or max(timestamps) == 0:
            return 0.0
        
        # Simple linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(timestamps, memory_values))
        sum_x2 = sum(x * x for x in timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Convert to growth rate percentage
        if memory_values[0] > 0:
            growth_rate_per_hour = slope / memory_values[0]
        else:
            growth_rate_per_hour = 0.0
        
        return growth_rate_per_hour
    
    def _check_monotonic_increase(self) -> float:
        """Check for monotonic memory increase pattern."""
        if len(self.snapshots) < 10:
            return 0.0
        
        increases = 0
        for i in range(1, len(self.snapshots)):
            if self.snapshots[i].rss_bytes > self.snapshots[i-1].rss_bytes:
                increases += 1
        
        return increases / (len(self.snapshots) - 1)
    
    def _find_suspicious_allocations(self) -> List[Tuple[str, int, float]]:
        """Find allocations that are growing suspiciously."""
        if not self.enable_tracemalloc or len(self.tracemalloc_snapshots) < 2:
            return []
        
        suspicious = []
        
        try:
            # Compare first and last snapshots
            first_snapshot = self.tracemalloc_snapshots[0]
            last_snapshot = self.tracemalloc_snapshots[-1]
            
            diff = last_snapshot.compare_to(first_snapshot, 'lineno')
            
            for stat in diff[:10]:  # Top 10 differences
                if stat.size_diff > 1024 * 1024:  # More than 1MB growth
                    location = str(stat.traceback)
                    growth_rate = stat.size_diff / max(stat.size - stat.size_diff, 1)
                    suspicious.append((location, stat.size_diff, growth_rate))
        
        except Exception as e:
            logger.error("Error analyzing allocations", error=str(e))
        
        return suspicious[:5]  # Return top 5 suspicious allocations
    
    def get_memory_trend(self, hours: int = 1) -> MemoryTrend:
        """Analyze memory trend over specified hours."""
        if not self.snapshots:
            return MemoryTrend(
                growth_rate_per_hour=0.0,
                average_usage_bytes=0,
                peak_usage_bytes=0,
                leak_detected=False,
                leak_confidence=0.0
            )
        
        # Filter snapshots for time window
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            recent_snapshots = list(self.snapshots)
        
        # Calculate statistics
        memory_values = [s.rss_bytes for s in recent_snapshots]
        average_usage = int(statistics.mean(memory_values)) if memory_values else 0
        peak_usage = max(memory_values) if memory_values else 0
        
        # Get growth rate
        growth_rate = self._calculate_growth_rate()
        
        # Detect leak
        leak_result = self.detect_leaks()
        
        # Estimate time to OOM if leak detected
        time_to_oom = None
        if leak_result.has_leak and growth_rate > 0:
            current_memory = self.snapshots[-1].rss_bytes if self.snapshots else 0
            available_memory = psutil.virtual_memory().total
            if current_memory > 0:
                remaining_memory = available_memory - current_memory
                time_to_oom = remaining_memory / (current_memory * growth_rate)
        
        return MemoryTrend(
            growth_rate_per_hour=growth_rate,
            average_usage_bytes=average_usage,
            peak_usage_bytes=peak_usage,
            leak_detected=leak_result.has_leak,
            leak_confidence=leak_result.confidence,
            estimated_time_to_oom=time_to_oom
        )
    
    def get_top_allocations(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top memory allocations."""
        if not self.snapshots or not self.snapshots[-1].top_allocations:
            return []
        
        return self.snapshots[-1].top_allocations[:limit]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        memory_stats = psutil.virtual_memory()
        
        stats = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent_used': memory_stats.percent,
            'available_gb': memory_stats.available / 1024 / 1024 / 1024,
            'snapshot_count': len(self.snapshots),
            'monitoring_duration_hours': 0.0
        }
        
        if self.start_time:
            stats['monitoring_duration_hours'] = \
                (datetime.now() - self.start_time).total_seconds() / 3600
        
        if self.baseline_memory:
            growth = (memory_info.rss - self.baseline_memory) / self.baseline_memory
            stats['growth_from_baseline'] = growth
        
        # Add trend analysis
        trend = self.get_memory_trend()
        stats['growth_rate_per_hour'] = trend.growth_rate_per_hour
        stats['leak_detected'] = trend.leak_detected
        stats['leak_confidence'] = trend.leak_confidence
        
        return stats
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before = self.process.memory_info().rss
        
        # Force full collection
        gc.collect(2)
        
        after = self.process.memory_info().rss
        
        stats = {
            'memory_before_mb': before / 1024 / 1024,
            'memory_after_mb': after / 1024 / 1024,
            'memory_freed_mb': (before - after) / 1024 / 1024,
            'gc_objects': len(gc.get_objects())
        }
        
        logger.info("Forced GC completed", **stats)
        return stats
    
    def set_alert_threshold(self, threshold_name: str, value: float) -> None:
        """Set alert threshold value."""
        if threshold_name in self.alert_thresholds:
            self.alert_thresholds[threshold_name] = value
            logger.info(f"Alert threshold updated", threshold=threshold_name, value=value)
        else:
            logger.warning(f"Unknown threshold name", threshold=threshold_name)
    
    def forecast_resource_usage(self, hours_ahead: float = 24.0) -> Dict[str, Any]:
        """Forecast resource usage based on historical trends.
        
        Args:
            hours_ahead: Number of hours to forecast ahead
            
        Returns:
            Dictionary with forecasted metrics
        """
        if len(self.snapshots) < 5:
            return {
                'forecast_available': False,
                'reason': 'Insufficient historical data'
            }
        
        # Calculate growth rate
        growth_rate = self._calculate_growth_rate()
        
        # Get current memory usage
        current_memory = self.snapshots[-1].rss_bytes if self.snapshots else 0
        
        # Forecast future memory usage
        forecasted_memory = current_memory * (1 + growth_rate * hours_ahead)
        
        # Get system limits
        total_memory = psutil.virtual_memory().total
        
        # Calculate time to resource exhaustion
        time_to_exhaustion = None
        if growth_rate > 0:
            remaining_memory = total_memory - current_memory
            if remaining_memory > 0:
                time_to_exhaustion = remaining_memory / (current_memory * growth_rate)
        
        # Analyze trend patterns
        trend_analysis = self._analyze_trend_patterns()
        
        forecast = {
            'forecast_available': True,
            'hours_ahead': hours_ahead,
            'current_memory_mb': current_memory / 1024 / 1024,
            'forecasted_memory_mb': forecasted_memory / 1024 / 1024,
            'growth_rate_per_hour': growth_rate,
            'time_to_exhaustion_hours': time_to_exhaustion,
            'confidence_level': self._calculate_forecast_confidence(),
            'trend_pattern': trend_analysis['pattern'],
            'seasonality_detected': trend_analysis['seasonality'],
            'recommendations': self._generate_capacity_recommendations(forecasted_memory, total_memory)
        }
        
        return forecast
    
    def _analyze_trend_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns for trends and seasonality."""
        if len(self.snapshots) < 10:
            return {'pattern': 'insufficient_data', 'seasonality': False}
        
        memory_values = [s.rss_bytes for s in self.snapshots]
        
        # Check for linear trend
        monotonic_score = self._check_monotonic_increase()
        
        # Check for cyclic patterns (simplified)
        seasonality = False
        if len(memory_values) >= 24:  # At least 24 data points
            # Simple check for repeating patterns
            period_lengths = [6, 12, 24]  # Check for 6h, 12h, 24h cycles
            for period in period_lengths:
                if self._check_seasonality(memory_values, period):
                    seasonality = True
                    break
        
        # Determine pattern type
        if monotonic_score > 0.8:
            pattern = 'linear_growth'
        elif monotonic_score < 0.2:
            pattern = 'stable'
        elif seasonality:
            pattern = 'cyclic'
        else:
            pattern = 'irregular'
        
        return {
            'pattern': pattern,
            'seasonality': seasonality,
            'monotonic_score': monotonic_score
        }
    
    def _check_seasonality(self, values: List[int], period: int) -> bool:
        """Check for seasonal patterns in data."""
        if len(values) < period * 2:
            return False
        
        # Compare values at period intervals
        correlations = []
        for i in range(len(values) - period):
            if values[i] > 0:
                ratio = values[i + period] / values[i]
                correlations.append(abs(ratio - 1.0))
        
        if correlations:
            avg_deviation = sum(correlations) / len(correlations)
            return avg_deviation < 0.1  # Less than 10% average deviation
        
        return False
    
    def _calculate_forecast_confidence(self) -> float:
        """Calculate confidence level for forecast."""
        confidence = 1.0
        
        # Reduce confidence for limited data
        if len(self.snapshots) < 20:
            confidence *= 0.7
        elif len(self.snapshots) < 50:
            confidence *= 0.85
        
        # Reduce confidence for irregular patterns
        trend_analysis = self._analyze_trend_patterns()
        if trend_analysis['pattern'] == 'irregular':
            confidence *= 0.6
        elif trend_analysis['pattern'] == 'cyclic':
            confidence *= 0.8
        
        # Reduce confidence for high variance
        if len(self.snapshots) >= 5:
            memory_values = [s.rss_bytes for s in self.snapshots[-10:]]
            if memory_values:
                mean_val = statistics.mean(memory_values)
                if mean_val > 0:
                    variance = statistics.variance(memory_values) / mean_val
                    if variance > 0.2:  # High variance
                        confidence *= 0.7
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_capacity_recommendations(self, forecasted_memory: float, total_memory: float) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        utilization = forecasted_memory / total_memory
        
        if utilization > 0.9:
            recommendations.append("CRITICAL: Forecasted memory usage exceeds 90%. Immediate action required.")
            recommendations.append("Consider scaling up resources or optimizing memory usage immediately.")
        elif utilization > 0.8:
            recommendations.append("WARNING: Forecasted memory usage exceeds 80%. Plan for scaling soon.")
            recommendations.append("Review and optimize memory-intensive operations.")
        elif utilization > 0.7:
            recommendations.append("INFO: Memory usage trending upward. Monitor closely.")
            recommendations.append("Consider implementing memory optimization strategies.")
        else:
            recommendations.append("Memory usage is within acceptable limits.")
        
        # Add growth-based recommendations
        growth_rate = self._calculate_growth_rate()
        if growth_rate > 0.1:  # 10% per hour
            recommendations.append(f"High growth rate detected ({growth_rate:.1%}/hour). Investigate cause.")
        elif growth_rate > 0.05:  # 5% per hour
            recommendations.append(f"Moderate growth rate ({growth_rate:.1%}/hour). Monitor for sustained growth.")
        
        return recommendations
    
    async def run_stability_test(self, duration_hours: float = 48.0) -> Dict[str, Any]:
        """Run extended stability test.
        
        Args:
            duration_hours: Duration of the stability test in hours (default: 48.0)
            
        Returns:
            Dict containing test results including pass/fail status, memory growth metrics,
            leak detection results, and resource forecasts
        """
        logger.info(f"Starting {duration_hours} hour stability test")
        
        await self.start_monitoring()
        start_memory = self.process.memory_info().rss
        start_time = time.time()
        
        results = {
            'duration_hours': duration_hours,
            'start_memory_mb': start_memory / 1024 / 1024,
            'end_memory_mb': 0,
            'peak_memory_mb': 0,
            'growth_percentage': 0,
            'leak_detected': False,
            'test_passed': False,
            'hourly_stats': [],
            'resource_forecast': {}
        }
        
        try:
            while time.time() - start_time < duration_hours * 3600:
                await asyncio.sleep(3600)  # Sleep for 1 hour
                
                current_memory = self.process.memory_info().rss
                growth = (current_memory - start_memory) / start_memory
                
                hourly_stat = {
                    'hour': int((time.time() - start_time) / 3600),
                    'memory_mb': current_memory / 1024 / 1024,
                    'growth_percentage': growth * 100
                }
                results['hourly_stats'].append(hourly_stat)
                
                # Check if growth exceeds threshold
                if growth > self.growth_threshold:
                    logger.warning(f"Memory growth exceeds threshold",
                                 growth=f"{growth:.2%}",
                                 threshold=f"{self.growth_threshold:.2%}")
                
                # Update peak memory
                if current_memory > results['peak_memory_mb'] * 1024 * 1024:
                    results['peak_memory_mb'] = current_memory / 1024 / 1024
            
            # Final analysis
            end_memory = self.process.memory_info().rss
            results['end_memory_mb'] = end_memory / 1024 / 1024
            results['growth_percentage'] = ((end_memory - start_memory) / start_memory) * 100
            
            leak_result = self.detect_leaks()
            results['leak_detected'] = leak_result.has_leak
            results['test_passed'] = (
                results['growth_percentage'] < self.growth_threshold * 100 and
                not leak_result.has_leak
            )
            
            # Add resource forecast
            results['resource_forecast'] = self.forecast_resource_usage(hours_ahead=24.0)
            
        finally:
            await self.stop_monitoring()
        
        logger.info("Stability test completed", **results)
        return results