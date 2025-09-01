"""Advanced performance profiling with py-spy and memory tracking for Project GENESIS."""

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Memory snapshot for leak detection."""
    timestamp: float
    rss_bytes: int
    vms_bytes: int
    percent: float
    top_allocations: List[Tuple[str, int, int]]  # (traceback, size, count)
    gc_stats: Dict[str, int]


@dataclass
class CPUProfile:
    """CPU profiling data."""
    timestamp: float
    duration_seconds: float
    samples: int
    top_functions: List[Tuple[str, float]]  # (function, percentage)
    flame_graph_path: Optional[str] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance profiling report."""
    start_time: datetime
    end_time: datetime
    cpu_profiles: List[CPUProfile] = field(default_factory=list)
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    memory_leaks_detected: List[Dict[str, Any]] = field(default_factory=list)
    slow_operations: List[Dict[str, float]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AdvancedPerformanceProfiler:
    """Advanced performance profiler with py-spy and tracemalloc integration."""
    
    def __init__(
        self,
        profile_dir: str = ".profiles",
        memory_threshold_mb: float = 100.0,
        cpu_threshold_percent: float = 80.0,
        snapshot_interval_seconds: int = 60
    ):
        """Initialize performance profiler.
        
        Args:
            profile_dir: Directory to store profile data
            memory_threshold_mb: Memory leak detection threshold in MB
            cpu_threshold_percent: High CPU usage threshold
            snapshot_interval_seconds: Interval between automatic snapshots
        """
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.snapshot_interval = snapshot_interval_seconds
        
        self._profiling_active = False
        self._py_spy_process: Optional[subprocess.Popen] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._memory_baseline: Optional[MemorySnapshot] = None
        
        # Performance data storage
        self.memory_history = deque(maxlen=100)
        self.cpu_history = deque(maxlen=100)
        self.operation_timings = defaultdict(list)
        
        # Memory tracking
        self._tracemalloc_started = False
        
        logger.info(
            "AdvancedPerformanceProfiler initialized",
            profile_dir=str(self.profile_dir),
            memory_threshold_mb=memory_threshold_mb,
            cpu_threshold_percent=cpu_threshold_percent
        )
    
    def start_profiling(self, profile_cpu: bool = True, profile_memory: bool = True) -> None:
        """Start comprehensive performance profiling.
        
        Args:
            profile_cpu: Enable CPU profiling with py-spy
            profile_memory: Enable memory profiling with tracemalloc
        """
        if self._profiling_active:
            logger.warning("Profiling already active")
            return
        
        self._profiling_active = True
        self._start_time = datetime.now()
        
        # Start memory profiling
        if profile_memory:
            self._start_memory_profiling()
        
        # Start CPU profiling with py-spy
        if profile_cpu:
            self._start_cpu_profiling()
        
        # Start snapshot thread
        self._start_snapshot_thread()
        
        logger.info("Performance profiling started", cpu=profile_cpu, memory=profile_memory)
    
    def stop_profiling(self) -> PerformanceReport:
        """Stop profiling and generate comprehensive report.
        
        Returns:
            Performance profiling report
        """
        if not self._profiling_active:
            logger.warning("Profiling not active")
            return PerformanceReport(datetime.now(), datetime.now())
        
        self._profiling_active = False
        end_time = datetime.now()
        
        # Stop snapshot thread
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5)
        
        # Stop CPU profiling
        cpu_profiles = self._stop_cpu_profiling()
        
        # Stop memory profiling
        memory_leaks = self._stop_memory_profiling()
        
        # Generate report
        report = PerformanceReport(
            start_time=self._start_time,
            end_time=end_time,
            cpu_profiles=cpu_profiles,
            memory_snapshots=list(self.memory_history),
            memory_leaks_detected=memory_leaks,
            slow_operations=self._analyze_slow_operations(),
            recommendations=self._generate_recommendations()
        )
        
        # Save report to file
        self._save_report(report)
        
        logger.info("Performance profiling stopped", duration=(end_time - self._start_time))
        return report
    
    def _start_memory_profiling(self) -> None:
        """Start memory profiling with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Store 10 frames for tracebacks
            self._tracemalloc_started = True
        
        # Capture baseline
        self._memory_baseline = self._capture_memory_snapshot()
        logger.debug("Memory profiling started with baseline captured")
    
    def _stop_memory_profiling(self) -> List[Dict[str, Any]]:
        """Stop memory profiling and detect leaks.
        
        Returns:
            List of detected memory leaks
        """
        leaks = []
        
        if self._tracemalloc_started and tracemalloc.is_tracing():
            # Capture final snapshot
            final_snapshot = self._capture_memory_snapshot()
            
            # Detect leaks by comparing with baseline
            if self._memory_baseline:
                leaks = self._detect_memory_leaks(
                    self._memory_baseline,
                    final_snapshot
                )
            
            tracemalloc.stop()
            self._tracemalloc_started = False
        
        return leaks
    
    def _start_cpu_profiling(self) -> None:
        """Start CPU profiling with py-spy."""
        try:
            # Check if py-spy is available
            result = subprocess.run(
                ["py-spy", "--version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Start py-spy profiling
                profile_path = self.profile_dir / f"cpu_profile_{int(time.time())}.svg"
                self._py_spy_process = subprocess.Popen(
                    [
                        "py-spy", "record",
                        "-o", str(profile_path),
                        "-d", "300",  # Duration 5 minutes max
                        "-r", "100",  # Sample rate 100Hz
                        "--", sys.executable, "-m", "genesis"
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info("CPU profiling started with py-spy", output=str(profile_path))
            else:
                logger.warning("py-spy not available, CPU profiling disabled")
        except FileNotFoundError:
            logger.warning("py-spy not installed, CPU profiling disabled")
        except Exception as e:
            logger.error("Failed to start CPU profiling", error=str(e))
    
    def _stop_cpu_profiling(self) -> List[CPUProfile]:
        """Stop CPU profiling and collect results.
        
        Returns:
            List of CPU profiles
        """
        profiles = []
        
        if self._py_spy_process:
            try:
                # Terminate py-spy
                self._py_spy_process.terminate()
                self._py_spy_process.wait(timeout=5)
                
                # Parse py-spy output if available
                # Note: Actual parsing would depend on py-spy output format
                logger.info("CPU profiling stopped")
            except Exception as e:
                logger.error("Error stopping CPU profiling", error=str(e))
        
        # Add CPU usage history as profiles
        for timestamp, cpu_percent in self.cpu_history:
            profiles.append(CPUProfile(
                timestamp=timestamp,
                duration_seconds=self.snapshot_interval,
                samples=1,
                top_functions=[("system", cpu_percent)]
            ))
        
        return profiles
    
    def _start_snapshot_thread(self) -> None:
        """Start background thread for periodic snapshots."""
        def snapshot_worker():
            while self._profiling_active:
                try:
                    # Capture memory snapshot
                    snapshot = self._capture_memory_snapshot()
                    self.memory_history.append(snapshot)
                    
                    # Capture CPU usage
                    cpu_percent = psutil.Process().cpu_percent(interval=1)
                    self.cpu_history.append((time.time(), cpu_percent))
                    
                    # Check for issues
                    self._check_performance_issues(snapshot, cpu_percent)
                    
                    # Sleep until next snapshot
                    time.sleep(self.snapshot_interval)
                except Exception as e:
                    logger.error("Error in snapshot thread", error=str(e))
        
        self._snapshot_thread = threading.Thread(target=snapshot_worker, daemon=True)
        self._snapshot_thread.start()
    
    def _capture_memory_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot.
        
        Returns:
            Memory snapshot
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get top memory allocations if tracemalloc is active
        top_allocations = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('traceback')
            
            for stat in top_stats[:10]:
                # Format traceback
                tb_lines = []
                for frame in stat.traceback:
                    tb_lines.append(f"{frame.filename}:{frame.lineno}")
                traceback_str = " -> ".join(tb_lines[-3:])  # Last 3 frames
                
                top_allocations.append((
                    traceback_str,
                    stat.size,
                    stat.count
                ))
        
        # Get garbage collection stats
        gc_stats = {
            f"generation_{i}": len(gc.get_objects(i))
            for i in range(gc.get_count().__len__())
        }
        gc_stats["total_objects"] = len(gc.get_objects())
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_bytes=memory_info.rss,
            vms_bytes=memory_info.vms,
            percent=process.memory_percent(),
            top_allocations=top_allocations,
            gc_stats=gc_stats
        )
    
    def _detect_memory_leaks(
        self,
        baseline: MemorySnapshot,
        current: MemorySnapshot
    ) -> List[Dict[str, Any]]:
        """Detect memory leaks by comparing snapshots.
        
        Args:
            baseline: Baseline memory snapshot
            current: Current memory snapshot
        
        Returns:
            List of detected memory leaks
        """
        leaks = []
        
        # Check for significant memory increase
        memory_increase_mb = (current.rss_bytes - baseline.rss_bytes) / (1024 * 1024)
        
        if memory_increase_mb > self.memory_threshold_mb:
            # Analyze top allocations for leak sources
            current_allocs = {tb: (size, count) for tb, size, count in current.top_allocations}
            baseline_allocs = {tb: (size, count) for tb, size, count in baseline.top_allocations}
            
            for traceback_str, (size, count) in current_allocs.items():
                baseline_size, baseline_count = baseline_allocs.get(traceback_str, (0, 0))
                size_increase_mb = (size - baseline_size) / (1024 * 1024)
                
                if size_increase_mb > 10:  # 10MB threshold for individual allocation
                    leaks.append({
                        "traceback": traceback_str,
                        "size_increase_mb": size_increase_mb,
                        "count_increase": count - baseline_count,
                        "current_size_mb": size / (1024 * 1024),
                        "severity": "high" if size_increase_mb > 50 else "medium"
                    })
        
        # Check for object accumulation
        if current.gc_stats["total_objects"] > baseline.gc_stats["total_objects"] * 2:
            leaks.append({
                "type": "object_accumulation",
                "baseline_objects": baseline.gc_stats["total_objects"],
                "current_objects": current.gc_stats["total_objects"],
                "increase_ratio": current.gc_stats["total_objects"] / baseline.gc_stats["total_objects"],
                "severity": "medium"
            })
        
        return leaks
    
    def _check_performance_issues(self, snapshot: MemorySnapshot, cpu_percent: float) -> None:
        """Check for performance issues and log warnings.
        
        Args:
            snapshot: Memory snapshot
            cpu_percent: CPU usage percentage
        """
        # Check memory usage
        if snapshot.percent > 80:
            logger.warning(
                "High memory usage detected",
                rss_mb=snapshot.rss_bytes / (1024 * 1024),
                percent=snapshot.percent
            )
        
        # Check CPU usage
        if cpu_percent > self.cpu_threshold_percent:
            logger.warning(
                "High CPU usage detected",
                cpu_percent=cpu_percent
            )
        
        # Check for memory growth pattern
        if len(self.memory_history) >= 5:
            recent_memory = [s.rss_bytes for s in list(self.memory_history)[-5:]]
            if all(recent_memory[i] < recent_memory[i+1] for i in range(4)):
                logger.warning(
                    "Consistent memory growth detected",
                    growth_mb=(recent_memory[-1] - recent_memory[0]) / (1024 * 1024)
                )
    
    def _analyze_slow_operations(self) -> List[Dict[str, float]]:
        """Analyze operation timings for slow operations.
        
        Returns:
            List of slow operations
        """
        slow_ops = []
        
        for operation, timings in self.operation_timings.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                max_time = max(timings)
                
                if max_time > 1.0:  # Operations taking more than 1 second
                    slow_ops.append({
                        "operation": operation,
                        "avg_time_seconds": avg_time,
                        "max_time_seconds": max_time,
                        "count": len(timings),
                        "total_time_seconds": sum(timings)
                    })
        
        return sorted(slow_ops, key=lambda x: x["max_time_seconds"], reverse=True)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Memory recommendations
        if self.memory_history:
            latest_memory = self.memory_history[-1]
            if latest_memory.percent > 70:
                recommendations.append(
                    "High memory usage detected. Consider optimizing data structures "
                    "or implementing pagination for large datasets."
                )
            
            # Check for memory growth
            if len(self.memory_history) >= 2:
                memory_growth = (
                    self.memory_history[-1].rss_bytes - 
                    self.memory_history[0].rss_bytes
                ) / (1024 * 1024)
                
                if memory_growth > 100:
                    recommendations.append(
                        f"Memory grew by {memory_growth:.1f}MB during profiling. "
                        "Review object lifecycle and implement proper cleanup."
                    )
        
        # CPU recommendations
        if self.cpu_history:
            avg_cpu = sum(cpu for _, cpu in self.cpu_history) / len(self.cpu_history)
            if avg_cpu > 70:
                recommendations.append(
                    f"Average CPU usage was {avg_cpu:.1f}%. Consider optimizing "
                    "hot code paths or implementing caching."
                )
        
        # Slow operation recommendations
        slow_ops = self._analyze_slow_operations()
        if slow_ops:
            slowest = slow_ops[0]
            recommendations.append(
                f"Operation '{slowest['operation']}' took up to "
                f"{slowest['max_time_seconds']:.2f}s. Consider optimization."
            )
        
        return recommendations
    
    def _save_report(self, report: PerformanceReport) -> str:
        """Save performance report to file.
        
        Args:
            report: Performance report
        
        Returns:
            Path to saved report
        """
        report_path = self.profile_dir / f"performance_report_{int(time.time())}.json"
        
        # Convert report to JSON-serializable format
        report_dict = {
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat(),
            "duration_seconds": (report.end_time - report.start_time).total_seconds(),
            "memory_leaks_detected": report.memory_leaks_detected,
            "slow_operations": report.slow_operations,
            "recommendations": report.recommendations,
            "memory_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_bytes / (1024 * 1024),
                    "percent": s.percent,
                    "top_allocations": [
                        {"traceback": tb, "size_mb": size / (1024 * 1024), "count": count}
                        for tb, size, count in s.top_allocations[:5]
                    ]
                }
                for s in report.memory_snapshots
            ],
            "cpu_profiles": [
                {
                    "timestamp": p.timestamp,
                    "duration_seconds": p.duration_seconds,
                    "samples": p.samples
                }
                for p in report.cpu_profiles
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info("Performance report saved", path=str(report_path))
        return str(report_path)
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile individual function performance.
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Record timing
                self.operation_timings[func.__name__].append(execution_time)
                
                # Log slow operations
                if execution_time > 1.0:
                    logger.warning(
                        "Slow operation detected",
                        function=func.__name__,
                        execution_time=execution_time
                    )
                
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self.operation_timings[f"{func.__name__}_error"].append(execution_time)
                raise
        
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Record timing
                self.operation_timings[func.__name__].append(execution_time)
                
                # Log slow operations
                if execution_time > 1.0:
                    logger.warning(
                        "Slow operation detected",
                        function=func.__name__,
                        execution_time=execution_time
                    )
                
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                self.operation_timings[f"{func.__name__}_error"].append(execution_time)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    def get_profiling_endpoints(self) -> Dict[str, str]:
        """Get endpoints for triggering profiling operations.
        
        Returns:
            Dictionary of endpoint paths
        """
        return {
            "start": "/profiling/start",
            "stop": "/profiling/stop",
            "status": "/profiling/status",
            "report": "/profiling/report",
            "memory_snapshot": "/profiling/memory",
            "cpu_snapshot": "/profiling/cpu"
        }


# Global profiler instance
_profiler_instance: Optional[AdvancedPerformanceProfiler] = None


def get_advanced_profiler() -> AdvancedPerformanceProfiler:
    """Get or create global performance profiler instance.
    
    Returns:
        Performance profiler instance
    """
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = AdvancedPerformanceProfiler()
    return _profiler_instance