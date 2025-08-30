"""Performance profiling for Project GENESIS."""

import asyncio
import cProfile
import io
import pstats
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, TypeVar

import psutil
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    function_name: str
    duration_ms: float
    cpu_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    call_count: int
    timestamp: datetime
    stats: pstats.Stats | None = None
    flame_graph_data: dict[str, Any] | None = None


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    rss_mb: float  # Resident set size
    vms_mb: float  # Virtual memory size
    available_mb: float
    percent: float
    top_allocations: list[dict[str, Any]] = field(default_factory=list)


class PerformanceProfiler:
    """Performance profiling and analysis."""

    def __init__(self, enabled: bool = True, overhead_target: float = 0.01):
        self.enabled = enabled
        self.overhead_target = overhead_target  # Max 1% overhead
        self.profiles: list[ProfileResult] = []
        self.memory_snapshots: list[MemorySnapshot] = []
        self._profiler: cProfile.Profile | None = None
        self._memory_tracking = False
        self._profile_task: asyncio.Task | None = None
        self._snapshot_interval = 60  # seconds

    async def start_continuous_profiling(self, sampling_rate: float = 0.01) -> None:
        """Start continuous profiling with sampling."""
        if not self.enabled:
            return

        if not self._profile_task:
            self._profile_task = asyncio.create_task(
                self._continuous_profiling_loop(sampling_rate)
            )
            logger.info("Started continuous profiling",
                       sampling_rate=sampling_rate)

    async def stop_continuous_profiling(self) -> None:
        """Stop continuous profiling."""
        if self._profile_task:
            self._profile_task.cancel()
            try:
                await self._profile_task
            except asyncio.CancelledError:
                pass
            self._profile_task = None
            logger.info("Stopped continuous profiling")

    async def _continuous_profiling_loop(self, sampling_rate: float) -> None:
        """Main profiling loop."""
        import random

        while True:
            try:
                # Sample-based profiling to reduce overhead
                if random.random() < sampling_rate:
                    await self._take_profile_snapshot()

                # Take memory snapshot
                await self._take_memory_snapshot()

                await asyncio.sleep(self._snapshot_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in profiling loop", error=str(e))
                await asyncio.sleep(self._snapshot_interval)

    async def _take_profile_snapshot(self) -> None:
        """Take a CPU profile snapshot."""
        profiler = cProfile.Profile()
        profiler.enable()

        # Profile for 1 second
        await asyncio.sleep(1)

        profiler.disable()

        # Analyze results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Store snapshot
        if len(self.profiles) > 1000:
            self.profiles.pop(0)

    async def _take_memory_snapshot(self) -> None:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            available_mb=psutil.virtual_memory().available / 1024 / 1024,
            percent=memory_percent
        )

        # Get top memory allocations if tracking enabled
        if self._memory_tracking and tracemalloc.is_tracing():
            snapshot.top_allocations = self._get_top_allocations(10)

        self.memory_snapshots.append(snapshot)

        # Keep limited history
        if len(self.memory_snapshots) > 1440:  # 24 hours at 1-minute intervals
            self.memory_snapshots.pop(0)

        # Check for memory leaks
        await self._check_memory_leak()

    def _get_top_allocations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top memory allocations."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:limit]

        allocations = []
        for stat in top_stats:
            allocations.append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })

        return allocations

    async def _check_memory_leak(self) -> None:
        """Check for potential memory leaks."""
        if len(self.memory_snapshots) < 10:
            return

        # Check memory growth over last 10 snapshots
        recent_snapshots = self.memory_snapshots[-10:]
        first_rss = recent_snapshots[0].rss_mb
        last_rss = recent_snapshots[-1].rss_mb

        growth_rate = (last_rss - first_rss) / first_rss if first_rss > 0 else 0

        # Alert if memory grew by more than 10%
        if growth_rate > 0.1:
            logger.warning("Potential memory leak detected",
                         growth_rate_percent=growth_rate * 100,
                         initial_mb=first_rss,
                         current_mb=last_rss)

    def profile_function(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to profile a function."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            if not self.enabled:
                return await func(*args, **kwargs)

            result = ProfileResult(
                function_name=func.__name__,
                duration_ms=0,
                cpu_time_ms=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_peak_mb=0,
                call_count=1,
                timestamp=datetime.now()
            )

            # Memory before
            process = psutil.Process()
            result.memory_before_mb = process.memory_info().rss / 1024 / 1024

            # CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.perf_counter()
            start_cpu = time.process_time()

            try:
                return_value = await func(*args, **kwargs)
            finally:
                # Timing
                result.duration_ms = (time.perf_counter() - start_time) * 1000
                result.cpu_time_ms = (time.process_time() - start_cpu) * 1000

                # Memory after
                result.memory_after_mb = process.memory_info().rss / 1024 / 1024
                result.memory_peak_mb = max(result.memory_before_mb, result.memory_after_mb)

                # CPU profiling
                profiler.disable()
                result.stats = pstats.Stats(profiler)

                # Store result
                self.profiles.append(result)

                # Log if slow
                if result.duration_ms > 100:
                    logger.warning("Slow function execution",
                                 function=func.__name__,
                                 duration_ms=result.duration_ms)

            return return_value

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            if not self.enabled:
                return func(*args, **kwargs)

            result = ProfileResult(
                function_name=func.__name__,
                duration_ms=0,
                cpu_time_ms=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_peak_mb=0,
                call_count=1,
                timestamp=datetime.now()
            )

            # Memory before
            process = psutil.Process()
            result.memory_before_mb = process.memory_info().rss / 1024 / 1024

            # CPU profiling
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.perf_counter()
            start_cpu = time.process_time()

            try:
                return_value = func(*args, **kwargs)
            finally:
                # Timing
                result.duration_ms = (time.perf_counter() - start_time) * 1000
                result.cpu_time_ms = (time.process_time() - start_cpu) * 1000

                # Memory after
                result.memory_after_mb = process.memory_info().rss / 1024 / 1024
                result.memory_peak_mb = max(result.memory_before_mb, result.memory_after_mb)

                # CPU profiling
                profiler.disable()
                result.stats = pstats.Stats(profiler)

                # Store result
                self.profiles.append(result)

                # Log if slow
                if result.duration_ms > 100:
                    logger.warning("Slow function execution",
                                 function=func.__name__,
                                 duration_ms=result.duration_ms)

            return return_value

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling code blocks."""
        if not self.enabled:
            yield
            return

        result = ProfileResult(
            function_name=name,
            duration_ms=0,
            cpu_time_ms=0,
            memory_before_mb=0,
            memory_after_mb=0,
            memory_peak_mb=0,
            call_count=1,
            timestamp=datetime.now()
        )

        # Memory before
        process = psutil.Process()
        result.memory_before_mb = process.memory_info().rss / 1024 / 1024

        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            yield
        finally:
            # Stop profiling
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            result.cpu_time_ms = (time.process_time() - start_cpu) * 1000

            # Memory after
            result.memory_after_mb = process.memory_info().rss / 1024 / 1024
            result.memory_peak_mb = max(result.memory_before_mb, result.memory_after_mb)

            # CPU profiling
            profiler.disable()
            result.stats = pstats.Stats(profiler)

            # Store result
            self.profiles.append(result)

    def start_memory_tracking(self) -> None:
        """Start detailed memory tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Store 10 frames
            self._memory_tracking = True
            logger.info("Started memory tracking")

    def stop_memory_tracking(self) -> None:
        """Stop memory tracking."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self._memory_tracking = False
            logger.info("Stopped memory tracking")

    def generate_flame_graph(self, profile: ProfileResult) -> dict[str, Any]:
        """Generate flame graph data from profile."""
        if not profile.stats:
            return {}

        # Convert pstats to flame graph format
        flame_data = {
            "name": "root",
            "value": 0,
            "children": []
        }

        # Process stats
        stats_dict = {}
        for (filename, line, name), (cc, nc, tt, ct, callers) in profile.stats.stats.items():
            key = f"{filename}:{line}:{name}"
            stats_dict[key] = {
                "calls": cc,
                "total_time": tt,
                "cumulative_time": ct
            }

        # Build tree structure for flame graph
        for key, data in stats_dict.items():
            flame_data["children"].append({
                "name": key,
                "value": int(data["cumulative_time"] * 1000),  # Convert to ms
                "calls": data["calls"]
            })

        flame_data["value"] = sum(c["value"] for c in flame_data["children"])

        return flame_data

    def get_hot_paths(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the hottest code paths."""
        hot_paths = []

        for profile in self.profiles[-100:]:  # Last 100 profiles
            if profile.stats:
                stream = io.StringIO()
                profile.stats.stream = stream
                profile.stats.sort_stats('cumulative')
                profile.stats.print_stats(limit)

                # Parse output
                for line in stream.getvalue().split('\n'):
                    if '/' in line and not line.startswith(' '):
                        parts = line.split()
                        if len(parts) >= 6:
                            hot_paths.append({
                                "function": parts[-1],
                                "calls": int(parts[0]) if parts[0].isdigit() else 0,
                                "total_time": float(parts[2]) if len(parts) > 2 else 0,
                                "cumulative_time": float(parts[4]) if len(parts) > 4 else 0
                            })

        # Sort by cumulative time
        hot_paths.sort(key=lambda x: x["cumulative_time"], reverse=True)
        return hot_paths[:limit]

    def detect_performance_regression(self, function_name: str, threshold: float = 0.2) -> bool:
        """Detect performance regression for a function."""
        recent_profiles = [
            p for p in self.profiles
            if p.function_name == function_name and
            (datetime.now() - p.timestamp) < timedelta(hours=1)
        ]

        if len(recent_profiles) < 10:
            return False

        # Compare recent vs historical performance
        recent = recent_profiles[-5:]
        historical = recent_profiles[-10:-5]

        recent_avg = sum(p.duration_ms for p in recent) / len(recent)
        historical_avg = sum(p.duration_ms for p in historical) / len(historical)

        # Check if performance degraded by more than threshold
        if historical_avg > 0:
            degradation = (recent_avg - historical_avg) / historical_avg
            if degradation > threshold:
                logger.warning("Performance regression detected",
                             function=function_name,
                             recent_avg_ms=recent_avg,
                             historical_avg_ms=historical_avg,
                             degradation_percent=degradation * 100)
                return True

        return False

    def calculate_overhead(self) -> float:
        """Calculate profiling overhead."""
        if not self.profiles:
            return 0.0

        # Calculate average overhead
        total_overhead = sum(
            p.duration_ms - p.cpu_time_ms
            for p in self.profiles[-100:]
        )

        total_duration = sum(p.duration_ms for p in self.profiles[-100:])

        if total_duration > 0:
            return (total_overhead / total_duration) * 100

        return 0.0

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.profiles:
            return {}

        recent_profiles = self.profiles[-100:]

        # Calculate statistics
        total_duration = sum(p.duration_ms for p in recent_profiles)
        total_cpu = sum(p.cpu_time_ms for p in recent_profiles)

        # Memory statistics
        current_memory = 0
        if self.memory_snapshots:
            current_memory = self.memory_snapshots[-1].rss_mb

        return {
            "total_profiles": len(self.profiles),
            "recent_duration_ms": total_duration,
            "recent_cpu_ms": total_cpu,
            "cpu_efficiency": (total_cpu / total_duration * 100) if total_duration > 0 else 0,
            "current_memory_mb": current_memory,
            "memory_snapshots": len(self.memory_snapshots),
            "profiling_overhead_percent": self.calculate_overhead(),
            "hot_paths": self.get_hot_paths(5)
        }
