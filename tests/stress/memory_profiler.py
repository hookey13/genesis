"""
Memory leak detection system for long-running tests.

This module provides memory profiling and leak detection capabilities
for 7-day continuous operation validation.
"""

import asyncio
import gc
import tracemalloc
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import psutil
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a point in time."""
    
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    heap_mb: float  # Python heap size
    top_allocations: List[Tuple[str, int]]  # Top memory allocations
    gc_stats: Dict[str, Any]
    object_counts: Dict[str, int]


@dataclass
class MemoryMetrics:
    """Tracks memory metrics over time."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    leak_candidates: List[Dict[str, Any]] = field(default_factory=list)
    memory_growth_rate: float = 0  # MB per hour
    peak_memory_mb: float = 0
    average_memory_mb: float = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    
    def add_snapshot(self, snapshot: MemorySnapshot):
        """Add a memory snapshot."""
        self.snapshots.append(snapshot)
        
        # Update peak memory
        self.peak_memory_mb = max(self.peak_memory_mb, snapshot.rss_mb)
        
        # Calculate average
        if self.snapshots:
            total_memory = sum(s.rss_mb for s in self.snapshots)
            self.average_memory_mb = total_memory / len(self.snapshots)
        
        # Calculate growth rate
        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            time_diff_hours = (last.timestamp - first.timestamp).total_seconds() / 3600
            
            if time_diff_hours > 0:
                memory_diff_mb = last.rss_mb - first.rss_mb
                self.memory_growth_rate = memory_diff_mb / time_diff_hours
    
    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 10:
            return []
        
        leaks = []
        
        # Check for continuous memory growth
        memory_values = [s.rss_mb for s in self.snapshots[-10:]]
        is_growing = all(memory_values[i] <= memory_values[i+1] 
                        for i in range(len(memory_values)-1))
        
        if is_growing and self.memory_growth_rate > 1:  # Growing > 1MB/hour
            leaks.append({
                "type": "continuous_growth",
                "growth_rate_mb_per_hour": self.memory_growth_rate,
                "severity": "high" if self.memory_growth_rate > 10 else "medium"
            })
        
        # Check for specific object growth
        if len(self.snapshots) >= 2:
            recent = self.snapshots[-1].object_counts
            old = self.snapshots[0].object_counts
            
            for obj_type, count in recent.items():
                old_count = old.get(obj_type, 0)
                growth = count - old_count
                
                if growth > 10000:  # More than 10k objects created
                    leaks.append({
                        "type": "object_leak",
                        "object_type": obj_type,
                        "growth": growth,
                        "severity": "high" if growth > 100000 else "medium"
                    })
        
        self.leak_candidates = leaks
        return leaks


class MemoryProfiler:
    """Memory profiler for long-running tests."""
    
    def __init__(self, snapshot_interval_seconds: int = 300):
        """
        Initialize memory profiler.
        
        Args:
            snapshot_interval_seconds: Interval between snapshots
        """
        self.snapshot_interval = snapshot_interval_seconds
        self.metrics = MemoryMetrics()
        self.running = False
        self.tracemalloc_started = False
        
    async def start_profiling(self, duration_days: int = 7):
        """
        Start memory profiling for specified duration.
        
        Args:
            duration_days: Profiling duration in days
        """
        logger.info(f"Starting memory profiling for {duration_days} days")
        
        # Start tracemalloc
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
        
        self.running = True
        self.metrics = MemoryMetrics()
        
        end_time = datetime.now() + timedelta(days=duration_days)
        
        try:
            while datetime.now() < end_time and self.running:
                # Take snapshot
                snapshot = await self.take_snapshot()
                self.metrics.add_snapshot(snapshot)
                
                # Check for leaks
                leaks = self.metrics.detect_leaks()
                if leaks:
                    logger.warning(f"Potential memory leaks detected: {leaks}")
                
                # Log current status
                logger.info(
                    "Memory profile",
                    current_mb=snapshot.rss_mb,
                    peak_mb=self.metrics.peak_memory_mb,
                    growth_rate=f"{self.metrics.memory_growth_rate:.2f} MB/hour"
                )
                
                # Wait for next snapshot
                await asyncio.sleep(self.snapshot_interval)
                
        finally:
            self.running = False
            self.metrics.end_time = datetime.now()
            
            # Generate report
            await self.generate_report()
            
            # Stop tracemalloc
            if self.tracemalloc_started:
                tracemalloc.stop()
                self.tracemalloc_started = False
    
    async def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get tracemalloc statistics
        top_stats = []
        if self.tracemalloc_started:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]  # Top 10 allocations
        
        # Get GC statistics
        gc_stats = {
            f"generation_{i}": gc.get_stats()[i] if i < len(gc.get_stats()) else {}
            for i in range(gc.get_count().__len__())
        }
        gc_stats["thresholds"] = gc.get_threshold()
        
        # Count objects by type
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Sort object counts and keep top 20
        sorted_counts = dict(sorted(
            object_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            heap_mb=sum(stat.size for stat in top_stats) / 1024 / 1024 if top_stats else 0,
            top_allocations=[(str(stat.traceback), stat.size) for stat in top_stats[:5]],
            gc_stats=gc_stats,
            object_counts=sorted_counts
        )
    
    async def analyze_heap_dump(self) -> Dict[str, Any]:
        """Analyze heap dump for memory issues."""
        if not self.tracemalloc_started:
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        analysis = {
            "total_blocks": sum(stat.count for stat in top_stats),
            "total_size_mb": sum(stat.size for stat in top_stats) / 1024 / 1024,
            "top_allocations": [],
            "by_file": {}
        }
        
        # Top allocations
        for stat in top_stats[:10]:
            analysis["top_allocations"].append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count,
                "average_size": stat.size / stat.count if stat.count > 0 else 0
            })
        
        # Group by file
        file_stats = snapshot.statistics('filename')
        for stat in file_stats[:10]:
            filename = stat.traceback[0].filename if stat.traceback else "unknown"
            analysis["by_file"][filename] = {
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            }
        
        return analysis
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics."""
        before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Force collection
        collected = {}
        for generation in range(gc.get_count().__len__()):
            collected[f"generation_{generation}"] = gc.collect(generation)
        
        after = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "memory_before_mb": before,
            "memory_after_mb": after,
            "memory_freed_mb": before - after,
            "objects_collected": collected
        }
    
    async def generate_report(self):
        """Generate memory profiling report."""
        report_path = Path("tests/stress/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"memory_profile_{timestamp}.json"
        
        # Final heap analysis
        heap_analysis = await self.analyze_heap_dump()
        
        # Detect leaks one final time
        leaks = self.metrics.detect_leaks()
        
        report = {
            "test_type": "memory_profiling",
            "duration_hours": (
                (self.metrics.end_time or datetime.now()) - self.metrics.start_time
            ).total_seconds() / 3600,
            "metrics": {
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "average_memory_mb": self.metrics.average_memory_mb,
                "memory_growth_rate_mb_per_hour": self.metrics.memory_growth_rate,
                "total_snapshots": len(self.metrics.snapshots),
                "potential_leaks": leaks
            },
            "heap_analysis": heap_analysis,
            "success": len(leaks) == 0 and self.metrics.memory_growth_rate < 1
        }
        
        # Add snapshot timeline
        report["timeline"] = [
            {
                "timestamp": s.timestamp.isoformat(),
                "rss_mb": s.rss_mb,
                "object_count": sum(s.object_counts.values())
            }
            for s in self.metrics.snapshots
        ]
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report generated: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 60)
        print(f"Duration: {report['duration_hours']:.1f} hours")
        print(f"Peak Memory: {self.metrics.peak_memory_mb:.1f} MB")
        print(f"Average Memory: {self.metrics.average_memory_mb:.1f} MB")
        print(f"Growth Rate: {self.metrics.memory_growth_rate:.2f} MB/hour")
        print(f"Potential Leaks: {len(leaks)}")
        if leaks:
            for leak in leaks:
                print(f"  - {leak['type']}: {leak.get('severity', 'unknown')} severity")
        print(f"Test Result: {'PASS' if report['success'] else 'FAIL'}")
        print("=" * 60)


async def main():
    """Run memory profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory leak detection")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Profiling duration in days (default: 7)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Snapshot interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick 1-hour test instead"
    )
    
    args = parser.parse_args()
    
    duration_days = 1/24 if args.quick else args.days  # 1 hour for quick test
    
    profiler = MemoryProfiler(snapshot_interval_seconds=args.interval)
    
    try:
        await profiler.start_profiling(duration_days=duration_days)
    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())