"""
Memory profiling and leak detection for load testing.
Monitors memory usage patterns and identifies potential leaks.
"""

import asyncio
import gc
import logging
import os
import time
import tracemalloc
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psutil
from memory_profiler import profile
from prometheus_client import Gauge, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Advanced memory profiling for load testing."""
    
    def __init__(self, threshold_mb: float = 100):
        """
        Initialize memory profiler.
        
        Args:
            threshold_mb: Memory increase threshold to trigger warnings (MB)
        """
        self.threshold_mb = threshold_mb
        self.process = psutil.Process(os.getpid())
        self.baseline_memory: Optional[float] = None
        self.memory_samples: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.tracemalloc_snapshots: List[tracemalloc.Snapshot] = []
        self.leak_candidates: List[Dict] = []
        
        # Prometheus metrics
        self.memory_usage_gauge = Gauge(
            'genesis_memory_usage_mb',
            'Current memory usage in MB'
        )
        self.memory_growth_gauge = Gauge(
            'genesis_memory_growth_mb',
            'Memory growth since baseline in MB'
        )
        self.gc_collections = Gauge(
            'genesis_gc_collections_total',
            'Total garbage collections',
            ['generation']
        )
        
        # Start tracemalloc for detailed tracking
        tracemalloc.start()
        
    def set_baseline(self):
        """Set baseline memory usage for comparison."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.get_memory_usage()
        logger.info(f"Memory baseline set: {self.baseline_memory:.2f} MB")
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
        
    def get_memory_info(self) -> Dict:
        """Get detailed memory information."""
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': mem_percent,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'gc_stats': gc.get_stats(),
            'gc_count': gc.get_count()
        }
        
    def take_snapshot(self) -> tracemalloc.Snapshot:
        """Take a tracemalloc snapshot for detailed analysis."""
        snapshot = tracemalloc.take_snapshot()
        self.tracemalloc_snapshots.append(snapshot)
        return snapshot
        
    def analyze_memory_growth(self) -> Dict:
        """Analyze memory growth patterns."""
        if not self.baseline_memory:
            self.set_baseline()
            
        current_memory = self.get_memory_usage()
        growth = current_memory - self.baseline_memory
        
        # Update Prometheus metrics
        self.memory_usage_gauge.set(current_memory)
        self.memory_growth_gauge.set(growth)
        
        # Update GC metrics
        for i, count in enumerate(gc.get_count()):
            self.gc_collections.labels(generation=str(i)).set(count)
        
        # Record sample
        self.memory_samples.append((time.time(), current_memory))
        
        # Check for potential leak
        is_leak = growth > self.threshold_mb
        
        if is_leak:
            logger.warning(f"Potential memory leak detected! Growth: {growth:.2f} MB")
            self.identify_leak_sources()
            
        return {
            'baseline_mb': self.baseline_memory,
            'current_mb': current_memory,
            'growth_mb': growth,
            'growth_percent': (growth / self.baseline_memory) * 100,
            'is_potential_leak': is_leak,
            'samples_count': len(self.memory_samples)
        }
        
    def identify_leak_sources(self):
        """Identify potential sources of memory leaks."""
        if len(self.tracemalloc_snapshots) < 2:
            logger.warning("Need at least 2 snapshots to identify leak sources")
            return
            
        # Compare last two snapshots
        old_snapshot = self.tracemalloc_snapshots[-2]
        new_snapshot = self.tracemalloc_snapshots[-1]
        
        top_stats = new_snapshot.compare_to(old_snapshot, 'lineno')
        
        leak_candidates = []
        for stat in top_stats[:10]:  # Top 10 memory increases
            if stat.size_diff > 1024 * 1024:  # More than 1MB increase
                leak_candidates.append({
                    'file': stat.traceback.format()[0] if stat.traceback else 'Unknown',
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'count_diff': stat.count_diff
                })
                
        if leak_candidates:
            self.leak_candidates.extend(leak_candidates)
            logger.warning(f"Top memory increases:")
            for candidate in leak_candidates:
                logger.warning(f"  {candidate['file']}: +{candidate['size_diff_mb']:.2f} MB")
                
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict]:
        """Get top memory consuming code locations."""
        if not self.tracemalloc_snapshots:
            return []
            
        snapshot = self.tracemalloc_snapshots[-1]
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for stat in top_stats[:limit]:
            consumers.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'Unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
            
        return consumers
        
    def calculate_memory_trend(self, window_seconds: float = 60) -> Dict:
        """Calculate memory usage trend over time window."""
        if len(self.memory_samples) < 2:
            return {'trend': 'insufficient_data'}
            
        current_time = time.time()
        recent_samples = [
            (t, m) for t, m in self.memory_samples 
            if current_time - t <= window_seconds
        ]
        
        if len(recent_samples) < 2:
            return {'trend': 'insufficient_data'}
            
        # Simple linear regression for trend
        n = len(recent_samples)
        sum_t = sum(t for t, _ in recent_samples)
        sum_m = sum(m for _, m in recent_samples)
        sum_tm = sum(t * m for t, m in recent_samples)
        sum_t2 = sum(t * t for t, _ in recent_samples)
        
        # Calculate slope (MB per second)
        slope = (n * sum_tm - sum_t * sum_m) / (n * sum_t2 - sum_t * sum_t)
        
        # Determine trend
        if abs(slope) < 0.001:  # Less than 1KB/s
            trend = 'stable'
        elif slope > 0.01:  # More than 10KB/s growth
            trend = 'increasing'
        elif slope < -0.01:  # More than 10KB/s decrease
            trend = 'decreasing'
        else:
            trend = 'slight_change'
            
        return {
            'trend': trend,
            'slope_mb_per_second': slope,
            'slope_mb_per_minute': slope * 60,
            'samples_analyzed': n,
            'window_seconds': window_seconds
        }
        
    def force_garbage_collection(self) -> Dict:
        """Force garbage collection and return statistics."""
        before_memory = self.get_memory_usage()
        before_objects = len(gc.get_objects())
        
        # Collect all generations
        collected = gc.collect(2)
        
        after_memory = self.get_memory_usage()
        after_objects = len(gc.get_objects())
        
        return {
            'collected_objects': collected,
            'memory_freed_mb': before_memory - after_memory,
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_freed': before_objects - after_objects
        }
        
    def generate_report(self) -> Dict:
        """Generate comprehensive memory profiling report."""
        current_analysis = self.analyze_memory_growth()
        trend = self.calculate_memory_trend()
        top_consumers = self.get_top_memory_consumers()
        gc_stats = self.force_garbage_collection()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_analysis': current_analysis,
            'trend': trend,
            'top_consumers': top_consumers,
            'gc_stats': gc_stats,
            'leak_candidates': self.leak_candidates[-10:] if self.leak_candidates else [],
            'memory_info': self.get_memory_info()
        }
        
        return report
        
    def reset(self):
        """Reset profiler state."""
        gc.collect()
        self.baseline_memory = None
        self.memory_samples.clear()
        self.tracemalloc_snapshots.clear()
        self.leak_candidates.clear()
        tracemalloc.clear_traces()
        

class MemoryLoadTest:
    """Memory-focused load testing scenarios."""
    
    def __init__(self, profiler: MemoryProfiler):
        self.profiler = profiler
        self.test_data: List[bytes] = []
        
    async def test_memory_accumulation(self, duration_seconds: int = 60):
        """Test for memory accumulation issues."""
        logger.info(f"Starting memory accumulation test for {duration_seconds} seconds")
        
        self.profiler.set_baseline()
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Simulate data accumulation
            data = b"x" * (1024 * 1024)  # 1MB of data
            self.test_data.append(data)
            
            # Take snapshot every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                self.profiler.take_snapshot()
                analysis = self.profiler.analyze_memory_growth()
                
                if analysis['is_potential_leak']:
                    logger.error(f"Memory leak detected during accumulation test!")
                    
            await asyncio.sleep(1)
            
        # Clear test data
        self.test_data.clear()
        gc.collect()
        
        # Final analysis
        final_report = self.profiler.generate_report()
        return final_report
        
    async def test_connection_memory(self, num_connections: int = 100):
        """Test memory usage with multiple connections."""
        logger.info(f"Testing memory with {num_connections} connections")
        
        self.profiler.set_baseline()
        connections = []
        
        # Simulate creating connections
        for i in range(num_connections):
            conn = {'id': i, 'data': b"x" * (1024 * 100)}  # 100KB per connection
            connections.append(conn)
            
            if i % 10 == 0:
                analysis = self.profiler.analyze_memory_growth()
                logger.info(f"Memory after {i} connections: {analysis['current_mb']:.2f} MB")
                
        # Hold connections for a while
        await asyncio.sleep(10)
        
        # Close connections
        connections.clear()
        gc.collect()
        
        final_report = self.profiler.generate_report()
        return final_report
        
    async def test_message_processing_memory(self, messages_per_second: int = 1000):
        """Test memory usage during message processing."""
        logger.info(f"Testing message processing at {messages_per_second} msg/s")
        
        self.profiler.set_baseline()
        message_count = 0
        start_time = time.time()
        test_duration = 60  # seconds
        
        while time.time() - start_time < test_duration:
            # Process batch of messages
            batch_size = messages_per_second // 10  # Process in batches
            
            for _ in range(batch_size):
                # Simulate message processing
                message = {
                    'id': message_count,
                    'timestamp': time.time(),
                    'data': 'x' * 1024  # 1KB message
                }
                
                # Process message (simulate work)
                processed = str(message).encode()
                del processed  # Immediately delete
                
                message_count += 1
                
            # Check memory every second
            if int(time.time() - start_time) % 1 == 0:
                self.profiler.take_snapshot()
                trend = self.profiler.calculate_memory_trend(window_seconds=10)
                
                if trend['trend'] == 'increasing':
                    logger.warning(f"Memory increasing during message processing: {trend['slope_mb_per_minute']:.2f} MB/min")
                    
            await asyncio.sleep(0.1)
            
        final_report = self.profiler.generate_report()
        final_report['messages_processed'] = message_count
        return final_report
        

@profile
def memory_intensive_operation():
    """Example memory-intensive operation for profiling."""
    data = []
    for i in range(1000000):
        data.append({'id': i, 'value': 'x' * 100})
    return len(data)


async def run_memory_profile_suite():
    """Run complete memory profiling test suite."""
    logger.info("Starting memory profiling suite")
    
    profiler = MemoryProfiler(threshold_mb=50)
    test = MemoryLoadTest(profiler)
    
    results = {}
    
    # Test 1: Memory accumulation
    logger.info("\n=== Test 1: Memory Accumulation ===")
    results['accumulation'] = await test.test_memory_accumulation(duration_seconds=30)
    
    # Reset between tests
    profiler.reset()
    await asyncio.sleep(5)
    
    # Test 2: Connection memory
    logger.info("\n=== Test 2: Connection Memory ===")
    results['connections'] = await test.test_connection_memory(num_connections=50)
    
    # Reset between tests
    profiler.reset()
    await asyncio.sleep(5)
    
    # Test 3: Message processing
    logger.info("\n=== Test 3: Message Processing ===")
    results['messages'] = await test.test_message_processing_memory(messages_per_second=500)
    
    # Generate final report
    logger.info("\n=== Final Memory Profile Report ===")
    for test_name, result in results.items():
        logger.info(f"\n{test_name.upper()} Test:")
        logger.info(f"  Final memory: {result['current_analysis']['current_mb']:.2f} MB")
        logger.info(f"  Memory growth: {result['current_analysis']['growth_mb']:.2f} MB")
        logger.info(f"  Trend: {result['trend']['trend']}")
        
        if result.get('leak_candidates'):
            logger.warning(f"  Potential leak sources:")
            for candidate in result['leak_candidates'][:3]:
                logger.warning(f"    {candidate['file']}: +{candidate['size_diff_mb']:.2f} MB")
                
    return results


if __name__ == "__main__":
    # Run memory profiling suite
    asyncio.run(run_memory_profile_suite())