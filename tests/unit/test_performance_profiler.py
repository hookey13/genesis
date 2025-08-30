"""Unit tests for Performance Profiler."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock, mock_open
import cProfile
import pstats
from io import StringIO

from genesis.monitoring.performance_profiler import (
    ProfileType,
    ProfileResult,
    PerformanceProfiler,
    CPUProfiler,
    MemoryProfiler,
    AsyncProfiler,
    FlameGraphGenerator,
    PerformanceBaseline,
    RegressionDetector
)


class TestProfileResult:
    """Test ProfileResult class."""
    
    def test_profile_result_creation(self):
        """Test creating a profile result."""
        result = ProfileResult(
            profile_type=ProfileType.CPU,
            duration_seconds=10.5,
            samples_collected=1000,
            top_functions=[
                ("function1", 45.0),
                ("function2", 30.0),
                ("function3", 15.0)
            ]
        )
        
        assert result.profile_type == ProfileType.CPU
        assert result.duration_seconds == 10.5
        assert result.samples_collected == 1000
        assert len(result.top_functions) == 3
        assert result.top_functions[0][1] == 45.0
    
    def test_profile_result_with_flame_graph(self):
        """Test profile result with flame graph data."""
        result = ProfileResult(
            profile_type=ProfileType.CPU,
            duration_seconds=5.0,
            samples_collected=500,
            flame_graph_data={"root": {"children": {}}}
        )
        
        assert result.flame_graph_data is not None
        assert "root" in result.flame_graph_data
    
    def test_profile_result_to_dict(self):
        """Test converting profile result to dictionary."""
        result = ProfileResult(
            profile_type=ProfileType.MEMORY,
            duration_seconds=3.0,
            samples_collected=100,
            memory_usage_mb=256.5,
            memory_peak_mb=512.0
        )
        
        data = result.to_dict()
        
        assert data["profile_type"] == "MEMORY"
        assert data["duration_seconds"] == 3.0
        assert data["memory_usage_mb"] == 256.5
        assert data["memory_peak_mb"] == 512.0


class TestCPUProfiler:
    """Test CPUProfiler class."""
    
    def test_cpu_profiler_initialization(self):
        """Test CPU profiler initialization."""
        profiler = CPUProfiler(sampling_interval=0.001)
        
        assert profiler.sampling_interval == 0.001
        assert profiler.profiler is None
        assert profiler.is_running is False
    
    def test_cpu_profiler_start_stop(self):
        """Test starting and stopping CPU profiler."""
        profiler = CPUProfiler()
        
        profiler.start()
        assert profiler.is_running is True
        assert profiler.profiler is not None
        
        # Do some work
        sum([i**2 for i in range(1000)])
        
        result = profiler.stop()
        assert profiler.is_running is False
        assert result is not None
        assert result.profile_type == ProfileType.CPU
        assert result.duration_seconds > 0
    
    def test_cpu_profiler_context_manager(self):
        """Test using CPU profiler as context manager."""
        profiler = CPUProfiler()
        
        with profiler:
            # Do some work
            for i in range(1000):
                _ = i ** 2
        
        result = profiler.get_results()
        assert result is not None
        assert result.samples_collected > 0
    
    def test_cpu_profiler_top_functions(self):
        """Test getting top functions from CPU profile."""
        profiler = CPUProfiler()
        
        def test_function():
            return sum(i**2 for i in range(10000))
        
        profiler.start()
        test_function()
        result = profiler.stop()
        
        assert len(result.top_functions) > 0
        # Function names and percentages should be present
        for func_name, percentage in result.top_functions:
            assert isinstance(func_name, str)
            assert isinstance(percentage, float)
            assert 0 <= percentage <= 100
    
    @patch('cProfile.Profile')
    def test_cpu_profiler_error_handling(self, mock_profile):
        """Test CPU profiler error handling."""
        mock_profile.side_effect = Exception("Profile error")
        
        profiler = CPUProfiler()
        profiler.start()
        
        # Should handle error gracefully
        result = profiler.stop()
        assert result is None


class TestMemoryProfiler:
    """Test MemoryProfiler class."""
    
    @patch('psutil.Process')
    def test_memory_profiler_initialization(self, mock_process):
        """Test memory profiler initialization."""
        profiler = MemoryProfiler(interval=0.5)
        
        assert profiler.interval == 0.5
        assert profiler.is_running is False
        assert len(profiler.memory_samples) == 0
    
    @patch('psutil.Process')
    def test_memory_profiler_start_stop(self, mock_process):
        """Test starting and stopping memory profiler."""
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value = mock_proc
        
        profiler = MemoryProfiler(interval=0.01)
        
        profiler.start()
        assert profiler.is_running is True
        
        time.sleep(0.05)  # Let it collect some samples
        
        result = profiler.stop()
        assert profiler.is_running is False
        assert result is not None
        assert result.profile_type == ProfileType.MEMORY
        assert result.memory_usage_mb > 0
    
    @patch('psutil.Process')
    def test_memory_profiler_peak_detection(self, mock_process):
        """Test memory peak detection."""
        mock_proc = MagicMock()
        memory_values = [50, 100, 200, 150, 100]  # Peak at 200 MB
        mock_proc.memory_info.side_effect = [
            MagicMock(rss=mb * 1024 * 1024) for mb in memory_values
        ]
        mock_process.return_value = mock_proc
        
        profiler = MemoryProfiler(interval=0.01)
        
        profiler.start()
        time.sleep(0.06)
        result = profiler.stop()
        
        assert result.memory_peak_mb >= 200
    
    @patch('psutil.Process')
    def test_memory_profiler_allocation_tracking(self, mock_process):
        """Test tracking memory allocations."""
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.return_value = mock_proc
        
        profiler = MemoryProfiler()
        
        # Track allocations
        profiler.track_allocation("buffer1", 10 * 1024 * 1024)  # 10 MB
        profiler.track_allocation("buffer2", 20 * 1024 * 1024)  # 20 MB
        
        allocations = profiler.get_allocations()
        assert "buffer1" in allocations
        assert "buffer2" in allocations
        assert allocations["buffer1"] == 10 * 1024 * 1024
    
    @patch('tracemalloc')
    @patch('psutil.Process')
    def test_memory_profiler_leak_detection(self, mock_process, mock_tracemalloc):
        """Test memory leak detection."""
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process.return_value = mock_proc
        
        profiler = MemoryProfiler()
        
        # Simulate potential leak
        leaks = profiler.detect_leaks(threshold_mb=50)
        assert isinstance(leaks, list)


class TestAsyncProfiler:
    """Test AsyncProfiler class."""
    
    @pytest.mark.asyncio
    async def test_async_profiler_initialization(self):
        """Test async profiler initialization."""
        profiler = AsyncProfiler()
        
        assert profiler.tasks_profiled == 0
        assert len(profiler.task_stats) == 0
    
    @pytest.mark.asyncio
    async def test_async_profiler_task_profiling(self):
        """Test profiling async tasks."""
        profiler = AsyncProfiler()
        
        async def test_task():
            await asyncio.sleep(0.01)
            return "result"
        
        # Profile the task
        result = await profiler.profile_task(test_task())
        
        assert result == "result"
        assert profiler.tasks_profiled == 1
        assert len(profiler.task_stats) == 1
    
    @pytest.mark.asyncio
    async def test_async_profiler_context_manager(self):
        """Test using async profiler as context manager."""
        profiler = AsyncProfiler()
        
        async with profiler.profile("test_operation"):
            await asyncio.sleep(0.01)
        
        stats = profiler.get_stats("test_operation")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["total_time"] > 0
    
    @pytest.mark.asyncio
    async def test_async_profiler_concurrent_tasks(self):
        """Test profiling concurrent tasks."""
        profiler = AsyncProfiler()
        
        async def task(n):
            await asyncio.sleep(0.01 * n)
            return n
        
        # Profile multiple concurrent tasks
        tasks = [profiler.profile_task(task(i)) for i in range(1, 4)]
        results = await asyncio.gather(*tasks)
        
        assert results == [1, 2, 3]
        assert profiler.tasks_profiled == 3
    
    @pytest.mark.asyncio
    async def test_async_profiler_event_loop_metrics(self):
        """Test collecting event loop metrics."""
        profiler = AsyncProfiler()
        
        metrics = await profiler.get_event_loop_metrics()
        
        assert "tasks_running" in metrics
        assert "tasks_pending" in metrics
        assert isinstance(metrics["tasks_running"], int)


class TestFlameGraphGenerator:
    """Test FlameGraphGenerator class."""
    
    def test_flame_graph_initialization(self):
        """Test flame graph generator initialization."""
        generator = FlameGraphGenerator()
        
        assert generator.stack_traces == []
        assert generator.folded_stacks == {}
    
    def test_flame_graph_add_stack_trace(self):
        """Test adding stack traces."""
        generator = FlameGraphGenerator()
        
        generator.add_stack_trace(["main", "function1", "function2"], 10)
        generator.add_stack_trace(["main", "function1", "function3"], 5)
        generator.add_stack_trace(["main", "function4"], 15)
        
        assert len(generator.stack_traces) == 3
        assert generator.stack_traces[0][1] == 10
    
    def test_flame_graph_fold_stacks(self):
        """Test folding stack traces."""
        generator = FlameGraphGenerator()
        
        generator.add_stack_trace(["main", "func1", "func2"], 10)
        generator.add_stack_trace(["main", "func1", "func2"], 5)
        generator.add_stack_trace(["main", "func1", "func3"], 8)
        
        generator.fold_stacks()
        
        assert "main;func1;func2" in generator.folded_stacks
        assert generator.folded_stacks["main;func1;func2"] == 15
        assert generator.folded_stacks["main;func1;func3"] == 8
    
    def test_flame_graph_generate_svg(self):
        """Test generating SVG flame graph."""
        generator = FlameGraphGenerator()
        
        generator.add_stack_trace(["main", "process", "calculate"], 100)
        generator.add_stack_trace(["main", "process", "validate"], 50)
        generator.add_stack_trace(["main", "output"], 30)
        
        svg_data = generator.generate_svg()
        
        # Should return SVG-like data (even if simplified)
        assert svg_data is not None
        assert isinstance(svg_data, str)
    
    def test_flame_graph_from_profile(self):
        """Test generating flame graph from profile data."""
        generator = FlameGraphGenerator()
        
        # Create a simple profile
        prof = cProfile.Profile()
        prof.enable()
        sum([i**2 for i in range(100)])
        prof.disable()
        
        # Generate flame graph from profile
        generator.from_profile(prof)
        
        assert len(generator.stack_traces) > 0
    
    def test_flame_graph_export_json(self):
        """Test exporting flame graph as JSON."""
        generator = FlameGraphGenerator()
        
        generator.add_stack_trace(["main", "func1"], 10)
        generator.add_stack_trace(["main", "func2"], 20)
        
        json_data = generator.export_json()
        
        assert "name" in json_data
        assert "value" in json_data
        assert "children" in json_data


class TestPerformanceBaseline:
    """Test PerformanceBaseline class."""
    
    def test_baseline_initialization(self):
        """Test performance baseline initialization."""
        baseline = PerformanceBaseline(name="test_function")
        
        assert baseline.name == "test_function"
        assert baseline.samples == []
        assert baseline.mean is None
        assert baseline.std_dev is None
    
    def test_baseline_add_samples(self):
        """Test adding samples to baseline."""
        baseline = PerformanceBaseline(name="test_function")
        
        samples = [0.1, 0.12, 0.11, 0.09, 0.10, 0.11]
        for sample in samples:
            baseline.add_sample(sample)
        
        assert len(baseline.samples) == 6
        baseline.calculate_statistics()
        
        assert baseline.mean is not None
        assert abs(baseline.mean - 0.105) < 0.01
        assert baseline.std_dev is not None
    
    def test_baseline_outlier_detection(self):
        """Test detecting outliers in baseline."""
        baseline = PerformanceBaseline(name="test_function")
        
        # Add normal samples
        normal_samples = [0.1, 0.11, 0.09, 0.10, 0.11]
        for sample in normal_samples:
            baseline.add_sample(sample)
        
        baseline.calculate_statistics()
        
        # Test outlier detection
        assert baseline.is_outlier(0.2) is True  # Too slow
        assert baseline.is_outlier(0.11) is False  # Normal
        assert baseline.is_outlier(0.01) is True  # Too fast (suspicious)
    
    def test_baseline_percentiles(self):
        """Test calculating percentiles."""
        baseline = PerformanceBaseline(name="test_function")
        
        samples = list(range(1, 101))  # 1 to 100
        for sample in samples:
            baseline.add_sample(sample)
        
        assert baseline.get_percentile(50) == 50.5  # Median
        assert baseline.get_percentile(95) == 95.5
        assert baseline.get_percentile(99) == 99.5


class TestRegressionDetector:
    """Test RegressionDetector class."""
    
    def test_regression_detector_initialization(self):
        """Test regression detector initialization."""
        detector = RegressionDetector(threshold_percent=10)
        
        assert detector.threshold_percent == 10
        assert len(detector.baselines) == 0
    
    def test_regression_detector_add_baseline(self):
        """Test adding baseline to detector."""
        detector = RegressionDetector()
        
        baseline = PerformanceBaseline(name="function1")
        baseline.add_sample(0.1)
        baseline.add_sample(0.11)
        baseline.calculate_statistics()
        
        detector.add_baseline(baseline)
        assert "function1" in detector.baselines
    
    def test_regression_detector_check_regression(self):
        """Test checking for performance regression."""
        detector = RegressionDetector(threshold_percent=20)
        
        # Create baseline
        baseline = PerformanceBaseline(name="function1")
        for _ in range(10):
            baseline.add_sample(0.1)  # 100ms baseline
        baseline.calculate_statistics()
        detector.add_baseline(baseline)
        
        # Check for regression
        assert detector.check_regression("function1", 0.11) is False  # 10% slower - OK
        assert detector.check_regression("function1", 0.13) is True   # 30% slower - Regression
        assert detector.check_regression("function1", 0.08) is False  # Faster - OK
    
    def test_regression_detector_compare_profiles(self):
        """Test comparing two profiles for regression."""
        detector = RegressionDetector()
        
        # Create baseline profile
        baseline_result = ProfileResult(
            profile_type=ProfileType.CPU,
            duration_seconds=10,
            samples_collected=1000,
            top_functions=[
                ("function1", 30.0),
                ("function2", 20.0),
                ("function3", 10.0)
            ]
        )
        
        # Create current profile with regression
        current_result = ProfileResult(
            profile_type=ProfileType.CPU,
            duration_seconds=10,
            samples_collected=1000,
            top_functions=[
                ("function1", 45.0),  # 50% increase - regression
                ("function2", 22.0),  # 10% increase - OK
                ("function3", 8.0)    # Decrease - OK
            ]
        )
        
        regressions = detector.compare_profiles(baseline_result, current_result)
        
        assert len(regressions) > 0
        assert any("function1" in r for r in regressions)
    
    def test_regression_detector_trend_analysis(self):
        """Test trend analysis for gradual regression."""
        detector = RegressionDetector()
        
        baseline = PerformanceBaseline(name="function1")
        
        # Add samples showing gradual degradation
        times = [0.1, 0.1, 0.11, 0.11, 0.12, 0.13, 0.14, 0.15]
        for t in times:
            baseline.add_sample(t)
        
        trend = detector.analyze_trend(baseline)
        
        assert trend["direction"] == "increasing"  # Performance getting worse
        assert trend["slope"] > 0  # Positive slope indicates degradation


class TestPerformanceProfiler:
    """Test main PerformanceProfiler class."""
    
    def test_profiler_initialization(self):
        """Test performance profiler initialization."""
        profiler = PerformanceProfiler(
            cpu_enabled=True,
            memory_enabled=True,
            async_enabled=True
        )
        
        assert profiler.cpu_enabled is True
        assert profiler.memory_enabled is True
        assert profiler.async_enabled is True
    
    @pytest.mark.asyncio
    async def test_profiler_profile_function(self):
        """Test profiling a function."""
        profiler = PerformanceProfiler()
        
        def test_function(n):
            return sum(i**2 for i in range(n))
        
        result = await profiler.profile_function(test_function, 1000)
        
        assert result is not None
        assert result["return_value"] == sum(i**2 for i in range(1000))
        assert result["duration"] > 0
    
    @pytest.mark.asyncio
    async def test_profiler_profile_async_function(self):
        """Test profiling an async function."""
        profiler = PerformanceProfiler()
        
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await profiler.profile_function(async_test_function)
        
        assert result["return_value"] == "async_result"
        assert result["duration"] >= 0.01
    
    @pytest.mark.asyncio
    async def test_profiler_continuous_profiling(self):
        """Test continuous profiling."""
        profiler = PerformanceProfiler()
        
        # Start continuous profiling
        await profiler.start_continuous_profiling(interval=0.1, duration=0.3)
        
        # Should have collected some profiles
        profiles = profiler.get_collected_profiles()
        assert len(profiles) > 0
    
    def test_profiler_decorator(self):
        """Test profiler decorator."""
        profiler = PerformanceProfiler()
        
        @profiler.profile
        def decorated_function(x, y):
            return x + y
        
        result = decorated_function(5, 3)
        assert result == 8
        
        # Check that profile was recorded
        profiles = profiler.get_collected_profiles()
        assert len(profiles) > 0