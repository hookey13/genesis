"""Resource validator for memory leaks, CPU usage, and container resource limits."""

import asyncio
import gc
import os
import psutil
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ResourceValidator:
    """Validates resource usage including memory, CPU, and container limits."""

    # Resource thresholds
    MAX_MEMORY_USAGE_MB = 1024  # 1GB max memory
    MAX_MEMORY_GROWTH_MB = 100  # Max 100MB growth during test
    MAX_CPU_USAGE_PERCENT = 80
    MAX_FILE_DESCRIPTORS = 1000
    MAX_THREAD_COUNT = 50
    MAX_CONNECTION_COUNT = 100
    
    # Container limits (for Docker/K8s environments)
    CONTAINER_MEMORY_LIMIT_MB = 512
    CONTAINER_CPU_LIMIT_CORES = 2

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize resource validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.process = psutil.Process()
        self.resource_history: List[Dict[str, Any]] = []
        self.memory_snapshots: List[tracemalloc.Snapshot] = []

    async def run_validation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive resource validation.
        
        Args:
            context: Optional context for validation
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting resource validation")
        start_time = datetime.utcnow()

        results = {
            "validator": "ResourceValidator",
            "timestamp": start_time.isoformat(),
            "status": "pending",
            "passed": False,
            "memory_analysis": {},
            "cpu_analysis": {},
            "resource_usage": {},
            "container_limits": {},
            "leaks_detected": [],
            "violations": [],
            "trends": {},
            "evidence": {},
        }

        try:
            # Start memory tracking
            tracemalloc.start()
            initial_snapshot = tracemalloc.take_snapshot()

            # Analyze current memory usage
            memory_analysis = await self._analyze_memory_usage(initial_snapshot)
            results["memory_analysis"] = memory_analysis

            # Detect memory leaks
            leaks = await self._detect_memory_leaks()
            results["leaks_detected"] = leaks

            # Analyze CPU usage
            cpu_analysis = await self._analyze_cpu_usage()
            results["cpu_analysis"] = cpu_analysis

            # Check overall resource usage
            resource_usage = self._check_resource_usage()
            results["resource_usage"] = resource_usage

            # Validate container resource limits
            container_limits = self._validate_container_limits()
            results["container_limits"] = container_limits

            # Analyze resource trends
            trends = self._analyze_resource_trends(memory_analysis, cpu_analysis, resource_usage)
            results["trends"] = trends

            # Check for violations
            violations = self._check_resource_violations(
                memory_analysis, cpu_analysis, resource_usage, leaks
            )
            results["violations"] = violations

            # Generate evidence report
            evidence = self._generate_resource_evidence(
                memory_analysis, cpu_analysis, resource_usage, leaks, violations
            )
            results["evidence"] = evidence

            # Determine pass/fail
            results["passed"] = len(violations) == 0 and len(leaks) == 0
            results["status"] = "passed" if results["passed"] else "failed"

            # Stop memory tracking
            tracemalloc.stop()

            logger.info(
                "Resource validation completed",
                passed=results["passed"],
                violations=len(violations),
                leaks=len(leaks),
            )

        except Exception as e:
            logger.error("Resource validation failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)
            tracemalloc.stop()

        return results

    async def _analyze_memory_usage(self, initial_snapshot: tracemalloc.Snapshot) -> Dict[str, Any]:
        """Analyze current memory usage patterns.
        
        Args:
            initial_snapshot: Initial memory snapshot
            
        Returns:
            Memory analysis results
        """
        logger.info("Analyzing memory usage")
        
        memory_data = {
            "current_usage_mb": 0,
            "peak_usage_mb": 0,
            "available_mb": 0,
            "percent_used": 0,
            "top_allocations": [],
            "by_category": {},
            "gc_stats": {},
        }

        try:
            # Get process memory info
            mem_info = self.process.memory_info()
            memory_data["current_usage_mb"] = mem_info.rss / 1024 / 1024
            memory_data["peak_usage_mb"] = mem_info.rss / 1024 / 1024  # Will track over time

            # Get system memory info
            virtual_mem = psutil.virtual_memory()
            memory_data["available_mb"] = virtual_mem.available / 1024 / 1024
            memory_data["percent_used"] = virtual_mem.percent

            # Take current snapshot and compare
            current_snapshot = tracemalloc.take_snapshot()
            self.memory_snapshots.append(current_snapshot)

            # Get top memory allocations
            top_stats = current_snapshot.statistics("lineno")[:10]
            for stat in top_stats:
                memory_data["top_allocations"].append({
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                })

            # Categorize memory usage by module
            stats_by_file = current_snapshot.statistics("filename")
            for stat in stats_by_file:
                module = self._categorize_module(stat.traceback)
                if module not in memory_data["by_category"]:
                    memory_data["by_category"][module] = 0
                memory_data["by_category"][module] += stat.size / 1024 / 1024

            # Get garbage collection stats
            gc_stats = gc.get_stats()
            if gc_stats:
                latest_gc = gc_stats[-1]
                memory_data["gc_stats"] = {
                    "collections": latest_gc.get("collections", 0),
                    "collected": latest_gc.get("collected", 0),
                    "uncollectable": latest_gc.get("uncollectable", 0),
                }

        except Exception as e:
            logger.error("Failed to analyze memory", error=str(e))

        return memory_data

    async def _detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks using tracemalloc.
        
        Returns:
            List of detected memory leaks
        """
        logger.info("Detecting memory leaks")
        leaks = []

        try:
            # Run a simple workload to detect leaks
            initial_mem = self.process.memory_info().rss / 1024 / 1024
            
            # Simulate workload
            for i in range(5):
                await self._simulate_workload()
                gc.collect()  # Force garbage collection
                await asyncio.sleep(1)

            # Check memory growth
            final_mem = self.process.memory_info().rss / 1024 / 1024
            growth_mb = final_mem - initial_mem

            if growth_mb > self.MAX_MEMORY_GROWTH_MB:
                leaks.append({
                    "type": "memory_growth",
                    "initial_mb": initial_mem,
                    "final_mb": final_mem,
                    "growth_mb": growth_mb,
                    "threshold_mb": self.MAX_MEMORY_GROWTH_MB,
                    "severity": "high",
                })

            # Analyze snapshots for leak patterns
            if len(self.memory_snapshots) >= 2:
                first_snapshot = self.memory_snapshots[0]
                last_snapshot = self.memory_snapshots[-1]
                
                diff = last_snapshot.compare_to(first_snapshot, "lineno")
                
                # Find growing allocations
                for stat in diff[:10]:
                    if stat.size_diff > 1024 * 1024:  # More than 1MB growth
                        leaks.append({
                            "type": "growing_allocation",
                            "location": stat.traceback.format()[0] if stat.traceback else "unknown",
                            "size_diff_mb": stat.size_diff / 1024 / 1024,
                            "count_diff": stat.count_diff,
                            "severity": "medium",
                        })

            # Check for uncollectable objects
            uncollectable = gc.collect()
            if uncollectable > 0:
                leaks.append({
                    "type": "uncollectable_objects",
                    "count": uncollectable,
                    "severity": "high",
                    "description": "Objects with circular references and __del__ methods",
                })

        except Exception as e:
            logger.error("Failed to detect memory leaks", error=str(e))

        return leaks

    async def _simulate_workload(self) -> None:
        """Simulate a typical workload for leak detection."""
        try:
            # Create some objects
            data = []
            for _ in range(1000):
                data.append({"id": os.urandom(16).hex(), "value": os.urandom(1024)})
            
            # Process data
            processed = [d["id"] for d in data if d["value"]]
            
            # Simulate async operations
            await asyncio.sleep(0.1)
            
            # Clear references
            del data
            del processed
            
        except Exception as e:
            logger.warning("Workload simulation failed", error=str(e))

    async def _analyze_cpu_usage(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns.
        
        Returns:
            CPU analysis results
        """
        logger.info("Analyzing CPU usage")
        
        cpu_data = {
            "current_percent": 0,
            "average_percent": 0,
            "peak_percent": 0,
            "per_core": [],
            "thread_count": 0,
            "context_switches": 0,
            "cpu_times": {},
        }

        try:
            # Get current CPU usage
            cpu_data["current_percent"] = self.process.cpu_percent(interval=1)
            
            # Sample CPU usage over time
            samples = []
            for _ in range(5):
                samples.append(self.process.cpu_percent(interval=0.5))
                await asyncio.sleep(0.5)
            
            cpu_data["average_percent"] = sum(samples) / len(samples) if samples else 0
            cpu_data["peak_percent"] = max(samples) if samples else 0

            # Get per-core usage
            per_core = psutil.cpu_percent(interval=1, percpu=True)
            cpu_data["per_core"] = [{"core": i, "percent": usage} for i, usage in enumerate(per_core)]

            # Get thread information
            cpu_data["thread_count"] = self.process.num_threads()

            # Get context switches
            ctx_switches = self.process.num_ctx_switches()
            cpu_data["context_switches"] = ctx_switches.voluntary + ctx_switches.involuntary

            # Get CPU times
            cpu_times = self.process.cpu_times()
            cpu_data["cpu_times"] = {
                "user": cpu_times.user,
                "system": cpu_times.system,
                "total": cpu_times.user + cpu_times.system,
            }

        except Exception as e:
            logger.error("Failed to analyze CPU", error=str(e))

        return cpu_data

    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check overall system resource usage.
        
        Returns:
            Resource usage information
        """
        resource_data = {
            "file_descriptors": 0,
            "max_file_descriptors": 0,
            "connections": 0,
            "disk_io": {},
            "network_io": {},
        }

        try:
            # Get file descriptor usage
            resource_data["file_descriptors"] = self.process.num_fds() if hasattr(self.process, "num_fds") else 0
            
            # Get soft limit for file descriptors
            import resource
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource_data["max_file_descriptors"] = soft_limit

            # Get network connections
            try:
                connections = self.process.connections()
                resource_data["connections"] = len(connections)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                resource_data["connections"] = 0

            # Get disk I/O stats
            try:
                io_counters = self.process.io_counters()
                resource_data["disk_io"] = {
                    "read_bytes_mb": io_counters.read_bytes / 1024 / 1024,
                    "write_bytes_mb": io_counters.write_bytes / 1024 / 1024,
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                }
            except (psutil.AccessDenied, AttributeError):
                pass

            # Get network I/O stats
            try:
                net_io = psutil.net_io_counters()
                resource_data["network_io"] = {
                    "bytes_sent_mb": net_io.bytes_sent / 1024 / 1024,
                    "bytes_recv_mb": net_io.bytes_recv / 1024 / 1024,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
            except Exception:
                pass

        except Exception as e:
            logger.error("Failed to check resource usage", error=str(e))

        return resource_data

    def _validate_container_limits(self) -> Dict[str, Any]:
        """Validate container resource limits if running in container.
        
        Returns:
            Container limit validation results
        """
        container_data = {
            "is_containerized": False,
            "memory_limit_mb": 0,
            "memory_usage_mb": 0,
            "cpu_limit_cores": 0,
            "cpu_usage_cores": 0,
            "compliance": True,
        }

        try:
            # Check if running in container (Docker/K8s)
            if os.path.exists("/.dockerenv") or os.environ.get("KUBERNETES_SERVICE_HOST"):
                container_data["is_containerized"] = True

                # Check cgroup limits (Docker/K8s)
                cgroup_memory_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                if cgroup_memory_limit.exists():
                    limit_bytes = int(cgroup_memory_limit.read_text().strip())
                    # Check if it's not the max value (no limit)
                    if limit_bytes < 9223372036854771712:  # Default max value
                        container_data["memory_limit_mb"] = limit_bytes / 1024 / 1024

                # Check current usage against limits
                mem_info = self.process.memory_info()
                container_data["memory_usage_mb"] = mem_info.rss / 1024 / 1024

                # Check CPU limits
                cgroup_cpu_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
                cgroup_cpu_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
                
                if cgroup_cpu_quota.exists() and cgroup_cpu_period.exists():
                    quota = int(cgroup_cpu_quota.read_text().strip())
                    period = int(cgroup_cpu_period.read_text().strip())
                    
                    if quota > 0:
                        container_data["cpu_limit_cores"] = quota / period

                # Check compliance with recommended limits
                if container_data["memory_limit_mb"] > 0:
                    if container_data["memory_usage_mb"] > container_data["memory_limit_mb"] * 0.9:
                        container_data["compliance"] = False

                if container_data["memory_limit_mb"] > self.CONTAINER_MEMORY_LIMIT_MB:
                    container_data["compliance"] = False

        except Exception as e:
            logger.warning("Failed to validate container limits", error=str(e))

        return container_data

    def _analyze_resource_trends(
        self,
        memory_analysis: Dict[str, Any],
        cpu_analysis: Dict[str, Any],
        resource_usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze resource usage trends over time.
        
        Returns:
            Resource trend analysis
        """
        current_snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_mb": memory_analysis.get("current_usage_mb", 0),
            "cpu_percent": cpu_analysis.get("average_percent", 0),
            "threads": cpu_analysis.get("thread_count", 0),
            "connections": resource_usage.get("connections", 0),
            "file_descriptors": resource_usage.get("file_descriptors", 0),
        }

        self.resource_history.append(current_snapshot)

        # Keep only last 30 snapshots
        if len(self.resource_history) > 30:
            self.resource_history = self.resource_history[-30:]

        trends = {}
        
        if len(self.resource_history) >= 2:
            # Calculate trends for each metric
            for metric in ["memory_mb", "cpu_percent", "threads", "connections", "file_descriptors"]:
                values = [h.get(metric, 0) for h in self.resource_history]
                current = values[-1]
                previous = values[-2]
                
                # Calculate trend
                if previous > 0:
                    change_percent = ((current - previous) / previous) * 100
                else:
                    change_percent = 0
                
                trend_direction = "increasing" if current > previous else \
                                "decreasing" if current < previous else "stable"
                
                # Check for concerning trends
                is_concerning = False
                if metric == "memory_mb" and len(values) >= 5:
                    # Check if memory is consistently increasing
                    recent_values = values[-5:]
                    is_concerning = all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values)))
                
                trends[metric] = {
                    "current": current,
                    "previous": previous,
                    "change": current - previous,
                    "change_percent": change_percent,
                    "trend": trend_direction,
                    "concerning": is_concerning,
                }
        else:
            # Baseline snapshot
            for metric in current_snapshot:
                if metric != "timestamp":
                    trends[metric] = {
                        "current": current_snapshot[metric],
                        "trend": "baseline",
                        "concerning": False,
                    }

        return trends

    def _check_resource_violations(
        self,
        memory_analysis: Dict[str, Any],
        cpu_analysis: Dict[str, Any],
        resource_usage: Dict[str, Any],
        leaks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for resource usage violations.
        
        Returns:
            List of resource violations
        """
        violations = []

        # Check memory violations
        current_memory = memory_analysis.get("current_usage_mb", 0)
        if current_memory > self.MAX_MEMORY_USAGE_MB:
            violations.append({
                "type": "excessive_memory",
                "value": current_memory,
                "threshold": self.MAX_MEMORY_USAGE_MB,
                "severity": "high",
                "impact": "Risk of OOM errors",
            })

        # Check for memory leaks
        if leaks:
            violations.append({
                "type": "memory_leak",
                "count": len(leaks),
                "severity": "critical",
                "impact": "Memory will exhaust over time",
            })

        # Check CPU violations
        avg_cpu = cpu_analysis.get("average_percent", 0)
        if avg_cpu > self.MAX_CPU_USAGE_PERCENT:
            violations.append({
                "type": "high_cpu_usage",
                "value": avg_cpu,
                "threshold": self.MAX_CPU_USAGE_PERCENT,
                "severity": "medium",
                "impact": "System responsiveness degradation",
            })

        # Check thread count
        thread_count = cpu_analysis.get("thread_count", 0)
        if thread_count > self.MAX_THREAD_COUNT:
            violations.append({
                "type": "excessive_threads",
                "value": thread_count,
                "threshold": self.MAX_THREAD_COUNT,
                "severity": "medium",
                "impact": "Thread management overhead",
            })

        # Check file descriptors
        fd_count = resource_usage.get("file_descriptors", 0)
        if fd_count > self.MAX_FILE_DESCRIPTORS:
            violations.append({
                "type": "excessive_file_descriptors",
                "value": fd_count,
                "threshold": self.MAX_FILE_DESCRIPTORS,
                "severity": "high",
                "impact": "Risk of file descriptor exhaustion",
            })

        # Check connections
        connection_count = resource_usage.get("connections", 0)
        if connection_count > self.MAX_CONNECTION_COUNT:
            violations.append({
                "type": "excessive_connections",
                "value": connection_count,
                "threshold": self.MAX_CONNECTION_COUNT,
                "severity": "medium",
                "impact": "Connection pool exhaustion",
            })

        return violations

    def _generate_resource_evidence(
        self,
        memory_analysis: Dict[str, Any],
        cpu_analysis: Dict[str, Any],
        resource_usage: Dict[str, Any],
        leaks: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate resource usage evidence report.
        
        Returns:
            Evidence dictionary
        """
        evidence = {
            "summary": {
                "memory_usage_mb": memory_analysis.get("current_usage_mb", 0),
                "cpu_usage_percent": cpu_analysis.get("average_percent", 0),
                "thread_count": cpu_analysis.get("thread_count", 0),
                "connection_count": resource_usage.get("connections", 0),
            },
            "memory_hotspots": memory_analysis.get("top_allocations", [])[:5],
            "memory_by_category": memory_analysis.get("by_category", {}),
            "detected_leaks": leaks[:5] if leaks else [],
            "critical_violations": [v for v in violations if v["severity"] == "critical"],
            "optimization_opportunities": self._identify_resource_optimizations(
                memory_analysis, cpu_analysis, violations
            ),
        }

        return evidence

    def _identify_resource_optimizations(
        self,
        memory_analysis: Dict[str, Any],
        cpu_analysis: Dict[str, Any],
        violations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify resource optimization opportunities.
        
        Returns:
            List of optimization suggestions
        """
        optimizations = []

        # Memory optimizations
        if memory_analysis.get("current_usage_mb", 0) > self.MAX_MEMORY_USAGE_MB * 0.7:
            optimizations.append({
                "area": "memory",
                "suggestion": "Implement object pooling for frequently created objects",
                "expected_benefit": "20-30% memory reduction",
            })
            optimizations.append({
                "area": "memory",
                "suggestion": "Use generators instead of lists for large datasets",
                "expected_benefit": "Reduced memory footprint",
            })

        # CPU optimizations
        if cpu_analysis.get("average_percent", 0) > self.MAX_CPU_USAGE_PERCENT * 0.7:
            optimizations.append({
                "area": "cpu",
                "suggestion": "Profile and optimize hot code paths",
                "expected_benefit": "15-25% CPU reduction",
            })
            optimizations.append({
                "area": "cpu",
                "suggestion": "Implement caching for expensive computations",
                "expected_benefit": "Reduced CPU cycles",
            })

        # Thread optimizations
        if cpu_analysis.get("thread_count", 0) > self.MAX_THREAD_COUNT * 0.7:
            optimizations.append({
                "area": "threading",
                "suggestion": "Use asyncio instead of threading where possible",
                "expected_benefit": "Lower thread overhead",
            })

        # Connection optimizations
        if resource_usage.get("connections", 0) > self.MAX_CONNECTION_COUNT * 0.7:
            optimizations.append({
                "area": "connections",
                "suggestion": "Implement connection pooling",
                "expected_benefit": "Reduced connection overhead",
            })

        return optimizations

    def _categorize_module(self, traceback) -> str:
        """Categorize a traceback into a module category.
        
        Args:
            traceback: Traceback object
            
        Returns:
            Module category name
        """
        if not traceback:
            return "unknown"
        
        filename = str(traceback) if traceback else ""
        
        if "genesis" in filename:
            if "engine" in filename:
                return "engine"
            elif "exchange" in filename:
                return "exchange"
            elif "ui" in filename:
                return "ui"
            elif "data" in filename:
                return "data"
            else:
                return "genesis_other"
        elif "site-packages" in filename:
            return "third_party"
        else:
            return "system"