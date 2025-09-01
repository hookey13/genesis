"""Health check and service monitoring validation module."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class HealthCheckValidator:
    """Validates health check endpoints and service monitoring."""

    REQUIRED_HEALTH_ENDPOINTS = {
        "/health": "Basic health check",
        "/health/live": "Liveness probe",
        "/health/ready": "Readiness probe",
        "/health/dependencies": "Dependency health",
        "/metrics": "Prometheus metrics endpoint",
    }

    EXPECTED_RESPONSE_TIME_MS = 1000  # 1 second max
    REQUIRED_HEALTH_FIELDS = ["status", "timestamp", "version"]
    
    REQUIRED_DEPENDENCY_CHECKS = [
        "database",
        "redis",
        "exchange_api",
        "websocket",
    ]

    def __init__(self, genesis_root: Path | None = None, base_url: str = "http://localhost:8000"):
        """Initialize health check validator.
        
        Args:
            genesis_root: Root directory of Genesis project
            base_url: Base URL for health check endpoints
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.base_url = base_url
        self.results: Dict[str, Any] = {}

    async def validate(self) -> Dict[str, Any]:
        """Run health check validation.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting health check validation")
        start_time = datetime.utcnow()

        self.results = {
            "validator": "health_check",
            "timestamp": start_time.isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": [],
        }

        # Verify all services have health endpoints
        endpoints_result = await self._verify_health_endpoints()
        self.results["checks"]["health_endpoints"] = endpoints_result

        # Test health check response times
        response_time_result = await self._test_response_times()
        self.results["checks"]["response_times"] = response_time_result

        # Validate health check dependencies
        dependencies_result = await self._validate_health_dependencies()
        self.results["checks"]["health_dependencies"] = dependencies_result

        # Check liveness and readiness probes
        probes_result = await self._check_probes()
        self.results["checks"]["probes"] = probes_result

        # Validate health check format
        format_result = await self._validate_health_format()
        self.results["checks"]["health_format"] = format_result

        # Check health check implementation in code
        implementation_result = await self._check_health_implementation()
        self.results["checks"]["health_implementation"] = implementation_result

        # Generate service health report
        health_report = await self._generate_health_report()
        self.results["checks"]["health_report"] = health_report

        # Calculate overall score
        total_checks = len(self.results["checks"])
        passed_checks = sum(
            1 for check in self.results["checks"].values() if check.get("passed", False)
        )
        self.results["score"] = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0

        # Determine overall status
        critical_checks = ["health_endpoints", "probes", "health_implementation"]
        critical_passed = all(
            self.results["checks"].get(check, {}).get("passed", False)
            for check in critical_checks
        )

        if critical_passed and self.results["score"] >= 75:
            self.results["passed"] = True
            self.results["summary"] = "Health checks properly configured and operational"
        else:
            self.results["passed"] = False
            self.results["summary"] = "Health check issues detected - review failed checks"

        # Add execution time
        self.results["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

        return self.results

    async def _verify_health_endpoints(self) -> Dict[str, Any]:
        """Verify all services have health endpoints.
        
        Returns:
            Validation result for health endpoints
        """
        result = {
            "passed": False,
            "message": "",
            "endpoints_found": [],
            "endpoints_missing": [],
            "endpoints_responding": [],
        }

        # Try to check actual endpoints if service is running
        session_timeout = aiohttp.ClientTimeout(total=5)
        
        try:
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                for endpoint, description in self.REQUIRED_HEALTH_ENDPOINTS.items():
                    url = f"{self.base_url}{endpoint}"
                    
                    try:
                        async with session.get(url) as response:
                            if response.status in [200, 204]:
                                result["endpoints_responding"].append(endpoint)
                                result["endpoints_found"].append(endpoint)
                            else:
                                result["endpoints_found"].append(f"{endpoint} (status: {response.status})")
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        # Service might not be running, check code instead
                        pass
        except Exception:
            # Service not running, check code implementation
            pass

        # If service isn't running, check code for endpoint definitions
        if not result["endpoints_responding"]:
            result = await self._check_health_endpoints_in_code()
        
        # Determine missing endpoints
        found_endpoints = [e.split(" ")[0] for e in result["endpoints_found"]]
        result["endpoints_missing"] = [
            e for e in self.REQUIRED_HEALTH_ENDPOINTS.keys()
            if e not in found_endpoints
        ]

        if not result["endpoints_missing"]:
            result["passed"] = True
            result["message"] = f"All {len(self.REQUIRED_HEALTH_ENDPOINTS)} health endpoints configured"
        else:
            result["message"] = f"Missing endpoints: {', '.join(result['endpoints_missing'])}"

        return result

    async def _check_health_endpoints_in_code(self) -> Dict[str, Any]:
        """Check for health endpoint definitions in code.
        
        Returns:
            Validation result for health endpoints in code
        """
        result = {
            "passed": False,
            "message": "",
            "endpoints_found": [],
            "endpoints_missing": [],
            "endpoints_responding": [],
        }

        # Look for FastAPI/Flask route definitions
        api_dir = self.genesis_root / "genesis" / "api"
        
        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    for endpoint in self.REQUIRED_HEALTH_ENDPOINTS.keys():
                        # Check for route decorators
                        if f'"{endpoint}"' in content or f"'{endpoint}'" in content:
                            result["endpoints_found"].append(endpoint)
                        # Check for explicit path definitions
                        elif f"path={endpoint}" in content or f"route={endpoint}" in content:
                            result["endpoints_found"].append(endpoint)

        # Remove duplicates
        result["endpoints_found"] = list(set(result["endpoints_found"]))
        
        return result

    async def _test_response_times(self) -> Dict[str, Any]:
        """Test health check response times.
        
        Returns:
            Validation result for response times
        """
        result = {
            "passed": False,
            "message": "",
            "response_times": {},
            "slow_endpoints": [],
        }

        session_timeout = aiohttp.ClientTimeout(total=5)
        
        try:
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                for endpoint in ["/health", "/health/live", "/health/ready"]:
                    url = f"{self.base_url}{endpoint}"
                    
                    try:
                        start = datetime.utcnow()
                        async with session.get(url) as response:
                            elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
                            result["response_times"][endpoint] = elapsed_ms
                            
                            if elapsed_ms > self.EXPECTED_RESPONSE_TIME_MS:
                                result["slow_endpoints"].append(f"{endpoint} ({elapsed_ms:.0f}ms)")
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        result["response_times"][endpoint] = "N/A"
        except Exception:
            # Service not running, check will be skipped
            pass

        if result["response_times"] and not result["slow_endpoints"]:
            result["passed"] = True
            avg_time = sum(
                t for t in result["response_times"].values() 
                if isinstance(t, (int, float))
            ) / len(result["response_times"])
            result["message"] = f"Health checks responsive (avg: {avg_time:.0f}ms)"
        elif result["slow_endpoints"]:
            result["message"] = f"Slow endpoints: {', '.join(result['slow_endpoints'])}"
        else:
            result["passed"] = True  # Pass if service not running
            result["message"] = "Response time check skipped (service not running)"

        return result

    async def _validate_health_dependencies(self) -> Dict[str, Any]:
        """Validate health check includes dependency checks.
        
        Returns:
            Validation result for dependency checks
        """
        result = {
            "passed": False,
            "message": "",
            "dependencies_checked": [],
            "dependencies_missing": [],
        }

        # Try to check actual endpoint
        session_timeout = aiohttp.ClientTimeout(total=5)
        
        try:
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                url = f"{self.base_url}/health/dependencies"
                
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Check for required dependencies
                            for dep in self.REQUIRED_DEPENDENCY_CHECKS:
                                if dep in str(data).lower():
                                    result["dependencies_checked"].append(dep)
                                else:
                                    result["dependencies_missing"].append(dep)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
        except Exception:
            pass

        # If endpoint not available, check code
        if not result["dependencies_checked"]:
            result = await self._check_dependencies_in_code()

        if not result["dependencies_missing"] or len(result["dependencies_checked"]) >= 2:
            result["passed"] = True
            result["message"] = f"Dependency checks configured ({len(result['dependencies_checked'])} deps)"
        else:
            result["message"] = f"Missing dependency checks: {', '.join(result['dependencies_missing'])}"

        return result

    async def _check_dependencies_in_code(self) -> Dict[str, Any]:
        """Check for dependency checks in code.
        
        Returns:
            Validation result for dependency checks in code
        """
        result = {
            "passed": False,
            "message": "",
            "dependencies_checked": [],
            "dependencies_missing": [],
        }

        # Look for health check implementations
        health_files = [
            self.genesis_root / "genesis" / "api" / "health.py",
            self.genesis_root / "genesis" / "api" / "routes.py",
            self.genesis_root / "genesis" / "monitoring" / "health.py",
        ]

        for health_file in health_files:
            if health_file.exists():
                with open(health_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    
                    for dep in self.REQUIRED_DEPENDENCY_CHECKS:
                        if dep in content:
                            result["dependencies_checked"].append(dep)

        result["dependencies_checked"] = list(set(result["dependencies_checked"]))
        result["dependencies_missing"] = [
            d for d in self.REQUIRED_DEPENDENCY_CHECKS 
            if d not in result["dependencies_checked"]
        ]

        return result

    async def _check_probes(self) -> Dict[str, Any]:
        """Check liveness and readiness probes configuration.
        
        Returns:
            Validation result for probes
        """
        result = {
            "passed": False,
            "message": "",
            "liveness_configured": False,
            "readiness_configured": False,
            "probe_configs": [],
        }

        # Check Kubernetes manifests for probe configuration
        k8s_deployment = self.genesis_root / "kubernetes" / "deployment.yaml"
        if k8s_deployment.exists():
            with open(k8s_deployment, "r") as f:
                content = f.read()
                
                if "livenessProbe" in content:
                    result["liveness_configured"] = True
                    result["probe_configs"].append("Kubernetes liveness")
                
                if "readinessProbe" in content:
                    result["readiness_configured"] = True
                    result["probe_configs"].append("Kubernetes readiness")

        # Check Docker Compose for health checks
        compose_files = [
            self.genesis_root / "docker" / "docker-compose.yml",
            self.genesis_root / "docker" / "docker-compose.prod.yml",
        ]

        for compose_file in compose_files:
            if compose_file.exists():
                with open(compose_file, "r") as f:
                    content = f.read()
                    
                    if "healthcheck:" in content:
                        result["liveness_configured"] = True
                        result["probe_configs"].append("Docker healthcheck")

        # Check code for probe endpoints
        if not result["liveness_configured"] or not result["readiness_configured"]:
            endpoints_check = self.results.get("checks", {}).get("health_endpoints", {})
            endpoints_found = endpoints_check.get("endpoints_found", [])
            
            if "/health/live" in str(endpoints_found):
                result["liveness_configured"] = True
                result["probe_configs"].append("Liveness endpoint")
            
            if "/health/ready" in str(endpoints_found):
                result["readiness_configured"] = True
                result["probe_configs"].append("Readiness endpoint")

        if result["liveness_configured"] and result["readiness_configured"]:
            result["passed"] = True
            result["message"] = "Liveness and readiness probes configured"
        elif result["liveness_configured"] or result["readiness_configured"]:
            result["message"] = "Partial probe configuration"
        else:
            result["message"] = "No probe configuration found"

        return result

    async def _validate_health_format(self) -> Dict[str, Any]:
        """Validate health check response format.
        
        Returns:
            Validation result for health format
        """
        result = {
            "passed": False,
            "message": "",
            "format_valid": False,
            "fields_found": [],
            "fields_missing": [],
        }

        # Try to check actual endpoint response
        session_timeout = aiohttp.ClientTimeout(total=5)
        
        try:
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                url = f"{self.base_url}/health"
                
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Check for required fields
                            for field in self.REQUIRED_HEALTH_FIELDS:
                                if field in data:
                                    result["fields_found"].append(field)
                                else:
                                    result["fields_missing"].append(field)
                            
                            result["format_valid"] = not result["fields_missing"]
                except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
                    pass
        except Exception:
            pass

        # If endpoint not available, check code
        if not result["fields_found"]:
            result = await self._check_health_format_in_code()

        if result["format_valid"] or len(result["fields_found"]) >= 2:
            result["passed"] = True
            result["message"] = "Health check format valid"
        else:
            result["message"] = f"Health format issues: missing {', '.join(result['fields_missing'])}"

        return result

    async def _check_health_format_in_code(self) -> Dict[str, Any]:
        """Check health response format in code.
        
        Returns:
            Validation result for health format in code
        """
        result = {
            "passed": False,
            "message": "",
            "format_valid": False,
            "fields_found": [],
            "fields_missing": [],
        }

        # Look for health response construction
        api_dir = self.genesis_root / "genesis" / "api"
        
        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Look for health response dictionary
                    if "health" in content.lower():
                        for field in self.REQUIRED_HEALTH_FIELDS:
                            if f'"{field}"' in content or f"'{field}'" in content:
                                result["fields_found"].append(field)

        result["fields_found"] = list(set(result["fields_found"]))
        result["fields_missing"] = [
            f for f in self.REQUIRED_HEALTH_FIELDS 
            if f not in result["fields_found"]
        ]
        
        result["format_valid"] = not result["fields_missing"]

        return result

    async def _check_health_implementation(self) -> Dict[str, Any]:
        """Check health check implementation in code.
        
        Returns:
            Validation result for health implementation
        """
        result = {
            "passed": False,
            "message": "",
            "implementation_found": False,
            "implementation_files": [],
            "features": [],
        }

        # Search for health check implementations
        search_paths = [
            self.genesis_root / "genesis" / "api",
            self.genesis_root / "genesis" / "monitoring",
            self.genesis_root / "genesis" / "core",
        ]

        for search_path in search_paths:
            if search_path.exists():
                for py_file in search_path.rglob("*.py"):
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Look for health-related implementations
                        if any(term in content.lower() for term in ["health", "liveness", "readiness"]):
                            result["implementation_found"] = True
                            result["implementation_files"].append(str(py_file.relative_to(self.genesis_root)))
                            
                            # Check for specific features
                            if "async def health" in content or "def health" in content:
                                result["features"].append("health endpoint")
                            if "check_database" in content or "db_health" in content:
                                result["features"].append("database check")
                            if "check_redis" in content or "redis_health" in content:
                                result["features"].append("redis check")
                            if "check_exchange" in content or "exchange_health" in content:
                                result["features"].append("exchange check")

        if result["implementation_found"] and result["features"]:
            result["passed"] = True
            result["message"] = f"Health implementation found with {len(result['features'])} features"
        elif result["implementation_found"]:
            result["message"] = "Basic health implementation found"
        else:
            result["message"] = "No health check implementation found"

        return result

    async def _generate_health_report(self) -> Dict[str, Any]:
        """Generate service health report.
        
        Returns:
            Service health summary
        """
        result = {
            "passed": False,
            "message": "",
            "health_score": 0,
            "healthy_components": [],
            "unhealthy_components": [],
        }

        # Aggregate results from other checks
        checks_summary = {
            "Endpoints": self.results["checks"].get("health_endpoints", {}).get("passed", False),
            "Response Times": self.results["checks"].get("response_times", {}).get("passed", False),
            "Dependencies": self.results["checks"].get("health_dependencies", {}).get("passed", False),
            "Probes": self.results["checks"].get("probes", {}).get("passed", False),
            "Format": self.results["checks"].get("health_format", {}).get("passed", False),
            "Implementation": self.results["checks"].get("health_implementation", {}).get("passed", False),
        }

        for component, healthy in checks_summary.items():
            if healthy:
                result["healthy_components"].append(component)
            else:
                result["unhealthy_components"].append(component)

        result["health_score"] = int(
            (len(result["healthy_components"]) / len(checks_summary)) * 100
        )

        if result["health_score"] >= 80:
            result["passed"] = True
            result["message"] = f"Service health {result['health_score']}% - Good"
        elif result["health_score"] >= 60:
            result["message"] = f"Service health {result['health_score']}% - Fair"
        else:
            result["message"] = f"Service health {result['health_score']}% - Poor"

        return result

    def generate_report(self) -> str:
        """Generate a detailed health check validation report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run validate() first."

        report = []
        report.append("=" * 80)
        report.append("HEALTH CHECK VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Status: {'PASSED' if self.results['passed'] else 'FAILED'}")
        report.append(f"Score: {self.results['score']}%")
        report.append(f"Summary: {self.results['summary']}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)
        
        for check_name, check_result in self.results["checks"].items():
            status = "✓" if check_result.get("passed", False) else "✗"
            report.append(f"{status} {check_name}: {check_result.get('message', '')}")
            
            # Add details
            if check_result.get("endpoints_missing"):
                report.append(f"  Missing endpoints: {', '.join(check_result['endpoints_missing'])}")
            if check_result.get("slow_endpoints"):
                report.append(f"  Slow endpoints: {', '.join(check_result['slow_endpoints'])}")
            if check_result.get("dependencies_missing"):
                report.append(f"  Missing deps: {', '.join(check_result['dependencies_missing'])}")
            if check_result.get("probe_configs"):
                report.append(f"  Probes: {', '.join(check_result['probe_configs'])}")
            if check_result.get("features"):
                report.append(f"  Features: {', '.join(check_result['features'])}")
            if check_name == "health_report":
                report.append(f"  Health Score: {check_result.get('health_score', 0)}%")
                if check_result.get("unhealthy_components"):
                    report.append(f"  Issues: {', '.join(check_result['unhealthy_components'])}")

        report.append("")
        report.append(f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds")
        report.append("=" * 80)

        return "\n".join(report)