"""Test coverage and test suite validation."""

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TestValidator:
    """Validates test coverage and test suite health."""

    def __init__(self):
        self.critical_paths = [
            "genesis/engine/risk_engine.py",
            "genesis/engine/executor/",
            "genesis/utils/math.py",
            "genesis/core/models.py",
            "genesis/exchange/gateway.py",
        ]
        self.risk_components = [
            "genesis/engine/risk_engine.py",
            "genesis/tilt/detector.py",
            "genesis/tilt/interventions.py",
            "genesis/engine/state_machine.py",
        ]

    async def validate(self) -> dict[str, Any]:
        """Run test suite and coverage validation."""
        try:
            # Run unit tests with coverage
            unit_results = await self._run_unit_tests()

            # Run integration tests
            integration_results = await self._run_integration_tests()

            # Parse coverage data
            coverage_analysis = await self._analyze_coverage()

            # Check critical path coverage
            critical_coverage = self._check_critical_path_coverage(coverage_analysis)

            # Check risk component coverage
            risk_coverage = self._check_risk_coverage(coverage_analysis)

            # Determine overall pass/fail
            passed = (
                unit_results["passed"]
                and integration_results["passed"]
                and critical_coverage["passed"]
                and risk_coverage["passed"]
                and coverage_analysis["overall_coverage"] >= 90
            )

            return {
                "passed": passed,
                "details": {
                    "unit_tests_passed": unit_results["passed"],
                    "unit_tests_total": unit_results["total"],
                    "unit_tests_failed": unit_results["failed"],
                    "unit_coverage": coverage_analysis["overall_coverage"],
                    "integration_tests_passed": integration_results["passed"],
                    "integration_tests_total": integration_results["total"],
                    "integration_pass_rate": integration_results["pass_rate"],
                    "critical_path_coverage": critical_coverage["average"],
                    "critical_paths_100": critical_coverage["paths_at_100"],
                    "risk_component_coverage": risk_coverage["average"],
                    "module_coverage": coverage_analysis["module_coverage"],
                },
                "recommendations": self._generate_recommendations(
                    unit_results,
                    integration_results,
                    coverage_analysis,
                    critical_coverage,
                    risk_coverage,
                ),
            }
        except Exception as e:
            logger.error("Test validation failed", error=str(e))
            return {
                "passed": False,
                "error": str(e),
                "details": {},
            }

    async def _run_unit_tests(self) -> dict[str, Any]:
        """Run unit tests with pytest."""
        try:
            # Run pytest with coverage
            result = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest",
                "tests/unit/",
                "--cov=genesis",
                "--cov-report=json",
                "--cov-report=term",
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=test_results.json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=300
            )

            # Parse test results
            if Path("test_results.json").exists():
                with open("test_results.json") as f:
                    test_data = json.load(f)

                passed = test_data.get("exitcode", 1) == 0
                total = test_data.get("summary", {}).get("total", 0)
                failed = test_data.get("summary", {}).get("failed", 0)

                return {
                    "passed": passed,
                    "total": total,
                    "failed": failed,
                    "pass_rate": ((total - failed) / total * 100) if total > 0 else 0,
                    "output": stdout.decode() if stdout else "",
                }
            else:
                # Fallback parsing from output
                stdout_text = stdout.decode() if stdout else ""
                lines = stdout_text.split("\n")
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Parse pytest summary line
                        parts = line.split()
                        passed_count = 0
                        failed_count = 0
                        for i, part in enumerate(parts):
                            if part == "passed":
                                passed_count = int(parts[i-1])
                            if part == "failed":
                                failed_count = int(parts[i-1])

                        total = passed_count + failed_count
                        return {
                            "passed": failed_count == 0,
                            "total": total,
                            "failed": failed_count,
                            "pass_rate": (passed_count / total * 100) if total > 0 else 0,
                            "output": stdout_text,
                        }

                # Could not parse results
                return {
                    "passed": result.returncode == 0,
                    "total": 0,
                    "failed": 0,
                    "pass_rate": 0,
                    "output": stdout_text,
                }

        except TimeoutError:
            return {
                "passed": False,
                "error": "Unit tests timed out after 5 minutes",
                "total": 0,
                "failed": 0,
                "pass_rate": 0,
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "total": 0,
                "failed": 0,
                "pass_rate": 0,
            }

    async def _run_integration_tests(self) -> dict[str, Any]:
        """Run integration tests."""
        try:
            result = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest",
                "tests/integration/",
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=integration_results.json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=600
            )

            # Parse test results
            if Path("integration_results.json").exists():
                with open("integration_results.json") as f:
                    test_data = json.load(f)

                passed = test_data.get("exitcode", 1) == 0
                total = test_data.get("summary", {}).get("total", 0)
                failed = test_data.get("summary", {}).get("failed", 0)

                return {
                    "passed": passed and failed == 0,
                    "total": total,
                    "failed": failed,
                    "pass_rate": ((total - failed) / total * 100) if total > 0 else 0,
                    "output": stdout.decode() if stdout else "",
                }
            else:
                # Fallback to return code
                return {
                    "passed": result.returncode == 0,
                    "total": 0,
                    "failed": 0,
                    "pass_rate": 100 if result.returncode == 0 else 0,
                    "output": stdout.decode() if stdout else "",
                }

        except TimeoutError:
            return {
                "passed": False,
                "error": "Integration tests timed out after 10 minutes",
                "total": 0,
                "failed": 0,
                "pass_rate": 0,
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "total": 0,
                "failed": 0,
                "pass_rate": 0,
            }

    async def _analyze_coverage(self) -> dict[str, Any]:
        """Analyze coverage data from coverage.json."""
        try:
            # Read coverage data
            coverage_file = Path("coverage.json")
            if not coverage_file.exists():
                # Try to generate it
                result = await asyncio.create_subprocess_exec(
                    "coverage", "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(result.communicate(), timeout=30)

            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                # Calculate overall coverage
                totals = coverage_data.get("totals", {})
                overall_coverage = totals.get("percent_covered", 0)

                # Get per-module coverage
                module_coverage = {}
                files = coverage_data.get("files", {})
                for filepath, file_data in files.items():
                    # Normalize path for comparison
                    normalized_path = filepath.replace("\\", "/")
                    if "genesis/" in normalized_path:
                        module_name = normalized_path.split("genesis/")[1]
                        coverage_percent = file_data.get("summary", {}).get("percent_covered", 0)
                        module_coverage[module_name] = coverage_percent

                return {
                    "overall_coverage": overall_coverage,
                    "module_coverage": module_coverage,
                    "lines_covered": totals.get("covered_lines", 0),
                    "lines_total": totals.get("num_statements", 0),
                }
            else:
                # No coverage data available
                return {
                    "overall_coverage": 0,
                    "module_coverage": {},
                    "lines_covered": 0,
                    "lines_total": 0,
                }

        except Exception as e:
            logger.error("Failed to analyze coverage", error=str(e))
            return {
                "overall_coverage": 0,
                "module_coverage": {},
                "error": str(e),
            }

    def _check_critical_path_coverage(self, coverage_analysis: dict) -> dict[str, Any]:
        """Check coverage for critical money paths."""
        module_coverage = coverage_analysis.get("module_coverage", {})

        critical_coverages = []
        paths_at_100 = 0

        for path in self.critical_paths:
            # Handle both file and directory paths
            normalized_path = path.replace("genesis/", "").replace("\\", "/")

            if normalized_path.endswith("/"):
                # Directory - check all files in it
                matching_modules = [
                    (module, cov)
                    for module, cov in module_coverage.items()
                    if module.startswith(normalized_path)
                ]
                if matching_modules:
                    avg_coverage = sum(cov for _, cov in matching_modules) / len(matching_modules)
                    critical_coverages.append(avg_coverage)
                    if avg_coverage == 100:
                        paths_at_100 += 1
            else:
                # Single file
                if normalized_path in module_coverage:
                    cov = module_coverage[normalized_path]
                    critical_coverages.append(cov)
                    if cov == 100:
                        paths_at_100 += 1

        if critical_coverages:
            average = sum(critical_coverages) / len(critical_coverages)
            passed = average == 100  # Critical paths must have 100% coverage
        else:
            average = 0
            passed = False

        return {
            "passed": passed,
            "average": average,
            "paths_at_100": paths_at_100,
            "total_paths": len(self.critical_paths),
            "details": dict(zip(self.critical_paths, critical_coverages, strict=False)) if critical_coverages else {},
        }

    def _check_risk_coverage(self, coverage_analysis: dict) -> dict[str, Any]:
        """Check coverage for risk and tilt components."""
        module_coverage = coverage_analysis.get("module_coverage", {})

        risk_coverages = []

        for component in self.risk_components:
            normalized_path = component.replace("genesis/", "").replace("\\", "/")
            if normalized_path in module_coverage:
                risk_coverages.append(module_coverage[normalized_path])

        if risk_coverages:
            average = sum(risk_coverages) / len(risk_coverages)
            passed = average >= 90  # Risk components need 90% coverage
        else:
            average = 0
            passed = False

        return {
            "passed": passed,
            "average": average,
            "components_above_90": sum(1 for c in risk_coverages if c >= 90),
            "total_components": len(self.risk_components),
            "details": dict(zip(self.risk_components, risk_coverages, strict=False)) if risk_coverages else {},
        }

    def _generate_recommendations(
        self,
        unit_results: dict,
        integration_results: dict,
        coverage_analysis: dict,
        critical_coverage: dict,
        risk_coverage: dict,
    ) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Unit test recommendations
        if not unit_results["passed"]:
            recommendations.append(
                f"Fix {unit_results['failed']} failing unit tests"
            )

        # Integration test recommendations
        if not integration_results["passed"]:
            recommendations.append(
                f"Fix {integration_results['failed']} failing integration tests"
            )

        # Coverage recommendations
        if coverage_analysis["overall_coverage"] < 90:
            recommendations.append(
                f"Increase overall coverage from {coverage_analysis['overall_coverage']:.1f}% to 90%"
            )

        # Critical path recommendations
        if not critical_coverage["passed"]:
            recommendations.append(
                f"Achieve 100% coverage for all critical money paths "
                f"(currently {critical_coverage['paths_at_100']}/{critical_coverage['total_paths']})"
            )

        # Risk component recommendations
        if not risk_coverage["passed"]:
            recommendations.append(
                f"Increase risk component coverage to 90% "
                f"(currently {risk_coverage['average']:.1f}%)"
            )

        # Module-specific recommendations
        module_coverage = coverage_analysis.get("module_coverage", {})
        low_coverage_modules = [
            (module, cov)
            for module, cov in module_coverage.items()
            if cov < 70
        ]
        if low_coverage_modules:
            worst_modules = sorted(low_coverage_modules, key=lambda x: x[1])[:3]
            for module, cov in worst_modules:
                recommendations.append(
                    f"Improve coverage for {module} (currently {cov:.1f}%)"
                )

        return recommendations
