"""Test coverage and test suite validation."""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class TestValidator:
    """Validates test coverage and test suite health with path-specific thresholds."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize test validator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()

        # Define coverage thresholds by path type
        self.coverage_thresholds = {
            "money_paths": {
                "threshold": 100,  # 100% coverage required
                "paths": [
                    "genesis/engine/risk_engine.py",
                    "genesis/engine/executor/",
                    "genesis/utils/math.py",
                    "genesis/core/models.py",
                    "genesis/exchange/gateway.py",
                ]
            },
            "risk_components": {
                "threshold": 90,  # 90% coverage required
                "paths": [
                    "genesis/engine/risk_engine.py",
                    "genesis/tilt/detector.py",
                    "genesis/tilt/interventions.py",
                    "genesis/engine/state_machine.py",
                    "genesis/engine/circuit_breaker.py",
                ]
            },
            "core_modules": {
                "threshold": 85,  # 85% coverage required
                "paths": [
                    "genesis/core/",
                    "genesis/engine/",
                    "genesis/exchange/",
                    "genesis/data/",
                ]
            },
            "ui_modules": {
                "threshold": 70,  # 70% coverage for UI
                "paths": [
                    "genesis/ui/",
                ]
            },
            "utility_modules": {
                "threshold": 80,  # 80% for utilities
                "paths": [
                    "genesis/utils/",
                    "genesis/analytics/",
                ]
            }
        }

        # Store coverage history for trend tracking
        self.coverage_history: list[dict[str, Any]] = []

    async def validate(self) -> dict[str, Any]:
        """Run test suite and coverage validation with path-specific thresholds."""
        logger.info("Starting test coverage validation")
        start_time = datetime.utcnow()

        try:
            # Run unit tests with coverage
            unit_results = await self._run_unit_tests()

            # Run integration tests
            integration_results = await self._run_integration_tests()

            # Parse coverage data (both JSON and XML)
            coverage_analysis = await self._analyze_coverage()

            # Check path-specific coverage thresholds
            threshold_results = self._check_path_thresholds(coverage_analysis)

            # Track coverage trends
            self._track_coverage_trend(coverage_analysis)

            # Generate detailed module breakdown
            module_breakdown = self._generate_module_breakdown(coverage_analysis)

            # Determine overall pass/fail based on all thresholds
            passed = (
                unit_results["passed"]
                and integration_results["passed"]
                and all(result["passed"] for result in threshold_results.values())
            )

            # Calculate overall score
            score = self._calculate_overall_score(
                unit_results,
                integration_results,
                threshold_results,
                coverage_analysis
            )

            return {
                "validator": "test_coverage",
                "timestamp": start_time.isoformat(),
                "passed": passed,
                "score": score,
                "checks": {
                    "unit_tests": {
                        "passed": unit_results["passed"],
                        "details": [
                            f"Total: {unit_results['total']}",
                            f"Failed: {unit_results['failed']}",
                            f"Pass rate: {unit_results['pass_rate']:.1f}%"
                        ]
                    },
                    "integration_tests": {
                        "passed": integration_results["passed"],
                        "details": [
                            f"Total: {integration_results['total']}",
                            f"Failed: {integration_results['failed']}",
                            f"Pass rate: {integration_results['pass_rate']:.1f}%"
                        ]
                    },
                    **{f"{category}_coverage": result for category, result in threshold_results.items()}
                },
                "details": {
                    "overall_coverage": coverage_analysis["overall_coverage"],
                    "lines_covered": coverage_analysis["lines_covered"],
                    "lines_total": coverage_analysis["lines_total"],
                    "module_breakdown": module_breakdown,
                    "coverage_trend": self._get_coverage_trend(),
                },
                "summary": self._generate_summary(passed, score, threshold_results),
                "recommendations": self._generate_recommendations(
                    unit_results,
                    integration_results,
                    coverage_analysis,
                    threshold_results,
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

    def _check_path_thresholds(self, coverage_analysis: dict[str, Any]) -> dict[str, Any]:
        """Check coverage against path-specific thresholds."""
        module_coverage = coverage_analysis.get("module_coverage", {})
        results = {}

        for category, config in self.coverage_thresholds.items():
            threshold = config["threshold"]
            paths = config["paths"]
            path_coverages = {}

            for path in paths:
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
                        path_coverages[path] = avg_coverage
                else:
                    # Single file
                    if normalized_path in module_coverage:
                        path_coverages[path] = module_coverage[normalized_path]
                    else:
                        # File might not exist or have no coverage
                        path_coverages[path] = 0

            # Calculate category results
            if path_coverages:
                average = sum(path_coverages.values()) / len(path_coverages)
                passed = all(cov >= threshold for cov in path_coverages.values())
                paths_meeting_threshold = sum(1 for cov in path_coverages.values() if cov >= threshold)
            else:
                average = 0
                passed = False
                paths_meeting_threshold = 0

            results[category] = {
                "passed": passed,
                "threshold": threshold,
                "average": average,
                "paths_meeting_threshold": paths_meeting_threshold,
                "total_paths": len(paths),
                "details": [
                    f"{path}: {cov:.1f}% {'✓' if cov >= threshold else '✗'}"
                    for path, cov in sorted(path_coverages.items())
                ]
            }

        return results

    async def _analyze_coverage_xml(self) -> dict[str, Any]:
        """Parse coverage data from XML report for more detailed analysis."""
        try:
            coverage_xml = self.genesis_root / "coverage.xml"

            # Generate XML report if it doesn't exist
            if not coverage_xml.exists():
                result = await asyncio.create_subprocess_exec(
                    "coverage", "xml",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(result.communicate(), timeout=30)

            if coverage_xml.exists():
                tree = ET.parse(coverage_xml)
                root = tree.getroot()

                # Extract detailed metrics
                packages = root.findall(".//package")
                package_coverage = {}

                for package in packages:
                    package_name = package.get("name")
                    lines_covered = int(package.get("line-rate", 0) * 100)
                    branch_coverage = int(package.get("branch-rate", 0) * 100)
                    complexity = package.get("complexity", "0")

                    package_coverage[package_name] = {
                        "line_coverage": lines_covered,
                        "branch_coverage": branch_coverage,
                        "complexity": complexity
                    }

                return package_coverage

            return {}

        except Exception as e:
            logger.error("Failed to parse XML coverage", error=str(e))
            return {}

    def _track_coverage_trend(self, coverage_analysis: dict[str, Any]) -> None:
        """Track coverage trends over time."""
        trend_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_coverage": coverage_analysis["overall_coverage"],
            "lines_covered": coverage_analysis["lines_covered"],
            "lines_total": coverage_analysis["lines_total"]
        }

        self.coverage_history.append(trend_data)

        # Keep only last 30 runs
        if len(self.coverage_history) > 30:
            self.coverage_history = self.coverage_history[-30:]

    def _get_coverage_trend(self) -> dict[str, Any]:
        """Get coverage trend analysis."""
        if len(self.coverage_history) < 2:
            return {"trend": "insufficient_data", "history": self.coverage_history}

        recent = self.coverage_history[-1]["overall_coverage"]
        previous = self.coverage_history[-2]["overall_coverage"]

        if recent > previous:
            trend = "improving"
        elif recent < previous:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "current": recent,
            "previous": previous,
            "change": recent - previous,
            "history_length": len(self.coverage_history)
        }

    def _generate_module_breakdown(self, coverage_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate detailed module coverage breakdown."""
        module_coverage = coverage_analysis.get("module_coverage", {})

        # Group modules by category
        breakdown = {
            "core": [],
            "engine": [],
            "exchange": [],
            "tilt": [],
            "ui": [],
            "utils": [],
            "analytics": [],
            "validation": [],
            "other": []
        }

        for module, coverage in module_coverage.items():
            category = "other"
            for cat in breakdown:
                if cat in module.lower():
                    category = cat
                    break

            breakdown[category].append({
                "module": module,
                "coverage": coverage,
                "status": "✓" if coverage >= 80 else "⚠" if coverage >= 60 else "✗"
            })

        # Sort each category by coverage
        for category in breakdown:
            breakdown[category].sort(key=lambda x: x["coverage"], reverse=True)

        return breakdown

    def _calculate_overall_score(
        self,
        unit_results: dict[str, Any],
        integration_results: dict[str, Any],
        threshold_results: dict[str, Any],
        coverage_analysis: dict[str, Any]
    ) -> float:
        """Calculate overall test validation score."""
        scores = []
        weights = []

        # Unit test score (weight: 25%)
        if unit_results["total"] > 0:
            scores.append(unit_results["pass_rate"])
            weights.append(0.25)

        # Integration test score (weight: 20%)
        if integration_results["total"] > 0:
            scores.append(integration_results["pass_rate"])
            weights.append(0.20)

        # Coverage threshold scores (weight: 40%)
        for category, result in threshold_results.items():
            if category == "money_paths":
                # Money paths are most critical
                scores.append(result["average"])
                weights.append(0.20)
            elif category == "risk_components":
                scores.append(result["average"])
                weights.append(0.10)
            else:
                scores.append(result["average"])
                weights.append(0.05)

        # Overall coverage score (weight: 15%)
        scores.append(coverage_analysis["overall_coverage"])
        weights.append(0.15)

        # Normalize weights if they don't sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Calculate weighted score
        if scores and weights:
            return sum(s * w for s, w in zip(scores, weights, strict=False))
        return 0

    def _generate_summary(
        self,
        passed: bool,
        score: float,
        threshold_results: dict[str, Any]
    ) -> str:
        """Generate summary message."""
        if passed:
            return f"Test validation passed with score {score:.1f}%"
        else:
            failed_categories = [
                cat for cat, result in threshold_results.items()
                if not result["passed"]
            ]
            if failed_categories:
                return f"Test validation failed. Categories not meeting thresholds: {', '.join(failed_categories)}"
            return f"Test validation failed with score {score:.1f}%"

    def _generate_recommendations(
        self,
        unit_results: dict[str, Any],
        integration_results: dict[str, Any],
        coverage_analysis: dict[str, Any],
        threshold_results: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Unit test recommendations
        if not unit_results["passed"]:
            recommendations.append(
                f"🔴 Fix {unit_results['failed']} failing unit tests"
            )

        # Integration test recommendations
        if not integration_results["passed"]:
            recommendations.append(
                f"🔴 Fix {integration_results['failed']} failing integration tests"
            )

        # Threshold-specific recommendations
        for category, result in threshold_results.items():
            if not result["passed"]:
                below_threshold = result["total_paths"] - result["paths_meeting_threshold"]
                recommendations.append(
                    f"⚠️ {category}: {below_threshold} paths below {result['threshold']}% threshold "
                    f"(current avg: {result['average']:.1f}%)"
                )

                # Add specific path recommendations for money paths
                if category == "money_paths" and "details" in result:
                    for detail in result["details"][:3]:  # Show top 3
                        if "✗" in detail:
                            recommendations.append(f"  - {detail}")

        # Overall coverage recommendation
        if coverage_analysis["overall_coverage"] < 80:
            recommendations.append(
                f"📊 Increase overall coverage from {coverage_analysis['overall_coverage']:.1f}% to at least 80%"
            )

        # Module-specific recommendations
        module_coverage = coverage_analysis.get("module_coverage", {})
        critical_low_coverage = [
            (module, cov)
            for module, cov in module_coverage.items()
            if ("risk" in module or "executor" in module or "math" in module) and cov < 90
        ]

        if critical_low_coverage:
            worst_critical = sorted(critical_low_coverage, key=lambda x: x[1])[:3]
            for module, cov in worst_critical:
                recommendations.append(
                    f"🚨 Critical module needs attention: {module} ({cov:.1f}% coverage)"
                )

        # Add positive feedback if doing well
        if not recommendations:
            recommendations.append("✅ Excellent test coverage! All thresholds met.")

        return recommendations
