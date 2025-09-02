"""Test coverage validator with path-specific thresholds for money and risk paths."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class TestCoverageValidator:
    """Validates test coverage meets path-specific requirements."""

    MONEY_PATH_THRESHOLD = 100.0  # 100% for money paths
    RISK_PATH_THRESHOLD = 90.0    # 90% for risk/tilt
    CORE_PATH_THRESHOLD = 85.0    # 85% for core modules
    UI_PATH_THRESHOLD = 70.0      # 70% for UI/analytics
    DEFAULT_THRESHOLD = 80.0      # 80% default

    MONEY_PATHS = [
        "genesis/engine/risk_engine.py",
        "genesis/engine/executor/",
        "genesis/utils/math.py",
        "genesis/exchange/gateway.py",
        "genesis/core/models.py",
        "genesis/data/repository.py",
        "genesis/engine/position_manager.py",  # Critical for position sizing
        "genesis/engine/pnl_calculator.py",     # Critical for P&L calculations
        "genesis/core/accounting.py",           # Critical for financial accounting
    ]

    RISK_PATHS = [
        "genesis/tilt/detector.py",
        "genesis/tilt/interventions.py",
        "genesis/engine/state_machine.py",
        "genesis/core/circuit_breaker.py",
        "genesis/core/recovery_manager.py",
    ]

    CORE_PATHS = [
        "genesis/core/",
        "genesis/engine/",
        "genesis/exchange/",
        "genesis/data/",
    ]

    UI_PATHS = [
        "genesis/ui/",
        "genesis/api/",
    ]

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test coverage validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.coverage_history: List[Dict[str, Any]] = []
        self.coverage_trends: Dict[str, List[float]] = {}

    async def run_validation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run test coverage validation with path-specific thresholds.
        
        Args:
            context: Optional context for validation
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting test coverage validation")
        start_time = datetime.utcnow()
        
        results = {
            "validator": "TestCoverageValidator",
            "timestamp": start_time.isoformat(),
            "status": "pending",
            "passed": False,
            "coverage_analysis": {},
            "threshold_violations": [],
            "coverage_trends": {},
            "evidence": {},
            "metadata": {},
        }

        try:
            # Parse coverage XML report
            coverage_data = await self._parse_coverage_xml()
            if not coverage_data:
                results["status"] = "failed"
                results["error"] = "Coverage report not found or invalid"
                return results

            # Analyze path-specific coverage
            path_analysis = self._analyze_path_coverage(coverage_data)
            results["coverage_analysis"] = path_analysis

            # Check threshold violations
            violations = self._check_thresholds(path_analysis)
            results["threshold_violations"] = violations

            # Generate coverage evidence report
            evidence = self._generate_evidence_report(coverage_data, path_analysis)
            results["evidence"] = evidence

            # Track coverage trends
            trends = self._update_coverage_trends(path_analysis)
            results["coverage_trends"] = trends

            # Determine pass/fail status
            results["passed"] = len(violations) == 0
            results["status"] = "passed" if results["passed"] else "failed"

            # Add metadata
            results["metadata"] = {
                "total_coverage": path_analysis.get("overall_coverage", 0),
                "files_analyzed": len(coverage_data.get("files", [])),
                "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "thresholds": {
                    "money_paths": self.MONEY_PATH_THRESHOLD,
                    "risk_paths": self.RISK_PATH_THRESHOLD,
                    "core_paths": self.CORE_PATH_THRESHOLD,
                    "ui_paths": self.UI_PATH_THRESHOLD,
                },
            }

            logger.info(
                "Coverage validation completed",
                passed=results["passed"],
                violations=len(violations),
                total_coverage=path_analysis.get("overall_coverage", 0),
            )

        except Exception as e:
            logger.error("Coverage validation failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)

        return results

    async def _parse_coverage_xml(self) -> Optional[Dict[str, Any]]:
        """Parse pytest coverage.xml report.
        
        Returns:
            Parsed coverage data or None if not found
        """
        coverage_file = self.project_root / "coverage.xml"
        
        if not coverage_file.exists():
            # Try alternative locations
            alt_locations = [
                self.project_root / "htmlcov" / "coverage.xml",
                self.project_root / ".coverage" / "coverage.xml",
                self.project_root / "test-results" / "coverage.xml",
            ]
            
            for alt_path in alt_locations:
                if alt_path.exists():
                    coverage_file = alt_path
                    break
            else:
                logger.warning("Coverage XML report not found")
                return None

        try:
            tree = ET.parse(coverage_file)
            root = tree.getroot()

            # Extract overall coverage metrics
            coverage_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall": {},
                "files": {},
            }

            # Parse overall coverage from root attributes
            for metric in ["line-rate", "branch-rate"]:
                if metric in root.attrib:
                    coverage_data["overall"][metric] = float(root.attrib[metric]) * 100

            # Parse file-specific coverage
            for package in root.findall(".//package"):
                for class_elem in package.findall(".//class"):
                    filename = class_elem.get("filename", "")
                    if not filename:
                        continue

                    # Normalize path to be relative to project root
                    file_path = Path(filename)
                    if file_path.is_absolute():
                        try:
                            file_path = file_path.relative_to(self.project_root)
                        except ValueError:
                            pass
                    
                    file_key = str(file_path).replace("\\", "/")

                    # Calculate file coverage
                    lines = class_elem.findall(".//line")
                    total_lines = len(lines)
                    covered_lines = sum(1 for line in lines if line.get("hits", "0") != "0")
                    
                    coverage_data["files"][file_key] = {
                        "line_rate": (covered_lines / total_lines * 100) if total_lines > 0 else 0,
                        "total_lines": total_lines,
                        "covered_lines": covered_lines,
                        "uncovered_lines": total_lines - covered_lines,
                    }

            return coverage_data

        except ET.ParseError as e:
            logger.error("Failed to parse coverage XML", error=str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error parsing coverage", error=str(e))
            return None

    def _analyze_path_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coverage for different path categories.
        
        Args:
            coverage_data: Parsed coverage data
            
        Returns:
            Path-specific coverage analysis
        """
        analysis = {
            "money_paths": {"files": [], "coverage": 0.0, "threshold": self.MONEY_PATH_THRESHOLD},
            "risk_paths": {"files": [], "coverage": 0.0, "threshold": self.RISK_PATH_THRESHOLD},
            "core_paths": {"files": [], "coverage": 0.0, "threshold": self.CORE_PATH_THRESHOLD},
            "ui_paths": {"files": [], "coverage": 0.0, "threshold": self.UI_PATH_THRESHOLD},
            "other_paths": {"files": [], "coverage": 0.0, "threshold": self.DEFAULT_THRESHOLD},
            "overall_coverage": coverage_data.get("overall", {}).get("line-rate", 0.0),
        }

        files = coverage_data.get("files", {})

        for file_path, file_data in files.items():
            categorized = False
            file_info = {
                "path": file_path,
                "coverage": file_data["line_rate"],
                "lines": file_data["total_lines"],
                "covered": file_data["covered_lines"],
            }

            # Check money paths (highest priority)
            for money_pattern in self.MONEY_PATHS:
                if money_pattern.endswith("/"):
                    if file_path.startswith(money_pattern):
                        analysis["money_paths"]["files"].append(file_info)
                        categorized = True
                        break
                elif file_path == money_pattern or file_path.endswith("/" + money_pattern):
                    analysis["money_paths"]["files"].append(file_info)
                    categorized = True
                    break

            # Check risk paths
            if not categorized:
                for risk_pattern in self.RISK_PATHS:
                    if risk_pattern.endswith("/"):
                        if file_path.startswith(risk_pattern):
                            analysis["risk_paths"]["files"].append(file_info)
                            categorized = True
                            break
                    elif file_path == risk_pattern or file_path.endswith("/" + risk_pattern):
                        analysis["risk_paths"]["files"].append(file_info)
                        categorized = True
                        break

            # Check core paths
            if not categorized:
                for core_pattern in self.CORE_PATHS:
                    if file_path.startswith(core_pattern):
                        analysis["core_paths"]["files"].append(file_info)
                        categorized = True
                        break

            # Check UI paths
            if not categorized:
                for ui_pattern in self.UI_PATHS:
                    if file_path.startswith(ui_pattern):
                        analysis["ui_paths"]["files"].append(file_info)
                        categorized = True
                        break

            # Other paths
            if not categorized:
                analysis["other_paths"]["files"].append(file_info)

        # Calculate weighted average coverage for each category
        for category in ["money_paths", "risk_paths", "core_paths", "ui_paths", "other_paths"]:
            files_in_category = analysis[category]["files"]
            if files_in_category:
                total_lines = sum(f["lines"] for f in files_in_category)
                total_covered = sum(f["covered"] for f in files_in_category)
                analysis[category]["coverage"] = (total_covered / total_lines * 100) if total_lines > 0 else 0.0

        return analysis

    def _check_thresholds(self, path_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for threshold violations in path-specific coverage.
        
        Args:
            path_analysis: Path coverage analysis
            
        Returns:
            List of threshold violations
        """
        violations = []

        for category in ["money_paths", "risk_paths", "core_paths", "ui_paths"]:
            category_data = path_analysis.get(category, {})
            coverage = category_data.get("coverage", 0.0)
            threshold = category_data.get("threshold", 0.0)

            if coverage < threshold:
                # Special handling for money paths - they are absolutely critical
                severity = "critical"
                if category == "money_paths":
                    severity = "blocker"  # Money paths below 100% is a blocker
                elif category == "risk_paths":
                    severity = "critical"
                else:
                    severity = "high" if category == "core_paths" else "medium"
                
                violation = {
                    "category": category,
                    "coverage": coverage,
                    "threshold": threshold,
                    "gap": threshold - coverage,
                    "severity": severity,
                    "files_below_threshold": [
                        f for f in category_data.get("files", [])
                        if f["coverage"] < threshold
                    ],
                }
                
                # Add specific recommendations for money path violations
                if category == "money_paths" and violation["files_below_threshold"]:
                    violation["recommendation"] = (
                        "CRITICAL: Money paths must have 100% test coverage. "
                        "These files handle financial calculations and MUST be fully tested. "
                        "Add comprehensive unit tests immediately covering all edge cases, "
                        "boundary conditions, and error scenarios."
                    )
                
                violations.append(violation)

        return violations

    def _generate_evidence_report(
        self, coverage_data: Dict[str, Any], path_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed coverage evidence report.
        
        Args:
            coverage_data: Raw coverage data
            path_analysis: Path-specific analysis
            
        Returns:
            Evidence report dictionary
        """
        evidence = {
            "summary": {
                "overall_coverage": path_analysis.get("overall_coverage", 0.0),
                "money_path_coverage": path_analysis["money_paths"]["coverage"],
                "risk_path_coverage": path_analysis["risk_paths"]["coverage"],
                "core_path_coverage": path_analysis["core_paths"]["coverage"],
                "ui_path_coverage": path_analysis["ui_paths"]["coverage"],
            },
            "critical_files": [],
            "improvement_opportunities": [],
            "coverage_gaps": [],
        }

        # Identify critical files with low coverage
        for category in ["money_paths", "risk_paths"]:
            for file_info in path_analysis[category]["files"]:
                if file_info["coverage"] < path_analysis[category]["threshold"]:
                    evidence["critical_files"].append({
                        "file": file_info["path"],
                        "category": category,
                        "current_coverage": file_info["coverage"],
                        "required_coverage": path_analysis[category]["threshold"],
                        "gap": path_analysis[category]["threshold"] - file_info["coverage"],
                        "uncovered_lines": file_info["lines"] - file_info["covered"],
                    })

        # Sort critical files by gap size
        evidence["critical_files"].sort(key=lambda x: x["gap"], reverse=True)

        # Identify improvement opportunities
        all_files = []
        for category in path_analysis:
            if category != "overall_coverage" and "files" in path_analysis[category]:
                all_files.extend([
                    (f, category) for f in path_analysis[category]["files"]
                ])

        # Find files close to threshold (within 5%)
        for file_info, category in all_files:
            threshold = path_analysis[category]["threshold"]
            if threshold - 5 <= file_info["coverage"] < threshold:
                evidence["improvement_opportunities"].append({
                    "file": file_info["path"],
                    "category": category,
                    "current_coverage": file_info["coverage"],
                    "threshold": threshold,
                    "lines_to_cover": int((threshold - file_info["coverage"]) * file_info["lines"] / 100),
                })

        # Identify coverage gaps (files with < 50% coverage)
        for file_info, category in all_files:
            if file_info["coverage"] < 50:
                evidence["coverage_gaps"].append({
                    "file": file_info["path"],
                    "category": category,
                    "coverage": file_info["coverage"],
                    "uncovered_lines": file_info["lines"] - file_info["covered"],
                })

        return evidence

    def _update_coverage_trends(self, path_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update and analyze coverage trends over time.
        
        Args:
            path_analysis: Current path coverage analysis
            
        Returns:
            Coverage trend analysis
        """
        current_snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": path_analysis.get("overall_coverage", 0.0),
            "money_paths": path_analysis["money_paths"]["coverage"],
            "risk_paths": path_analysis["risk_paths"]["coverage"],
            "core_paths": path_analysis["core_paths"]["coverage"],
            "ui_paths": path_analysis["ui_paths"]["coverage"],
        }

        # Add to history
        self.coverage_history.append(current_snapshot)

        # Keep only last 30 snapshots
        if len(self.coverage_history) > 30:
            self.coverage_history = self.coverage_history[-30:]

        # Update trends for each category
        trends = {}
        for category in ["overall", "money_paths", "risk_paths", "core_paths", "ui_paths"]:
            values = [h[category] for h in self.coverage_history]
            
            if len(values) >= 2:
                # Calculate trend metrics
                current = values[-1]
                previous = values[-2]
                change = current - previous
                
                # Calculate moving average
                window_size = min(5, len(values))
                moving_avg = sum(values[-window_size:]) / window_size
                
                # Determine trend direction
                if len(values) >= 3:
                    recent_changes = [values[i] - values[i-1] for i in range(len(values)-2, len(values))]
                    trend_direction = "improving" if sum(recent_changes) > 0 else "declining" if sum(recent_changes) < 0 else "stable"
                else:
                    trend_direction = "improving" if change > 0 else "declining" if change < 0 else "stable"
                
                trends[category] = {
                    "current": current,
                    "previous": previous,
                    "change": change,
                    "change_percent": (change / previous * 100) if previous > 0 else 0,
                    "moving_average": moving_avg,
                    "trend": trend_direction,
                    "history_points": len(values),
                }
            else:
                trends[category] = {
                    "current": values[0] if values else 0,
                    "trend": "baseline",
                    "history_points": len(values),
                }

        return trends

    async def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable coverage validation report.
        
        Args:
            validation_results: Results from run_validation
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 80,
            "TEST COVERAGE VALIDATION REPORT",
            "=" * 80,
            f"Timestamp: {validation_results['timestamp']}",
            f"Status: {validation_results['status'].upper()}",
            "",
        ]

        # Summary section
        lines.extend([
            "COVERAGE SUMMARY",
            "-" * 40,
        ])
        
        analysis = validation_results.get("coverage_analysis", {})
        for category in ["money_paths", "risk_paths", "core_paths", "ui_paths"]:
            if category in analysis:
                coverage = analysis[category]["coverage"]
                threshold = analysis[category]["threshold"]
                status = "✓" if coverage >= threshold else "✗"
                lines.append(
                    f"{status} {category.replace('_', ' ').title()}: "
                    f"{coverage:.1f}% (threshold: {threshold}%)"
                )

        # Violations section
        violations = validation_results.get("threshold_violations", [])
        if violations:
            lines.extend([
                "",
                "THRESHOLD VIOLATIONS",
                "-" * 40,
            ])
            for violation in violations:
                lines.append(
                    f"[{violation['severity'].upper()}] {violation['category']}: "
                    f"{violation['coverage']:.1f}% < {violation['threshold']}% "
                    f"(gap: {violation['gap']:.1f}%)"
                )
                if violation.get("files_below_threshold"):
                    for file_info in violation["files_below_threshold"][:5]:
                        lines.append(f"  - {file_info['path']}: {file_info['coverage']:.1f}%")
                    if len(violation["files_below_threshold"]) > 5:
                        lines.append(f"  ... and {len(violation['files_below_threshold']) - 5} more files")

        # Critical files section
        evidence = validation_results.get("evidence", {})
        critical_files = evidence.get("critical_files", [])
        if critical_files:
            lines.extend([
                "",
                "CRITICAL FILES REQUIRING ATTENTION",
                "-" * 40,
            ])
            for file_data in critical_files[:10]:
                lines.append(
                    f"- {file_data['file']}: {file_data['current_coverage']:.1f}% "
                    f"(needs {file_data['gap']:.1f}% more, {file_data['uncovered_lines']} lines)"
                )

        # Trends section
        trends = validation_results.get("coverage_trends", {})
        if trends and any(t.get("trend") != "baseline" for t in trends.values()):
            lines.extend([
                "",
                "COVERAGE TRENDS",
                "-" * 40,
            ])
            for category, trend_data in trends.items():
                if trend_data.get("trend") != "baseline":
                    symbol = "↑" if trend_data["trend"] == "improving" else "↓" if trend_data["trend"] == "declining" else "→"
                    lines.append(
                        f"{symbol} {category.replace('_', ' ').title()}: "
                        f"{trend_data['current']:.1f}% ({trend_data['change']:+.1f}%)"
                    )

        lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(lines)