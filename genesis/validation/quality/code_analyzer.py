"""Code quality analyzer for complexity, duplication, and standards compliance."""

import ast
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


class CodeQualityAnalyzer:
    """Analyzes code quality including complexity, duplication, and standards compliance."""

    # Thresholds for code quality metrics
    MAX_CYCLOMATIC_COMPLEXITY = 10
    MAX_FUNCTION_LENGTH = 50
    MAX_FILE_LENGTH = 500
    MAX_LINE_LENGTH = 100
    MAX_DUPLICATE_PERCENTAGE = 5.0
    MAX_COGNITIVE_COMPLEXITY = 15

    # Code smell patterns
    CODE_SMELL_PATTERNS = {
        "bare_except": re.compile(r"except\s*:"),
        "print_statement": re.compile(r"\bprint\s*\("),
        "hardcoded_credentials": re.compile(r"(password|secret|key|token)\s*=\s*['\"][\w]+['\"]", re.IGNORECASE),
        "todo_fixme": re.compile(r"(TODO|FIXME|XXX|HACK)", re.IGNORECASE),
        "long_parameter_list": re.compile(r"def\s+\w+\s*\([^)]{100,}\)"),
        "magic_numbers": re.compile(r"\b(?<!\.)\d{3,}(?![\.\d])"),
        "global_variables": re.compile(r"^global\s+", re.MULTILINE),
        "mutable_defaults": re.compile(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\})[^)]*\)"),
    }

    # Anti-patterns to detect
    ANTI_PATTERNS = {
        "float_for_money": re.compile(r"(price|amount|balance|cost|fee|total)\s*[:=]\s*\d+\.\d+"),
        "missing_type_hints": re.compile(r"def\s+\w+\s*\([^)]*\)\s*(?!->)"),
        "no_docstring": re.compile(r"(def|class)\s+\w+[^:]*:\s*\n\s*(?![\"\'][\"\'][\"\'])"),
        "sync_io_in_async": re.compile(r"async\s+def.*\n.*(?:open|read|write)\s*\("),
    }

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize code quality analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.genesis_path = self.project_root / "genesis"

    async def run_validation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive code quality analysis.
        
        Args:
            context: Optional context for validation
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Starting code quality analysis")
        start_time = datetime.utcnow()

        results = {
            "validator": "CodeQualityAnalyzer",
            "timestamp": start_time.isoformat(),
            "status": "pending",
            "passed": False,
            "complexity_analysis": {},
            "duplication_analysis": {},
            "code_smells": [],
            "standards_violations": [],
            "metrics": {},
            "evidence": {},
        }

        try:
            # Run complexity analysis using radon
            complexity_results = await self._analyze_complexity()
            results["complexity_analysis"] = complexity_results

            # Run duplication detection using pylint
            duplication_results = await self._detect_duplication()
            results["duplication_analysis"] = duplication_results

            # Detect code smells and anti-patterns
            smell_results = await self._detect_code_smells()
            results["code_smells"] = smell_results

            # Check coding standards compliance
            standards_results = await self._check_coding_standards()
            results["standards_violations"] = standards_results

            # Generate overall metrics
            metrics = self._calculate_metrics(
                complexity_results, duplication_results, smell_results, standards_results
            )
            results["metrics"] = metrics

            # Generate evidence report
            evidence = self._generate_evidence(
                complexity_results, duplication_results, smell_results, standards_results
            )
            results["evidence"] = evidence

            # Determine pass/fail
            results["passed"] = self._determine_pass_status(metrics)
            results["status"] = "passed" if results["passed"] else "failed"

            logger.info(
                "Code quality analysis completed",
                passed=results["passed"],
                complexity_issues=len(complexity_results.get("violations", [])),
                code_smells=len(smell_results),
            )

        except Exception as e:
            logger.error("Code quality analysis failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)

        return results

    async def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity using radon.
        
        Returns:
            Complexity analysis results
        """
        logger.info("Analyzing code complexity")
        
        complexity_data = {
            "cyclomatic_complexity": [],
            "cognitive_complexity": [],
            "maintainability_index": [],
            "violations": [],
            "summary": {},
        }

        try:
            # Run radon cyclomatic complexity
            cc_result = subprocess.run(
                ["radon", "cc", str(self.genesis_path), "-j", "--no-assert"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if cc_result.returncode == 0 and cc_result.stdout:
                import json
                cc_data = json.loads(cc_result.stdout)
                
                for file_path, functions in cc_data.items():
                    for func in functions:
                        complexity = func.get("complexity", 0)
                        
                        func_info = {
                            "file": file_path,
                            "function": func.get("name", "unknown"),
                            "complexity": complexity,
                            "rank": func.get("rank", ""),
                            "line": func.get("lineno", 0),
                        }
                        
                        complexity_data["cyclomatic_complexity"].append(func_info)
                        
                        # Check for violations
                        if complexity > self.MAX_CYCLOMATIC_COMPLEXITY:
                            complexity_data["violations"].append({
                                "type": "cyclomatic_complexity",
                                "file": file_path,
                                "function": func.get("name"),
                                "value": complexity,
                                "threshold": self.MAX_CYCLOMATIC_COMPLEXITY,
                                "severity": "high" if complexity > 15 else "medium",
                            })

            # Run radon maintainability index
            mi_result = subprocess.run(
                ["radon", "mi", str(self.genesis_path), "-j"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if mi_result.returncode == 0 and mi_result.stdout:
                import json
                mi_data = json.loads(mi_result.stdout)
                
                for file_path, metrics in mi_data.items():
                    mi_score = metrics.get("mi", 0)
                    complexity_data["maintainability_index"].append({
                        "file": file_path,
                        "score": mi_score,
                        "rank": metrics.get("rank", ""),
                    })
                    
                    # Flag low maintainability
                    if mi_score < 20:
                        complexity_data["violations"].append({
                            "type": "maintainability",
                            "file": file_path,
                            "value": mi_score,
                            "threshold": 20,
                            "severity": "high",
                        })

            # Calculate summary statistics
            if complexity_data["cyclomatic_complexity"]:
                complexities = [f["complexity"] for f in complexity_data["cyclomatic_complexity"]]
                complexity_data["summary"] = {
                    "avg_complexity": sum(complexities) / len(complexities),
                    "max_complexity": max(complexities),
                    "functions_analyzed": len(complexities),
                    "high_complexity_functions": len([c for c in complexities if c > self.MAX_CYCLOMATIC_COMPLEXITY]),
                }

        except subprocess.TimeoutExpired:
            logger.warning("Radon analysis timed out")
        except Exception as e:
            logger.error("Failed to analyze complexity", error=str(e))

        return complexity_data

    async def _detect_duplication(self) -> Dict[str, Any]:
        """Detect code duplication using pylint.
        
        Returns:
            Duplication analysis results
        """
        logger.info("Detecting code duplication")
        
        duplication_data = {
            "duplicated_blocks": [],
            "duplication_percentage": 0.0,
            "violations": [],
            "summary": {},
        }

        try:
            # Run pylint duplicate-code checker
            result = subprocess.run(
                [
                    "pylint",
                    str(self.genesis_path),
                    "--disable=all",
                    "--enable=duplicate-code",
                    "--output-format=json",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                import json
                try:
                    pylint_output = json.loads(result.stdout)
                    
                    for message in pylint_output:
                        if message.get("message-id") == "R0801":  # duplicate-code
                            duplication_data["duplicated_blocks"].append({
                                "files": [message.get("path", "")],
                                "lines": message.get("line", 0),
                                "message": message.get("message", ""),
                            })
                except json.JSONDecodeError:
                    # Try parsing line by line for older pylint versions
                    for line in result.stdout.splitlines():
                        if "duplicate-code" in line or "R0801" in line:
                            duplication_data["duplicated_blocks"].append({
                                "message": line,
                            })

            # Calculate duplication percentage
            total_lines = self._count_code_lines()
            duplicated_lines = sum(block.get("lines", 0) for block in duplication_data["duplicated_blocks"])
            
            if total_lines > 0:
                duplication_data["duplication_percentage"] = (duplicated_lines / total_lines) * 100
                
                if duplication_data["duplication_percentage"] > self.MAX_DUPLICATE_PERCENTAGE:
                    duplication_data["violations"].append({
                        "type": "high_duplication",
                        "percentage": duplication_data["duplication_percentage"],
                        "threshold": self.MAX_DUPLICATE_PERCENTAGE,
                        "severity": "medium",
                    })

            duplication_data["summary"] = {
                "total_blocks": len(duplication_data["duplicated_blocks"]),
                "duplication_percentage": duplication_data["duplication_percentage"],
                "total_duplicated_lines": duplicated_lines,
            }

        except subprocess.TimeoutExpired:
            logger.warning("Pylint duplication detection timed out")
        except Exception as e:
            logger.error("Failed to detect duplication", error=str(e))

        return duplication_data

    async def _detect_code_smells(self) -> List[Dict[str, Any]]:
        """Detect code smells and anti-patterns.
        
        Returns:
            List of detected code smells
        """
        logger.info("Detecting code smells")
        code_smells = []

        try:
            # Scan Python files for code smells
            for py_file in self.genesis_path.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8")
                    
                    # Check for code smell patterns
                    for smell_name, pattern in self.CODE_SMELL_PATTERNS.items():
                        matches = pattern.finditer(content)
                        for match in matches:
                            line_num = content[:match.start()].count("\n") + 1
                            code_smells.append({
                                "type": smell_name,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "snippet": match.group(0)[:100],
                                "severity": self._get_smell_severity(smell_name),
                            })

                    # Check for anti-patterns
                    for pattern_name, pattern in self.ANTI_PATTERNS.items():
                        matches = pattern.finditer(content)
                        for match in matches:
                            line_num = content[:match.start()].count("\n") + 1
                            code_smells.append({
                                "type": f"anti_pattern_{pattern_name}",
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "snippet": match.group(0)[:100],
                                "severity": "high" if pattern_name == "float_for_money" else "medium",
                            })

                    # Check file length
                    lines = content.splitlines()
                    if len(lines) > self.MAX_FILE_LENGTH:
                        code_smells.append({
                            "type": "file_too_long",
                            "file": str(py_file.relative_to(self.project_root)),
                            "lines": len(lines),
                            "threshold": self.MAX_FILE_LENGTH,
                            "severity": "low",
                        })

                    # Check line lengths
                    long_lines = [i for i, line in enumerate(lines, 1) if len(line) > self.MAX_LINE_LENGTH]
                    if long_lines:
                        code_smells.append({
                            "type": "long_lines",
                            "file": str(py_file.relative_to(self.project_root)),
                            "lines": long_lines[:5],  # First 5 long lines
                            "count": len(long_lines),
                            "severity": "low",
                        })

                except Exception as e:
                    logger.warning(f"Failed to analyze file {py_file}", error=str(e))

        except Exception as e:
            logger.error("Failed to detect code smells", error=str(e))

        return code_smells

    async def _check_coding_standards(self) -> List[Dict[str, Any]]:
        """Check compliance with coding standards.
        
        Returns:
            List of coding standards violations
        """
        logger.info("Checking coding standards compliance")
        violations = []

        try:
            # Check Python files for standards compliance
            for py_file in self.genesis_path.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(content)
                    
                    # Check for missing docstrings
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node)
                            if not docstring:
                                violations.append({
                                    "type": "missing_docstring",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": node.lineno,
                                    "name": node.name,
                                    "severity": "medium",
                                })

                            # Check for type hints in functions
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if not node.returns and node.name != "__init__":
                                    violations.append({
                                        "type": "missing_return_type",
                                        "file": str(py_file.relative_to(self.project_root)),
                                        "line": node.lineno,
                                        "function": node.name,
                                        "severity": "medium",
                                    })

                                # Check parameter type hints
                                for arg in node.args.args:
                                    if not arg.annotation and arg.arg != "self":
                                        violations.append({
                                            "type": "missing_param_type",
                                            "file": str(py_file.relative_to(self.project_root)),
                                            "line": node.lineno,
                                            "function": node.name,
                                            "parameter": arg.arg,
                                            "severity": "medium",
                                        })

                    # Check naming conventions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if not self._is_pascal_case(node.name):
                                violations.append({
                                    "type": "class_naming",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": node.lineno,
                                    "name": node.name,
                                    "expected": "PascalCase",
                                    "severity": "low",
                                })

                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not self._is_snake_case(node.name) and not node.name.startswith("_"):
                                violations.append({
                                    "type": "function_naming",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": node.lineno,
                                    "name": node.name,
                                    "expected": "snake_case",
                                    "severity": "low",
                                })

                except Exception as e:
                    logger.warning(f"Failed to check standards for {py_file}", error=str(e))

        except Exception as e:
            logger.error("Failed to check coding standards", error=str(e))

        return violations

    def _calculate_metrics(
        self,
        complexity_results: Dict[str, Any],
        duplication_results: Dict[str, Any],
        smell_results: List[Dict[str, Any]],
        standards_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate overall code quality metrics.
        
        Returns:
            Code quality metrics
        """
        metrics = {
            "complexity": {
                "average": complexity_results.get("summary", {}).get("avg_complexity", 0),
                "max": complexity_results.get("summary", {}).get("max_complexity", 0),
                "violations": len(complexity_results.get("violations", [])),
            },
            "duplication": {
                "percentage": duplication_results.get("duplication_percentage", 0),
                "blocks": len(duplication_results.get("duplicated_blocks", [])),
            },
            "code_smells": {
                "total": len(smell_results),
                "by_severity": self._count_by_severity(smell_results),
                "by_type": self._count_by_type(smell_results),
            },
            "standards": {
                "violations": len(standards_results),
                "by_type": self._count_by_type(standards_results),
            },
            "overall_score": self._calculate_overall_score(
                complexity_results, duplication_results, smell_results, standards_results
            ),
        }

        return metrics

    def _generate_evidence(
        self,
        complexity_results: Dict[str, Any],
        duplication_results: Dict[str, Any],
        smell_results: List[Dict[str, Any]],
        standards_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate evidence report for code quality.
        
        Returns:
            Evidence dictionary
        """
        evidence = {
            "high_complexity_functions": [
                f for f in complexity_results.get("cyclomatic_complexity", [])
                if f.get("complexity", 0) > self.MAX_CYCLOMATIC_COMPLEXITY
            ][:10],
            "duplicated_code": duplication_results.get("duplicated_blocks", [])[:10],
            "critical_smells": [
                s for s in smell_results if s.get("severity") in ["critical", "high"]
            ][:10],
            "top_violations": standards_results[:10],
            "improvement_opportunities": self._identify_improvements(
                complexity_results, smell_results, standards_results
            ),
        }

        return evidence

    def _determine_pass_status(self, metrics: Dict[str, Any]) -> bool:
        """Determine if code quality meets standards.
        
        Returns:
            True if passed, False otherwise
        """
        # Fail if complexity is too high
        if metrics["complexity"]["max"] > 20:
            return False

        # Fail if duplication is excessive
        if metrics["duplication"]["percentage"] > self.MAX_DUPLICATE_PERCENTAGE:
            return False

        # Fail if there are critical code smells
        if metrics["code_smells"]["by_severity"].get("critical", 0) > 0:
            return False

        # Fail if overall score is too low
        if metrics.get("overall_score", 0) < 70:
            return False

        return True

    def _count_code_lines(self) -> int:
        """Count total lines of Python code.
        
        Returns:
            Total line count
        """
        total_lines = 0
        try:
            for py_file in self.genesis_path.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    total_lines += len(py_file.read_text().splitlines())
        except Exception:
            pass
        return total_lines

    def _get_smell_severity(self, smell_type: str) -> str:
        """Get severity level for a code smell type.
        
        Returns:
            Severity level
        """
        severity_map = {
            "bare_except": "high",
            "hardcoded_credentials": "critical",
            "print_statement": "medium",
            "todo_fixme": "low",
            "magic_numbers": "low",
            "global_variables": "medium",
            "mutable_defaults": "high",
        }
        return severity_map.get(smell_type, "medium")

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name follows PascalCase convention."""
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))

    def _count_by_severity(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count items by severity level."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for item in items:
            severity = item.get("severity", "medium")
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _count_by_type(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count items by type."""
        counts = {}
        for item in items:
            item_type = item.get("type", "unknown")
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts

    def _calculate_overall_score(
        self,
        complexity_results: Dict[str, Any],
        duplication_results: Dict[str, Any],
        smell_results: List[Dict[str, Any]],
        standards_results: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall code quality score (0-100).
        
        Returns:
            Overall score
        """
        score = 100.0

        # Deduct for complexity violations
        complexity_violations = len(complexity_results.get("violations", []))
        score -= min(complexity_violations * 2, 20)

        # Deduct for duplication
        duplication_pct = duplication_results.get("duplication_percentage", 0)
        if duplication_pct > self.MAX_DUPLICATE_PERCENTAGE:
            score -= min((duplication_pct - self.MAX_DUPLICATE_PERCENTAGE) * 2, 15)

        # Deduct for code smells
        smell_severity_counts = self._count_by_severity(smell_results)
        score -= smell_severity_counts.get("critical", 0) * 10
        score -= smell_severity_counts.get("high", 0) * 5
        score -= smell_severity_counts.get("medium", 0) * 2
        score -= smell_severity_counts.get("low", 0) * 0.5

        # Deduct for standards violations
        score -= min(len(standards_results) * 0.5, 10)

        return max(score, 0)

    def _identify_improvements(
        self,
        complexity_results: Dict[str, Any],
        smell_results: List[Dict[str, Any]],
        standards_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify top improvement opportunities.
        
        Returns:
            List of improvement suggestions
        """
        improvements = []

        # High complexity functions
        high_complexity = [
            f for f in complexity_results.get("cyclomatic_complexity", [])
            if f.get("complexity", 0) > self.MAX_CYCLOMATIC_COMPLEXITY
        ]
        if high_complexity:
            improvements.append({
                "category": "complexity",
                "suggestion": f"Refactor {len(high_complexity)} high-complexity functions",
                "impact": "high",
                "files": list(set(f["file"] for f in high_complexity[:5])),
            })

        # Code smells by type
        smell_types = self._count_by_type(smell_results)
        for smell_type, count in sorted(smell_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            improvements.append({
                "category": "code_smell",
                "suggestion": f"Fix {count} instances of {smell_type}",
                "impact": "medium",
            })

        # Standards violations
        standards_types = self._count_by_type(standards_results)
        if "missing_docstring" in standards_types:
            improvements.append({
                "category": "documentation",
                "suggestion": f"Add docstrings to {standards_types['missing_docstring']} functions/classes",
                "impact": "medium",
            })

        return improvements