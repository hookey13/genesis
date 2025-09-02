"""Documentation completeness validation module."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from genesis.validation.base import (
    CheckStatus,
    ValidationCheck,
    ValidationContext,
    ValidationEvidence,
    ValidationMetadata,
    ValidationResult,
    Validator,
)

logger = structlog.get_logger(__name__)


class DocumentationValidator(Validator):
    """Validates documentation completeness and quality."""

    REQUIRED_DOCS = {
        "README.md": {
            "sections": [
                "Installation",
                "Configuration",
                "Usage",
                "Architecture",
                "Testing",
                "Deployment",
            ],
            "min_lines": 100,
        },
        "docs/runbook.md": {
            "sections": [
                "System Startup",
                "Shutdown Procedures",
                "Emergency Position Closure",
                "API Rate Limit Handling",
                "Database Recovery",
                "Tilt Intervention",
                "Performance Degradation",
                "Security Incident Response",
                "Rollback Procedures",
                "Data Corruption Recovery",
            ],
            "min_lines": 200,
        },
        "docs/architecture.md": {
            "sections": [
                "System Overview",
                "Components",
                "Data Flow",
                "Security",
                "Scalability",
            ],
            "min_lines": 150,
        },
        "docs/deployment.md": {
            "sections": [
                "Prerequisites",
                "Environment Setup",
                "Deployment Steps",
                "Verification",
                "Rollback",
            ],
            "min_lines": 50,
        },
        "docs/troubleshooting.md": {
            "sections": [
                "Common Issues",
                "Error Messages",
                "Performance Issues",
                "Connectivity Problems",
                "Recovery Procedures",
            ],
            "min_lines": 100,
        },
        "docs/monitoring.md": {
            "sections": [
                "Metrics",
                "Dashboards",
                "Alerts",
                "Log Analysis",
                "Health Checks",
            ],
            "min_lines": 75,
        },
    }

    API_DOC_PATTERNS = {
        "endpoints": r"(GET|POST|PUT|DELETE|PATCH)\s+/[\w/\-{}]+",
        "parameters": r"(Parameters?|Query Params?|Body|Headers?):",
        "responses": r"(Response|Returns?):",
        "examples": r"(Example|Sample|Usage):",
        "authentication": r"(Auth|Authentication|Authorization):",
    }

    RUNBOOK_SCENARIOS = [
        "startup",
        "shutdown",
        "emergency",
        "rate limit",
        "database",
        "tilt",
        "performance",
        "security",
        "rollback",
        "corruption",
    ]

    def __init__(self, genesis_root: Path | None = None):
        """Initialize documentation validator.

        Args:
            genesis_root: Root directory of Genesis project
        """
        super().__init__(
            validator_id="OPS-003",
            name="DocumentationValidator",
            description="Validates documentation completeness including runbooks, API docs, README, and operational guides",
        )
        self.genesis_root = genesis_root or Path.cwd()
        self.set_timeout(60)  # 1 minute for doc validation
        self.set_retry_policy(retry_count=1, retry_delay_seconds=5)

    async def run_validation(self, context: ValidationContext) -> ValidationResult:
        """Run documentation validation checks.

        Args:
            context: Validation context with configuration

        Returns:
            ValidationResult with checks and evidence
        """
        logger.info("Starting documentation validation")
        start_time = datetime.utcnow()

        # Update genesis_root from context if available
        if context.genesis_root:
            self.genesis_root = Path(context.genesis_root)

        checks = []
        evidence = ValidationEvidence()

        # Create validation checks
        checks.append(await self._create_required_docs_check())
        checks.append(await self._create_runbook_check())
        checks.append(await self._create_api_docs_check())
        checks.append(await self._create_readme_check())
        checks.append(await self._create_operational_guides_check())
        checks.append(await self._create_gap_analysis_check())

        # Determine overall status
        failed_checks = [c for c in checks if c.status == CheckStatus.FAILED]
        warning_checks = [c for c in checks if c.status == CheckStatus.WARNING]

        if not failed_checks and not warning_checks:
            overall_status = CheckStatus.PASSED
            message = "Documentation complete and comprehensive"
        elif failed_checks:
            overall_status = CheckStatus.FAILED
            message = (
                f"Documentation gaps detected - {len(failed_checks)} checks failed"
            )
        else:
            overall_status = CheckStatus.WARNING
            message = (
                f"Documentation partially complete - {len(warning_checks)} warnings"
            )

        # Create metadata
        metadata = ValidationMetadata(
            version="1.0.0",
            environment=context.environment,
            run_id=context.metadata.run_id if context.metadata else "local-run",
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            machine_info={},
            additional_info={"genesis_root": str(self.genesis_root)},
        )

        # Create result
        result = ValidationResult(
            validator_id=self.validator_id,
            validator_name=self.name,
            status=overall_status,
            message=message,
            checks=checks,
            evidence=evidence,
            metadata=metadata,
        )

        # Update counts and score
        result.update_counts()

        return result

    async def _create_required_docs_check(self) -> ValidationCheck:
        """Create validation check for required documentation."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._check_required_docs()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.FAILED
        evidence.metrics["docs_found"] = result.get("docs_found", [])
        evidence.metrics["docs_missing"] = result.get("docs_missing", [])

        return ValidationCheck(
            id="DOC-001",
            name="Required Documentation",
            description="Check all required documentation files exist",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=True,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="high" if not result["passed"] else "low",
            remediation=(
                "Create missing documentation files" if not result["passed"] else None
            ),
            tags=["documentation", "required"],
        )

    async def _create_runbook_check(self) -> ValidationCheck:
        """Create validation check for runbook completeness."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._check_runbook_completeness()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.FAILED
        evidence.metrics["scenarios_covered"] = result.get("scenarios_covered", [])
        evidence.metrics["scenarios_missing"] = result.get("scenarios_missing", [])

        return ValidationCheck(
            id="DOC-002",
            name="Runbook Completeness",
            description="Verify runbook covers all required scenarios",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=True,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="critical" if not result["passed"] else "low",
            remediation=(
                "Add missing runbook scenarios" if not result["passed"] else None
            ),
            tags=["documentation", "runbook", "operations"],
        )

    async def _create_api_docs_check(self) -> ValidationCheck:
        """Create validation check for API documentation."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._verify_api_documentation()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.WARNING
        evidence.metrics["coverage_percentage"] = result.get("coverage_percentage", 0)
        evidence.metrics["missing_sections"] = result.get("missing_sections", [])

        return ValidationCheck(
            id="DOC-003",
            name="API Documentation",
            description="Verify API documentation completeness",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=False,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="medium" if status == CheckStatus.WARNING else "low",
            remediation=(
                "Complete API documentation" if status == CheckStatus.WARNING else None
            ),
            tags=["documentation", "api"],
        )

    async def _create_readme_check(self) -> ValidationCheck:
        """Create validation check for README."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._validate_readme()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.WARNING
        evidence.metrics["sections_found"] = result.get("sections_found", [])
        evidence.metrics["sections_missing"] = result.get("sections_missing", [])

        return ValidationCheck(
            id="DOC-004",
            name="README Validation",
            description="Validate README file completeness",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=False,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="medium" if status == CheckStatus.WARNING else "low",
            remediation=(
                "Update README with missing sections"
                if status == CheckStatus.WARNING
                else None
            ),
            tags=["documentation", "readme"],
        )

    async def _create_operational_guides_check(self) -> ValidationCheck:
        """Create validation check for operational guides."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._check_operational_guides()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.WARNING
        evidence.metrics["guides_found"] = result.get("guides_found", [])
        evidence.metrics["guides_missing"] = result.get("guides_missing", [])

        return ValidationCheck(
            id="DOC-005",
            name="Operational Guides",
            description="Check operational documentation completeness",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=False,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="medium" if status == CheckStatus.WARNING else "low",
            remediation=(
                "Create missing operational guides"
                if status == CheckStatus.WARNING
                else None
            ),
            tags=["documentation", "operations"],
        )

    async def _create_gap_analysis_check(self) -> ValidationCheck:
        """Create validation check for documentation gaps."""
        start_time = datetime.utcnow()
        evidence = ValidationEvidence()
        result = await self._generate_gap_analysis()

        status = CheckStatus.PASSED if result["passed"] else CheckStatus.WARNING
        evidence.metrics["gaps_identified"] = result.get("gaps", [])
        evidence.metrics["recommendations"] = result.get("recommendations", [])

        return ValidationCheck(
            id="DOC-006",
            name="Gap Analysis",
            description="Documentation gap analysis",
            category="documentation",
            status=status,
            details=result["message"],
            is_blocking=False,
            evidence=evidence,
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            timestamp=datetime.utcnow(),
            severity="low",
            remediation=(
                "Address identified documentation gaps"
                if status == CheckStatus.WARNING
                else None
            ),
            tags=["documentation", "gap-analysis"],
        )

    async def _check_required_docs(self) -> dict[str, Any]:
        """Check if all required documentation files exist.

        Returns:
            Validation result for required documentation
        """
        result = {
            "passed": False,
            "message": "",
            "docs_found": [],
            "docs_missing": [],
            "docs_incomplete": [],
        }

        for doc_path, requirements in self.REQUIRED_DOCS.items():
            full_path = self.genesis_root / doc_path

            if full_path.exists():
                result["docs_found"].append(doc_path)

                # Check file size/content
                with open(full_path, encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                    if len(lines) < requirements["min_lines"]:
                        result["docs_incomplete"].append(
                            f"{doc_path} (only {len(lines)} lines, need {requirements['min_lines']})"
                        )
            else:
                result["docs_missing"].append(doc_path)

        if not result["docs_missing"] and not result["docs_incomplete"]:
            result["passed"] = True
            result["message"] = (
                f"All {len(self.REQUIRED_DOCS)} required documents present and complete"
            )
        else:
            issues = []
            if result["docs_missing"]:
                issues.append(f"{len(result['docs_missing'])} missing")
            if result["docs_incomplete"]:
                issues.append(f"{len(result['docs_incomplete'])} incomplete")
            result["message"] = f"Documentation issues: {', '.join(issues)}"

        return result

    async def _check_runbook_completeness(self) -> dict[str, Any]:
        """Check runbook completeness for all scenarios.

        Returns:
            Validation result for runbook completeness
        """
        result = {
            "passed": False,
            "message": "",
            "scenarios_covered": [],
            "scenarios_missing": [],
        }

        runbook_path = self.genesis_root / "docs" / "runbook.md"

        if runbook_path.exists():
            with open(runbook_path, encoding="utf-8") as f:
                content = f.read().lower()

                for scenario in self.RUNBOOK_SCENARIOS:
                    # Check if scenario is covered with sufficient detail
                    if scenario in content:
                        # Look for procedure steps (numbered lists or bullet points)
                        scenario_section = content[content.find(scenario) :]
                        if re.search(r"(\d+\.|[-*])\s+\w+", scenario_section[:500]):
                            result["scenarios_covered"].append(scenario)
                        else:
                            result["scenarios_missing"].append(
                                f"{scenario} (no procedures)"
                            )
                    else:
                        result["scenarios_missing"].append(scenario)

            if not result["scenarios_missing"]:
                result["passed"] = True
                result["message"] = (
                    f"All {len(self.RUNBOOK_SCENARIOS)} scenarios documented with procedures"
                )
            else:
                result["message"] = (
                    f"Missing scenarios: {', '.join(result['scenarios_missing'])}"
                )
        else:
            result["message"] = "Runbook file not found"
            result["scenarios_missing"] = self.RUNBOOK_SCENARIOS

        return result

    async def _verify_api_documentation(self) -> dict[str, Any]:
        """Verify API documentation coverage.

        Returns:
            Validation result for API documentation
        """
        result = {
            "passed": False,
            "message": "",
            "coverage": {},
            "missing_sections": [],
        }

        api_doc_dir = self.genesis_root / "docs" / "api"

        if api_doc_dir.exists():
            api_files = (
                list(api_doc_dir.glob("*.md"))
                + list(api_doc_dir.glob("*.yaml"))
                + list(api_doc_dir.glob("*.json"))
            )

            if api_files:
                total_coverage = dict.fromkeys(self.API_DOC_PATTERNS, 0)

                for api_file in api_files:
                    with open(api_file, encoding="utf-8") as f:
                        content = f.read()

                        for pattern_name, pattern in self.API_DOC_PATTERNS.items():
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            total_coverage[pattern_name] += len(matches)

                result["coverage"] = total_coverage

                # Check if all patterns are covered
                missing = [name for name, count in total_coverage.items() if count == 0]

                if not missing:
                    result["passed"] = True
                    result["message"] = (
                        "API documentation complete with all required sections"
                    )
                else:
                    result["missing_sections"] = missing
                    result["message"] = (
                        f"API documentation missing: {', '.join(missing)}"
                    )
            else:
                result["message"] = "No API documentation files found"
        else:
            # Check if API docs are in a different location
            swagger_file = self.genesis_root / "swagger.yaml"
            openapi_file = self.genesis_root / "openapi.json"

            if swagger_file.exists() or openapi_file.exists():
                result["passed"] = True
                result["message"] = "API documentation found (OpenAPI/Swagger)"
            else:
                result["message"] = "API documentation directory not found"

        return result

    async def _validate_readme(self) -> dict[str, Any]:
        """Validate README completeness.

        Returns:
            Validation result for README
        """
        result = {
            "passed": False,
            "message": "",
            "sections_found": [],
            "sections_missing": [],
        }

        readme_path = self.genesis_root / "README.md"

        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                content = f.read()

                required_sections = self.REQUIRED_DOCS["README.md"]["sections"]

                for section in required_sections:
                    # Check for section headers
                    if re.search(rf"#+\s*{section}", content, re.IGNORECASE):
                        result["sections_found"].append(section)
                    else:
                        result["sections_missing"].append(section)

                # Check for installation commands
                has_install_commands = bool(
                    re.search(r"(pip install|npm install|make install)", content)
                )

                # Check for usage examples
                has_usage_examples = bool(
                    re.search(r"```[\w]*\n.*\n```", content, re.DOTALL)
                )

                if (
                    not result["sections_missing"]
                    and has_install_commands
                    and has_usage_examples
                ):
                    result["passed"] = True
                    result["message"] = (
                        "README complete with all required sections and examples"
                    )
                else:
                    issues = []
                    if result["sections_missing"]:
                        issues.append(
                            f"missing sections: {', '.join(result['sections_missing'])}"
                        )
                    if not has_install_commands:
                        issues.append("no installation commands")
                    if not has_usage_examples:
                        issues.append("no usage examples")
                    result["message"] = f"README issues: {', '.join(issues)}"
        else:
            result["message"] = "README.md not found"
            result["sections_missing"] = self.REQUIRED_DOCS["README.md"]["sections"]

        return result

    async def _check_operational_guides(self) -> dict[str, Any]:
        """Check for required operational guides.

        Returns:
            Validation result for operational guides
        """
        result = {
            "passed": False,
            "message": "",
            "guides_found": [],
            "guides_missing": [],
        }

        required_guides = [
            "deployment",
            "monitoring",
            "troubleshooting",
            "backup",
            "recovery",
            "scaling",
        ]

        docs_dir = self.genesis_root / "docs"

        if docs_dir.exists():
            for guide in required_guides:
                # Check for guide file or section in existing docs
                guide_file = docs_dir / f"{guide}.md"

                if guide_file.exists():
                    result["guides_found"].append(guide)
                else:
                    # Check if guide is covered in other docs
                    found_in_other = False
                    for doc_file in docs_dir.glob("*.md"):
                        with open(doc_file, encoding="utf-8") as f:
                            if guide in f.read().lower():
                                found_in_other = True
                                break

                    if found_in_other:
                        result["guides_found"].append(f"{guide} (in other docs)")
                    else:
                        result["guides_missing"].append(guide)

            if not result["guides_missing"]:
                result["passed"] = True
                result["message"] = (
                    f"All {len(required_guides)} operational guides present"
                )
            else:
                result["message"] = (
                    f"Missing guides: {', '.join(result['guides_missing'])}"
                )
        else:
            result["message"] = "Documentation directory not found"
            result["guides_missing"] = required_guides

        return result

    async def _generate_gap_analysis(self) -> dict[str, Any]:
        """Generate documentation gap analysis.

        Returns:
            Gap analysis results
        """
        result = {
            "passed": False,
            "message": "",
            "total_coverage": 0,
            "gaps": [],
            "recommendations": [],
        }

        gaps = []
        recommendations = []

        # Check code documentation coverage
        code_files = list((self.genesis_root / "genesis").rglob("*.py"))
        documented_files = 0

        for code_file in code_files:
            with open(code_file, encoding="utf-8") as f:
                content = f.read()
                # Check for docstrings
                if '"""' in content or "'''" in content:
                    documented_files += 1
                else:
                    relative_path = code_file.relative_to(self.genesis_root)
                    gaps.append(f"No docstrings in {relative_path}")

        code_coverage = (documented_files / len(code_files) * 100) if code_files else 0

        # Check for architecture decision records
        adr_dir = self.genesis_root / "docs" / "adr"
        if not adr_dir.exists():
            gaps.append("No Architecture Decision Records (ADR) directory")
            recommendations.append("Create docs/adr/ for architectural decisions")

        # Check for changelog
        changelog = self.genesis_root / "CHANGELOG.md"
        if not changelog.exists():
            gaps.append("No CHANGELOG.md file")
            recommendations.append("Create CHANGELOG.md to track version history")

        # Check for contributing guide
        contributing = self.genesis_root / "CONTRIBUTING.md"
        if not contributing.exists():
            gaps.append("No CONTRIBUTING.md file")
            recommendations.append("Create CONTRIBUTING.md for contribution guidelines")

        result["gaps"] = gaps[:10]  # Limit to top 10 gaps
        result["recommendations"] = recommendations
        result["total_coverage"] = int(code_coverage)

        if code_coverage >= 80 and len(gaps) <= 5:
            result["passed"] = True
            result["message"] = (
                f"Documentation coverage: {code_coverage:.1f}% with {len(gaps)} minor gaps"
            )
        else:
            result["message"] = (
                f"Documentation coverage: {code_coverage:.1f}% with {len(gaps)} gaps"
            )

        return result

    def generate_report(self) -> str:
        """Generate a detailed documentation validation report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run validate() first."

        report = []
        report.append("=" * 80)
        report.append("DOCUMENTATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(
            f"Overall Status: {'PASSED' if self.results['passed'] else 'FAILED'}"
        )
        report.append(f"Score: {self.results['score']}%")
        report.append(f"Summary: {self.results['summary']}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)

        for check_name, check_result in self.results["checks"].items():
            status = "✓" if check_result.get("passed", False) else "✗"
            report.append(f"{status} {check_name}: {check_result.get('message', '')}")

            # Add details for failed checks
            if not check_result.get("passed", False):
                if check_result.get("docs_missing"):
                    report.append(
                        f"  Missing docs: {', '.join(check_result['docs_missing'])}"
                    )
                if check_result.get("sections_missing"):
                    report.append(
                        f"  Missing sections: {', '.join(check_result['sections_missing'])}"
                    )
                if check_result.get("scenarios_missing"):
                    report.append(
                        f"  Missing scenarios: {', '.join(check_result['scenarios_missing'][:5])}"
                    )
                if check_result.get("guides_missing"):
                    report.append(
                        f"  Missing guides: {', '.join(check_result['guides_missing'])}"
                    )

            # Add gap analysis details
            if check_name == "gap_analysis":
                report.append(f"  Coverage: {check_result.get('total_coverage', 0)}%")
                if check_result.get("recommendations"):
                    report.append("  Recommendations:")
                    for rec in check_result["recommendations"][:3]:
                        report.append(f"    - {rec}")

        report.append("")
        report.append(
            f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds"
        )
        report.append("=" * 80)

        return "\n".join(report)
