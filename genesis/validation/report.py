"""Report generation for validation framework."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from genesis.validation.base import CheckStatus, ValidationResult


class ReportGenerator:
    """Generates validation reports in various formats."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize report generator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.reports_dir = self.genesis_root / "docs" / "validation_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(
        self,
        results: dict[str, ValidationResult],
        metadata: dict[str, Any]
    ) -> str:
        """Generate JSON format report.
        
        Args:
            results: Dictionary of validation results by validator ID
            metadata: Report metadata
            
        Returns:
            JSON string representation
        """
        report_data = self._prepare_report_data(results, metadata)

        # Custom JSON encoder for Decimal and datetime
        def json_encoder(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(report_data, indent=2, default=json_encoder)

    def generate_yaml_report(
        self,
        results: dict[str, ValidationResult],
        metadata: dict[str, Any]
    ) -> str:
        """Generate YAML format report.
        
        Args:
            results: Dictionary of validation results by validator ID
            metadata: Report metadata
            
        Returns:
            YAML string representation
        """
        report_data = self._prepare_report_data(results, metadata)

        # Convert Decimals to floats for YAML
        def convert_decimals(data):
            if isinstance(data, dict):
                return {k: convert_decimals(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_decimals(item) for item in data]
            elif isinstance(data, Decimal):
                return float(data)
            return data

        report_data = convert_decimals(report_data)

        return yaml.dump(report_data, default_flow_style=False, sort_keys=False)

    def generate_markdown_report(
        self,
        results: dict[str, ValidationResult],
        metadata: dict[str, Any]
    ) -> str:
        """Generate Markdown format report.
        
        Args:
            results: Dictionary of validation results by validator ID
            metadata: Report metadata
            
        Returns:
            Markdown string representation
        """
        lines = []

        # Header
        lines.append("# Validation Report")
        lines.append("")
        lines.append(f"**Run ID:** `{metadata.get('run_id', 'N/A')}`")
        lines.append(f"**Date:** {metadata.get('timestamp', 'N/A')}")
        lines.append(f"**Duration:** {metadata.get('duration_seconds', 0):.2f} seconds")
        lines.append(f"**Environment:** {metadata.get('environment', 'N/A')}")
        lines.append(f"**Mode:** {metadata.get('mode', 'N/A')}")
        lines.append("")

        # Overall Summary
        lines.append("## Summary")
        lines.append("")

        overall_status = metadata.get('overall_status', 'unknown')
        status_emoji = self._get_status_emoji(overall_status)
        lines.append(f"**Overall Status:** {status_emoji} {overall_status.upper()}")
        lines.append(f"**Overall Score:** {metadata.get('overall_score', 0):.1f}%")
        lines.append("")

        # Statistics
        lines.append("### Statistics")
        lines.append("")
        lines.append(f"- **Validators Run:** {metadata.get('validators_run', 0)}")
        lines.append(f"- **Passed:** {metadata.get('validators_passed', 0)}")
        lines.append(f"- **Failed:** {metadata.get('validators_failed', 0)}")
        lines.append(f"- **Warnings:** {metadata.get('validators_warning', 0)}")
        lines.append(f"- **Skipped:** {metadata.get('validators_skipped', 0)}")
        lines.append("")

        # Overrides if any
        overrides = metadata.get('overrides', {})
        if overrides:
            lines.append("### Overrides")
            lines.append("")
            for vid, override_info in overrides.items():
                lines.append(f"- **{vid}:** {override_info.get('reason', 'No reason')}")
                lines.append(f"  - Authorized by: {override_info.get('authorized_by', 'Unknown')}")
                lines.append(f"  - Level: {override_info.get('authorization_level', 'Unknown')}")
            lines.append("")

        # Detailed Results
        lines.append("## Detailed Results")
        lines.append("")

        # Group by status
        passed_validators = []
        failed_validators = []
        warning_validators = []
        skipped_validators = []

        for vid, result in results.items():
            if result.status == CheckStatus.PASSED:
                passed_validators.append((vid, result))
            elif result.status == CheckStatus.FAILED:
                failed_validators.append((vid, result))
            elif result.status == CheckStatus.WARNING:
                warning_validators.append((vid, result))
            elif result.status == CheckStatus.SKIPPED:
                skipped_validators.append((vid, result))

        # Failed validators first (most important)
        if failed_validators:
            lines.append("### ❌ Failed Validators")
            lines.append("")
            for vid, result in failed_validators:
                lines.extend(self._format_validator_result(vid, result))

        # Warning validators
        if warning_validators:
            lines.append("### ⚠️ Warning Validators")
            lines.append("")
            for vid, result in warning_validators:
                lines.extend(self._format_validator_result(vid, result))

        # Passed validators (summary only)
        if passed_validators:
            lines.append("### ✅ Passed Validators")
            lines.append("")
            for vid, result in passed_validators:
                lines.append(f"- **{result.validator_name}** - Score: {result.score:.1f}%")
            lines.append("")

        # Skipped validators
        if skipped_validators:
            lines.append("### ⏭️ Skipped Validators")
            lines.append("")
            for vid, result in skipped_validators:
                lines.append(f"- **{result.validator_name}** - {result.message}")
            lines.append("")

        # Blocking Issues Summary
        blocking_issues = []
        for result in results.values():
            if result.has_blocking_failures():
                for check in result.checks:
                    if check.is_blocking and check.status in [CheckStatus.FAILED, CheckStatus.ERROR]:
                        blocking_issues.append((result.validator_name, check))

        if blocking_issues:
            lines.append("## 🚨 Blocking Issues")
            lines.append("")
            lines.append("These issues must be resolved before go-live:")
            lines.append("")
            for validator_name, check in blocking_issues:
                lines.append(f"- **[{validator_name}] {check.name}**")
                lines.append(f"  - {check.details}")
                if check.remediation:
                    lines.append(f"  - **Remediation:** {check.remediation}")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.extend(self._generate_recommendations(results))

        return "\n".join(lines)

    def _prepare_report_data(
        self,
        results: dict[str, ValidationResult],
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare report data structure.
        
        Args:
            results: Dictionary of validation results
            metadata: Report metadata
            
        Returns:
            Report data dictionary
        """
        return {
            "metadata": metadata,
            "results": {
                vid: result.to_dict()
                for vid, result in results.items()
            }
        }

    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status.
        
        Args:
            status: Status string
            
        Returns:
            Emoji character
        """
        emoji_map = {
            "passed": "✅",
            "failed": "❌",
            "warning": "⚠️",
            "skipped": "⏭️",
            "error": "🔥",
            "unknown": "❓"
        }
        return emoji_map.get(status.lower(), "❓")

    def _format_validator_result(self, vid: str, result: ValidationResult) -> list[str]:
        """Format a single validator result for markdown.
        
        Args:
            vid: Validator ID
            result: Validation result
            
        Returns:
            List of markdown lines
        """
        lines = []

        lines.append(f"#### {result.validator_name}")
        lines.append("")
        lines.append(f"- **Status:** {self._get_status_emoji(result.status.value)} {result.status.value}")
        lines.append(f"- **Score:** {result.score:.1f}%")
        lines.append(f"- **Checks:** {result.passed_checks} passed, {result.failed_checks} failed, {result.warning_checks} warnings")

        if result.message:
            lines.append(f"- **Message:** {result.message}")

        # Show failed checks
        failed_checks = [c for c in result.checks if c.status == CheckStatus.FAILED]
        if failed_checks:
            lines.append("")
            lines.append("**Failed Checks:**")
            for check in failed_checks[:5]:  # Limit to first 5
                lines.append(f"- {check.name}: {check.details}")
                if check.remediation:
                    lines.append(f"  - Remediation: {check.remediation}")

        # Show errors
        if result.error_checks > 0:
            error_checks = [c for c in result.checks if c.status == CheckStatus.ERROR]
            lines.append("")
            lines.append("**Errors:**")
            for check in error_checks[:3]:
                lines.append(f"- {check.error_message}")

        lines.append("")
        return lines

    def _generate_recommendations(self, results: dict[str, ValidationResult]) -> list[str]:
        """Generate recommendations based on results.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            List of recommendation lines
        """
        recommendations = []

        # Analyze results
        total_score = sum(r.score for r in results.values())
        avg_score = total_score / len(results) if results else Decimal("0")

        failed_validators = [r for r in results.values() if r.status == CheckStatus.FAILED]
        warning_validators = [r for r in results.values() if r.status == CheckStatus.WARNING]

        # Generate recommendations
        if avg_score >= Decimal("95"):
            recommendations.append("✅ System is in excellent condition and ready for production.")
        elif avg_score >= Decimal("80"):
            recommendations.append("⚠️ System is generally ready but has some issues to address:")
            if failed_validators:
                recommendations.append(f"- Fix {len(failed_validators)} failed validators")
            if warning_validators:
                recommendations.append(f"- Review {len(warning_validators)} warning validators")
        else:
            recommendations.append("❌ System requires significant work before production:")
            recommendations.append(f"- Current score: {avg_score:.1f}% (minimum 80% recommended)")
            recommendations.append(f"- Failed validators: {len(failed_validators)}")
            recommendations.append(f"- Warning validators: {len(warning_validators)}")

        # Specific recommendations by category
        security_issues = [
            r for r in results.values()
            if "security" in r.validator_id.lower() and r.status == CheckStatus.FAILED
        ]
        if security_issues:
            recommendations.append("")
            recommendations.append("**Security Priority:**")
            recommendations.append("- Address security validators immediately")
            recommendations.append("- Review all secrets and credentials")
            recommendations.append("- Ensure encryption is properly configured")

        performance_issues = [
            r for r in results.values()
            if "performance" in r.validator_id.lower() and r.status != CheckStatus.PASSED
        ]
        if performance_issues:
            recommendations.append("")
            recommendations.append("**Performance Optimization:**")
            recommendations.append("- Review performance test results")
            recommendations.append("- Optimize database queries if needed")
            recommendations.append("- Consider caching strategies")

        return recommendations

    async def save_report(
        self,
        results: dict[str, ValidationResult],
        metadata: dict[str, Any],
        format: str = "json",
        filename: str | None = None
    ) -> Path:
        """Save validation report to file.
        
        Args:
            results: Dictionary of validation results
            metadata: Report metadata
            format: Report format ('json', 'yaml', 'markdown')
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved report
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_id = metadata.get('run_id', 'unknown')
            filename = f"validation_{run_id}_{timestamp}.{format}"

        file_path = self.reports_dir / filename

        # Generate report content
        if format == "json":
            content = self.generate_json_report(results, metadata)
        elif format == "yaml":
            content = self.generate_yaml_report(results, metadata)
        elif format == "markdown" or format == "md":
            content = self.generate_markdown_report(results, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path

    def compare_reports(
        self,
        report1_path: Path,
        report2_path: Path
    ) -> dict[str, Any]:
        """Compare two validation reports for trend analysis.
        
        Args:
            report1_path: Path to first report
            report2_path: Path to second report
            
        Returns:
            Comparison results
        """
        # Load reports
        with open(report1_path) as f:
            if report1_path.suffix == '.json':
                report1 = json.load(f)
            else:
                report1 = yaml.safe_load(f)

        with open(report2_path) as f:
            if report2_path.suffix == '.json':
                report2 = json.load(f)
            else:
                report2 = yaml.safe_load(f)

        # Compare overall scores
        score1 = report1['metadata'].get('overall_score', 0)
        score2 = report2['metadata'].get('overall_score', 0)
        score_change = score2 - score1

        # Compare validator statuses
        results1 = report1.get('results', {})
        results2 = report2.get('results', {})

        improved_validators = []
        degraded_validators = []
        new_validators = []
        removed_validators = []

        # Check each validator
        all_validators = set(results1.keys()) | set(results2.keys())

        for vid in all_validators:
            if vid in results1 and vid in results2:
                # Compare scores
                score1 = results1[vid].get('score', 0)
                score2 = results2[vid].get('score', 0)

                if score2 > score1:
                    improved_validators.append({
                        'id': vid,
                        'name': results2[vid].get('validator_name', vid),
                        'old_score': score1,
                        'new_score': score2,
                        'improvement': score2 - score1
                    })
                elif score2 < score1:
                    degraded_validators.append({
                        'id': vid,
                        'name': results2[vid].get('validator_name', vid),
                        'old_score': score1,
                        'new_score': score2,
                        'degradation': score1 - score2
                    })
            elif vid in results2:
                new_validators.append(vid)
            else:
                removed_validators.append(vid)

        return {
            'overall_score_change': score_change,
            'improved_validators': improved_validators,
            'degraded_validators': degraded_validators,
            'new_validators': new_validators,
            'removed_validators': removed_validators,
            'report1_date': report1['metadata'].get('timestamp'),
            'report2_date': report2['metadata'].get('timestamp')
        }


class ValidationReport:
    """Comprehensive validation report with multiple validators."""

    def __init__(
        self,
        run_id: str,
        pipeline_name: str,
        timestamp: datetime,
        mode: str = "standard"
    ):
        """Initialize validation report.
        
        Args:
            run_id: Unique run identifier
            pipeline_name: Name of the pipeline executed
            timestamp: Report timestamp
            mode: Validation mode used
        """
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.timestamp = timestamp
        self.mode = mode
        self.results: dict[str, ValidationResult] = {}
        self.duration_seconds: float = 0
        self.overall_status: str = "pending"
        self.overall_score: Decimal = Decimal("0")
        self.ready_for_production: bool = False
        self.blocking_issues: list[dict[str, Any]] = []
        self.overrides: dict[str, dict[str, Any]] = {}

    def add_result(self, validator_id: str, result: ValidationResult) -> None:
        """Add a validator result to the report.
        
        Args:
            validator_id: Validator identifier
            result: Validation result
        """
        self.results[validator_id] = result
        self._update_summary()

    def _update_summary(self) -> None:
        """Update report summary based on current results."""
        if not self.results:
            return

        # Calculate overall score
        total_score = sum(r.score for r in self.results.values())
        self.overall_score = total_score / Decimal(len(self.results))

        # Determine overall status
        has_failures = any(r.status == CheckStatus.FAILED for r in self.results.values())
        has_warnings = any(r.status == CheckStatus.WARNING for r in self.results.values())
        all_passed = all(r.status == CheckStatus.PASSED for r in self.results.values())

        if all_passed:
            self.overall_status = "passed"
        elif has_failures:
            self.overall_status = "failed"
        elif has_warnings:
            self.overall_status = "warning"
        else:
            self.overall_status = "mixed"

        # Check for blocking issues
        self.blocking_issues = []
        for vid, result in self.results.items():
            if result.has_blocking_failures():
                for check in result.checks:
                    if check.is_blocking and check.status in [CheckStatus.FAILED, CheckStatus.ERROR]:
                        self.blocking_issues.append({
                            'validator_id': vid,
                            'validator_name': result.validator_name,
                            'check_name': check.name,
                            'message': check.details,
                            'severity': check.severity
                        })

        # Determine production readiness
        self.ready_for_production = (
            self.overall_score >= Decimal("80") and
            len(self.blocking_issues) == 0 and
            not has_failures
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'metadata': {
                'run_id': self.run_id,
                'pipeline_name': self.pipeline_name,
                'timestamp': self.timestamp.isoformat(),
                'mode': self.mode,
                'duration_seconds': self.duration_seconds,
                'environment': 'production',
                'overall_status': self.overall_status,
                'overall_score': float(self.overall_score),
                'validators_run': len(self.results),
                'validators_passed': sum(1 for r in self.results.values() if r.status == CheckStatus.PASSED),
                'validators_failed': sum(1 for r in self.results.values() if r.status == CheckStatus.FAILED),
                'validators_warning': sum(1 for r in self.results.values() if r.status == CheckStatus.WARNING),
                'validators_skipped': sum(1 for r in self.results.values() if r.status == CheckStatus.SKIPPED),
                'ready_for_production': self.ready_for_production,
                'overrides': self.overrides
            },
            'blocking_issues': self.blocking_issues,
            'results': {
                vid: result.to_dict()
                for vid, result in self.results.items()
            }
        }
