"""Comprehensive report generation for validation results."""

import json
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Template

from genesis.validation.decision import GoLiveDecision
from genesis.validation.history import ValidationHistory
from genesis.validation.orchestrator import ValidationReport

logger = structlog.get_logger(__name__)


class ReportGenerator:
    """Generates validation reports in multiple formats."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize report generator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.reports_dir = self.genesis_root / "docs" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.history = ValidationHistory(genesis_root)

    def generate_markdown(
        self,
        report: ValidationReport,
        decision: GoLiveDecision | None = None
    ) -> str:
        """Generate Markdown report for documentation.
        
        Args:
            report: Validation report
            decision: Optional go-live decision
            
        Returns:
            Markdown formatted report
        """
        md_lines = []

        # Header
        md_lines.append("# Go-Live Readiness Report")
        md_lines.append("")
        md_lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        md_lines.append(f"**Pipeline:** {report.pipeline_name}")
        md_lines.append(f"**Duration:** {report.duration_seconds:.1f} seconds")
        md_lines.append("")

        # Executive Summary
        md_lines.append("## Executive Summary")
        md_lines.append("")

        status_icon = "✅" if report.ready else "❌"
        status_text = "**READY FOR DEPLOYMENT**" if report.ready else "**NOT READY FOR DEPLOYMENT**"
        md_lines.append(f"{status_icon} {status_text}")
        md_lines.append("")

        md_lines.append(f"- **Overall Score:** {report.overall_score:.1f}%")
        md_lines.append(f"- **Validators Passed:** {sum(1 for r in report.results if r.passed)}/{len(report.results)}")
        md_lines.append(f"- **Blocking Issues:** {len(report.blocking_issues)}")
        md_lines.append("")

        # Go-Live Decision
        if decision:
            md_lines.append("## Go-Live Decision")
            md_lines.append("")
            md_lines.append(f"- **Decision:** {'GO' if decision.ready else 'NO-GO'}")
            md_lines.append(f"- **Deployment Target:** {decision.deployment_target.value if decision.deployment_target else 'Not specified'}")
            md_lines.append(f"- **Deployment Allowed:** {'Yes' if decision.deployment_allowed else 'No'}")

            if decision.override:
                md_lines.append("- **Override Applied:** Yes")
                md_lines.append(f"  - **Authorized By:** {decision.override.authorized_by}")
                md_lines.append(f"  - **Reason:** {decision.override.reason}")
            md_lines.append("")

        # Validation Summary Table
        md_lines.append("## Validation Summary")
        md_lines.append("")
        md_lines.append("| Category | Validator | Status | Score | Duration |")
        md_lines.append("|----------|-----------|--------|-------|----------|")

        for result in report.results:
            status = "✅ Pass" if result.passed else "❌ Fail"
            md_lines.append(
                f"| {result.category} | {result.validator_name} | {status} | "
                f"{result.score:.1f}% | {result.duration_seconds:.2f}s |"
            )
        md_lines.append("")

        # Blocking Issues
        if report.blocking_issues:
            md_lines.append("## 🚨 Blocking Issues")
            md_lines.append("")
            md_lines.append("These issues must be resolved before deployment:")
            md_lines.append("")

            for i, issue in enumerate(report.blocking_issues, 1):
                md_lines.append(f"{i}. **{issue.name}**")
                md_lines.append(f"   - {issue.message}")
                md_lines.append(f"   - Severity: {issue.severity}")
                md_lines.append("")

        # Detailed Results
        md_lines.append("## Detailed Validation Results")
        md_lines.append("")

        for category in ["technical", "security", "operational", "business"]:
            category_results = [r for r in report.results if r.category == category]

            if category_results:
                md_lines.append(f"### {category.capitalize()} Validators")
                md_lines.append("")

                for result in category_results:
                    md_lines.append(f"#### {result.validator_name}")
                    md_lines.append("")
                    md_lines.append(f"- **Status:** {'PASSED' if result.passed else 'FAILED'}")
                    md_lines.append(f"- **Score:** {result.score:.1f}%")
                    md_lines.append(f"- **Duration:** {result.duration_seconds:.2f} seconds")

                    if result.errors:
                        md_lines.append("- **Errors:**")
                        for error in result.errors[:5]:  # Limit to 5 errors
                            md_lines.append(f"  - {error}")

                    if result.warnings:
                        md_lines.append("- **Warnings:**")
                        for warning in result.warnings[:5]:  # Limit to 5 warnings
                            md_lines.append(f"  - {warning}")

                    md_lines.append("")

        # Historical Trend
        md_lines.append("## Historical Trend")
        md_lines.append("")

        try:
            trend_graph = self.history.generate_trend_graph(report.pipeline_name, days=7)
            md_lines.append("```")
            md_lines.append(trend_graph)
            md_lines.append("```")
        except Exception as e:
            md_lines.append(f"_Unable to generate trend graph: {e!s}_")

        md_lines.append("")

        # Recommendations
        md_lines.append("## Recommendations")
        md_lines.append("")

        recommendations = self._generate_recommendations(report)
        for i, rec in enumerate(recommendations, 1):
            md_lines.append(f"{i}. {rec}")

        md_lines.append("")

        # Footer
        md_lines.append("---")
        md_lines.append("_This report was automatically generated by the Genesis Validation Framework._")

        return "\n".join(md_lines)

    def generate_json(
        self,
        report: ValidationReport,
        decision: GoLiveDecision | None = None
    ) -> str:
        """Generate JSON report for API consumption.
        
        Args:
            report: Validation report
            decision: Optional go-live decision
            
        Returns:
            JSON formatted report
        """
        report_dict = report.to_dict()

        if decision:
            report_dict["decision"] = decision.to_dict()

        # Add metadata
        report_dict["metadata"] = {
            "generator": "Genesis Validation Framework",
            "version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat()
        }

        # Add recommendations
        report_dict["recommendations"] = self._generate_recommendations(report)

        return json.dumps(report_dict, indent=2, default=str)

    def generate_html(
        self,
        report: ValidationReport,
        decision: GoLiveDecision | None = None
    ) -> str:
        """Generate HTML report for web viewing.
        
        Args:
            report: Validation report
            decision: Optional go-live decision
            
        Returns:
            HTML formatted report
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Go-Live Readiness Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .summary-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .status-ready {
            color: #10b981;
            font-weight: bold;
        }
        .status-not-ready {
            color: #ef4444;
            font-weight: bold;
        }
        .score-high { color: #10b981; }
        .score-medium { color: #f59e0b; }
        .score-low { color: #ef4444; }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 1rem 0;
        }
        th {
            background: #f3f4f6;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 0.75rem;
            border-top: 1px solid #e5e7eb;
        }
        .pass { color: #10b981; }
        .fail { color: #ef4444; }
        .warning { color: #f59e0b; }
        .blocking-issue {
            background: #fee2e2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            margin: 1rem 0;
        }
        .recommendation {
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Go-Live Readiness Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Pipeline: {{ pipeline_name }}</p>
    </div>
    
    <div class="summary-card">
        <h2>Executive Summary</h2>
        <p class="{{ status_class }}">{{ status_text }}</p>
        <p>Overall Score: <span class="{{ score_class }}">{{ overall_score }}%</span></p>
        <p>Validators Passed: {{ passed_count }}/{{ total_count }}</p>
        <p>Blocking Issues: {{ blocking_count }}</p>
    </div>
    
    {% if decision %}
    <div class="summary-card">
        <h2>Deployment Decision</h2>
        <p>Decision: <strong>{{ decision_text }}</strong></p>
        <p>Target: {{ deployment_target }}</p>
        {% if override %}
        <p class="warning">Override Applied by {{ override_user }}</p>
        {% endif %}
    </div>
    {% endif %}
    
    <div class="summary-card">
        <h2>Validation Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Validator</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>
                {{ results_table }}
            </tbody>
        </table>
    </div>
    
    {% if blocking_issues %}
    <div class="summary-card">
        <h2>🚨 Blocking Issues</h2>
        {{ blocking_issues_html }}
    </div>
    {% endif %}
    
    <div class="summary-card">
        <h2>Recommendations</h2>
        {{ recommendations_html }}
    </div>
</body>
</html>
        """

        template = Template(html_template)

        # Prepare template variables
        status_class = "status-ready" if report.ready else "status-not-ready"
        status_text = "✅ READY FOR DEPLOYMENT" if report.ready else "❌ NOT READY FOR DEPLOYMENT"

        score_class = "score-high" if report.overall_score >= 90 else "score-medium" if report.overall_score >= 70 else "score-low"

        # Build results table
        results_table = []
        for result in report.results:
            status = '<span class="pass">✅ Pass</span>' if result.passed else '<span class="fail">❌ Fail</span>'
            score_class = "score-high" if result.score >= 90 else "score-medium" if result.score >= 70 else "score-low"
            results_table.append(f"""
                <tr>
                    <td>{result.category}</td>
                    <td>{result.validator_name}</td>
                    <td>{status}</td>
                    <td class="{score_class}">{result.score:.1f}%</td>
                    <td>{result.duration_seconds:.2f}s</td>
                </tr>
            """)

        # Build blocking issues
        blocking_issues_html = ""
        if report.blocking_issues:
            for issue in report.blocking_issues:
                blocking_issues_html += f"""
                    <div class="blocking-issue">
                        <strong>{issue.name}</strong><br>
                        {issue.message}<br>
                        <small>Severity: {issue.severity}</small>
                    </div>
                """

        # Build recommendations
        recommendations = self._generate_recommendations(report)
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"""
                <div class="recommendation">
                    {rec}
                </div>
            """

        # Render template
        html = template.render(
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            pipeline_name=report.pipeline_name,
            status_class=status_class,
            status_text=status_text,
            score_class=score_class,
            overall_score=f"{report.overall_score:.1f}",
            passed_count=sum(1 for r in report.results if r.passed),
            total_count=len(report.results),
            blocking_count=len(report.blocking_issues),
            decision=decision,
            decision_text="GO" if decision and decision.ready else "NO-GO",
            deployment_target=decision.deployment_target.value if decision and decision.deployment_target else "Not specified",
            override=decision.override if decision else None,
            override_user=decision.override.authorized_by if decision and decision.override else "",
            results_table="".join(results_table),
            blocking_issues=report.blocking_issues,
            blocking_issues_html=blocking_issues_html,
            recommendations_html=recommendations_html
        )

        return html

    def generate_pdf(
        self,
        report: ValidationReport,
        decision: GoLiveDecision | None = None,
        output_path: Path | None = None
    ) -> Path:
        """Generate PDF report (simplified version without reportlab).
        
        Args:
            report: Validation report
            decision: Optional go-live decision
            output_path: Optional output path
            
        Returns:
            Path to generated text file (PDF generation requires reportlab)
        """
        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = self.reports_dir / f"validation_report_{timestamp}.txt"

        # For now, save as text file (PDF generation would require reportlab)
        content = self.generate_markdown(report, decision)
        output_path.write_text(content)

        logger.info("Report saved as text (PDF generation requires reportlab)", path=str(output_path))
        return output_path

    def save_all_formats(
        self,
        report: ValidationReport,
        decision: GoLiveDecision | None = None
    ) -> dict[str, Path]:
        """Save report in all available formats.
        
        Args:
            report: Validation report
            decision: Optional go-live decision
            
        Returns:
            Dictionary of format to file path
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"validation_report_{timestamp}"

        paths = {}

        # Save Markdown
        md_path = self.reports_dir / f"{base_name}.md"
        md_content = self.generate_markdown(report, decision)
        md_path.write_text(md_content)
        paths["markdown"] = md_path

        # Save JSON
        json_path = self.reports_dir / f"{base_name}.json"
        json_content = self.generate_json(report, decision)
        json_path.write_text(json_content)
        paths["json"] = json_path

        # Save HTML
        html_path = self.reports_dir / f"{base_name}.html"
        html_content = self.generate_html(report, decision)
        html_path.write_text(html_content)
        paths["html"] = html_path

        # Save PDF (as text for now)
        pdf_path = self.generate_pdf(report, decision)
        paths["pdf"] = pdf_path

        logger.info("Reports saved in all formats", paths=paths)
        return paths

    def distribute_report(
        self,
        report_paths: dict[str, Path],
        recipients: list[str],
        smtp_config: dict[str, Any] | None = None
    ):
        """Distribute reports via email.
        
        Args:
            report_paths: Dictionary of format to file path
            recipients: List of email addresses
            smtp_config: SMTP configuration
        """
        if not smtp_config:
            logger.warning("No SMTP config provided, skipping email distribution")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f"Go-Live Readiness Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
            msg['From'] = smtp_config.get('from_address', 'genesis@example.com')
            msg['To'] = ', '.join(recipients)

            # Body
            body = """
            The latest Go-Live Readiness Report is attached.
            
            Please review the report and take appropriate action based on the findings.
            
            This is an automated message from the Genesis Validation Framework.
            """
            msg.attach(MIMEText(body, 'plain'))

            # Attach reports
            for format_name, path in report_paths.items():
                if path.exists():
                    with open(path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment.add_header(
                            'Content-Disposition',
                            'attachment',
                            filename=path.name
                        )
                        msg.attach(attachment)

            # Send email
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                if smtp_config.get('use_tls'):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)

            logger.info("Reports distributed via email", recipients=recipients)

        except Exception as e:
            logger.error("Failed to distribute reports", error=str(e))

    def _generate_recommendations(self, report: ValidationReport) -> list[str]:
        """Generate recommendations based on validation results.
        
        Args:
            report: Validation report
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Check overall score
        if report.overall_score < 95:
            recommendations.append(
                f"Improve overall validation score from {report.overall_score:.1f}% to at least 95% for production deployment"
            )

        # Check for failed validators
        failed_validators = [r for r in report.results if not r.passed]
        if failed_validators:
            validator_names = ", ".join(v.validator_name for v in failed_validators[:3])
            recommendations.append(
                f"Address failures in validators: {validator_names}"
            )

        # Check for blocking issues
        if report.blocking_issues:
            recommendations.append(
                f"Resolve {len(report.blocking_issues)} blocking issue(s) before deployment"
            )

        # Check specific categories
        for category in ["security", "operational"]:
            category_results = [r for r in report.results if r.category == category]
            if category_results:
                avg_score = sum(r.score for r in category_results) / len(category_results)
                if avg_score < 90:
                    recommendations.append(
                        f"Improve {category} validation scores (current average: {avg_score:.1f}%)"
                    )

        # Check for slow validators
        slow_validators = [r for r in report.results if r.duration_seconds > 60]
        if slow_validators:
            recommendations.append(
                "Optimize slow validators to improve validation pipeline performance"
            )

        # If all good
        if not recommendations:
            recommendations.append("System appears ready for deployment. Proceed with standard deployment procedures.")

        return recommendations
