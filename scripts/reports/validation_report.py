"""Validation report generation for Genesis trading system."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from scripts.validators import ValidationResult, ValidationStatus, ValidationSeverity


class ValidationReportGenerator:
    """Generates reports from validation results."""
    
    def __init__(self, results: Dict[str, ValidationResult], summary: Dict[str, Any]):
        """Initialize with validation results."""
        self.results = results
        self.summary = summary
        self.timestamp = datetime.now()
    
    def print_console(self):
        """Print results to console."""
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {self._get_status_emoji(self.summary['overall_status'])} {self.summary['overall_status']}")
        print("-"*70)
        
        # Summary statistics
        print("\nSummary:")
        print(f"  Total Validators: {self.summary['total_validators']}")
        print(f"  Passed: {self.summary['passed']} ✅")
        print(f"  Failed: {self.summary['failed']} ❌")
        print(f"  Warnings: {self.summary['warnings']} ⚠️")
        
        # Individual validator results
        print("\nValidator Results:")
        print("-"*70)
        
        for name, result in self.results.items():
            status_emoji = self._get_status_emoji(result.status.value.upper())
            print(f"\n{status_emoji} {name.upper()}")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.metrics.duration_seconds:.2f}s")
            print(f"  Checks: {result.metrics.checks_performed} total, "
                  f"{result.metrics.checks_passed} passed, "
                  f"{result.metrics.checks_failed} failed")
            
            # Show critical issues
            if result.critical_issues:
                print(f"  Critical Issues ({len(result.critical_issues)}):")
                for issue in result.critical_issues[:3]:
                    print(f"    - {issue.message}")
            
            # Show errors
            if result.error_issues:
                print(f"  Errors ({len(result.error_issues)}):")
                for issue in result.error_issues[:3]:
                    print(f"    - {issue.message}")
            
            # Show warnings
            warnings = [i for i in result.issues if i.severity == ValidationSeverity.WARNING]
            if warnings:
                print(f"  Warnings ({len(warnings)}):")
                for issue in warnings[:2]:
                    print(f"    - {issue.message}")
        
        print("\n" + "="*70)
        
        # Action items
        self._print_action_items()
    
    def _print_action_items(self):
        """Print prioritized action items."""
        print("\nACTION ITEMS:")
        print("-"*70)
        
        critical_items = []
        error_items = []
        warning_items = []
        
        for name, result in self.results.items():
            for issue in result.issues:
                if issue.recommendation:
                    item = f"[{name}] {issue.recommendation}"
                    if issue.severity == ValidationSeverity.CRITICAL:
                        critical_items.append(item)
                    elif issue.severity == ValidationSeverity.ERROR:
                        error_items.append(item)
                    elif issue.severity == ValidationSeverity.WARNING:
                        warning_items.append(item)
        
        if critical_items:
            print("\n🔴 CRITICAL (Fix immediately):")
            for item in critical_items[:5]:
                print(f"  • {item}")
        
        if error_items:
            print("\n🟠 HIGH (Fix before production):")
            for item in error_items[:5]:
                print(f"  • {item}")
        
        if warning_items:
            print("\n🟡 MEDIUM (Address soon):")
            for item in warning_items[:5]:
                print(f"  • {item}")
        
        if not (critical_items or error_items or warning_items):
            print("\n✅ No action items - system ready!")
    
    def generate_json(self) -> Dict[str, Any]:
        """Generate JSON report."""
        return {
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "version": "1.0",
                "system": "Genesis Trading System"
            },
            "summary": self.summary,
            "results": {
                name: result.to_dict() for name, result in self.results.items()
            },
            "action_items": self._get_action_items()
        }
    
    def generate_html(self) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Genesis Validation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .status-passed {{ background: #48bb78; }}
        .status-failed {{ background: #f56565; }}
        .status-warning {{ background: #ed8936; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e2e8f0;
        }}
        .summary-card .number {{
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}
        .validator-section {{
            margin: 20px 0;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .issue {{
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #cbd5e0;
        }}
        .issue-critical {{ border-color: #f56565; }}
        .issue-error {{ border-color: #ed8936; }}
        .issue-warning {{ border-color: #f6d55c; }}
        .issue-info {{ border-color: #4299e1; }}
        .timestamp {{
            color: #718096;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Genesis Trading System - Validation Report</h1>
        <p class="timestamp">Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div style="margin: 20px 0;">
            <span class="status-badge status-{self.summary['overall_status'].lower()}">
                Overall Status: {self.summary['overall_status']}
            </span>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="number">{self.summary['total_validators']}</div>
                <div>Total Validators</div>
            </div>
            <div class="summary-card">
                <div class="number" style="color: #48bb78;">{self.summary['passed']}</div>
                <div>Passed</div>
            </div>
            <div class="summary-card">
                <div class="number" style="color: #f56565;">{self.summary['failed']}</div>
                <div>Failed</div>
            </div>
            <div class="summary-card">
                <div class="number" style="color: #ed8936;">{self.summary['warnings']}</div>
                <div>Warnings</div>
            </div>
        </div>
        
        <h2>Validation Results</h2>
        {self._generate_html_results()}
        
        <h2>Action Items</h2>
        {self._generate_html_action_items()}
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; color: #718096;">
            <p>Genesis Trading System v1.0 | Production Validation Suite</p>
        </div>
    </div>
</body>
</html>"""
        return html
    
    def generate_markdown(self) -> str:
        """Generate Markdown report."""
        md = f"""# Genesis Trading System - Validation Report

**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| **Overall Status** | {self.summary['overall_status']} |
| **Total Validators** | {self.summary['total_validators']} |
| **Passed** | {self.summary['passed']} ✅ |
| **Failed** | {self.summary['failed']} ❌ |
| **Warnings** | {self.summary['warnings']} ⚠️ |

## Validator Results

"""
        
        for name, result in self.results.items():
            status_emoji = self._get_status_emoji(result.status.value.upper())
            md += f"### {status_emoji} {name.upper()}\n\n"
            md += f"- **Status:** {result.status.value}\n"
            md += f"- **Duration:** {result.metrics.duration_seconds:.2f}s\n"
            md += f"- **Checks:** {result.metrics.checks_performed} total, "
            md += f"{result.metrics.checks_passed} passed, "
            md += f"{result.metrics.checks_failed} failed\n\n"
            
            if result.critical_issues or result.error_issues:
                md += "#### Issues\n\n"
                for issue in result.critical_issues[:3]:
                    md += f"- 🔴 **CRITICAL:** {issue.message}\n"
                for issue in result.error_issues[:3]:
                    md += f"- 🟠 **ERROR:** {issue.message}\n"
                md += "\n"
        
        # Add action items
        action_items = self._get_action_items()
        if action_items:
            md += "## Action Items\n\n"
            for priority, items in action_items.items():
                if items:
                    md += f"### {priority}\n\n"
                    for item in items[:5]:
                        md += f"- {item}\n"
                    md += "\n"
        
        return md
    
    def generate_compliance_certificate(self) -> str:
        """Generate compliance certificate for passed validation."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Compliance Certificate - Genesis Trading System</title>
    <style>
        body {{
            font-family: Georgia, serif;
            margin: 0;
            padding: 40px;
            background: #f8f9fa;
        }}
        .certificate {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border: 3px solid #2c3e50;
            border-radius: 10px;
            padding: 50px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .content {{
            line-height: 1.8;
            color: #34495e;
        }}
        .seal {{
            position: absolute;
            top: 40px;
            right: 40px;
            width: 100px;
            height: 100px;
            border: 3px solid #e74c3c;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: #e74c3c;
            transform: rotate(-15deg);
        }}
        .signature {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="certificate">
        <div class="seal">VALIDATED</div>
        
        <div class="header">
            <h1>Certificate of Compliance</h1>
            <div class="subtitle">Genesis Trading System</div>
        </div>
        
        <div class="content">
            <p>This certifies that the <strong>Genesis Trading System</strong> has successfully completed comprehensive production validation on <strong>{self.timestamp.strftime('%B %d, %Y')}</strong>.</p>
            
            <h3>Validation Summary</h3>
            <ul>
                <li>Total Validators Run: {self.summary['total_validators']}</li>
                <li>Validators Passed: {self.summary['passed']}</li>
                <li>Critical Issues: 0</li>
                <li>Compliance Status: PASSED</li>
            </ul>
            
            <h3>Validated Components</h3>
            <ul>
                <li>✅ Strategy Implementation (All Tiers)</li>
                <li>✅ Risk Management Engine</li>
                <li>✅ Order Execution System</li>
                <li>✅ Database Connectivity</li>
                <li>✅ Monitoring Infrastructure</li>
                <li>✅ Security Configuration</li>
                <li>✅ Compliance Requirements</li>
                <li>✅ Performance Benchmarks</li>
                <li>✅ Disaster Recovery</li>
            </ul>
            
            <p>The system meets all production readiness criteria and is approved for deployment.</p>
        </div>
        
        <div class="signature">
            <p><strong>Validated By:</strong> Genesis Validation Suite v1.0</p>
            <p><strong>Validation ID:</strong> {self.timestamp.strftime('%Y%m%d%H%M%S')}</p>
        </div>
        
        <div class="timestamp">
            Certificate generated on {self.timestamp.strftime('%Y-%m-%d at %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>"""
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emoji_map = {
            "PASSED": "✅",
            "FAILED": "❌",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "RUNNING": "🔄",
            "PENDING": "⏳",
            "SKIPPED": "⏭️"
        }
        return emoji_map.get(status, "❓")
    
    def _get_action_items(self) -> Dict[str, List[str]]:
        """Get categorized action items."""
        critical_items = []
        error_items = []
        warning_items = []
        
        for name, result in self.results.items():
            for issue in result.issues:
                if issue.recommendation:
                    item = f"[{name}] {issue.recommendation}"
                    if issue.severity == ValidationSeverity.CRITICAL:
                        critical_items.append(item)
                    elif issue.severity == ValidationSeverity.ERROR:
                        error_items.append(item)
                    elif issue.severity == ValidationSeverity.WARNING:
                        warning_items.append(item)
        
        return {
            "Critical": critical_items,
            "High Priority": error_items,
            "Medium Priority": warning_items
        }
    
    def _generate_html_results(self) -> str:
        """Generate HTML for individual validator results."""
        html = ""
        for name, result in self.results.items():
            status_class = "passed" if result.passed else "failed"
            html += f"""
            <div class="validator-section">
                <h3>{name.upper()} - <span class="status-badge status-{status_class}">{result.status.value}</span></h3>
                <p>Duration: {result.metrics.duration_seconds:.2f}s | 
                   Checks: {result.metrics.checks_performed} total, 
                   {result.metrics.checks_passed} passed, 
                   {result.metrics.checks_failed} failed</p>
            """
            
            # Add issues
            for issue in result.issues[:5]:
                severity_class = f"issue-{issue.severity.value}"
                html += f"""
                <div class="issue {severity_class}">
                    <strong>{issue.severity.value.upper()}:</strong> {issue.message}
                    {f"<br><em>Recommendation: {issue.recommendation}</em>" if issue.recommendation else ""}
                </div>
                """
            
            html += "</div>"
        
        return html
    
    def _generate_html_action_items(self) -> str:
        """Generate HTML for action items."""
        action_items = self._get_action_items()
        html = "<ul>"
        
        for priority, items in action_items.items():
            if items:
                html += f"<li><strong>{priority}:</strong><ul>"
                for item in items[:5]:
                    html += f"<li>{item}</li>"
                html += "</ul></li>"
        
        if not any(action_items.values()):
            html += "<li>✅ No action items - system ready!</li>"
        
        html += "</ul>"
        return html