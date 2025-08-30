#!/usr/bin/env python3
"""Production readiness validation script."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from decimal import Decimal

import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.validation.test_validator import TestValidator
from genesis.validation.stability_tester import StabilityTester
from genesis.validation.security_scanner import SecurityScanner
from genesis.validation.performance_validator import PerformanceValidator
from genesis.validation.dr_validator import DisasterRecoveryValidator
from genesis.validation.paper_trading_validator import PaperTradingValidator

logger = structlog.get_logger(__name__)
console = Console()


class ProductionReadinessValidator:
    """Comprehensive production readiness validation."""
    
    def __init__(self):
        self.validators = {
            "test_coverage": TestValidator(),
            "stability": StabilityTester(),
            "security": SecurityScanner(),
            "performance": PerformanceValidator(),
            "disaster_recovery": DisasterRecoveryValidator(),
            "paper_trading": PaperTradingValidator(),
        }
        self.results: Dict[str, Dict[str, Any]] = {}
        self.checklist_items = [
            {
                "id": "AC1",
                "name": "Unit Tests",
                "description": "All unit tests passing (>90% coverage)",
                "validator": "test_coverage",
                "required": True,
            },
            {
                "id": "AC2",
                "name": "Integration Tests",
                "description": "Integration test suite 100% green",
                "validator": "test_coverage",
                "required": True,
            },
            {
                "id": "AC3",
                "name": "Stability Test",
                "description": "48-hour stability test completed",
                "validator": "stability",
                "required": True,
            },
            {
                "id": "AC4",
                "name": "Security Scan",
                "description": "Security scan with zero critical issues",
                "validator": "security",
                "required": True,
            },
            {
                "id": "AC5",
                "name": "Performance",
                "description": "Performance benchmarks met (<50ms p99 latency)",
                "validator": "performance",
                "required": True,
            },
            {
                "id": "AC6",
                "name": "Disaster Recovery",
                "description": "Disaster recovery tested successfully",
                "validator": "disaster_recovery",
                "required": True,
            },
            {
                "id": "AC7",
                "name": "Operations Ready",
                "description": "Operations team trained on runbooks",
                "validator": None,  # Manual verification
                "required": True,
            },
            {
                "id": "AC8",
                "name": "Paper Trading",
                "description": "$10,000 paper trading profit demonstrated",
                "validator": "paper_trading",
                "required": True,
            },
            {
                "id": "AC9",
                "name": "Legal Review",
                "description": "Legal review of terms and compliance",
                "validator": None,  # Manual verification
                "required": True,
            },
            {
                "id": "AC10",
                "name": "Insurance",
                "description": "Insurance coverage confirmed (E&O, Cyber)",
                "validator": None,  # Manual verification
                "required": True,
            },
        ]
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        console.print(Panel.fit(
            "[bold cyan]Production Readiness Validation[/bold cyan]\n"
            "Running comprehensive checks...",
            box=box.ROUNDED
        ))
        
        # Run automated validators
        for name, validator in track(
            self.validators.items(),
            description="Running validators...",
        ):
            try:
                result = await validator.validate()
                self.results[name] = result
                logger.info(f"Validator completed: {name}", result=result)
            except Exception as e:
                self.results[name] = {
                    "status": "error",
                    "error": str(e),
                    "passed": False,
                }
                logger.error(f"Validator failed: {name}", error=str(e))
        
        # Process checklist items
        checklist_results = []
        for item in self.checklist_items:
            if item["validator"]:
                # Get result from validator
                validator_result = self.results.get(item["validator"], {})
                
                # Determine if specific check passed
                passed = self._check_item_passed(item["id"], validator_result)
                status = "✅" if passed else "❌"
                details = validator_result.get("details", {})
            else:
                # Manual verification required
                status = "⚠️"
                passed = None
                details = {"note": "Manual verification required"}
            
            checklist_results.append({
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "status": status,
                "passed": passed,
                "required": item["required"],
                "details": details,
            })
        
        # Generate overall assessment
        assessment = self._generate_assessment(checklist_results)
        
        # Display results
        self._display_results(checklist_results, assessment)
        
        # Generate report
        report = self._generate_report(checklist_results, assessment)
        
        return {
            "checklist": checklist_results,
            "assessment": assessment,
            "report": report,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _check_item_passed(self, item_id: str, validator_result: Dict) -> bool:
        """Check if specific acceptance criteria passed."""
        if not validator_result.get("passed", False):
            return False
        
        # Specific checks per acceptance criteria
        if item_id == "AC1":
            # Unit tests with >90% coverage
            coverage = validator_result.get("details", {}).get("unit_coverage", 0)
            return coverage >= 90
        elif item_id == "AC2":
            # Integration tests 100% passing
            integration_pass_rate = validator_result.get("details", {}).get(
                "integration_pass_rate", 0
            )
            return integration_pass_rate == 100
        elif item_id == "AC3":
            # 48-hour stability test
            hours_stable = validator_result.get("details", {}).get("hours_stable", 0)
            return hours_stable >= 48
        elif item_id == "AC4":
            # No critical security issues
            critical_issues = validator_result.get("details", {}).get(
                "critical_issues", 0
            )
            return critical_issues == 0
        elif item_id == "AC5":
            # Performance p99 < 50ms
            p99_latency = validator_result.get("details", {}).get("p99_latency_ms", 999)
            return p99_latency < 50
        elif item_id == "AC6":
            # DR test passed
            return validator_result.get("passed", False)
        elif item_id == "AC8":
            # $10,000 paper trading profit
            profit = validator_result.get("details", {}).get("total_profit", 0)
            return profit >= 10000
        
        return validator_result.get("passed", False)
    
    def _generate_assessment(self, checklist_results: List[Dict]) -> Dict[str, Any]:
        """Generate overall go/no-go assessment."""
        required_items = [r for r in checklist_results if r["required"]]
        passed_items = [r for r in required_items if r["passed"] is True]
        manual_items = [r for r in required_items if r["passed"] is None]
        failed_items = [r for r in required_items if r["passed"] is False]
        
        # Calculate readiness score
        if manual_items:
            # Can't determine readiness with manual items pending
            readiness_score = None
            recommendation = "PENDING"
            reason = f"{len(manual_items)} manual verification items pending"
        elif failed_items:
            readiness_score = len(passed_items) / len(required_items) * 100
            recommendation = "NO-GO"
            reason = f"{len(failed_items)} required items failed"
        else:
            readiness_score = 100
            recommendation = "GO"
            reason = "All required items passed"
        
        return {
            "recommendation": recommendation,
            "readiness_score": readiness_score,
            "reason": reason,
            "passed_count": len(passed_items),
            "failed_count": len(failed_items),
            "manual_count": len(manual_items),
            "total_required": len(required_items),
        }
    
    def _display_results(self, checklist_results: List[Dict], assessment: Dict):
        """Display validation results in console."""
        # Create results table
        table = Table(title="Production Readiness Checklist", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Item", style="white")
        table.add_column("Description", style="dim")
        table.add_column("Details", style="dim")
        
        for item in checklist_results:
            details_str = ""
            if isinstance(item["details"], dict):
                for key, value in item["details"].items():
                    if key != "note":
                        details_str += f"{key}: {value}\n"
                    else:
                        details_str = value
            
            table.add_row(
                item["id"],
                item["status"],
                item["name"],
                item["description"],
                details_str.strip(),
            )
        
        console.print(table)
        
        # Display assessment
        recommendation = assessment["recommendation"]
        if recommendation == "GO":
            style = "bold green"
            icon = "✅"
        elif recommendation == "NO-GO":
            style = "bold red"
            icon = "❌"
        else:
            style = "bold yellow"
            icon = "⚠️"
        
        console.print(Panel.fit(
            f"{icon} [{ style}]Recommendation: {recommendation}[/{style}]\n"
            f"Reason: {assessment['reason']}\n"
            f"Readiness Score: {assessment['readiness_score']:.1f}%" if assessment['readiness_score'] else "Readiness Score: Pending\n",
            title="Go/No-Go Assessment",
            box=box.DOUBLE,
        ))
    
    def _generate_report(self, checklist_results: List[Dict], assessment: Dict) -> str:
        """Generate detailed HTML report."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Build HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Production Readiness Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #00a0cc; margin-top: 30px; }}
        .assessment {{ background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .go {{ border-left: 5px solid #00ff00; }}
        .no-go {{ border-left: 5px solid #ff0000; }}
        .pending {{ border-left: 5px solid #ffaa00; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #333; padding: 12px; text-align: left; border: 1px solid #555; }}
        td {{ padding: 10px; border: 1px solid #555; }}
        tr:nth-child(even) {{ background: #2a2a2a; }}
        .status-pass {{ color: #00ff00; font-weight: bold; }}
        .status-fail {{ color: #ff0000; font-weight: bold; }}
        .status-pending {{ color: #ffaa00; font-weight: bold; }}
        .details {{ font-size: 0.9em; color: #aaa; }}
        .summary {{ background: #333; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Production Readiness Report</h1>
    <p>Generated: {timestamp}</p>
    
    <div class="assessment {assessment['recommendation'].lower().replace('-', '')}">
        <h2>Assessment: {assessment['recommendation']}</h2>
        <div class="summary">
            <p><strong>Reason:</strong> {assessment['reason']}</p>
            <p><strong>Readiness Score:</strong> {assessment['readiness_score']:.1f}%</p>
            <p><strong>Items Passed:</strong> {assessment['passed_count']}/{assessment['total_required']}</p>
            <p><strong>Items Failed:</strong> {assessment['failed_count']}</p>
            <p><strong>Manual Verification Pending:</strong> {assessment['manual_count']}</p>
        </div>
    </div>
    
    <h2>Checklist Details</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Status</th>
            <th>Item</th>
            <th>Description</th>
            <th>Details</th>
        </tr>
"""
        
        for item in checklist_results:
            if item["passed"] is True:
                status_class = "status-pass"
                status_text = "PASS"
            elif item["passed"] is False:
                status_class = "status-fail"
                status_text = "FAIL"
            else:
                status_class = "status-pending"
                status_text = "PENDING"
            
            details_html = ""
            if isinstance(item["details"], dict):
                for key, value in item["details"].items():
                    details_html += f"<div>{key}: {value}</div>"
            
            html_content += f"""
        <tr>
            <td>{item['id']}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{item['name']}</td>
            <td>{item['description']}</td>
            <td class="details">{details_html}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Validator Results</h2>
"""
        
        for validator_name, result in self.results.items():
            status = "PASS" if result.get("passed") else "FAIL"
            html_content += f"""
    <div class="summary">
        <h3>{validator_name.replace('_', ' ').title()}</h3>
        <p><strong>Status:</strong> {status}</p>
"""
            if "error" in result:
                html_content += f"<p><strong>Error:</strong> {result['error']}</p>"
            if "details" in result:
                html_content += "<p><strong>Details:</strong></p><ul>"
                for key, value in result["details"].items():
                    html_content += f"<li>{key}: {value}</li>"
                html_content += "</ul>"
            html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        # Save report
        report_dir = Path("docs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"production_readiness_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        report_file.write_text(html_content)
        
        console.print(f"\n📄 Report saved to: {report_file}")
        
        return str(report_file)


async def main():
    """Main entry point."""
    validator = ProductionReadinessValidator()
    
    try:
        results = await validator.run_validation()
        
        # Write results to JSON for programmatic access
        results_file = Path("docs/reports/production_readiness_latest.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any Decimal values to string for JSON serialization
        def decimal_default(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError
        
        results_file.write_text(
            json.dumps(results, indent=2, default=decimal_default)
        )
        
        # Exit with appropriate code
        if results["assessment"]["recommendation"] == "GO":
            sys.exit(0)
        elif results["assessment"]["recommendation"] == "NO-GO":
            sys.exit(1)
        else:
            sys.exit(2)  # Pending manual verification
            
    except Exception as e:
        logger.error("Production readiness validation failed", error=str(e))
        console.print(f"[bold red]❌ Validation failed: {e}[/bold red]")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())