#!/usr/bin/env python3
"""
Production validation script for Genesis trading system.

Runs comprehensive validation checks to ensure system readiness for production.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.validators import ValidationOrchestrator
from scripts.validators.strategy_validator import StrategyValidator
from scripts.validators.risk_validator import RiskValidator
from scripts.validators.execution_validator import ExecutionValidator
from scripts.validators.database_validator import DatabaseValidator
from scripts.validators.monitoring_validator import MonitoringValidator
from scripts.validators.security_validator import SecurityValidator
from scripts.validators.compliance_validator import ComplianceValidator
from scripts.validators.performance_validator import PerformanceValidator
from scripts.validators.disaster_recovery_validator import DisasterRecoveryValidator
from scripts.reports.validation_report import ValidationReportGenerator


async def main(
    mode: str = "standard",
    validators: Optional[list] = None,
    output_format: str = "console",
    output_file: Optional[str] = None,
    config_path: Optional[str] = None
):
    """Run production validation suite."""
    print(f"\n{'='*60}")
    print(f"Genesis Production Validation Suite")
    print(f"Mode: {mode} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Load configuration
    config_file = Path(config_path) if config_path else Path("scripts/config/validation_criteria.yaml")
    
    # Initialize orchestrator
    orchestrator = ValidationOrchestrator(config_file)
    
    # Register all validators
    print("Registering validators...")
    orchestrator.register_validator(StrategyValidator())
    orchestrator.register_validator(RiskValidator())
    orchestrator.register_validator(ExecutionValidator())
    orchestrator.register_validator(DatabaseValidator())
    orchestrator.register_validator(MonitoringValidator())
    orchestrator.register_validator(SecurityValidator())
    orchestrator.register_validator(ComplianceValidator())
    orchestrator.register_validator(PerformanceValidator())
    orchestrator.register_validator(DisasterRecoveryValidator())
    print(f"Registered {len(orchestrator.validators)} validators\n")
    
    # Run validation
    print(f"Running validation in '{mode}' mode...")
    if validators:
        print(f"Validators: {', '.join(validators)}")
    print()
    
    try:
        results = await orchestrator.validate(
            mode=mode,
            validators=validators
        )
        
        # Get summary
        summary = orchestrator.get_summary()
        
        # Generate report
        report_generator = ValidationReportGenerator(results, summary)
        
        # Output results
        if output_format == "console":
            report_generator.print_console()
        elif output_format == "json":
            json_report = report_generator.generate_json()
            if output_file:
                with open(output_file, "w") as f:
                    json.dump(json_report, f, indent=2)
                print(f"\nJSON report saved to: {output_file}")
            else:
                print(json.dumps(json_report, indent=2))
        elif output_format == "html":
            html_report = report_generator.generate_html()
            if output_file:
                with open(output_file, "w") as f:
                    f.write(html_report)
                print(f"\nHTML report saved to: {output_file}")
            else:
                print(html_report)
        elif output_format == "markdown":
            md_report = report_generator.generate_markdown()
            if output_file:
                with open(output_file, "w") as f:
                    f.write(md_report)
                print(f"\nMarkdown report saved to: {output_file}")
            else:
                print(md_report)
        
        # Generate compliance certificate if all passed
        if summary["overall_status"] == "PASSED" and mode == "thorough":
            cert_path = Path("scripts/reports/compliance_certificate.html")
            cert_html = report_generator.generate_compliance_certificate()
            cert_path.write_text(cert_html)
            print(f"\n[OK] Compliance certificate generated: {cert_path}")
        
        # Determine exit code
        if summary["overall_status"] == "PASSED":
            if summary["warnings"] > 0:
                sys.exit(1)  # Warning
            else:
                sys.exit(0)  # Success
        else:
            critical_count = sum(
                len(r.critical_issues) for r in results.values()
            )
            if critical_count > 0:
                sys.exit(3)  # Critical
            else:
                sys.exit(2)  # Failure
    
    except Exception as e:
        print(f"\n[ERROR] Validation suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genesis Production Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Modes:
  quick     - Quick smoke tests (5 minutes)
  standard  - Standard validation (30 minutes)
  thorough  - Complete validation with load tests (2 hours)
  security  - Security-focused validation (1 hour)
  performance - Performance benchmarks (45 minutes)

Exit Codes:
  0 - All validations passed
  1 - Passed with warnings
  2 - Validation failures
  3 - Critical failures or errors

Examples:
  # Run standard validation
  python validate_production.py
  
  # Run quick smoke tests
  python validate_production.py --mode quick
  
  # Run specific validators
  python validate_production.py --validators strategy risk database
  
  # Generate HTML report
  python validate_production.py --format html --output report.html
  
  # Run thorough validation for production release
  python validate_production.py --mode thorough --format html
        """
    )
    
    parser.add_argument(
        "--mode",
        "-m",
        choices=["quick", "standard", "thorough", "security", "performance"],
        default="standard",
        help="Validation mode to run"
    )
    
    parser.add_argument(
        "--validators",
        "-v",
        nargs="+",
        choices=[
            "strategy", "risk", "execution", "database", "monitoring",
            "security", "compliance", "performance", "disaster_recovery"
        ],
        help="Specific validators to run"
    )
    
    parser.add_argument(
        "--format",
        "-f",
        choices=["console", "json", "html", "markdown"],
        default="console",
        help="Output format for results"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (for non-console formats)"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Path to custom validation configuration"
    )
    
    args = parser.parse_args()
    
    # Run validation
    asyncio.run(main(
        mode=args.mode,
        validators=args.validators,
        output_format=args.format,
        output_file=args.output,
        config_path=args.config
    ))