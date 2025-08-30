#!/usr/bin/env python3
"""Validation pipeline script for CI/CD integration."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add genesis to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.validation import ValidationOrchestrator


async def run_validation(
    validators: Optional[List[str]] = None,
    output_file: Optional[Path] = None,
    critical_only: bool = False,
    parallel: bool = True
) -> int:
    """Run validation pipeline.
    
    Args:
        validators: Specific validators to run (None = all)
        output_file: Output file for results
        critical_only: Run only critical validators
        parallel: Run validators in parallel
        
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    try:
        print("🚀 Starting validation pipeline...")
        orchestrator = ValidationOrchestrator()
        
        if critical_only:
            print("Running critical validators only...")
            results = await orchestrator.run_critical_validators()
        else:
            print(f"Running {'all' if not validators else ', '.join(validators)} validators...")
            results = await orchestrator.run_all_validators(parallel=parallel)
        
        # Save results if output file specified
        if output_file:
            await orchestrator.save_report(results, output_file)
            print(f"📝 Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if "summary" in results:
            print(results["summary"])
        
        # Print individual validator results
        if "validators" in results:
            print("\nValidator Results:")
            for name, validator in results["validators"].items():
                status = "✅" if validator.get("passed") else "❌"
                score = validator.get("score", 0)
                print(f"  {status} {name}: {score:.1f}%")
        
        # Determine exit code
        overall_passed = results.get("overall_passed", False)
        
        if overall_passed:
            print("\n✅ All validations PASSED!")
            return 0
        else:
            print("\n❌ Some validations FAILED!")
            
            # Print recommendations if available
            if "validators" in results:
                print("\n📋 Recommendations:")
                for name, validator in results["validators"].items():
                    if not validator.get("passed") and "recommendations" in validator:
                        print(f"\n{name}:")
                        for rec in validator["recommendations"][:3]:
                            print(f"  - {rec}")
            
            return 1
            
    except Exception as e:
        print(f"❌ Validation pipeline failed: {str(e)}")
        return 2


def main():
    """Main entry point for validation pipeline."""
    parser = argparse.ArgumentParser(description="Genesis Validation Pipeline")
    parser.add_argument(
        "--validators",
        nargs="+",
        choices=[
            "test_coverage",
            "stability",
            "security",
            "performance",
            "disaster_recovery",
            "paper_trading",
            "compliance",
            "operational"
        ],
        help="Specific validators to run"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Run only critical validators"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run validators sequentially instead of in parallel"
    )
    parser.add_argument(
        "--github-action",
        action="store_true",
        help="Format output for GitHub Actions"
    )
    
    args = parser.parse_args()
    
    # GitHub Actions specific formatting
    if args.github_action:
        print("::group::Validation Pipeline")
    
    # Run validation
    exit_code = asyncio.run(
        run_validation(
            validators=args.validators,
            output_file=args.output,
            critical_only=args.critical_only,
            parallel=not args.sequential
        )
    )
    
    if args.github_action:
        print("::endgroup::")
        
        if exit_code == 0:
            print("::notice::✅ All validations passed successfully!")
        else:
            print("::error::❌ Validation pipeline failed! Check the logs for details.")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()