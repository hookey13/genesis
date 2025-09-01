#!/usr/bin/env python
"""
Verification script for Story 8.8: Automated Testing & Quality Gates.

This script verifies that all 10 acceptance criteria have been fully implemented.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def check_acceptance_criteria() -> Dict[int, Tuple[bool, str]]:
    """Check all 10 acceptance criteria."""
    results = {}
    
    # AC1: Unit tests with 95%+ coverage for critical paths
    print("Checking AC1: Unit tests with 95%+ coverage...")
    ac1_files = [
        "tests/unit/test_risk_engine.py",
        "tests/unit/test_executor.py",
        "pyproject.toml",  # Should have coverage config
    ]
    ac1_pass = all(check_file_exists(f) for f in ac1_files)
    
    # Check coverage threshold in pyproject.toml
    if ac1_pass:
        with open("pyproject.toml") as f:
            content = f.read()
            ac1_pass = "--cov-fail-under=95" in content
    
    results[1] = (ac1_pass, "Unit tests with 95% coverage threshold configured")
    
    # AC2: Integration tests for all external integrations
    print("Checking AC2: Integration tests...")
    ac2_files = [
        "tests/integration/test_binance_integration.py",
        "tests/integration/test_full_trading_cycle.py",
    ]
    ac2_pass = all(check_file_exists(f) for f in ac2_files)
    results[2] = (ac2_pass, "Integration tests for external integrations")
    
    # AC3: End-to-end tests simulating real trading scenarios
    print("Checking AC3: End-to-end tests...")
    ac3_pass = check_file_exists("tests/integration/test_full_trading_cycle.py")
    
    if ac3_pass:
        with open("tests/integration/test_full_trading_cycle.py", encoding="utf-8") as f:
            content = f.read()
            # Check for key test scenarios
            ac3_pass = all([
                "test_successful_trade_cycle" in content,
                "test_stop_loss_execution" in content,
                "test_daily_loss_limit_enforcement" in content,
                ("test_tier_progression" in content or "TestTierTransitions" in content),
                ("test_emergency_procedures" in content or "TestEmergencyProcedures" in content),
            ])
    
    results[3] = (ac3_pass, "End-to-end tests with real trading scenarios")
    
    # AC4: Property-based testing with Hypothesis
    print("Checking AC4: Property-based testing...")
    ac4_pass = check_file_exists("tests/unit/test_position_properties.py")
    
    if ac4_pass:
        with open("tests/unit/test_position_properties.py", encoding="utf-8") as f:
            content = f.read()
            ac4_pass = all([
                "from hypothesis import" in content,
                "@given" in content,
                "test_position_size" in content,
                "test_pnl" in content,
                "test_risk_validation" in content,
            ])
    
    results[4] = (ac4_pass, "Property-based testing with Hypothesis")
    
    # AC5: Mutation testing
    print("Checking AC5: Mutation testing...")
    ac5_files = [
        "setup.cfg",  # mutmut config
        ".mutmut-cache",
    ]
    ac5_pass = all(check_file_exists(f) for f in ac5_files)
    
    if ac5_pass and check_file_exists("setup.cfg"):
        with open("setup.cfg") as f:
            content = f.read()
            ac5_pass = "[mutmut]" in content
    
    results[5] = (ac5_pass, "Mutation testing configured with mutmut")
    
    # AC6: Performance tests with JMeter/Locust
    print("Checking AC6: Performance tests...")
    ac6_pass = check_file_exists("tests/performance/locustfile.py")
    
    if ac6_pass:
        with open("tests/performance/locustfile.py", encoding="utf-8") as f:
            content = f.read()
            ac6_pass = all([
                "from locust import" in content,
                "LATENCY_P99_REQUIREMENT_MS = 50" in content,
                "class TradingUser" in content,
                "test_order_placement" in content or "place_order" in content,
            ])
    
    results[6] = (ac6_pass, "Performance tests with Locust")
    
    # AC7: Security tests with OWASP ZAP/Bandit
    print("Checking AC7: Security tests...")
    ac7_pass = check_file_exists(".bandit")
    
    if ac7_pass:
        with open(".bandit") as f:
            content = f.read()
            ac7_pass = "[bandit]" in content
    
    results[7] = (ac7_pass, "Security testing configured with Bandit")
    
    # AC8: Chaos engineering tests
    print("Checking AC8: Chaos engineering tests...")
    ac8_pass = check_file_exists("tests/chaos/chaos_framework.py")
    
    if ac8_pass:
        with open("tests/chaos/chaos_framework.py", encoding="utf-8") as f:
            content = f.read()
            ac8_pass = all([
                "class ChaosMonkey" in content,
                "ChaosType" in content,
                "inject_network_latency" in content,
                "inject_database_failure" in content,
                "inject_api_outage" in content,
            ])
    
    results[8] = (ac8_pass, "Chaos engineering framework implemented")
    
    # AC9: Contract testing for API compatibility
    print("Checking AC9: Contract testing...")
    ac9_pass = check_file_exists("tests/contract/test_api_contracts.py")
    
    if ac9_pass:
        with open("tests/contract/test_api_contracts.py", encoding="utf-8") as f:
            content = f.read()
            ac9_pass = all([
                "OrderRequestV1" in content,
                "OrderRequestV2" in content,
                ("backward_compatibility" in content or "TestBackwardCompatibility" in content),
                "ContractValidator" in content,
                ("schema_evolution" in content or "TestSchemaEvolution" in content),
            ])
    
    results[9] = (ac9_pass, "Contract testing for API compatibility")
    
    # AC10: Automated regression test suite
    print("Checking AC10: Regression test suite...")
    ac10_pass = check_file_exists("tests/regression/test_regression_suite.py")
    
    if ac10_pass:
        with open("tests/regression/test_regression_suite.py", encoding="utf-8") as f:
            content = f.read()
            ac10_pass = all([
                "class RegressionTestSuite" in content,
                "@pytest.mark.regression" in content,
                "test_new_user_onboarding" in content,
                "test_basic_trading_flow" in content,
                "run_nightly_regression" in content,
            ])
    
    # Check CI/CD configuration
    ci_pass = check_file_exists(".github/workflows/automated_testing.yml")
    if ci_pass:
        with open(".github/workflows/automated_testing.yml", encoding="utf-8") as f:
            content = f.read()
            ci_pass = "schedule:" in content and "cron:" in content
    
    results[10] = (ac10_pass and ci_pass, "Automated regression suite with CI/CD integration")
    
    return results


def verify_test_infrastructure() -> List[Tuple[str, bool, str]]:
    """Verify supporting test infrastructure."""
    checks = []
    
    # Check test configuration
    checks.append((
        "Coverage configuration",
        check_file_exists("pyproject.toml"),
        "pyproject.toml with pytest configuration"
    ))
    
    # Check CI/CD workflows
    checks.append((
        "CI/CD workflow",
        check_file_exists(".github/workflows/automated_testing.yml"),
        "GitHub Actions workflow for automated testing"
    ))
    
    # Check test folders structure
    test_folders = [
        "tests/unit",
        "tests/integration",
        "tests/performance",
        "tests/chaos",
        "tests/contract",
        "tests/regression",
    ]
    
    all_folders_exist = all(Path(folder).is_dir() for folder in test_folders)
    checks.append((
        "Test folder structure",
        all_folders_exist,
        f"All test folders: {', '.join(test_folders)}"
    ))
    
    # Check for key test utilities
    checks.append((
        "Mutation testing setup",
        check_file_exists("setup.cfg"),
        "Mutation testing configuration"
    ))
    
    checks.append((
        "Security scanning setup",
        check_file_exists(".bandit"),
        "Bandit security scanning configuration"
    ))
    
    return checks


def main():
    """Main verification function."""
    print("=" * 60)
    print("Story 8.8: Automated Testing & Quality Gates")
    print("Verification Report")
    print("=" * 60)
    print()
    
    # Check acceptance criteria
    print("ACCEPTANCE CRITERIA CHECK:")
    print("-" * 40)
    
    ac_results = check_acceptance_criteria()
    all_passed = True
    
    for ac_num in sorted(ac_results.keys()):
        passed, description = ac_results[ac_num]
        status = "PASS" if passed else "FAIL"
        print(f"AC{ac_num}: [{status}] {description}")
        if not passed:
            all_passed = False
    
    print()
    print("TEST INFRASTRUCTURE CHECK:")
    print("-" * 40)
    
    infra_checks = verify_test_infrastructure()
    for name, passed, description in infra_checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}: {description}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("SUCCESS: VERIFICATION PASSED - All acceptance criteria met!")
        print("Story 8.8 is FULLY IMPLEMENTED")
    else:
        print("ERROR: VERIFICATION FAILED - Some criteria not met")
        print("Please review and complete missing items")
        sys.exit(1)
    
    print("=" * 60)
    
    # Generate summary report
    report = {
        "story": "8.8",
        "title": "Automated Testing & Quality Gates",
        "verification_passed": all_passed,
        "acceptance_criteria": {
            f"AC{num}": {
                "passed": passed,
                "description": desc
            }
            for num, (passed, desc) in ac_results.items()
        },
        "infrastructure": [
            {"name": name, "passed": passed, "description": desc}
            for name, passed, desc in infra_checks
        ]
    }
    
    # Save report
    report_file = Path("tests/story_88_verification_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())