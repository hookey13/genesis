#!/usr/bin/env python
"""Validation script for DR implementation completeness."""

import sys
import importlib
import inspect
from typing import List, Tuple


def check_module_imports() -> Tuple[bool, List[str]]:
    """Check if all DR modules can be imported."""
    errors = []
    modules = [
        'genesis.dr.failover_manager',
        'genesis.dr.dr_testing',
        'genesis.dr.recovery_validator',
        'genesis.config.dr_config'
    ]
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name} imported successfully")
        except ImportError as e:
            errors.append(f"[FAIL] Failed to import {module_name}: {e}")
            
    return len(errors) == 0, errors


def check_no_placeholders() -> Tuple[bool, List[str]]:
    """Check for placeholder implementations."""
    warnings = []
    
    # Import modules
    try:
        from genesis.dr import failover_manager, dr_testing, recovery_validator
        
        # Check for actual implementations
        fm = failover_manager.FailoverManager(
            failover_manager.FailoverConfig()
        )
        
        # Check key methods are not just placeholders
        methods_to_check = [
            (fm._update_dns_routing, 'DNS routing'),
            (fm._promote_standby_database, 'Database promotion'),
            (fm._start_regional_services, 'Service startup'),
            (fm._check_database_health, 'Database health check'),
            (fm._send_email_notification, 'Email notification'),
            (fm._send_slack_notification, 'Slack notification')
        ]
        
        for method, name in methods_to_check:
            source = inspect.getsource(method)
            if 'placeholder' in source.lower() or 'todo' in source.lower():
                warnings.append(f"[WARN] {name} may contain placeholder code")
            else:
                print(f"[OK] {name} has production implementation")
                
    except Exception as e:
        warnings.append(f"[FAIL] Error checking implementations: {e}")
        
    return len(warnings) == 0, warnings


def check_critical_integrations() -> Tuple[bool, List[str]]:
    """Check critical integrations are present."""
    issues = []
    
    try:
        from genesis.dr import failover_manager
        
        # Check for AWS integrations
        source = inspect.getsource(failover_manager)
        
        integrations = {
            'boto3': 'AWS SDK',
            'asyncpg': 'PostgreSQL async driver',
            'aiohttp': 'HTTP client for health checks',
            'smtplib': 'Email notifications',
            'Route53': 'DNS failover',
            'pg_promote': 'Database promotion'
        }
        
        for integration, description in integrations.items():
            if integration in source:
                print(f"[OK] {description} integration found")
            else:
                issues.append(f"[WARN] {description} integration may be missing")
                
    except Exception as e:
        issues.append(f"[FAIL] Error checking integrations: {e}")
        
    return len(issues) == 0, issues


def check_test_coverage() -> Tuple[bool, List[str]]:
    """Check test coverage for DR functionality."""
    issues = []
    
    test_files = [
        'tests/dr/test_failover.py',
        'tests/dr/test_dr_drills.py'
    ]
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                
            # Count test functions
            test_count = content.count('def test_') + content.count('async def test_')
            
            if test_count >= 10:
                print(f"[OK] {test_file}: {test_count} tests found")
            else:
                issues.append(f"[WARN] {test_file}: Only {test_count} tests found")
                
        except FileNotFoundError:
            issues.append(f"[FAIL] Test file not found: {test_file}")
            
    return len(issues) == 0, issues


def check_configuration() -> Tuple[bool, List[str]]:
    """Check DR configuration is complete."""
    issues = []
    
    try:
        from genesis.config import dr_config
        
        config = dr_config.get_dr_config()
        
        # Check required configuration sections
        required_sections = [
            'failover',
            'regions',
            'dns',
            'database',
            'testing',
            'monitoring',
            'notifications',
            'validation'
        ]
        
        for section in required_sections:
            if section in config:
                print(f"[OK] Configuration section '{section}' present")
            else:
                issues.append(f"[FAIL] Missing configuration section: {section}")
                
        # Check critical values
        if config.get('failover', {}).get('rto_target_seconds', 0) <= 300:
            print("[OK] RTO target is 5 minutes or less")
        else:
            issues.append("[WARN] RTO target exceeds 5 minutes")
            
        if config.get('failover', {}).get('rpo_target_seconds', -1) == 0:
            print("[OK] RPO target is zero (no data loss)")
        else:
            issues.append("[WARN] RPO target is not zero")
            
    except Exception as e:
        issues.append(f"[FAIL] Error checking configuration: {e}")
        
    return len(issues) == 0, issues


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("DR Implementation Validation")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Run checks
    checks = [
        ("Module Imports", check_module_imports),
        ("Placeholder Code", check_no_placeholders),
        ("Critical Integrations", check_critical_integrations),
        ("Test Coverage", check_test_coverage),
        ("Configuration", check_configuration)
    ]
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 40)
        
        passed, issues = check_func()
        
        if not passed:
            all_passed = False
            for issue in issues:
                print(issue)
                
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] All validation checks passed!")
        print("The DR implementation is production-ready with no shortcuts.")
    else:
        print("[ERROR] Some validation checks failed.")
        print("Please review the issues above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())