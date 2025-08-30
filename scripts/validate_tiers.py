#!/usr/bin/env python3
"""Validate tier-specific requirements consistency and dependencies."""

import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
import re


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def parse_requirements_file(file_path: Path) -> Tuple[Set[str], Dict[str, str]]:
    """
    Parse a requirements file to extract package names and versions.
    
    Returns:
        Tuple of (inherited_files, packages_dict)
    """
    inherited = set()
    packages = {}
    
    if not file_path.exists():
        return inherited, packages
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Check for inheritance
            if line.startswith('-r '):
                inherited_file = line[3:].strip()
                inherited.add(inherited_file)
                continue
            
            # Parse package requirement
            # Handle different formats: package==1.0.0, package>=1.0.0, package[extra]==1.0.0
            match = re.match(r'^([a-zA-Z0-9\-_]+)(?:\[.*?\])?([><=~!]+.*)?$', line)
            if match:
                package_name = match.group(1).lower()
                version_spec = match.group(2) or ""
                packages[package_name] = version_spec
    
    return inherited, packages


def validate_tier_inheritance() -> bool:
    """Validate that tier requirements properly inherit from each other."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    errors = []
    
    # Define expected inheritance
    expected_inheritance = {
        "sniper.txt": ["base.txt"],
        "hunter.txt": ["base.txt"],
        "strategist.txt": ["hunter.txt"],
        "dev.txt": []  # Dev doesn't inherit from production requirements
    }
    
    for tier_file, expected_parents in expected_inheritance.items():
        file_path = req_dir / tier_file
        if not file_path.exists():
            errors.append(f"Missing tier file: {tier_file}")
            continue
        
        inherited, _ = parse_requirements_file(file_path)
        
        for expected_parent in expected_parents:
            if expected_parent not in inherited:
                # Check if file is empty (which means it inherits implicitly)
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content and expected_parent not in content:
                        errors.append(
                            f"{tier_file} should inherit from {expected_parent}"
                        )
    
    if errors:
        print("❌ Tier inheritance validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Tier inheritance validation passed")
    return True


def validate_tier_dependencies() -> bool:
    """Validate tier-specific dependencies are correct."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    errors = []
    
    # Load base requirements
    _, base_packages = parse_requirements_file(req_dir / "base.txt")
    
    # Define tier-specific expected packages
    tier_specific_packages = {
        "hunter.txt": {
            "scipy": "Scientific computing for advanced strategies",
            "statsmodels": "Statistical modeling for mean reversion"
        },
        "strategist.txt": {
            "scipy": "Scientific computing for advanced strategies",
            "statsmodels": "Statistical modeling",
            "scikit-learn": "Machine learning for pattern recognition",
        }
    }
    
    # Validate each tier
    for tier_file, expected_packages in tier_specific_packages.items():
        file_path = req_dir / tier_file
        if not file_path.exists():
            continue
        
        _, tier_packages = parse_requirements_file(file_path)
        
        # For hunter and strategist, check they have tier-specific packages
        for package, description in expected_packages.items():
            if package not in tier_packages:
                # Check if it's inherited
                if tier_file == "strategist.txt":
                    # Strategist inherits from hunter
                    _, hunter_packages = parse_requirements_file(req_dir / "hunter.txt")
                    if package not in hunter_packages:
                        errors.append(
                            f"{tier_file} missing {package}: {description}"
                        )
                else:
                    errors.append(
                        f"{tier_file} missing {package}: {description}"
                    )
    
    # Validate sniper doesn't have advanced packages
    _, sniper_packages = parse_requirements_file(req_dir / "sniper.txt")
    forbidden_sniper_packages = ["scipy", "statsmodels", "scikit-learn", "ta-lib"]
    for package in forbidden_sniper_packages:
        if package in sniper_packages:
            errors.append(
                f"sniper.txt should not have {package} (tier too low)"
            )
    
    if errors:
        print("❌ Tier dependency validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Tier dependency validation passed")
    return True


def validate_no_version_conflicts() -> bool:
    """Ensure no version conflicts between tiers."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    errors = []
    all_packages = {}
    
    # Collect all package versions across tiers
    tier_files = ["base.txt", "sniper.txt", "hunter.txt", "strategist.txt", "dev.txt"]
    
    for tier_file in tier_files:
        file_path = req_dir / tier_file
        if not file_path.exists():
            continue
        
        _, packages = parse_requirements_file(file_path)
        
        for package, version in packages.items():
            if package in all_packages:
                # Check for version conflict
                if version != all_packages[package]["version"]:
                    # Allow dev to have different versions
                    if tier_file != "dev.txt" and all_packages[package]["file"] != "dev.txt":
                        errors.append(
                            f"Version conflict for {package}: "
                            f"{version} in {tier_file} vs "
                            f"{all_packages[package]['version']} in {all_packages[package]['file']}"
                        )
            else:
                all_packages[package] = {"version": version, "file": tier_file}
    
    if errors:
        print("❌ Version conflict validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ No version conflicts detected")
    return True


def validate_conditional_loading() -> bool:
    """Validate that tier-specific loading is implemented correctly."""
    project_root = get_project_root()
    
    errors = []
    warnings = []
    
    # Check for tier loading logic in key files
    files_to_check = [
        project_root / "genesis" / "strategies" / "loader.py",
        project_root / "genesis" / "engine" / "state_machine.py",
        project_root / "config" / "settings.py",
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Check for tier-based conditionals
                if "TIER" not in content and "tier" not in content.lower():
                    warnings.append(
                        f"{file_path.name} may not have tier-based loading"
                    )
    
    # Check for @requires_tier decorator usage
    strategies_dir = project_root / "genesis" / "strategies"
    if strategies_dir.exists():
        for strategy_file in strategies_dir.rglob("*.py"):
            with open(strategy_file, 'r') as f:
                content = f.read()
                
                # Check strategist-level strategies
                if "strategist" in str(strategy_file):
                    if "@requires_tier" not in content:
                        warnings.append(
                            f"{strategy_file.name} in strategist tier should use @requires_tier decorator"
                        )
    
    if errors:
        print("❌ Conditional loading validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if warnings:
        print("⚠️  Conditional loading warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("✅ Conditional loading validation passed")
    return True


def generate_tier_report():
    """Generate a report of tier configurations."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    print("\n" + "=" * 60)
    print("TIER REQUIREMENTS REPORT")
    print("=" * 60)
    
    tier_info = {
        "base.txt": "Core dependencies for all tiers",
        "sniper.txt": "$500-$2k: Single pair, basic strategies",
        "hunter.txt": "$2k-$10k: Multi-pair, statistical analysis",
        "strategist.txt": "$10k+: ML-based strategies, market making",
        "dev.txt": "Development and testing tools"
    }
    
    for tier_file, description in tier_info.items():
        file_path = req_dir / tier_file
        if not file_path.exists():
            print(f"\n❌ {tier_file}: NOT FOUND")
            continue
        
        inherited, packages = parse_requirements_file(file_path)
        
        print(f"\n📦 {tier_file}")
        print(f"   {description}")
        
        if inherited:
            print(f"   Inherits from: {', '.join(inherited)}")
        
        print(f"   Packages: {len(packages)}")
        
        # Show key packages for each tier
        key_packages = {
            "base.txt": ["ccxt", "pydantic", "structlog", "SQLAlchemy"],
            "hunter.txt": ["scipy", "statsmodels"],
            "strategist.txt": ["scikit-learn", "ta-lib"],
            "dev.txt": ["pytest", "black", "ruff", "mypy"]
        }
        
        if tier_file in key_packages:
            print("   Key packages:")
            for pkg in key_packages[tier_file]:
                if pkg in packages:
                    print(f"     ✅ {pkg}{packages[pkg]}")
                else:
                    print(f"     ❌ {pkg} (missing)")
    
    print("\n" + "=" * 60)


def main():
    """Main validation entry point."""
    print("🔍 Validating Tier Requirements")
    print("-" * 40)
    
    all_valid = True
    
    # Run validations
    if not validate_tier_inheritance():
        all_valid = False
    
    if not validate_tier_dependencies():
        all_valid = False
    
    if not validate_no_version_conflicts():
        all_valid = False
    
    if not validate_conditional_loading():
        all_valid = False
    
    # Generate report
    generate_tier_report()
    
    # Final result
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL TIER VALIDATIONS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ TIER VALIDATION FAILED")
        print("Please fix the issues above before proceeding")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()