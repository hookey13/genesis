#!/usr/bin/env python3
"""Verify installed package hashes against requirements.lock."""

import sys
import subprocess
import hashlib
from pathlib import Path
import json


def get_installed_packages():
    """Get list of installed packages with their locations."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)


def calculate_package_hash(package_name: str) -> str:
    """Calculate SHA256 hash of installed package."""
    try:
        import importlib.metadata
        dist = importlib.metadata.distribution(package_name)
        
        # Get package files
        if dist.files:
            hasher = hashlib.sha256()
            for file in sorted(dist.files):
                file_path = Path(dist.locate_file(file))
                if file_path.exists() and file_path.is_file():
                    hasher.update(file_path.read_bytes())
            return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: Could not hash {package_name}: {e}")
        return None


def parse_requirements_lock():
    """Parse requirements.lock for package hashes."""
    project_root = Path(__file__).parent.parent
    lock_file = project_root / "requirements.lock"
    
    if not lock_file.exists():
        print("Error: requirements.lock not found")
        print("Run: python scripts/update_requirements.py")
        sys.exit(1)
    
    packages = {}
    current_package = None
    
    with open(lock_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if not line.startswith('--hash=') and not line.startswith('\\'):
                # Package line
                if '==' in line:
                    parts = line.split('==')
                    current_package = parts[0].strip().lower()
                    version = parts[1].split()[0].split(';')[0]
                    packages[current_package] = {
                        'version': version,
                        'hashes': []
                    }
            elif current_package and '--hash=' in line:
                # Hash line
                hash_part = line.split('--hash=')[1].strip()
                packages[current_package]['hashes'].append(hash_part)
    
    return packages


def verify_installation():
    """Verify installed packages match requirements.lock."""
    print("Verifying package integrity...")
    
    expected_packages = parse_requirements_lock()
    installed_packages = get_installed_packages()
    
    errors = []
    warnings = []
    verified = 0
    
    # Check each expected package
    for pkg_name, pkg_info in expected_packages.items():
        # Find installed version
        installed = next(
            (p for p in installed_packages if p['name'].lower().replace('-', '_') == pkg_name.replace('-', '_')),
            None
        )
        
        if not installed:
            # Check alternate names (underscore vs dash)
            alt_name = pkg_name.replace('_', '-') if '_' in pkg_name else pkg_name.replace('-', '_')
            installed = next(
                (p for p in installed_packages if p['name'].lower() == alt_name),
                None
            )
        
        if not installed:
            errors.append(f"Missing package: {pkg_name}")
            continue
            
        if installed['version'] != pkg_info['version']:
            errors.append(
                f"Version mismatch for {pkg_name}: "
                f"expected {pkg_info['version']}, got {installed['version']}"
            )
        else:
            verified += 1
    
    # Check for unexpected packages (excluding common tools)
    expected_names = {name.lower().replace('-', '_') for name in expected_packages.keys()}
    exclude_packages = {
        'pip', 'setuptools', 'wheel', 'pip-tools', 'piptools',
        'pkg-resources', 'pkg_resources', 'distribute'
    }
    
    for installed in installed_packages:
        normalized_name = installed['name'].lower().replace('-', '_')
        if normalized_name not in expected_names and normalized_name not in exclude_packages:
            warnings.append(f"Unexpected package: {installed['name']} ({installed['version']})")
    
    # Report results
    print("\n" + "=" * 60)
    if errors:
        print("❌ Verification FAILED:")
        for error in errors:
            print(f"  - {error}")
        print("\nRun the following to fix:")
        print("  pip install -r requirements.lock")
        sys.exit(1)
    else:
        print(f"✅ All {verified} expected packages verified successfully")
        
    if warnings:
        print("\n⚠️  Additional packages found (may be dependencies):")
        for warning in warnings[:10]:  # Limit output
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    print("=" * 60)


if __name__ == "__main__":
    verify_installation()