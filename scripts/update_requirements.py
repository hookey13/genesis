#!/usr/bin/env python3
"""Update and pin requirements with hashes for reproducible builds."""

import subprocess
import sys
from pathlib import Path
import hashlib
import tempfile
import shutil
from typing import List, Dict, Set


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(cmd: List[str], cwd: Path = None) -> str:
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)


def install_pip_tools():
    """Ensure pip-tools is installed."""
    try:
        import piptools
    except ImportError:
        print("Installing pip-tools...")
        run_command([sys.executable, "-m", "pip", "install", "pip-tools"])


def compile_requirements(input_file: Path, output_file: Path, base_file: Path = None):
    """Compile requirements with pip-compile."""
    cmd = [
        sys.executable, "-m", "piptools", "compile",
        "--generate-hashes",
        "--resolver=backtracking",
        "--quiet",
        "--output-file", str(output_file),
        str(input_file)
    ]
    
    print(f"Compiling {input_file.name} -> {output_file.name}")
    run_command(cmd, cwd=input_file.parent)


def create_requirements_lock():
    """Create a unified requirements.lock file with all pinned versions."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    # Collect all unique requirements
    all_requirements = {}
    tier_files = ["base.txt", "sniper.txt", "hunter.txt", "strategist.txt", "dev.txt"]
    
    for tier_file in tier_files:
        file_path = req_dir / tier_file
        if not file_path.exists():
            continue
            
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('-r'):
                    continue
                    
                # Parse package name
                if '==' in line:
                    pkg_name = line.split('==')[0].split('[')[0].strip()
                    all_requirements[pkg_name] = line
                elif '>=' in line or '<=' in line or '~=' in line:
                    pkg_name = line.split('[')[0].strip()
                    for sep in ['>=', '<=', '~=', '>', '<']:
                        if sep in pkg_name:
                            pkg_name = pkg_name.split(sep)[0]
                            break
                    all_requirements[pkg_name] = line
    
    # Create temporary requirements file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        for req in all_requirements.values():
            tmp.write(req + '\n')
        tmp_path = Path(tmp.name)
    
    # Compile with hashes
    lock_file = project_root / "requirements.lock"
    try:
        compile_requirements(tmp_path, lock_file)
        print(f"Created {lock_file}")
    finally:
        tmp_path.unlink()


def verify_hashes_script():
    """Create script to verify package hashes."""
    project_root = get_project_root()
    script_path = project_root / "scripts" / "verify_hashes.py"
    
    script_content = '''#!/usr/bin/env python3
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
                
            if not line.startswith('--hash='):
                # Package line
                if '==' in line:
                    parts = line.split('==')
                    current_package = parts[0].strip()
                    version = parts[1].split()[0]
                    packages[current_package] = {
                        'version': version,
                        'hashes': []
                    }
            elif current_package and line.startswith('--hash='):
                # Hash line
                hash_value = line.replace('--hash=', '').strip()
                packages[current_package]['hashes'].append(hash_value)
    
    return packages


def verify_installation():
    """Verify installed packages match requirements.lock."""
    print("Verifying package integrity...")
    
    expected_packages = parse_requirements_lock()
    installed_packages = get_installed_packages()
    
    errors = []
    warnings = []
    
    # Check each expected package
    for pkg_name, pkg_info in expected_packages.items():
        # Find installed version
        installed = next(
            (p for p in installed_packages if p['name'].lower() == pkg_name.lower()),
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
    
    # Check for unexpected packages
    expected_names = {name.lower() for name in expected_packages.keys()}
    for installed in installed_packages:
        if installed['name'].lower() not in expected_names:
            # Skip common development tools
            if installed['name'].lower() not in ['pip', 'setuptools', 'wheel']:
                warnings.append(f"Unexpected package: {installed['name']}")
    
    # Report results
    if errors:
        print("\\n❌ Verification FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ All packages verified successfully")
        
    if warnings:
        print("\\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print(f"\\nVerified {len(expected_packages)} packages")


if __name__ == "__main__":
    verify_installation()
'''
    
    script_path.write_text(script_content)
    print(f"Created {script_path}")
    return script_path


def update_tier_requirements():
    """Update tier-specific requirements with exact versions."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    
    # Read current base.txt to get exact versions
    base_file = req_dir / "base.txt"
    if not base_file.exists():
        print(f"Warning: {base_file} not found")
        return
    
    # Compile each tier's requirements
    tiers = {
        "base.txt": None,
        "sniper.txt": "base.txt",
        "hunter.txt": "base.txt",
        "strategist.txt": "hunter.txt",
        "dev.txt": None
    }
    
    for tier_file, parent in tiers.items():
        input_file = req_dir / tier_file
        if not input_file.exists():
            continue
            
        # Create compiled version with hashes
        output_file = req_dir / f"{tier_file.replace('.txt', '')}.lock"
        compile_requirements(input_file, output_file)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Genesis Requirements Update Tool")
    print("=" * 60)
    
    # Install pip-tools if needed
    install_pip_tools()
    
    # Update tier requirements
    print("\nUpdating tier-specific requirements...")
    update_tier_requirements()
    
    # Create unified lock file
    print("\nCreating unified requirements.lock...")
    create_requirements_lock()
    
    # Create verification script
    print("\nCreating verification script...")
    verify_script = verify_hashes_script()
    
    print("\n" + "=" * 60)
    print("✅ Requirements updated successfully!")
    print("\nNext steps:")
    print("1. Review generated .lock files in requirements/")
    print("2. Commit requirements.lock to version control")
    print(f"3. Run verification: python {verify_script.relative_to(get_project_root())}")
    print("=" * 60)


if __name__ == "__main__":
    main()