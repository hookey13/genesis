#!/usr/bin/env python3
"""Migrate from pip/requirements.txt to Poetry dependency management."""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_poetry_installed() -> bool:
    """Check if Poetry is installed."""
    try:
        result = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_poetry():
    """Install Poetry if not present."""
    if not check_poetry_installed():
        print("Poetry not found. Installing Poetry...")
        
        # Install using pip
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "poetry"],
                check=True
            )
            print("✅ Poetry installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Poetry")
            print("\nPlease install Poetry manually:")
            print("  curl -sSL https://install.python-poetry.org | python3 -")
            print("  or")
            print("  pip install poetry")
            sys.exit(1)
    else:
        print("✅ Poetry is already installed")


def backup_requirements():
    """Backup existing requirements files."""
    project_root = get_project_root()
    req_dir = project_root / "requirements"
    backup_dir = project_root / "requirements.backup"
    
    if req_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(req_dir, backup_dir)
        print(f"✅ Backed up requirements to {backup_dir}")


def verify_pyproject_toml():
    """Verify pyproject.toml has Poetry configuration."""
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    content = pyproject_path.read_text()
    if "[tool.poetry]" not in content:
        print("❌ pyproject.toml missing Poetry configuration")
        return False
    
    print("✅ pyproject.toml has Poetry configuration")
    return True


def install_dependencies(tier: str = "sniper"):
    """Install dependencies using Poetry."""
    project_root = get_project_root()
    
    print(f"\nInstalling dependencies for tier: {tier}")
    
    # Base command
    cmd = ["poetry", "install"]
    
    # Add tier-specific groups
    if tier == "hunter":
        cmd.extend(["--with", "hunter"])
    elif tier == "strategist":
        cmd.extend(["--with", "strategist"])
    
    # Always include dev dependencies in development
    if os.environ.get("ENV") != "production":
        cmd.extend(["--with", "dev"])
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
        print(f"✅ Dependencies installed for {tier} tier")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True


def generate_lock_file():
    """Generate poetry.lock file."""
    project_root = get_project_root()
    
    print("\nGenerating poetry.lock file...")
    
    try:
        subprocess.run(
            ["poetry", "lock", "--no-update"],
            cwd=project_root,
            check=True
        )
        print("✅ poetry.lock generated successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Could not generate lock file, will be created on first install")


def verify_installation():
    """Verify Poetry installation and dependencies."""
    project_root = get_project_root()
    
    print("\nVerifying installation...")
    
    # Check Poetry environment
    result = subprocess.run(
        ["poetry", "env", "info"],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Poetry environment configured")
        print(result.stdout)
    else:
        print("⚠️  Poetry environment not yet configured")
    
    # Check installed packages
    result = subprocess.run(
        ["poetry", "show"],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        package_count = len([l for l in lines if l.strip()])
        print(f"✅ {package_count} packages installed")
    
    return True


def create_activation_wrapper():
    """Create wrapper scripts for Poetry activation."""
    project_root = get_project_root()
    scripts_dir = project_root / "scripts"
    
    # Create poetry_activate.sh for Unix
    bash_wrapper = scripts_dir / "poetry_activate.sh"
    bash_content = """#!/bin/bash
# Poetry activation wrapper for Genesis

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Activating Poetry environment for Genesis"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed"
    echo "Run: pip install poetry"
    exit 1
fi

# Navigate to project root
cd "$PROJECT_ROOT"

# Activate Poetry shell
poetry shell

echo "✅ Poetry environment activated"
echo "Run 'exit' to deactivate"
"""
    bash_wrapper.write_text(bash_content)
    print(f"✅ Created {bash_wrapper}")
    
    # Create poetry_activate.ps1 for Windows
    ps1_wrapper = scripts_dir / "poetry_activate.ps1"
    ps1_content = """# Poetry activation wrapper for Genesis

$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "🚀 Activating Poetry environment for Genesis" -ForegroundColor Green

# Check if Poetry is installed
try {
    $poetryVersion = poetry --version
} catch {
    Write-Host "❌ Poetry is not installed" -ForegroundColor Red
    Write-Host "Run: pip install poetry"
    exit 1
}

# Navigate to project root
Set-Location $ProjectRoot

# Activate Poetry shell
poetry shell

Write-Host "✅ Poetry environment activated" -ForegroundColor Green
Write-Host "Run 'exit' to deactivate"
"""
    ps1_wrapper.write_text(ps1_content)
    print(f"✅ Created {ps1_wrapper}")


def update_documentation():
    """Update README with Poetry instructions."""
    project_root = get_project_root()
    
    migration_doc = project_root / "docs" / "setup" / "poetry_migration.md"
    migration_doc.parent.mkdir(parents=True, exist_ok=True)
    
    content = """# Poetry Migration Guide

## Overview

Project Genesis has migrated to Poetry for dependency management. Poetry provides:
- Deterministic dependency resolution
- Lock file for reproducible builds
- Simplified virtual environment management
- Better dependency conflict resolution
- Built-in packaging and publishing tools

## Installation

### Install Poetry

**Using pip:**
```bash
pip install poetry
```

**Using official installer:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Dependencies

**For Sniper tier (default):**
```bash
poetry install
```

**For Hunter tier:**
```bash
poetry install --with hunter
```

**For Strategist tier:**
```bash
poetry install --with strategist
```

**With development dependencies:**
```bash
poetry install --with dev
```

## Usage

### Activate Environment

**Using Poetry shell:**
```bash
poetry shell
```

**Using activation scripts:**
```bash
# Unix/Linux/macOS
source scripts/poetry_activate.sh

# Windows PowerShell
.\\scripts\\poetry_activate.ps1
```

### Run Commands

**Without activating shell:**
```bash
poetry run python -m genesis
poetry run pytest
poetry run black genesis/
```

### Add New Dependencies

**Add to main dependencies:**
```bash
poetry add package-name
```

**Add to specific group:**
```bash
poetry add --group dev pytest-benchmark
poetry add --group hunter scipy
```

### Update Dependencies

**Update all:**
```bash
poetry update
```

**Update specific package:**
```bash
poetry update package-name
```

### Export Requirements

**Export to requirements.txt format:**
```bash
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --with dev --output requirements-dev.txt
```

## Tier Management

Dependencies are organized by tier:

- **Base**: Core dependencies for all tiers
- **Sniper**: Base dependencies only (default)
- **Hunter**: Base + hunter group (scipy, statsmodels)
- **Strategist**: Base + strategist group (includes hunter deps + ML libs)
- **Dev**: Development and testing tools

## Migration from pip

If you have an existing pip/venv environment:

1. Backup existing environment:
   ```bash
   pip freeze > requirements.backup.txt
   ```

2. Run migration script:
   ```bash
   python scripts/migrate_to_poetry.py
   ```

3. Verify installation:
   ```bash
   poetry show
   ```

## Troubleshooting

### Poetry not found
```bash
pip install poetry
# or
curl -sSL https://install.python-poetry.org | python3 -
```

### Wrong Python version
```bash
poetry env use python3.11
```

### Clear Poetry cache
```bash
poetry cache clear pypi --all
```

### Reinstall from lock file
```bash
poetry install --sync
```

## Benefits

1. **Reproducible builds**: poetry.lock ensures exact same versions
2. **Dependency resolution**: Automatic conflict resolution
3. **Virtual environment**: Automatic venv management
4. **Groups**: Separate dev, test, and tier dependencies
5. **Scripts**: Define entry points in pyproject.toml
6. **Publishing**: Built-in package publishing to PyPI
"""
    
    migration_doc.write_text(content)
    print(f"✅ Created migration documentation at {migration_doc}")


def main():
    """Main migration process."""
    import os
    
    print("=" * 60)
    print("Genesis - Poetry Migration Tool")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info[:2] != (3, 11):
        print(f"⚠️  Warning: Using Python {sys.version_info.major}.{sys.version_info.minor}")
        print("   Genesis requires Python 3.11.8")
    
    # Step 1: Install Poetry
    install_poetry()
    
    # Step 2: Backup existing requirements
    backup_requirements()
    
    # Step 3: Verify pyproject.toml
    if not verify_pyproject_toml():
        print("\n❌ Migration failed: pyproject.toml not configured")
        print("   Please ensure pyproject.toml has [tool.poetry] section")
        sys.exit(1)
    
    # Step 4: Generate lock file
    generate_lock_file()
    
    # Step 5: Install dependencies
    tier = os.environ.get("TIER", "sniper")
    if not install_dependencies(tier):
        print("\n⚠️  Dependencies installation had issues")
        print("   You may need to install manually with: poetry install")
    
    # Step 6: Create activation wrappers
    create_activation_wrapper()
    
    # Step 7: Update documentation
    update_documentation()
    
    # Step 8: Verify installation
    verify_installation()
    
    print("\n" + "=" * 60)
    print("✅ Poetry migration completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate Poetry environment: poetry shell")
    print("2. Verify dependencies: poetry show")
    print("3. Run tests: poetry run pytest")
    print("4. Commit poetry.lock to version control")
    print("\nSee docs/setup/poetry_migration.md for detailed usage")


if __name__ == "__main__":
    main()