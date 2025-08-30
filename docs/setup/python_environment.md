# Python Environment Management

## Overview

Genesis Trading System requires Python 3.11.8 for optimal performance and compatibility. This document covers environment setup, version management, and best practices.

## Table of Contents
1. [Python Version Management](#python-version-management)
2. [Virtual Environment Setup](#virtual-environment-setup)
3. [Environment Activation](#environment-activation)
4. [Tier-Based Dependencies](#tier-based-dependencies)
5. [Troubleshooting](#troubleshooting)

## Python Version Management

### Using pyenv (Recommended)

pyenv provides seamless Python version management across different projects.

#### Installation

**macOS:**
```bash
brew install pyenv
```

**Linux:**
```bash
curl https://pyenv.run | bash
```

**Windows:**
```powershell
git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
```

#### Setup

1. Install Python 3.11.8:
```bash
pyenv install 3.11.8
```

2. Set local version for Genesis:
```bash
cd /path/to/genesis
pyenv local 3.11.8
```

This creates a `.python-version` file that automatically activates Python 3.11.8 when you enter the project directory.

### Manual Installation

If you prefer not to use pyenv:

1. Download Python 3.11.8 from [python.org](https://www.python.org/downloads/release/python-3118/)
2. Install with "Add Python to PATH" option checked
3. Verify installation:
```bash
python --version  # Should show: Python 3.11.8
```

## Virtual Environment Setup

Virtual environments isolate project dependencies from system Python.

### Automatic Setup

Use our provided activation scripts for automatic setup:

**Linux/macOS:**
```bash
source scripts/activate_env.sh
```

**Windows PowerShell:**
```powershell
.\scripts\activate_env.ps1
```

These scripts will:
- Check Python version
- Create virtual environment if needed
- Install tier-appropriate dependencies
- Set environment variables

### Manual Setup

1. Create virtual environment:
```bash
python3.11 -m venv .venv
```

2. Activate environment:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

3. Upgrade pip:
```bash
pip install --upgrade pip
```

4. Install dependencies:
```bash
# For Sniper tier (default)
pip install -r requirements/sniper.txt

# For Hunter tier
pip install -r requirements/hunter.txt

# For Strategist tier
pip install -r requirements/strategist.txt

# Development dependencies
pip install -r requirements/dev.txt
```

## Environment Activation

### Environment Variables

Set the trading tier before activation:

```bash
export TIER=sniper  # Linux/macOS
$env:TIER = "sniper"  # Windows PowerShell
```

Available tiers:
- `sniper`: $500-$2k capital (basic strategies)
- `hunter`: $2k-$10k capital (statistical analysis)
- `strategist`: $10k+ capital (ML-based strategies)

### Quick Activation Commands

Add these aliases to your shell profile for convenience:

**~/.bashrc or ~/.zshrc:**
```bash
alias genesis-activate='cd /path/to/genesis && source scripts/activate_env.sh'
alias genesis-sniper='TIER=sniper genesis-activate'
alias genesis-hunter='TIER=hunter genesis-activate'
alias genesis-strategist='TIER=strategist genesis-activate'
```

**Windows PowerShell Profile:**
```powershell
function Genesis-Activate {
    Set-Location C:\path\to\genesis
    .\scripts\activate_env.ps1
}

function Genesis-Sniper {
    $env:TIER = "sniper"
    Genesis-Activate
}
```

## Tier-Based Dependencies

### Dependency Structure

```
requirements/
├── base.txt         # Core dependencies (all tiers)
├── sniper.txt       # Inherits from base.txt
├── hunter.txt       # Inherits from base.txt + statistical libs
├── strategist.txt   # Inherits from hunter.txt + ML libs
└── dev.txt          # Development tools
```

### Core Dependencies (base.txt)

Essential packages for all tiers:
- `ccxt==4.4.0` - Exchange API integration
- `pydantic==2.5.3` - Configuration validation
- `structlog==24.1.0` - Structured logging
- `SQLAlchemy==2.0.25` - Database ORM
- `aiohttp==3.10.11` - Async HTTP client
- `websockets==13.1` - WebSocket support

### Tier-Specific Packages

**Hunter Tier:**
- `scipy` - Scientific computing
- `statsmodels` - Statistical modeling

**Strategist Tier:**
- `scikit-learn` - Machine learning
- `ta-lib` - Technical analysis (requires system library)

### Installing TA-Lib

TA-Lib requires additional system libraries:

**macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install ta-lib
```

**Windows:**
Download the appropriate wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install:
```powershell
pip install TA_Lib‑0.4.24‑cp311‑cp311‑win_amd64.whl
```

## Troubleshooting

### Common Issues

#### 1. Python Version Mismatch

**Error:** `Python 3.11.8 is required but not found`

**Solution:**
```bash
# Check current Python version
python --version

# If using pyenv, set correct version
pyenv local 3.11.8
pyenv rehash

# Verify
python --version
```

#### 2. Virtual Environment Not Activating

**Error:** `No module named 'genesis'`

**Solution:**
```bash
# Ensure you're in the project directory
cd /path/to/genesis

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# Verify activation
which python  # Should show .venv/bin/python
```

#### 3. Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'ccxt'`

**Solution:**
```bash
# Ensure virtual environment is activated
# Then reinstall dependencies
pip install -r requirements/base.txt
pip install -r requirements/${TIER}.txt
```

#### 4. Permission Errors

**Error:** `Permission denied` when installing packages

**Solution:**
```bash
# Never use sudo with pip in virtual environment
# Instead, ensure you own the virtual environment
chown -R $(whoami) .venv/

# Then retry installation
pip install -r requirements/base.txt
```

#### 5. SSL Certificate Errors

**Error:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution:**
```bash
# Upgrade certificates
pip install --upgrade certifi

# Or temporarily (not recommended for production)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Environment Validation

Run this command to validate your environment:

```bash
python -c "
import sys
assert sys.version_info[:3] == (3, 11, 8), f'Python 3.11.8 required, got {sys.version}'
print('✅ Python version correct')

try:
    import genesis
    print('✅ Genesis package importable')
except ImportError:
    print('❌ Genesis package not found')

import ccxt, pydantic, structlog
print('✅ Core dependencies installed')

print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
"
```

### Getting Help

If you encounter issues:

1. Check this documentation
2. Run the validation script: `python scripts/validate_tiers.py`
3. Check the debug log: `.ai/debug-log.md`
4. Review the [troubleshooting guide](../troubleshooting.md)

## Best Practices

1. **Always use Python 3.11.8** - The system is tested and optimized for this version
2. **Use virtual environments** - Never install packages globally
3. **Activate the correct tier** - Dependencies vary by trading tier
4. **Keep dependencies updated** - Run `pip install --upgrade -r requirements/base.txt` regularly
5. **Use the activation scripts** - They handle all setup automatically
6. **Document environment changes** - Update `.env.example` when adding new variables

## Next Steps

After setting up your Python environment:

1. [Configure your API credentials](../setup/api_configuration.md)
2. [Set up development tools](../setup/dev_tools.md)
3. [Run the test suite](../testing/running_tests.md)
4. [Start trading](../quickstart.md)