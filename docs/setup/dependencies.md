# Dependency Management Guide

## Overview

Genesis uses a multi-layered approach to dependency management with Poetry, pip-tools, and automated security scanning. This guide covers installation, updates, and security practices.

## Table of Contents
1. [Dependency Organization](#dependency-organization)
2. [Using Poetry](#using-poetry)
3. [Using pip-tools](#using-pip-tools)
4. [Security Scanning](#security-scanning)
5. [Automated Updates](#automated-updates)
6. [Best Practices](#best-practices)

## Dependency Organization

### Structure

```
genesis/
├── pyproject.toml           # Poetry configuration & tool settings
├── poetry.lock              # Poetry lock file
├── requirements.lock        # pip-tools lock file with hashes
└── requirements/
    ├── base.txt            # Core dependencies
    ├── sniper.txt          # Tier: $500-$2k
    ├── hunter.txt          # Tier: $2k-$10k
    ├── strategist.txt      # Tier: $10k+
    └── dev.txt             # Development tools
```

### Dependency Groups

**Core (base.txt):**
- Exchange integration (ccxt)
- Data processing (pandas, numpy)
- Async operations (aiohttp, websockets)
- Configuration (pydantic)
- Logging (structlog)
- Database (SQLAlchemy, alembic)

**Development (dev.txt):**
- Testing (pytest, pytest-asyncio, pytest-cov)
- Code quality (black, ruff, mypy)
- Security (safety, pip-audit, bandit)
- Documentation (mkdocs)

## Using Poetry

### Installation

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Or with pip
pip install poetry
```

### Basic Commands

```bash
# Install all dependencies
poetry install

# Install with specific groups
poetry install --with hunter
poetry install --with strategist
poetry install --with dev

# Add a new dependency
poetry add package-name

# Add to specific group
poetry add --group dev pytest-benchmark

# Update dependencies
poetry update

# Export to requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### Configuration

Poetry configuration in `pyproject.toml`:

```toml
[tool.poetry]
name = "genesis"
version = "1.0.0"
description = "Evolutionary cryptocurrency trading system"
authors = ["Genesis Team"]
python = "~3.11.8"

[tool.poetry.dependencies]
python = "~3.11.8"
ccxt = "4.4.0"
pydantic = "2.5.3"
# ... more dependencies

[tool.poetry.group.hunter]
optional = true

[tool.poetry.group.hunter.dependencies]
scipy = "^1.11.0"
statsmodels = "^0.14.0"
```

### Virtual Environment Management

```bash
# Show environment info
poetry env info

# Use specific Python version
poetry env use python3.11

# List environments
poetry env list

# Remove environment
poetry env remove python3.11
```

## Using pip-tools

### Installation

```bash
pip install pip-tools
```

### Generating Lock Files

```bash
# Generate requirements.lock with hashes
python scripts/update_requirements.py

# Or manually
pip-compile --generate-hashes requirements/base.txt -o requirements/base.lock
```

### Installing from Lock Files

```bash
# Install with hash verification
pip-sync requirements/base.lock

# Or manually
pip install -r requirements/base.lock
```

### Updating Dependencies

```bash
# Update all packages
pip-compile --upgrade requirements/base.txt

# Update specific package
pip-compile --upgrade-package ccxt requirements/base.txt

# Update to specific version
pip-compile --upgrade-package ccxt==4.5.0 requirements/base.txt
```

## Security Scanning

### Manual Scanning

```bash
# Run comprehensive security scan
python -m genesis.security.dependency_scanner

# Individual scanners
safety check
pip-audit
bandit -r genesis/
```

### Pre-commit Hooks

Automatic scanning on every commit:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

### CI/CD Integration

GitHub Actions workflows run automatically:
- On every push to main/develop
- On pull requests
- Daily scheduled scans
- When dependencies change

### Vulnerability Thresholds

| Severity | Threshold | Action |
|----------|-----------|--------|
| CRITICAL | 0 | Block deployment |
| HIGH | 3 | Block deployment |
| MEDIUM | 10 | Warning |
| LOW | Unlimited | Info only |

## Automated Updates

### Dependabot Configuration

Located in `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    groups:
      dev-dependencies:
        patterns:
          - "pytest*"
          - "black"
          - "ruff"
```

### Update Schedule

- **Security updates**: Immediate
- **Production dependencies**: Weekly review
- **Development dependencies**: Monthly
- **Major versions**: Manual review required

### Testing Updates

All dependency updates trigger:
1. Unit tests
2. Integration tests
3. Security scans
4. Compatibility checks
5. Performance benchmarks

### Rollback Procedure

If an update causes issues:

```bash
# Revert to previous lock file
git checkout HEAD~1 poetry.lock
poetry install --sync

# Or with pip-tools
git checkout HEAD~1 requirements.lock
pip-sync requirements.lock
```

## Best Practices

### 1. Version Pinning

**DO:**
```txt
ccxt==4.4.0           # Exact version for production
pytest>=8.0.0,<9.0.0  # Range for dev dependencies
```

**DON'T:**
```txt
ccxt                  # No version = unpredictable
ccxt>=4.0.0          # Too broad for production
```

### 2. Security First

- Run security scans before deployment
- Never ignore CRITICAL vulnerabilities
- Document security exceptions
- Keep audit trail of updates

### 3. Dependency Hygiene

```bash
# Remove unused dependencies
poetry show --tree  # View dependency tree
pip-autoremove package-name  # Remove with dependencies

# Check for conflicts
pip check

# Audit installed packages
pip list --outdated
```

### 4. Lock File Management

- **Always commit lock files** - Ensures reproducible builds
- **Update regularly** - Weekly for security patches
- **Test thoroughly** - Run full test suite after updates
- **Document changes** - Note breaking changes in commits

### 5. Tier Isolation

```bash
# Validate tier requirements
python scripts/validate_tiers.py

# Test tier-specific installation
TIER=sniper pip install -r requirements/sniper.txt
TIER=hunter pip install -r requirements/hunter.txt
```

### 6. Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/update-deps

# 2. Update dependencies
poetry update
# or
pip-compile --upgrade requirements/base.txt

# 3. Run tests
pytest tests/

# 4. Run security scan
python -m genesis.security.dependency_scanner

# 5. Commit if all pass
git add poetry.lock requirements.lock
git commit -m "chore: update dependencies

- Updated ccxt from 4.4.0 to 4.4.1
- Security: Fixed CVE-2024-001 in urllib3
- All tests passing"
```

## Troubleshooting

### Common Issues

#### 1. Hash Mismatch

**Error:** `THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE`

**Solution:**
```bash
# Clear pip cache
pip cache purge

# Reinstall with --force-reinstall
pip install --force-reinstall -r requirements.lock
```

#### 2. Dependency Conflicts

**Error:** `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solution:**
```bash
# Use Poetry's resolver
poetry lock --no-update
poetry install

# Or resolve manually
pip-compile --resolver=backtracking requirements/base.txt
```

#### 3. Platform-Specific Dependencies

**Issue:** Package not available for your platform

**Solution:**
```python
# In requirements.txt
numpy==1.26.4; platform_system != "Windows"
numpy==1.26.4; platform_system == "Windows"

# Or use Poetry markers
[tool.poetry.dependencies]
numpy = {version = "1.26.4", markers = "platform_system == 'Linux'"}
```

#### 4. Private Package Access

**Issue:** Can't access private repositories

**Solution:**
```bash
# Configure pip
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url https://your-private-repo.com/simple

# For Poetry
poetry config repositories.private https://your-private-repo.com/simple
poetry config http-basic.private username password
```

### Validation Commands

```bash
# Verify environment
python scripts/validate_tiers.py

# Check installed versions
pip freeze | grep -E "(ccxt|pydantic|structlog)"

# Verify hashes
python scripts/verify_hashes.py

# Test imports
python -c "import genesis; print('✅ Genesis importable')"
```

## Migration Guide

### From pip to Poetry

```bash
# 1. Run migration script
python scripts/migrate_to_poetry.py

# 2. Verify installation
poetry show

# 3. Test application
poetry run python -m genesis
```

### From requirements.txt to lock files

```bash
# 1. Generate lock files
python scripts/update_requirements.py

# 2. Install from lock files
pip install -r requirements.lock

# 3. Verify installation
python scripts/verify_hashes.py
```

## Next Steps

- [Docker deployment](../deployment/docker.md)
- [Security scanning](../security/scanning.md)
- [CI/CD setup](../deployment/cicd.md)