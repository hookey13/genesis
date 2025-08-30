# CI/CD Security Pipeline

This document describes the security scanning integrated into the CI/CD pipeline for Project GENESIS.

## Overview

The security pipeline implements defense-in-depth with multiple scanning layers:
- **SAST** (Static Application Security Testing)
- **Dependency scanning** for known vulnerabilities  
- **Secret detection** to prevent credential leaks
- **License compliance** checking
- **Container scanning** for Docker images
- **Security gates** that block dangerous code

## GitHub Actions Workflow

### Trigger Events

Security scans run on:
- Every push to `main` or `develop` branches
- All pull requests to `main`
- Daily scheduled scan at 2 AM UTC
- Manual trigger via workflow dispatch

### Security Jobs

#### 1. SAST - Bandit Scan

Performs static analysis of Python code for security issues.

**Configuration**: `.bandit`
```yaml
- Severity threshold: MEDIUM
- Confidence threshold: MEDIUM  
- Excludes: tests, venv, build directories
```

**Checks for**:
- Hardcoded passwords and secrets
- SQL injection vulnerabilities
- Command injection risks
- Insecure cryptography usage
- Dangerous function calls
- Path traversal vulnerabilities

**Failure conditions**:
- Any HIGH severity findings
- CRITICAL severity findings

#### 2. Dependency Vulnerability Scan

Uses `pip-audit` to check for known vulnerabilities in dependencies.

**Scans**:
- All requirements files (base, dev, production)
- Direct and transitive dependencies
- CVE database cross-reference

**Failure conditions**:
- CRITICAL severity CVEs
- HIGH severity CVEs in production dependencies

#### 3. Secret Detection

Multiple tools to prevent credential leaks:

**TruffleHog**:
- Scans entire git history
- Entropy-based detection
- Regex pattern matching

**Gitleaks**:
- Pre-commit hook compatible
- Custom rule support
- Allowlist configuration

**Checks for**:
- API keys and tokens
- Private keys
- Passwords
- Connection strings
- AWS credentials
- Database URLs

#### 4. License Compliance

Ensures all dependencies have acceptable licenses.

**Allowed licenses**:
- MIT
- Apache 2.0
- BSD (2-clause, 3-clause)
- ISC
- Python Software Foundation

**Forbidden licenses**:
- GPL (all versions)
- AGPL
- LGPL
- SSPL
- Commercial licenses

#### 5. Code Quality & Security

Additional security-focused quality checks:

**Ruff**:
- Security-focused linting rules
- Complexity analysis
- Dead code detection

**mypy**:
- Type safety verification
- Null safety checks
- Type confusion prevention

**Safety**:
- Known vulnerability database
- Insecure package detection
- Version pinning recommendations

#### 6. OWASP Dependency Check

Comprehensive vulnerability detection:
- CVE scanning
- NVD database integration
- CVSS scoring
- Retired package detection

### Security Gates

Pipeline fails if:
1. HIGH or CRITICAL vulnerabilities found
2. Secrets detected in code
3. Forbidden licenses present
4. Security tests fail
5. Type safety violations (strict mode)

## Local Security Scanning

### Pre-commit Hooks

Install pre-commit hooks for local scanning:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-c', '.bandit']
        
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```

### Manual Scanning

#### Run Bandit locally
```bash
# Install
pip install bandit[toml]

# Scan
bandit -r genesis/ -f json -o report.json

# With custom config
bandit -c .bandit -r genesis/
```

#### Run pip-audit locally
```bash
# Install
pip install pip-audit

# Scan current environment
pip-audit

# Scan requirements file
pip-audit -r requirements/base.txt

# Fix vulnerabilities
pip-audit --fix
```

#### Run safety check
```bash
# Install
pip install safety

# Scan
safety check

# With detailed output
safety check --json --detailed

# Scan specific file
safety check -r requirements/base.txt
```

## Security Exceptions

### Creating Exceptions

For false positives, create exceptions:

#### Bandit exceptions
```python
# nosec B105 - Not a hardcoded password
test_password = "test123"  # nosec
```

#### Gitleaks exceptions
`.gitleaksignore`:
```
path/to/file.py:line_number:commit_hash
```

#### pip-audit exceptions
```bash
# Ignore specific vulnerability
pip-audit --ignore-vuln VULN-ID
```

### Exception Documentation

All security exceptions must be documented:

`docs/security/exceptions.md`:
```markdown
## Security Exception Log

### Exception 1
- **Tool**: Bandit
- **Issue**: B105 - Hardcoded password
- **File**: tests/fixtures.py:42
- **Reason**: Test fixture, not production code
- **Approved by**: Security Team
- **Date**: 2024-01-15
```

## CI/CD Security Best Practices

### 1. Secret Management

**Never commit secrets**:
- Use environment variables
- Use GitHub Secrets for CI/CD
- Use HashiCorp Vault for production

**GitHub Secrets setup**:
```yaml
env:
  VAULT_TOKEN: ${{ secrets.VAULT_TOKEN }}
  BINANCE_API_KEY: ${{ secrets.BINANCE_API_KEY }}
```

### 2. Dependency Management

**Pin versions**:
```txt
# Good
cryptography==41.0.5

# Bad  
cryptography>=41.0.0
```

**Regular updates**:
```bash
# Check for updates
pip list --outdated

# Update with audit
pip-audit --fix
```

### 3. Security Notifications

Configure GitHub security alerts:
1. Settings → Security → Code security and analysis
2. Enable:
   - Dependency graph
   - Dependabot alerts
   - Dependabot security updates
   - Code scanning alerts
   - Secret scanning alerts

### 4. Branch Protection

Enforce security checks:
```yaml
# .github/branch-protection.yml
protection_rules:
  - name: main
    required_status_checks:
      - security-scan / sast-scan
      - security-scan / dependency-scan
      - security-scan / secret-scan
      - security-scan / security-gate
```

## Security Metrics

### Dashboard Metrics

Track in Grafana/Prometheus:
- Vulnerabilities by severity
- Time to remediation
- Dependency update lag
- Security scan coverage
- False positive rate

### KPIs

- **MTTD** (Mean Time To Detect): < 1 hour
- **MTTR** (Mean Time To Remediate): < 24 hours for CRITICAL
- **Vulnerability density**: < 1 per 1000 LOC
- **Dependency freshness**: > 90% up-to-date
- **Security gate pass rate**: > 95%

## Incident Response

### Security Finding Process

1. **Detection**: Automated scan finds issue
2. **Triage**: Assess severity and impact
3. **Notification**: Alert relevant team
4. **Remediation**: Fix or create exception
5. **Verification**: Re-scan to confirm fix
6. **Documentation**: Update security log

### Severity Levels

| Level | Response Time | Action |
|-------|--------------|--------|
| CRITICAL | Immediate | Stop deployment, hotfix |
| HIGH | 24 hours | Fix in next release |
| MEDIUM | 1 week | Schedule fix |
| LOW | 1 month | Backlog |

## Compliance Reports

### Generate Security Report

```bash
# Aggregate all security reports
python scripts/generate_security_report.py \
  --bandit-report bandit-report.json \
  --audit-report audit-report.json \
  --output security-report.html
```

### Report Contents

- Executive summary
- Vulnerability breakdown by severity
- Dependency risk assessment
- License compliance status
- Remediation recommendations
- Trend analysis

## Tools Reference

### Security Scanning Tools

| Tool | Purpose | Language | License |
|------|---------|----------|---------|
| Bandit | SAST | Python | Apache 2.0 |
| pip-audit | Dependency scan | Python | Apache 2.0 |
| Safety | Vulnerability DB | Python | MIT |
| TruffleHog | Secret detection | Go | GPL 2.0 |
| Gitleaks | Secret detection | Go | MIT |
| Trivy | Container scan | Go | Apache 2.0 |
| OWASP DC | Dependency check | Java | Apache 2.0 |

### Integration Examples

#### GitLab CI
```yaml
security-scan:
  stage: test
  script:
    - bandit -r genesis/
    - pip-audit
    - safety check
```

#### Jenkins
```groovy
pipeline {
  stages {
    stage('Security Scan') {
      steps {
        sh 'bandit -r genesis/ -f json -o report.json'
        sh 'pip-audit --desc'
        publishHTML([reportFiles: 'report.json'])
      }
    }
  }
}
```

#### CircleCI
```yaml
version: 2.1
jobs:
  security-scan:
    docker:
      - image: python:3.11
    steps:
      - checkout
      - run: pip install bandit pip-audit
      - run: bandit -r genesis/
      - run: pip-audit
```

## Troubleshooting

### Common Issues

#### 1. False Positives

**Problem**: Legitimate code flagged as vulnerable
**Solution**: Add inline exception with documentation

#### 2. Dependency Conflicts

**Problem**: Security fix breaks compatibility
**Solution**: Use version ranges, test thoroughly

#### 3. Slow Scans

**Problem**: Security scans timeout
**Solution**: Parallelize jobs, cache dependencies

#### 4. Secret Detection Noise

**Problem**: Too many false positives for secrets
**Solution**: Tune entropy thresholds, add patterns

## Future Enhancements

- [ ] DAST (Dynamic Application Security Testing)
- [ ] IAST (Interactive Application Security Testing)  
- [ ] Fuzzing integration
- [ ] Security benchmarking
- [ ] Automated penetration testing
- [ ] Supply chain security (SBOM)
- [ ] Runtime security monitoring