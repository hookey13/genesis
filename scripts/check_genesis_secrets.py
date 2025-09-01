#!/usr/bin/env python3
"""
Custom secret detection script for Genesis-specific patterns.
Checks for trading platform credentials, API keys, and sensitive configuration.
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Genesis-specific secret patterns
GENESIS_SECRET_PATTERNS = [
    # API Keys and Secrets
    (r'(?i)binance[_\-]?api[_\-]?(key|secret)', 'Binance API credentials'),
    (r'(?i)coinbase[_\-]?api[_\-]?(key|secret)', 'Coinbase API credentials'),
    (r'(?i)kraken[_\-]?api[_\-]?(key|secret)', 'Kraken API credentials'),
    (r'(?i)ftx[_\-]?api[_\-]?(key|secret)', 'FTX API credentials'),
    
    # Database credentials
    (r'(?i)database[_\-]?(password|key|secret)', 'Database credentials'),
    (r'(?i)postgres[_\-]?(password|key|secret)', 'PostgreSQL credentials'),
    (r'(?i)sqlite[_\-]?(password|key|encryption)', 'SQLite encryption key'),
    
    # Vault and encryption
    (r'(?i)vault[_\-]?token\s*=\s*["\'][^"\']+["\']', 'Vault token literal'),
    (r'(?i)encryption[_\-]?key\s*=\s*["\'][^"\']+["\']', 'Encryption key literal'),
    (r'(?i)master[_\-]?key\s*=\s*["\'][^"\']+["\']', 'Master key literal'),
    
    # Trading specific
    (r'(?i)trading[_\-]?(password|secret|key)', 'Trading credentials'),
    (r'(?i)webhook[_\-]?secret', 'Webhook secret'),
    (r'(?i)jwt[_\-]?secret', 'JWT secret'),
    
    # AWS/Cloud
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key ID'),
    (r'(?i)aws[_\-]?secret[_\-]?access[_\-]?key', 'AWS Secret Access Key'),
    (r'(?i)digitalocean[_\-]?token', 'DigitalOcean token'),
    
    # Private keys
    (r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----', 'Private key block'),
    (r'(?i)private[_\-]?key\s*=\s*["\'][^"\']+["\']', 'Private key literal'),
    
    # Genesis paths
    (r'/genesis/exchange/api-keys', 'Genesis API key path'),
    (r'/genesis/database/encryption-key', 'Genesis encryption key path'),
    (r'~/.genesis/.secrets/', 'Genesis secrets directory'),
]

# Patterns that are allowed (false positives)
ALLOWED_PATTERNS = [
    r'(?i)example',
    r'(?i)test',
    r'(?i)fake',
    r'(?i)dummy',
    r'(?i)mock',
    r'(?i)placeholder',
    r'(?i)todo',
    r'(?i)fixme',
    r'<[^>]+>',  # Template variables
    r'\{[^}]+\}',  # Format strings
]

# Files to skip
SKIP_FILES = {
    '.pre-commit-config.yaml',
    '.secrets.baseline',
    'check_genesis_secrets.py',
    'requirements.txt',
    'poetry.lock',
    'package-lock.json',
}

# Directories to skip
SKIP_DIRS = {
    '.git',
    '__pycache__',
    '.pytest_cache',
    'node_modules',
    'venv',
    '.venv',
    'env',
    '.env',
}


def is_allowed(line: str) -> bool:
    """Check if a line contains allowed patterns (false positives)."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def check_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Check a single file for secret patterns.
    
    Args:
        filepath: Path to the file to check
    
    Returns:
        List of (line_number, pattern_description, line_content) tuples
    """
    violations = []
    
    # Skip if file should be ignored
    if filepath.name in SKIP_FILES:
        return violations
    
    # Skip if in ignored directory
    for skip_dir in SKIP_DIRS:
        if skip_dir in filepath.parts:
            return violations
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines and comments
                stripped = line.strip()
                if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                    continue
                
                # Skip if line contains allowed patterns
                if is_allowed(line):
                    continue
                
                # Check against secret patterns
                for pattern, description in GENESIS_SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Double-check it's not in a docstring or comment
                        if '"""' in line or "'''" in line or '# ' in line:
                            # Could be documentation, do additional check
                            if not re.search(r'=\s*["\'][^"\']+["\']', line):
                                continue
                        
                        violations.append((line_num, description, line.strip()))
                        break  # Only report first match per line
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    
    return violations


def check_json_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Special handling for JSON files to check keys and values.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        List of violations
    """
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def check_dict(obj: Dict[str, Any], path: str = ""):
            """Recursively check dictionary for secrets."""
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check key names
                for pattern, description in GENESIS_SECRET_PATTERNS:
                    if re.search(pattern, key, re.IGNORECASE):
                        if not is_allowed(key):
                            violations.append((0, f"{description} in key", current_path))
                
                # Check string values
                if isinstance(value, str):
                    for pattern, description in GENESIS_SECRET_PATTERNS:
                        if re.search(pattern, value):
                            if not is_allowed(value):
                                violations.append((0, f"{description} in value", f"{current_path}={value[:50]}..."))
                
                # Recurse into nested objects
                elif isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            check_dict(item, f"{current_path}[{i}]")
        
        if isinstance(data, dict):
            check_dict(data)
    
    except Exception as e:
        # Not a valid JSON file or error reading, skip
        pass
    
    return violations


def main(files: List[str]) -> int:
    """
    Main function to check files for secrets.
    
    Args:
        files: List of file paths to check
    
    Returns:
        Exit code (0 for success, 1 for violations found)
    """
    all_violations = []
    
    for file_str in files:
        filepath = Path(file_str)
        
        if not filepath.exists() or not filepath.is_file():
            continue
        
        # Special handling for JSON files
        if filepath.suffix == '.json':
            violations = check_json_file(filepath)
        else:
            violations = check_file(filepath)
        
        if violations:
            all_violations.append((filepath, violations))
    
    # Report violations
    if all_violations:
        print("\n❌ Potential secrets detected:\n", file=sys.stderr)
        
        for filepath, violations in all_violations:
            print(f"  📄 {filepath}:", file=sys.stderr)
            for line_num, description, content in violations:
                if line_num > 0:
                    print(f"    Line {line_num}: {description}", file=sys.stderr)
                    print(f"      > {content[:100]}{'...' if len(content) > 100 else ''}", file=sys.stderr)
                else:
                    print(f"    {description}: {content}", file=sys.stderr)
            print(file=sys.stderr)
        
        print("⚠️  Please review and remove any actual secrets before committing.", file=sys.stderr)
        print("   If these are false positives, add them to the allowed patterns.", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))