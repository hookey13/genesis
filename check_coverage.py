#!/usr/bin/env python
"""Check test coverage for paper trading modules only."""

import subprocess
import sys

# Run tests with coverage for paper trading only
result = subprocess.run(
    [sys.executable, "-m", "pytest", 
     "tests/unit/test_paper_trading_simulator.py",
     "tests/integration/test_paper_trading.py",
     "--cov=genesis.paper_trading",
     "--cov-report=term",
     "-q"],
    capture_output=True,
    text=True
)

# Parse output to find coverage lines
lines = result.stdout.split('\n')
for line in lines:
    if 'genesis/paper_trading' in line or 'TOTAL' in line or '---' in line:
        print(line)

# Check if tests passed
if result.returncode != 0:
    print("\nSome tests failed")
    sys.exit(1)