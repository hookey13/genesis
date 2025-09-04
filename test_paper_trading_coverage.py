#!/usr/bin/env python
"""Test script to measure ONLY paper trading module coverage."""

import subprocess
import sys
import os

# Change to genesis directory
os.chdir('.')

# Run coverage for ONLY paper trading modules
print("Running tests with coverage for paper_trading modules only...")
result = subprocess.run([
    sys.executable, '-m', 'coverage', 'run',
    '--source=genesis/paper_trading',
    '-m', 'pytest',
    'tests/unit/test_paper_trading_simulator.py',
    'tests/integration/test_paper_trading.py',
    '-q'
], capture_output=True, text=True)

print("Test output:", result.stdout[:500] if result.stdout else "No output")
if result.stderr:
    print("Errors:", result.stderr[:500])

# Generate coverage report for ONLY paper trading
print("\n" + "="*60)
print("PAPER TRADING MODULE COVERAGE REPORT")
print("="*60)

report_result = subprocess.run([
    sys.executable, '-m', 'coverage', 'report',
    '--include=genesis/paper_trading/*'
], capture_output=True, text=True)

print(report_result.stdout)

# Parse the coverage percentage
lines = report_result.stdout.split('\n')
for line in lines:
    if 'TOTAL' in line:
        parts = line.split()
        if len(parts) >= 5:
            coverage_pct = parts[-1].replace('%', '')
            try:
                coverage = float(coverage_pct)
                print(f"\n{'='*60}")
                print(f"PAPER TRADING MODULE COVERAGE: {coverage:.2f}%")
                print(f"TARGET: 95.00%")
                print(f"STATUS: {'PASS ✓' if coverage >= 95 else 'FAIL ✗'}")
                print(f"{'='*60}")
                
                if coverage >= 95:
                    print("\n✓ COVERAGE TARGET MET!")
                else:
                    print(f"\n✗ Need to increase coverage by {95-coverage:.2f}%")
            except:
                pass

sys.exit(result.returncode)