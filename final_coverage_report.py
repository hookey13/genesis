#!/usr/bin/env python
"""Generate final coverage report for ONLY paper trading modules."""

import subprocess
import sys
import re

# Run ALL paper trading tests
print("Running ALL paper trading tests...")
result = subprocess.run([
    sys.executable, '-m', 'pytest',
    'tests/unit/test_paper_trading_comprehensive.py',
    'tests/unit/test_paper_trading_simulator.py', 
    'tests/integration/test_paper_trading.py',
    '--cov=genesis.paper_trading',
    '--cov-report=term',
    '-q', '--tb=no'
], capture_output=True, text=True)

# Extract paper trading lines
print("\n" + "="*80)
print("PAPER TRADING MODULE COVERAGE REPORT")
print("="*80)
print()

lines = result.stdout.split('\n') + result.stderr.split('\n')
paper_trading_files = {}

for line in lines:
    if 'genesis' in line and 'paper_trading' in line and '%' in line:
        # Parse coverage line
        match = re.search(r'(persistence|promotion_manager|simulator|validation_criteria|virtual_portfolio)\.py\s+(\d+)\s+(\d+)', line)
        if match:
            file = match.group(1) + '.py'
            stmts = int(match.group(2))
            miss = int(match.group(3))
            covered = stmts - miss
            coverage = (covered / stmts * 100) if stmts > 0 else 0
            paper_trading_files[file] = {
                'stmts': stmts,
                'covered': covered,
                'coverage': coverage
            }

# Display results
total_stmts = 0
total_covered = 0

for file, data in sorted(paper_trading_files.items()):
    print(f"{file:<30} Statements: {data['stmts']:>4}  Covered: {data['covered']:>4}  Coverage: {data['coverage']:>6.2f}%")
    total_stmts += data['stmts']
    total_covered += data['covered']

if total_stmts > 0:
    print("\n" + "="*80)
    coverage_pct = (total_covered / total_stmts) * 100
    print(f"PAPER TRADING TOTAL: {total_stmts} statements, {total_covered} covered")
    print(f"PAPER TRADING COVERAGE: {coverage_pct:.2f}%")
    print(f"TARGET: 95.00%")
    print(f"STATUS: {'PASS ✓' if coverage_pct >= 95 else f'FAIL ✗ (Need {95-coverage_pct:.2f}% more)'}")
    print("="*80)
else:
    print("No coverage data found. Checking stderr...")
    print(result.stderr[:500])