#!/usr/bin/env python
"""Quick check of paper trading coverage."""

import subprocess
import sys
import re

# Run coverage with all tests
print("Calculating paper trading coverage...")
result = subprocess.run([
    sys.executable, '-m', 'pytest',
    'tests/unit/test_paper_trading_comprehensive.py',
    'tests/unit/test_paper_trading_coverage_95.py',
    'tests/unit/test_paper_trading_simulator.py',
    'tests/integration/test_paper_trading.py',
    '--cov=genesis.paper_trading',
    '--cov-report=term',
    '--tb=no', '-q'
], capture_output=True, text=True, timeout=60)

# Parse output
lines = result.stdout.split('\n') + result.stderr.split('\n')

# Find coverage lines
coverage_data = {}
for line in lines:
    for module in ['persistence', 'promotion_manager', 'simulator', 'validation_criteria', 'virtual_portfolio']:
        if module + '.py' in line and 'genesis' in line:
            # Extract numbers
            parts = line.split()
            for i, part in enumerate(parts):
                if module + '.py' in part:
                    try:
                        stmts = int(parts[i+1])
                        miss = int(parts[i+2])
                        coverage = 100 * (stmts - miss) / stmts
                        coverage_data[module] = {
                            'stmts': stmts,
                            'miss': miss,
                            'coverage': coverage
                        }
                        break
                    except:
                        pass

# Display results
total_stmts = 0
total_covered = 0

print("\nPAPER TRADING MODULE COVERAGE:")
print("="*60)
for module in sorted(coverage_data.keys()):
    data = coverage_data[module]
    covered = data['stmts'] - data['miss']
    print(f"{module + '.py':<30} {data['coverage']:>6.2f}% ({covered}/{data['stmts']})")
    total_stmts += data['stmts']
    total_covered += covered

if total_stmts > 0:
    total_coverage = 100 * total_covered / total_stmts
    print("="*60)
    print(f"TOTAL COVERAGE: {total_coverage:.2f}% ({total_covered}/{total_stmts} statements)")
    print(f"TARGET: 95.00%")
    
    if total_coverage >= 95:
        print("✓ TARGET MET!")
    else:
        print(f"Need {95 - total_coverage:.2f}% more ({int((0.95 * total_stmts) - total_covered)} statements)")