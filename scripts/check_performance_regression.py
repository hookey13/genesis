#!/usr/bin/env python3
"""Performance regression detection script for CI/CD pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_benchmark_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_benchmarks(
    current: Dict,
    baseline: Dict,
    threshold: float = 0.1
) -> Tuple[bool, List[Dict]]:
    """Compare current benchmark results with baseline.
    
    Args:
        current: Current benchmark results
        baseline: Baseline benchmark results
        threshold: Regression threshold (0.1 = 10% degradation allowed)
    
    Returns:
        Tuple of (passed, regressions)
    """
    regressions = []
    
    # Extract metrics from both benchmarks
    current_metrics = extract_metrics(current)
    baseline_metrics = extract_metrics(baseline)
    
    # Compare each metric
    for metric_name, current_value in current_metrics.items():
        if metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name]
            
            # Calculate degradation percentage
            if baseline_value > 0:
                degradation = (current_value - baseline_value) / baseline_value
                
                # Check if regression exceeds threshold
                if degradation > threshold:
                    regressions.append({
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "degradation": degradation * 100,
                        "threshold": threshold * 100
                    })
    
    passed = len(regressions) == 0
    return passed, regressions


def extract_metrics(benchmark: Dict) -> Dict[str, float]:
    """Extract performance metrics from benchmark results.
    
    Args:
        benchmark: Benchmark results
    
    Returns:
        Dictionary of metric name to value
    """
    metrics = {}
    
    # Handle pytest-benchmark format
    if "benchmarks" in benchmark:
        for test in benchmark["benchmarks"]:
            name = test.get("name", "unknown")
            # Use mean time as the metric
            metrics[f"{name}_mean"] = test.get("stats", {}).get("mean", 0)
            metrics[f"{name}_max"] = test.get("stats", {}).get("max", 0)
            metrics[f"{name}_min"] = test.get("stats", {}).get("min", 0)
    
    # Handle custom format
    elif "metrics" in benchmark:
        for key, value in benchmark["metrics"].items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, dict):
                # Handle nested metrics
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics[f"{key}_{subkey}"] = subvalue
    
    # Handle simple key-value format
    else:
        for key, value in benchmark.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
    
    return metrics


def format_regression_report(regressions: List[Dict]) -> str:
    """Format regression report for display.
    
    Args:
        regressions: List of regression details
    
    Returns:
        Formatted report string
    """
    if not regressions:
        return "✅ No performance regressions detected!"
    
    report = ["❌ Performance regressions detected:", ""]
    
    # Create table header
    report.append("| Metric | Baseline | Current | Degradation | Threshold |")
    report.append("|--------|----------|---------|-------------|-----------|")
    
    # Add regression details
    for reg in regressions:
        report.append(
            f"| {reg['metric']} | "
            f"{reg['baseline']:.4f} | "
            f"{reg['current']:.4f} | "
            f"{reg['degradation']:.1f}% | "
            f"{reg['threshold']:.1f}% |"
        )
    
    report.append("")
    report.append(f"Total regressions: {len(regressions)}")
    
    return "\n".join(report)


def generate_github_comment(
    passed: bool,
    regressions: List[Dict],
    current_file: str,
    baseline_file: str
) -> str:
    """Generate GitHub PR comment with performance results.
    
    Args:
        passed: Whether tests passed
        regressions: List of regressions
        current_file: Current benchmark file path
        baseline_file: Baseline benchmark file path
    
    Returns:
        GitHub-formatted comment
    """
    status_emoji = "✅" if passed else "❌"
    status_text = "passed" if passed else "failed"
    
    comment = [
        f"## {status_emoji} Performance Test Results",
        "",
        f"Performance tests **{status_text}**.",
        "",
        f"- **Current benchmark**: `{current_file}`",
        f"- **Baseline benchmark**: `{baseline_file}`",
        ""
    ]
    
    if regressions:
        comment.append("### ⚠️ Performance Regressions Detected")
        comment.append("")
        comment.append("| Metric | Baseline | Current | Degradation |")
        comment.append("|--------|----------|---------|-------------|")
        
        for reg in regressions:
            comment.append(
                f"| {reg['metric']} | "
                f"{reg['baseline']:.4f} | "
                f"{reg['current']:.4f} | "
                f"**+{reg['degradation']:.1f}%** |"
            )
        
        comment.append("")
        comment.append("### 📊 Recommendations")
        comment.append("")
        comment.append("- Review changes that might have impacted performance")
        comment.append("- Consider profiling the affected code paths")
        comment.append("- Run benchmarks locally to reproduce the regression")
    else:
        comment.append("### ✨ Performance Maintained")
        comment.append("")
        comment.append("No performance regressions detected. Good job!")
    
    comment.append("")
    comment.append("---")
    comment.append("*Generated by GENESIS Performance CI*")
    
    return "\n".join(comment)


def update_baseline(current_file: str, baseline_file: str) -> None:
    """Update baseline with current results.
    
    Args:
        current_file: Current benchmark file
        baseline_file: Baseline file to update
    """
    current = load_benchmark_results(current_file)
    
    # Save as new baseline
    with open(baseline_file, 'w') as f:
        json.dump(current, f, indent=2)
    
    print(f"✅ Updated baseline: {baseline_file}")


def main():
    """Main entry point for performance regression detection."""
    parser = argparse.ArgumentParser(
        description="Check for performance regressions in benchmark results"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current benchmark results JSON"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline benchmark results JSON"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Regression threshold (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current results"
    )
    parser.add_argument(
        "--github-comment",
        action="store_true",
        help="Generate GitHub PR comment format"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.current).exists():
        print(f"❌ Current benchmark file not found: {args.current}")
        sys.exit(1)
    
    # Load current benchmark
    try:
        current = load_benchmark_results(args.current)
    except Exception as e:
        print(f"❌ Failed to load current benchmark: {e}")
        sys.exit(1)
    
    # Handle baseline
    if Path(args.baseline).exists():
        try:
            baseline = load_benchmark_results(args.baseline)
        except Exception as e:
            print(f"❌ Failed to load baseline benchmark: {e}")
            sys.exit(1)
    else:
        print(f"⚠️ Baseline not found, creating new baseline: {args.baseline}")
        update_baseline(args.current, args.baseline)
        sys.exit(0)
    
    # Compare benchmarks
    passed, regressions = compare_benchmarks(current, baseline, args.threshold)
    
    # Generate output
    if args.github_comment:
        output = generate_github_comment(
            passed, regressions, args.current, args.baseline
        )
    else:
        output = format_regression_report(regressions)
    
    # Display or save output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to: {args.output}")
    else:
        print(output)
    
    # Update baseline if requested
    if args.update_baseline:
        update_baseline(args.current, args.baseline)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()