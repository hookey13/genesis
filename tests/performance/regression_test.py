"""
Automated performance regression detection for Genesis trading system.
Compares current performance against historical baselines.
"""

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegressionThresholds:
    """Configurable thresholds for regression detection."""
    p95_degradation_percent: float = 10.0  # Flag if P95 degrades by >10%
    p99_degradation_percent: float = 15.0  # Flag if P99 degrades by >15%
    mean_degradation_percent: float = 20.0  # Flag if mean degrades by >20%
    error_rate_increase_percent: float = 5.0  # Flag if error rate increases by >5%
    throughput_decrease_percent: float = 10.0  # Flag if throughput decreases by >10%
    confidence_level: float = 0.95  # Statistical confidence level


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    is_regression: bool
    confidence: float
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, baseline_dir: Path = None, thresholds: RegressionThresholds = None):
        self.baseline_dir = baseline_dir or Path("test_results/baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or RegressionThresholds()
        self.historical_data: Dict[str, List] = {}
        self.load_historical_data()
        
    def load_historical_data(self):
        """Load historical performance data."""
        history_file = self.baseline_dir / "performance_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.historical_data = json.load(f)
                logger.info(f"Loaded {len(self.historical_data)} historical metrics")
        else:
            logger.info("No historical data found")
            
    def save_historical_data(self):
        """Save historical performance data."""
        history_file = self.baseline_dir / "performance_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2)
            
    def add_performance_data(self, metric_name: str, value: float, timestamp: str = None):
        """Add new performance data point."""
        if metric_name not in self.historical_data:
            self.historical_data[metric_name] = []
            
        self.historical_data[metric_name].append({
            'value': value,
            'timestamp': timestamp or datetime.now().isoformat()
        })
        
        # Keep only last 30 days of data
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        self.historical_data[metric_name] = [
            d for d in self.historical_data[metric_name]
            if d['timestamp'] > cutoff
        ]
        
    def detect_regression(self, metric_name: str, current_value: float) -> Optional[RegressionResult]:
        """Detect if current value represents a regression."""
        if metric_name not in self.historical_data or len(self.historical_data[metric_name]) < 5:
            logger.warning(f"Insufficient historical data for {metric_name}")
            return None
            
        historical_values = [d['value'] for d in self.historical_data[metric_name][-30:]]
        
        # Calculate baseline statistics
        baseline_mean = statistics.mean(historical_values)
        baseline_stdev = statistics.stdev(historical_values)
        baseline_p95 = np.percentile(historical_values, 95)
        baseline_p99 = np.percentile(historical_values, 99)
        
        # Perform statistical tests
        is_regression, confidence = self._statistical_test(historical_values, current_value)
        
        # Calculate change percentage
        change_percent = ((current_value - baseline_mean) / baseline_mean) * 100
        
        # Determine severity
        severity = self._determine_severity(metric_name, change_percent)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(metric_name, change_percent, severity)
        
        return RegressionResult(
            metric_name=metric_name,
            baseline_value=baseline_mean,
            current_value=current_value,
            change_percent=change_percent,
            is_regression=is_regression,
            confidence=confidence,
            severity=severity,
            recommendation=recommendation
        )
        
    def _statistical_test(self, historical_values: List[float], current_value: float) -> Tuple[bool, float]:
        """Perform statistical test for regression."""
        # Use z-test if enough samples, otherwise t-test
        n = len(historical_values)
        mean = statistics.mean(historical_values)
        stdev = statistics.stdev(historical_values)
        
        if stdev == 0:
            return current_value != mean, 1.0
            
        if n >= 30:
            # Z-test
            z_score = (current_value - mean) / (stdev / np.sqrt(n))
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            # T-test
            t_stat, p_value = scipy_stats.ttest_1samp(historical_values, current_value)
            
        # Check if statistically significant degradation
        is_regression = (current_value > mean) and (p_value < (1 - self.thresholds.confidence_level))
        confidence = 1 - p_value
        
        return is_regression, confidence
        
    def _determine_severity(self, metric_name: str, change_percent: float) -> str:
        """Determine severity of regression."""
        if 'p99' in metric_name.lower():
            if change_percent > 50:
                return 'critical'
            elif change_percent > 30:
                return 'high'
            elif change_percent > 15:
                return 'medium'
            else:
                return 'low'
        elif 'error' in metric_name.lower():
            if change_percent > 20:
                return 'critical'
            elif change_percent > 10:
                return 'high'
            elif change_percent > 5:
                return 'medium'
            else:
                return 'low'
        else:
            if change_percent > 40:
                return 'critical'
            elif change_percent > 25:
                return 'high'
            elif change_percent > 10:
                return 'medium'
            else:
                return 'low'
                
    def _generate_recommendation(self, metric_name: str, change_percent: float, severity: str) -> str:
        """Generate actionable recommendation."""
        if severity == 'critical':
            return f"CRITICAL: Immediate investigation required. {metric_name} degraded by {change_percent:.1f}%. Consider rolling back recent changes."
        elif severity == 'high':
            return f"HIGH: Performance regression detected in {metric_name}. Review recent code changes and database queries."
        elif severity == 'medium':
            return f"MEDIUM: Notable performance change in {metric_name}. Monitor closely and investigate if trend continues."
        else:
            return f"LOW: Minor performance variation in {metric_name}. Continue monitoring."
            
    def analyze_load_test_results(self, results_file: Path) -> List[RegressionResult]:
        """Analyze load test results for regressions."""
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        regressions = []
        
        # Analyze each metric
        metrics_to_check = [
            ('response_time_p95', results.get('summary', {}).get('p95_response_time')),
            ('response_time_p99', results.get('summary', {}).get('p99_response_time')),
            ('error_rate', results.get('summary', {}).get('error_rate')),
            ('throughput', results.get('summary', {}).get('requests_per_second'))
        ]
        
        for metric_name, current_value in metrics_to_check:
            if current_value is None:
                continue
                
            # Add to historical data
            self.add_performance_data(metric_name, current_value)
            
            # Check for regression
            regression = self.detect_regression(metric_name, current_value)
            if regression and regression.is_regression:
                regressions.append(regression)
                
        # Save updated historical data
        self.save_historical_data()
        
        return regressions
        
    def generate_trend_analysis(self, metric_name: str, days: int = 7) -> Dict:
        """Generate trend analysis for a metric."""
        if metric_name not in self.historical_data:
            return {'error': 'No data available'}
            
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent_data = [
            d for d in self.historical_data[metric_name]
            if d['timestamp'] > cutoff
        ]
        
        if len(recent_data) < 2:
            return {'error': 'Insufficient data for trend analysis'}
            
        values = [d['value'] for d in recent_data]
        timestamps = [datetime.fromisoformat(d['timestamp']).timestamp() for d in recent_data]
        
        # Calculate linear regression for trend
        slope, intercept = np.polyfit(timestamps, values, 1)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
            
        # Calculate statistics
        return {
            'metric': metric_name,
            'trend': trend,
            'slope': slope,
            'current_value': values[-1],
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'data_points': len(values),
            'days_analyzed': days
        }
        
    def create_regression_report(self, regressions: List[RegressionResult]) -> str:
        """Create detailed regression report."""
        if not regressions:
            return "No performance regressions detected. All metrics within acceptable thresholds."
            
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REGRESSION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Regressions Detected: {len(regressions)}")
        report.append("")
        
        # Group by severity
        critical = [r for r in regressions if r.severity == 'critical']
        high = [r for r in regressions if r.severity == 'high']
        medium = [r for r in regressions if r.severity == 'medium']
        low = [r for r in regressions if r.severity == 'low']
        
        if critical:
            report.append("CRITICAL REGRESSIONS")
            report.append("-" * 40)
            for r in critical:
                report.append(f"• {r.metric_name}:")
                report.append(f"  Baseline: {r.baseline_value:.2f}")
                report.append(f"  Current: {r.current_value:.2f}")
                report.append(f"  Change: {r.change_percent:+.1f}%")
                report.append(f"  Confidence: {r.confidence:.1%}")
                report.append(f"  {r.recommendation}")
                report.append("")
                
        if high:
            report.append("HIGH SEVERITY REGRESSIONS")
            report.append("-" * 40)
            for r in high:
                report.append(f"• {r.metric_name}: {r.change_percent:+.1f}% degradation")
                report.append(f"  {r.recommendation}")
                report.append("")
                
        if medium:
            report.append("MEDIUM SEVERITY REGRESSIONS")
            report.append("-" * 40)
            for r in medium:
                report.append(f"• {r.metric_name}: {r.change_percent:+.1f}% change")
                
        if low:
            report.append("")
            report.append(f"Low severity: {len(low)} minor variations detected")
            
        report.append("")
        report.append("RECOMMENDED ACTIONS")
        report.append("-" * 40)
        
        if critical:
            report.append("1. IMMEDIATE: Review and potentially rollback recent deployments")
            report.append("2. Analyze database query performance and indexes")
            report.append("3. Check for memory leaks or resource exhaustion")
            
        if high or critical:
            report.append("4. Profile code changes in affected areas")
            report.append("5. Review recent dependency updates")
            report.append("6. Verify infrastructure capacity and scaling")
            
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


class ContinuousRegressionMonitor:
    """Continuous monitoring for performance regressions."""
    
    def __init__(self, detector: PerformanceRegressionDetector):
        self.detector = detector
        self.alert_history: List[Dict] = []
        
    async def monitor_metrics(self, metrics: Dict[str, float]):
        """Monitor metrics and alert on regressions."""
        regressions = []
        
        for metric_name, value in metrics.items():
            self.detector.add_performance_data(metric_name, value)
            regression = self.detector.detect_regression(metric_name, value)
            
            if regression and regression.is_regression:
                regressions.append(regression)
                await self.send_alert(regression)
                
        self.detector.save_historical_data()
        return regressions
        
    async def send_alert(self, regression: RegressionResult):
        """Send alert for detected regression."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'metric': regression.metric_name,
            'severity': regression.severity,
            'change_percent': regression.change_percent,
            'message': regression.recommendation
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        if regression.severity == 'critical':
            logger.critical(f"REGRESSION ALERT: {regression.recommendation}")
        elif regression.severity == 'high':
            logger.error(f"REGRESSION: {regression.recommendation}")
        else:
            logger.warning(f"Performance change: {regression.metric_name} ({regression.change_percent:+.1f}%)")
            
        # Here you would integrate with alerting systems (PagerDuty, Slack, etc.)
        
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent alerts."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > cutoff]
        
        return {
            'total_alerts': len(recent_alerts),
            'critical': sum(1 for a in recent_alerts if a['severity'] == 'critical'),
            'high': sum(1 for a in recent_alerts if a['severity'] == 'high'),
            'medium': sum(1 for a in recent_alerts if a['severity'] == 'medium'),
            'low': sum(1 for a in recent_alerts if a['severity'] == 'low'),
            'alerts': recent_alerts[-10:]  # Last 10 alerts
        }


def main():
    """Run regression detection on latest test results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect performance regressions')
    parser.add_argument('--results', required=True, help='Path to test results file')
    parser.add_argument('--baseline-dir', help='Directory for baseline data')
    parser.add_argument('--report', help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PerformanceRegressionDetector(
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None
    )
    
    # Analyze results
    results_file = Path(args.results)
    regressions = detector.analyze_load_test_results(results_file)
    
    # Generate report
    report = detector.create_regression_report(regressions)
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    else:
        print(report)
        
    # Exit with error if critical regressions found
    if any(r.severity == 'critical' for r in regressions):
        logger.error("Critical performance regressions detected!")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())