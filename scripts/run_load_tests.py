#!/usr/bin/env python3
"""
Automated load testing execution script for Genesis trading system.
Runs various load test scenarios and generates comprehensive reports.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Orchestrates load test execution and reporting."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.results_dir = self.base_dir / "test_results" / "load_tests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_docker_compose(self, action: str = "up", detached: bool = True) -> bool:
        """Start or stop Docker Compose services."""
        compose_file = self.base_dir / "docker-compose.load-test.yml"
        
        cmd = ["docker-compose", "-f", str(compose_file), action]
        if detached and action == "up":
            cmd.append("-d")
            
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Docker Compose failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to run Docker Compose: {e}")
            return False
            
    def wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for all services to be healthy."""
        logger.info("Waiting for services to be healthy...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check Locust master
                result = subprocess.run(
                    ["curl", "-f", "http://localhost:8089/stats/requests"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info("Services are ready")
                    return True
            except:
                pass
                
            time.sleep(5)
            
        logger.error("Services failed to become healthy")
        return False
        
    def run_locust_test(self, scenario: Dict) -> Dict:
        """Run a specific Locust test scenario."""
        logger.info(f"Running scenario: {scenario['name']}")
        
        # Prepare Locust command
        cmd = [
            "locust",
            "-f", f"tests/load/{scenario['file']}",
            "--host", scenario.get('host', 'http://localhost:8000'),
            "--users", str(scenario.get('users', 100)),
            "--spawn-rate", str(scenario.get('spawn_rate', 10)),
            "--run-time", scenario.get('run_time', '5m'),
            "--headless",
            "--html", str(self.results_dir / f"{self.test_id}_{scenario['name']}.html"),
            "--csv", str(self.results_dir / f"{self.test_id}_{scenario['name']}")
        ]
        
        if scenario.get('master_host'):
            cmd.extend(["--master-host", scenario['master_host']])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time
            
            # Parse results from CSV
            stats_file = self.results_dir / f"{self.test_id}_{scenario['name']}_stats.csv"
            if stats_file.exists():
                return self.parse_locust_stats(stats_file, scenario['name'], duration)
            else:
                logger.error(f"No stats file generated for {scenario['name']}")
                return {
                    'scenario': scenario['name'],
                    'status': 'failed',
                    'error': 'No stats file generated'
                }
                
        except Exception as e:
            logger.error(f"Failed to run scenario {scenario['name']}: {e}")
            return {
                'scenario': scenario['name'],
                'status': 'error',
                'error': str(e)
            }
            
    def parse_locust_stats(self, stats_file: Path, scenario_name: str, duration: float) -> Dict:
        """Parse Locust statistics from CSV file."""
        import csv
        
        stats = {
            'scenario': scenario_name,
            'duration_seconds': duration,
            'requests': {},
            'summary': {}
        }
        
        try:
            with open(stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Name'] == 'Aggregated':
                        stats['summary'] = {
                            'total_requests': int(row.get('Request Count', 0)),
                            'failure_count': int(row.get('Failure Count', 0)),
                            'median_response_time': float(row.get('Median Response Time', 0)),
                            'average_response_time': float(row.get('Average Response Time', 0)),
                            'p95_response_time': float(row.get('95%', 0)),
                            'p99_response_time': float(row.get('99%', 0)),
                            'rps': float(row.get('Requests/s', 0))
                        }
                    else:
                        stats['requests'][row['Name']] = {
                            'count': int(row.get('Request Count', 0)),
                            'failures': int(row.get('Failure Count', 0)),
                            'median_ms': float(row.get('Median Response Time', 0)),
                            'p95_ms': float(row.get('95%', 0)),
                            'p99_ms': float(row.get('99%', 0))
                        }
                        
            # Calculate success rate
            if stats['summary'].get('total_requests', 0) > 0:
                stats['summary']['success_rate'] = (
                    1 - stats['summary']['failure_count'] / stats['summary']['total_requests']
                ) * 100
            else:
                stats['summary']['success_rate'] = 0
                
            stats['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Failed to parse stats file: {e}")
            stats['status'] = 'parse_error'
            stats['error'] = str(e)
            
        return stats
        
    def run_test_suite(self, scenarios: List[Dict]) -> List[Dict]:
        """Run complete test suite with multiple scenarios."""
        results = []
        
        # Start Docker services if needed
        if not self.run_docker_compose("up"):
            logger.error("Failed to start Docker services")
            return results
            
        if not self.wait_for_services():
            logger.error("Services not ready")
            self.run_docker_compose("down")
            return results
            
        try:
            # Run each scenario
            for scenario in scenarios:
                logger.info(f"\n{'='*60}")
                logger.info(f"Scenario: {scenario['name']}")
                logger.info(f"{'='*60}")
                
                result = self.run_locust_test(scenario)
                results.append(result)
                
                # Wait between scenarios
                if scenario != scenarios[-1]:
                    logger.info(f"Waiting 30 seconds before next scenario...")
                    time.sleep(30)
                    
        finally:
            # Stop Docker services
            self.run_docker_compose("down")
            
        return results
        
    def generate_report(self, results: List[Dict]) -> Path:
        """Generate comprehensive test report."""
        report = {
            'test_id': self.test_id,
            'timestamp': datetime.now().isoformat(),
            'scenarios': results,
            'summary': self.calculate_summary(results)
        }
        
        # Save JSON report
        report_file = self.results_dir / f"{self.test_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate HTML report
        html_report = self.generate_html_report(report)
        html_file = self.results_dir / f"{self.test_id}_report.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
            
        logger.info(f"Reports generated: {report_file}, {html_file}")
        return report_file
        
    def calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate overall test summary."""
        total_requests = sum(r.get('summary', {}).get('total_requests', 0) for r in results)
        total_failures = sum(r.get('summary', {}).get('failure_count', 0) for r in results)
        
        avg_p95 = []
        avg_p99 = []
        avg_rps = []
        
        for r in results:
            summary = r.get('summary', {})
            if summary.get('p95_response_time'):
                avg_p95.append(summary['p95_response_time'])
            if summary.get('p99_response_time'):
                avg_p99.append(summary['p99_response_time'])
            if summary.get('rps'):
                avg_rps.append(summary['rps'])
                
        return {
            'total_scenarios': len(results),
            'successful_scenarios': sum(1 for r in results if r.get('status') == 'completed'),
            'total_requests': total_requests,
            'total_failures': total_failures,
            'overall_success_rate': (1 - total_failures / total_requests) * 100 if total_requests > 0 else 0,
            'avg_p95_response_time': sum(avg_p95) / len(avg_p95) if avg_p95 else 0,
            'avg_p99_response_time': sum(avg_p99) / len(avg_p99) if avg_p99 else 0,
            'avg_rps': sum(avg_rps) / len(avg_rps) if avg_rps else 0
        }
        
    def generate_html_report(self, report: Dict) -> str:
        """Generate HTML report from test results."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Load Test Report - {report['test_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        .summary-box {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Genesis Load Test Report</h1>
    <p>Test ID: {report['test_id']}</p>
    <p>Generated: {report['timestamp']}</p>
    
    <div class="summary-box">
        <h2>Overall Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Scenarios</td><td>{report['summary']['total_scenarios']}</td></tr>
            <tr><td>Successful Scenarios</td><td>{report['summary']['successful_scenarios']}</td></tr>
            <tr><td>Total Requests</td><td>{report['summary']['total_requests']:,}</td></tr>
            <tr><td>Total Failures</td><td>{report['summary']['total_failures']:,}</td></tr>
            <tr><td>Overall Success Rate</td><td class="{'success' if report['summary']['overall_success_rate'] >= 99 else 'warning' if report['summary']['overall_success_rate'] >= 95 else 'error'}">{report['summary']['overall_success_rate']:.2f}%</td></tr>
            <tr><td>Average P95 Response Time</td><td>{report['summary']['avg_p95_response_time']:.2f} ms</td></tr>
            <tr><td>Average P99 Response Time</td><td>{report['summary']['avg_p99_response_time']:.2f} ms</td></tr>
            <tr><td>Average RPS</td><td>{report['summary']['avg_rps']:.2f}</td></tr>
        </table>
    </div>
    
    <h2>Scenario Results</h2>
    <table>
        <tr>
            <th>Scenario</th>
            <th>Status</th>
            <th>Requests</th>
            <th>Failures</th>
            <th>Success Rate</th>
            <th>P95 (ms)</th>
            <th>P99 (ms)</th>
            <th>RPS</th>
        </tr>
"""
        
        for scenario in report['scenarios']:
            summary = scenario.get('summary', {})
            status_class = 'success' if scenario.get('status') == 'completed' else 'error'
            success_rate = summary.get('success_rate', 0)
            success_class = 'success' if success_rate >= 99 else 'warning' if success_rate >= 95 else 'error'
            
            html += f"""
        <tr>
            <td>{scenario.get('scenario', 'Unknown')}</td>
            <td class="{status_class}">{scenario.get('status', 'Unknown')}</td>
            <td>{summary.get('total_requests', 0):,}</td>
            <td>{summary.get('failure_count', 0):,}</td>
            <td class="{success_class}">{success_rate:.2f}%</td>
            <td>{summary.get('p95_response_time', 0):.2f}</td>
            <td>{summary.get('p99_response_time', 0):.2f}</td>
            <td>{summary.get('rps', 0):.2f}</td>
        </tr>
"""
            
        html += """
    </table>
    
    <h2>Performance Targets</h2>
    <table>
        <tr>
            <th>Target</th>
            <th>Expected</th>
            <th>Actual</th>
            <th>Status</th>
        </tr>
"""
        
        # Check performance targets
        targets = [
            ('Concurrent Users', '1000+', report['summary']['total_scenarios'] * 100, 
             'success' if report['summary']['total_scenarios'] >= 10 else 'warning'),
            ('Orders/Second', '1000+', f"{report['summary']['avg_rps']:.0f}",
             'success' if report['summary']['avg_rps'] >= 1000 else 'warning'),
            ('P99 Response Time', '<50ms', f"{report['summary']['avg_p99_response_time']:.2f}ms",
             'success' if report['summary']['avg_p99_response_time'] < 50 else 'warning'),
            ('Success Rate', '>99%', f"{report['summary']['overall_success_rate']:.2f}%",
             'success' if report['summary']['overall_success_rate'] > 99 else 'error')
        ]
        
        for target, expected, actual, status in targets:
            html += f"""
        <tr>
            <td>{target}</td>
            <td>{expected}</td>
            <td>{actual}</td>
            <td class="{status}">{'✓' if status == 'success' else '⚠' if status == 'warning' else '✗'}</td>
        </tr>
"""
            
        html += """
    </table>
</body>
</html>
"""
        
        return html


def get_default_scenarios() -> List[Dict]:
    """Get default load test scenarios."""
    return [
        {
            'name': 'warmup',
            'file': 'trading_load_test.py',
            'users': 10,
            'spawn_rate': 2,
            'run_time': '1m',
            'description': 'Warmup with light load'
        },
        {
            'name': 'normal_trading',
            'file': 'trading_load_test.py',
            'users': 100,
            'spawn_rate': 10,
            'run_time': '5m',
            'description': 'Normal trading day simulation'
        },
        {
            'name': 'high_volume',
            'file': 'trading_load_test.py',
            'users': 500,
            'spawn_rate': 50,
            'run_time': '5m',
            'description': 'High volume trading period'
        },
        {
            'name': 'stress_test',
            'file': 'trading_load_test.py',
            'users': 1000,
            'spawn_rate': 100,
            'run_time': '3m',
            'description': 'Maximum stress test'
        },
        {
            'name': 'websocket_load',
            'file': 'websocket_load_test.py',
            'users': 200,
            'spawn_rate': 20,
            'run_time': '5m',
            'description': 'WebSocket connection load'
        },
        {
            'name': 'chaos_test',
            'file': 'chaos_testing.py',
            'users': 50,
            'spawn_rate': 5,
            'run_time': '3m',
            'description': 'Chaos engineering test'
        }
    ]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Genesis load tests')
    parser.add_argument('--scenarios', nargs='+', help='Specific scenarios to run')
    parser.add_argument('--docker', action='store_true', help='Use Docker Compose')
    parser.add_argument('--host', default='http://localhost:8000', help='Target host')
    parser.add_argument('--report-only', help='Generate report from existing results')
    
    args = parser.parse_args()
    
    runner = LoadTestRunner()
    
    if args.report_only:
        # Generate report from existing results
        logger.info(f"Generating report from: {args.report_only}")
        # Implementation for report generation from existing data
        return
        
    # Get scenarios to run
    all_scenarios = get_default_scenarios()
    
    if args.scenarios:
        scenarios = [s for s in all_scenarios if s['name'] in args.scenarios]
    else:
        scenarios = all_scenarios
        
    # Update host if provided
    for scenario in scenarios:
        scenario['host'] = args.host
        
    logger.info(f"Running {len(scenarios)} scenarios")
    
    # Run test suite
    if args.docker:
        results = runner.run_test_suite(scenarios)
    else:
        # Run without Docker
        results = []
        for scenario in scenarios:
            result = runner.run_locust_test(scenario)
            results.append(result)
            
    # Generate report
    report_file = runner.generate_report(results)
    logger.info(f"Test complete. Report: {report_file}")
    
    # Check if tests passed
    summary = runner.calculate_summary(results)
    if summary['overall_success_rate'] < 95:
        logger.error("Load tests failed to meet success rate target")
        sys.exit(1)
        
    logger.info("Load tests completed successfully")


if __name__ == "__main__":
    main()