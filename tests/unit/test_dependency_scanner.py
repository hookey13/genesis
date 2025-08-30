"""Unit tests for dependency vulnerability scanner."""

import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, call
import pytest

from genesis.security.dependency_scanner import (
    DependencyScanner,
    VulnerabilityReport,
    Vulnerability,
    SeverityLevel,
    scan_dependencies,
)


class TestSeverityLevel:
    """Test severity level enum."""
    
    def test_severity_scores(self):
        """Test severity level scoring."""
        assert SeverityLevel.CRITICAL.score == 4
        assert SeverityLevel.HIGH.score == 3
        assert SeverityLevel.MEDIUM.score == 2
        assert SeverityLevel.LOW.score == 1
        assert SeverityLevel.UNKNOWN.score == 0
    
    def test_from_cvss_score(self):
        """Test CVSS score conversion to severity level."""
        assert SeverityLevel.from_cvss(10.0) == SeverityLevel.CRITICAL
        assert SeverityLevel.from_cvss(9.0) == SeverityLevel.CRITICAL
        assert SeverityLevel.from_cvss(7.5) == SeverityLevel.HIGH
        assert SeverityLevel.from_cvss(5.0) == SeverityLevel.MEDIUM
        assert SeverityLevel.from_cvss(2.0) == SeverityLevel.LOW
        assert SeverityLevel.from_cvss(0.0) == SeverityLevel.UNKNOWN


class TestVulnerability:
    """Test vulnerability data class."""
    
    def test_vulnerability_creation(self):
        """Test creating vulnerability object."""
        vuln = Vulnerability(
            package="test-package",
            installed_version="1.0.0",
            affected_versions="<1.2.0",
            vulnerability_id="VULN-001",
            description="Test vulnerability",
            severity=SeverityLevel.HIGH,
            cvss_score=7.5,
            cve_id="CVE-2024-001",
            fixed_version="1.2.0",
            advisory_url="https://example.com/advisory",
        )
        
        assert vuln.package == "test-package"
        assert vuln.severity == SeverityLevel.HIGH
        assert vuln.cvss_score == 7.5
        assert vuln.fixed_version == "1.2.0"


class TestVulnerabilityReport:
    """Test vulnerability report class."""
    
    def test_report_creation(self):
        """Test creating vulnerability report."""
        report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=5,
        )
        
        assert report.total_packages == 100
        assert report.vulnerable_packages == 5
        assert report.critical_count == 0
        assert report.high_count == 0
        assert len(report.vulnerabilities) == 0
    
    def test_add_vulnerability(self):
        """Test adding vulnerabilities to report."""
        report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=0,
        )
        
        # Add critical vulnerability
        critical_vuln = Vulnerability(
            package="pkg1",
            installed_version="1.0.0",
            affected_versions="<2.0.0",
            vulnerability_id="VULN-001",
            description="Critical issue",
            severity=SeverityLevel.CRITICAL,
        )
        report.add_vulnerability(critical_vuln)
        
        assert len(report.vulnerabilities) == 1
        assert report.critical_count == 1
        
        # Add high vulnerability
        high_vuln = Vulnerability(
            package="pkg2",
            installed_version="2.0.0",
            affected_versions="<3.0.0",
            vulnerability_id="VULN-002",
            description="High issue",
            severity=SeverityLevel.HIGH,
        )
        report.add_vulnerability(high_vuln)
        
        assert len(report.vulnerabilities) == 2
        assert report.high_count == 1
    
    def test_should_block_deployment(self):
        """Test deployment blocking logic."""
        report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=0,
        )
        
        # No vulnerabilities - should not block
        assert not report.should_block_deployment()
        
        # Add critical vulnerability - should block (default threshold=0)
        report.critical_count = 1
        assert report.should_block_deployment()
        
        # Reset and add high vulnerabilities
        report.critical_count = 0
        report.high_count = 3
        assert not report.should_block_deployment()  # Default high threshold=3
        
        report.high_count = 4
        assert report.should_block_deployment()  # Exceeds threshold
        
        # Custom thresholds
        assert not report.should_block_deployment(
            critical_threshold=5, 
            high_threshold=10
        )
    
    def test_get_summary(self):
        """Test report summary generation."""
        scan_date = datetime.now()
        report = VulnerabilityReport(
            scan_date=scan_date,
            total_packages=100,
            vulnerable_packages=5,
            critical_count=1,
            high_count=2,
            medium_count=3,
            low_count=4,
            scan_duration_seconds=10.5,
        )
        
        summary = report.get_summary()
        assert f"Scan Date: {scan_date.isoformat()}" in summary
        assert "Total Packages: 100" in summary
        assert "Vulnerable Packages: 5" in summary
        assert "Critical: 1, High: 2" in summary
        assert "Medium: 3, Low: 4" in summary
        assert "Scan Duration: 10.50s" in summary


class TestDependencyScanner:
    """Test dependency scanner class."""
    
    @patch('subprocess.run')
    def test_detect_available_scanners(self, mock_run):
        """Test scanner detection."""
        # Mock successful scanner detection
        mock_run.side_effect = [
            MagicMock(returncode=0),  # safety found
            MagicMock(returncode=0),  # pip-audit found
            MagicMock(returncode=1),  # bandit not found
        ]
        
        scanner = DependencyScanner()
        
        # Should have called subprocess.run for each scanner
        assert mock_run.call_count >= 2
        
    @patch('subprocess.run')
    def test_scan_with_safety(self, mock_run):
        """Test Safety scanner integration."""
        # Mock Safety output
        safety_output = {
            "vulnerabilities": [
                {
                    "package": "requests",
                    "installed_version": "2.0.0",
                    "affected_versions": "<2.31.0",
                    "vulnerability_id": "51457",
                    "description": "Requests vulnerability",
                    "cve": "CVE-2023-32681",
                    "fixed_version": "2.31.0",
                    "more_info_url": "https://github.com/advisories/GHSA-j8r2-6x86-q33q",
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(safety_output),
            returncode=0,
        )
        
        scanner = DependencyScanner()
        vulnerabilities = scanner.scan_with_safety()
        
        assert len(vulnerabilities) == 1
        assert vulnerabilities[0].package == "requests"
        assert vulnerabilities[0].cve_id == "CVE-2023-32681"
        assert vulnerabilities[0].fixed_version == "2.31.0"
    
    @patch('subprocess.run')
    def test_scan_with_pip_audit(self, mock_run):
        """Test pip-audit scanner integration."""
        # Mock pip-audit output
        pip_audit_output = {
            "dependencies": [
                {
                    "name": "django",
                    "version": "3.0.0",
                    "vulns": [
                        {
                            "id": "PYSEC-2021-1",
                            "description": "Django SQL injection",
                            "affected": "<3.2.0",
                            "fix_versions": ["3.2.0"],
                            "link": "https://osv.dev/vulnerability/PYSEC-2021-1",
                        }
                    ],
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(pip_audit_output),
            returncode=0,
        )
        
        scanner = DependencyScanner()
        vulnerabilities = scanner.scan_with_pip_audit()
        
        assert len(vulnerabilities) == 1
        assert vulnerabilities[0].package == "django"
        assert vulnerabilities[0].vulnerability_id == "PYSEC-2021-1"
        assert vulnerabilities[0].fixed_version == "3.2.0"
    
    @patch('subprocess.run')
    def test_scan_code_with_bandit(self, mock_run):
        """Test Bandit code scanner integration."""
        # Mock Bandit output
        bandit_output = {
            "results": [
                {
                    "filename": "genesis/test.py",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Use of exec",
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(bandit_output),
            returncode=0,
        )
        
        scanner = DependencyScanner()
        results = scanner.scan_code_with_bandit()
        
        assert "results" in results
        assert len(results["results"]) == 1
    
    @patch.object(DependencyScanner, 'scan_with_safety')
    @patch.object(DependencyScanner, 'scan_with_pip_audit')
    @patch.object(DependencyScanner, '_count_installed_packages')
    def test_scan_all(self, mock_count, mock_pip_audit, mock_safety):
        """Test combined scanning."""
        # Mock package count
        mock_count.return_value = 50
        
        # Mock scanner results
        mock_safety.return_value = [
            Vulnerability(
                package="pkg1",
                installed_version="1.0.0",
                affected_versions="<2.0.0",
                vulnerability_id="VULN-001",
                description="Test",
                severity=SeverityLevel.CRITICAL,
            )
        ]
        
        mock_pip_audit.return_value = [
            Vulnerability(
                package="pkg2",
                installed_version="2.0.0",
                affected_versions="<3.0.0",
                vulnerability_id="VULN-002",
                description="Test",
                severity=SeverityLevel.HIGH,
            )
        ]
        
        scanner = DependencyScanner()
        scanner.available_scanners = ["safety", "pip-audit"]
        
        report = scanner.scan_all(include_code_scan=False)
        
        assert report.total_packages == 50
        assert len(report.vulnerabilities) == 2
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.vulnerable_packages == 2
    
    @patch('subprocess.run')
    def test_count_installed_packages(self, mock_run):
        """Test package counting."""
        # Mock pip list output
        packages = [
            {"name": "package1", "version": "1.0.0"},
            {"name": "package2", "version": "2.0.0"},
        ]
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(packages),
            returncode=0,
        )
        
        scanner = DependencyScanner()
        count = scanner._count_installed_packages()
        
        assert count == 2
    
    def test_parse_safety_severity(self):
        """Test Safety severity parsing."""
        scanner = DependencyScanner()
        
        # Test critical severity detection
        vuln_data = {"description": "Remote code execution vulnerability"}
        assert scanner._parse_safety_severity(vuln_data) == SeverityLevel.CRITICAL
        
        # Test high severity detection
        vuln_data = {"description": "SQL injection vulnerability"}
        assert scanner._parse_safety_severity(vuln_data) == SeverityLevel.HIGH
        
        # Test medium severity detection
        vuln_data = {"description": "Denial of service vulnerability"}
        assert scanner._parse_safety_severity(vuln_data) == SeverityLevel.MEDIUM
        
        # Test low severity detection
        vuln_data = {"description": "Low severity information disclosure"}
        assert scanner._parse_safety_severity(vuln_data) == SeverityLevel.LOW
        
        # Test unknown severity
        vuln_data = {"description": "Some vulnerability"}
        assert scanner._parse_safety_severity(vuln_data) == SeverityLevel.UNKNOWN
    
    def test_parse_pip_audit_severity(self):
        """Test pip-audit severity parsing."""
        scanner = DependencyScanner()
        
        # Test CVSS-based severity
        vuln_data = {"cvss": 9.5}
        assert scanner._parse_pip_audit_severity(vuln_data) == SeverityLevel.CRITICAL
        
        vuln_data = {"cvss": 7.5}
        assert scanner._parse_pip_audit_severity(vuln_data) == SeverityLevel.HIGH
        
        # Test description-based severity
        vuln_data = {"description": "Critical vulnerability"}
        assert scanner._parse_pip_audit_severity(vuln_data) == SeverityLevel.CRITICAL
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_generate_report_file(self, mock_json_dump, mock_open):
        """Test report file generation."""
        report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=5,
            critical_count=1,
            high_count=2,
        )
        
        scanner = DependencyScanner()
        output_path = scanner.generate_report_file(report)
        
        # Verify file was opened for writing
        mock_open.assert_called_once()
        
        # Verify JSON dump was called
        mock_json_dump.assert_called_once()
        
        # Check the data structure passed to json.dump
        call_args = mock_json_dump.call_args[0][0]
        assert call_args["total_packages"] == 100
        assert call_args["vulnerable_packages"] == 5
        assert call_args["summary"]["critical"] == 1
        assert call_args["summary"]["high"] == 2


class TestScanDependencies:
    """Test the main scan_dependencies function."""
    
    @patch.object(DependencyScanner, 'scan_all')
    @patch.object(DependencyScanner, 'generate_report_file')
    def test_scan_dependencies_pass(self, mock_generate, mock_scan):
        """Test successful dependency scan."""
        # Mock report with no critical vulnerabilities
        mock_report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=0,
        )
        mock_scan.return_value = mock_report
        
        passed, report = scan_dependencies()
        
        assert passed is True
        assert report == mock_report
        mock_generate.assert_called_once_with(mock_report)
    
    @patch.object(DependencyScanner, 'scan_all')
    @patch.object(DependencyScanner, 'generate_report_file')
    def test_scan_dependencies_fail(self, mock_generate, mock_scan):
        """Test failed dependency scan due to vulnerabilities."""
        # Mock report with critical vulnerabilities
        mock_report = VulnerabilityReport(
            scan_date=datetime.now(),
            total_packages=100,
            vulnerable_packages=1,
            critical_count=1,
        )
        mock_scan.return_value = mock_report
        
        passed, report = scan_dependencies(critical_threshold=0)
        
        assert passed is False
        assert report == mock_report
    
    @patch.object(DependencyScanner, '__init__')
    def test_scan_dependencies_no_scanners(self, mock_init):
        """Test behavior when no scanners are available."""
        # Mock scanner with no available scanners
        mock_init.return_value = None
        
        with patch.object(DependencyScanner, 'available_scanners', []):
            passed, report = scan_dependencies()
            
            assert passed is True
            assert report is None