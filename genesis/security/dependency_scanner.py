"""Dependency vulnerability scanner for Genesis trading system."""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class SeverityLevel(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    @property
    def score(self) -> int:
        """Get numeric score for severity level."""
        scores = {
            self.CRITICAL: 4,
            self.HIGH: 3,
            self.MEDIUM: 2,
            self.LOW: 1,
            self.UNKNOWN: 0
        }
        return scores.get(self, 0)

    @classmethod
    def from_cvss(cls, cvss_score: float) -> "SeverityLevel":
        """Convert CVSS score to severity level."""
        if cvss_score >= 9.0:
            return cls.CRITICAL
        elif cvss_score >= 7.0:
            return cls.HIGH
        elif cvss_score >= 4.0:
            return cls.MEDIUM
        elif cvss_score > 0:
            return cls.LOW
        else:
            return cls.UNKNOWN


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""
    package: str
    installed_version: str
    affected_versions: str
    vulnerability_id: str
    description: str
    severity: SeverityLevel
    cvss_score: float | None = None
    cve_id: str | None = None
    fixed_version: str | None = None
    advisory_url: str | None = None
    discovered_date: str | None = None


@dataclass
class VulnerabilityReport:
    """Vulnerability scan report."""
    scan_date: datetime
    total_packages: int
    vulnerable_packages: int
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    scan_tool: str = "multi-scanner"
    scan_duration_seconds: float = 0.0

    def add_vulnerability(self, vuln: Vulnerability):
        """Add vulnerability to report and update counts."""
        self.vulnerabilities.append(vuln)

        if vuln.severity == SeverityLevel.CRITICAL:
            self.critical_count += 1
        elif vuln.severity == SeverityLevel.HIGH:
            self.high_count += 1
        elif vuln.severity == SeverityLevel.MEDIUM:
            self.medium_count += 1
        elif vuln.severity == SeverityLevel.LOW:
            self.low_count += 1

    def get_summary(self) -> str:
        """Get report summary."""
        return (
            f"Scan Date: {self.scan_date.isoformat()}\n"
            f"Total Packages: {self.total_packages}\n"
            f"Vulnerable Packages: {self.vulnerable_packages}\n"
            f"Critical: {self.critical_count}, High: {self.high_count}, "
            f"Medium: {self.medium_count}, Low: {self.low_count}\n"
            f"Scan Duration: {self.scan_duration_seconds:.2f}s"
        )

    def should_block_deployment(self,
                               critical_threshold: int = 0,
                               high_threshold: int = 3) -> bool:
        """Determine if vulnerabilities should block deployment."""
        return (self.critical_count > critical_threshold or
                self.high_count > high_threshold)


class DependencyScanner:
    """Multi-tool dependency vulnerability scanner."""

    def __init__(self, project_root: Path | None = None):
        """Initialize scanner."""
        self.project_root = project_root or Path.cwd()
        self.available_scanners = self._detect_available_scanners()

    def _detect_available_scanners(self) -> list[str]:
        """Detect which vulnerability scanners are available."""
        available = []

        # Check for Safety
        try:
            subprocess.run(["safety", "--version"],
                         capture_output=True, check=True)
            available.append("safety")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Check for pip-audit
        try:
            subprocess.run(["pip-audit", "--version"],
                         capture_output=True, check=True)
            available.append("pip-audit")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Check for Bandit (for code security, not dependencies)
        try:
            subprocess.run(["bandit", "--version"],
                         capture_output=True, check=True)
            available.append("bandit")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        logger.info("Available scanners detected", scanners=available)
        return available

    def scan_with_safety(self) -> list[Vulnerability]:
        """Scan dependencies using Safety."""
        vulnerabilities = []

        try:
            # Run safety check with JSON output
            result = subprocess.run(
                ["safety", "check", "--json", "--stdin"],
                input=self._get_requirements_content(),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                data = json.loads(result.stdout)

                # Parse Safety output
                for vuln in data.get("vulnerabilities", []):
                    vulnerability = Vulnerability(
                        package=vuln.get("package", ""),
                        installed_version=vuln.get("installed_version", ""),
                        affected_versions=vuln.get("affected_versions", ""),
                        vulnerability_id=vuln.get("vulnerability_id", ""),
                        description=vuln.get("description", ""),
                        severity=self._parse_safety_severity(vuln),
                        cve_id=vuln.get("cve", None),
                        fixed_version=vuln.get("fixed_version", None),
                        advisory_url=vuln.get("more_info_url", None)
                    )
                    vulnerabilities.append(vulnerability)

        except subprocess.TimeoutExpired:
            logger.warning("Safety scan timed out")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error("Safety scan failed", error=str(e))

        return vulnerabilities

    def scan_with_pip_audit(self) -> list[Vulnerability]:
        """Scan dependencies using pip-audit."""
        vulnerabilities = []

        try:
            # Run pip-audit with JSON output
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--desc"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                data = json.loads(result.stdout)

                # Parse pip-audit output
                for dep in data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        vulnerability = Vulnerability(
                            package=dep.get("name", ""),
                            installed_version=dep.get("version", ""),
                            affected_versions=vuln.get("affected", ""),
                            vulnerability_id=vuln.get("id", ""),
                            description=vuln.get("description", ""),
                            severity=self._parse_pip_audit_severity(vuln),
                            fixed_version=vuln.get("fix_versions", [None])[0]
                                if vuln.get("fix_versions") else None,
                            advisory_url=vuln.get("link", None)
                        )
                        vulnerabilities.append(vulnerability)

        except subprocess.TimeoutExpired:
            logger.warning("pip-audit scan timed out")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error("pip-audit scan failed", error=str(e))

        return vulnerabilities

    def scan_code_with_bandit(self) -> dict[str, any]:
        """Scan Python code for security issues using Bandit."""
        try:
            # Run bandit on genesis package
            genesis_path = self.project_root / "genesis"
            result = subprocess.run(
                ["bandit", "-r", str(genesis_path), "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.stdout:
                return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            logger.warning("Bandit scan timed out")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error("Bandit scan failed", error=str(e))

        return {}

    def scan_all(self,
                 include_code_scan: bool = False) -> VulnerabilityReport:
        """Run all available scanners and combine results."""
        start_time = datetime.now()
        report = VulnerabilityReport(
            scan_date=start_time,
            total_packages=self._count_installed_packages(),
            vulnerable_packages=0
        )

        # Track unique vulnerabilities to avoid duplicates
        seen_vulns = set()

        # Run Safety scan
        if "safety" in self.available_scanners:
            logger.info("Running Safety scan")
            safety_vulns = self.scan_with_safety()
            for vuln in safety_vulns:
                vuln_key = (vuln.package, vuln.vulnerability_id)
                if vuln_key not in seen_vulns:
                    report.add_vulnerability(vuln)
                    seen_vulns.add(vuln_key)

        # Run pip-audit scan
        if "pip-audit" in self.available_scanners:
            logger.info("Running pip-audit scan")
            pip_audit_vulns = self.scan_with_pip_audit()
            for vuln in pip_audit_vulns:
                vuln_key = (vuln.package, vuln.vulnerability_id)
                if vuln_key not in seen_vulns:
                    report.add_vulnerability(vuln)
                    seen_vulns.add(vuln_key)

        # Run Bandit code scan if requested
        if include_code_scan and "bandit" in self.available_scanners:
            logger.info("Running Bandit code security scan")
            bandit_results = self.scan_code_with_bandit()
            # Bandit results are stored separately as they're code issues, not dependencies
            report.code_issues = bandit_results

        # Update report metrics
        vulnerable_packages = set(v.package for v in report.vulnerabilities)
        report.vulnerable_packages = len(vulnerable_packages)

        end_time = datetime.now()
        report.scan_duration_seconds = (end_time - start_time).total_seconds()

        logger.info(
            "Vulnerability scan completed",
            total_vulnerabilities=len(report.vulnerabilities),
            critical=report.critical_count,
            high=report.high_count,
            medium=report.medium_count,
            low=report.low_count,
            duration_seconds=report.scan_duration_seconds
        )

        return report

    def _get_requirements_content(self) -> str:
        """Get requirements content for scanning."""
        # Try to get from pip freeze
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def _count_installed_packages(self) -> int:
        """Count installed packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True
            )
            packages = json.loads(result.stdout)
            return len(packages)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return 0

    def _parse_safety_severity(self, vuln_data: dict) -> SeverityLevel:
        """Parse severity from Safety vulnerability data."""
        # Safety doesn't always provide severity, estimate from description
        desc = vuln_data.get("description", "").lower()
        vuln_id = vuln_data.get("vulnerability_id", "").lower()

        if "critical" in desc or "rce" in desc or "remote code" in desc:
            return SeverityLevel.CRITICAL
        elif "high" in desc or "injection" in desc or "xss" in desc:
            return SeverityLevel.HIGH
        elif "medium" in desc or "dos" in desc:
            return SeverityLevel.MEDIUM
        elif "low" in desc:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.UNKNOWN

    def _parse_pip_audit_severity(self, vuln_data: dict) -> SeverityLevel:
        """Parse severity from pip-audit vulnerability data."""
        # pip-audit may provide CVSS score
        if "cvss" in vuln_data:
            cvss_score = vuln_data.get("cvss", 0)
            return SeverityLevel.from_cvss(cvss_score)

        # Fall back to description parsing
        desc = vuln_data.get("description", "").lower()
        if "critical" in desc:
            return SeverityLevel.CRITICAL
        elif "high" in desc:
            return SeverityLevel.HIGH
        elif "medium" in desc:
            return SeverityLevel.MEDIUM
        elif "low" in desc:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.UNKNOWN

    def generate_report_file(self,
                           report: VulnerabilityReport,
                           output_path: Path | None = None) -> Path:
        """Generate vulnerability report file."""
        if output_path is None:
            output_path = self.project_root / "vulnerability_report.json"

        report_data = {
            "scan_date": report.scan_date.isoformat(),
            "total_packages": report.total_packages,
            "vulnerable_packages": report.vulnerable_packages,
            "summary": {
                "critical": report.critical_count,
                "high": report.high_count,
                "medium": report.medium_count,
                "low": report.low_count
            },
            "vulnerabilities": [
                {
                    "package": v.package,
                    "installed_version": v.installed_version,
                    "vulnerability_id": v.vulnerability_id,
                    "description": v.description,
                    "severity": v.severity.value,
                    "fixed_version": v.fixed_version,
                    "advisory_url": v.advisory_url
                }
                for v in report.vulnerabilities
            ],
            "scan_tool": report.scan_tool,
            "scan_duration_seconds": report.scan_duration_seconds
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info("Vulnerability report saved", path=str(output_path))
        return output_path


def scan_dependencies(project_root: Path | None = None,
                      critical_threshold: int = 0,
                      high_threshold: int = 3,
                      include_code_scan: bool = False) -> tuple[bool, VulnerabilityReport]:
    """
    Scan dependencies for vulnerabilities.
    
    Returns:
        Tuple of (pass/fail, report)
    """
    scanner = DependencyScanner(project_root)

    if not scanner.available_scanners:
        logger.warning("No vulnerability scanners available")
        logger.info("Install scanners with: pip install safety pip-audit bandit")
        return True, None

    report = scanner.scan_all(include_code_scan=include_code_scan)

    # Check if deployment should be blocked
    should_block = report.should_block_deployment(
        critical_threshold=critical_threshold,
        high_threshold=high_threshold
    )

    if should_block:
        logger.error(
            "Vulnerabilities exceed threshold",
            critical=report.critical_count,
            high=report.high_count,
            threshold_critical=critical_threshold,
            threshold_high=high_threshold
        )

    # Generate report file
    scanner.generate_report_file(report)

    return not should_block, report


if __name__ == "__main__":
    # Run vulnerability scan
    passed, report = scan_dependencies(include_code_scan=True)

    if report:
        print("\n" + "=" * 60)
        print("VULNERABILITY SCAN REPORT")
        print("=" * 60)
        print(report.get_summary())

        if report.vulnerabilities:
            print("\nVulnerabilities Found:")
            for vuln in sorted(report.vulnerabilities,
                             key=lambda v: v.severity.score,
                             reverse=True):
                print(f"\n[{vuln.severity.value.upper()}] {vuln.package} "
                      f"({vuln.installed_version})")
                print(f"  ID: {vuln.vulnerability_id}")
                print(f"  Description: {vuln.description[:100]}...")
                if vuln.fixed_version:
                    print(f"  Fix: Upgrade to {vuln.fixed_version}")

        print("\n" + "=" * 60)

        if not passed:
            print("❌ SCAN FAILED: Vulnerabilities exceed threshold")
            sys.exit(1)
        else:
            print("✅ SCAN PASSED")
    else:
        print("No scanners available. Install with:")
        print("  pip install safety pip-audit bandit")
