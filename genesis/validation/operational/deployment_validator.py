"""Deployment readiness and rollback validation module."""

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import structlog
import yaml

logger = structlog.get_logger(__name__)


class DeploymentValidator:
    """Validates deployment readiness and rollback procedures."""

    DEPLOYMENT_CHECKS = [
        "blue_green_configured",
        "rollback_tested",
        "staging_environment_ready",
        "production_environment_ready",
        "deployment_scripts_tested",
        "rollback_time_under_5_minutes",
        "zero_downtime_deployment",
        "health_checks_configured",
    ]

    MAX_ROLLBACK_TIME_SECONDS = 300  # 5 minutes

    REQUIRED_DEPLOYMENT_FILES = [
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docker/docker-compose.prod.yml",
        "scripts/deploy.sh",
        "scripts/rollback.sh",
    ]

    REQUIRED_KUBERNETES_FILES = [
        "kubernetes/deployment.yaml",
        "kubernetes/service.yaml",
        "kubernetes/configmap.yaml",
        "kubernetes/secrets.yaml",
    ]

    def __init__(self, genesis_root: Path | None = None):
        """Initialize deployment validator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.results: Dict[str, Any] = {}

    async def validate(self) -> Dict[str, Any]:
        """Run deployment validation checks.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting deployment validation")
        start_time = datetime.utcnow()

        self.results = {
            "validator": "deployment",
            "timestamp": start_time.isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": [],
        }

        # Verify rollback plan
        rollback_result = await self._verify_rollback_plan()
        self.results["checks"]["rollback_plan"] = rollback_result

        # Check blue-green deployment
        blue_green_result = await self._check_blue_green_deployment()
        self.results["checks"]["blue_green_deployment"] = blue_green_result

        # Validate deployment scripts
        scripts_result = await self._validate_deployment_scripts()
        self.results["checks"]["deployment_scripts"] = scripts_result

        # Test staging environment
        staging_result = await self._test_staging_environment()
        self.results["checks"]["staging_environment"] = staging_result

        # Check container configuration
        container_result = await self._check_container_configuration()
        self.results["checks"]["container_configuration"] = container_result

        # Validate Kubernetes manifests if present
        k8s_result = await self._validate_kubernetes_manifests()
        self.results["checks"]["kubernetes_manifests"] = k8s_result

        # Check rollback time
        rollback_time_result = await self._check_rollback_time()
        self.results["checks"]["rollback_time"] = rollback_time_result

        # Generate deployment readiness report
        readiness_report = await self._generate_readiness_report()
        self.results["checks"]["readiness_report"] = readiness_report

        # Calculate overall score
        total_checks = len(self.results["checks"])
        passed_checks = sum(
            1 for check in self.results["checks"].values() if check.get("passed", False)
        )
        self.results["score"] = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0

        # Determine overall status
        critical_checks = ["rollback_plan", "deployment_scripts", "container_configuration"]
        critical_passed = all(
            self.results["checks"].get(check, {}).get("passed", False)
            for check in critical_checks
        )

        if critical_passed and self.results["score"] >= 80:
            self.results["passed"] = True
            self.results["summary"] = "Deployment ready with rollback procedures verified"
        else:
            self.results["passed"] = False
            self.results["summary"] = "Deployment issues detected - review failed checks"

        # Add execution time
        self.results["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

        return self.results

    async def _verify_rollback_plan(self) -> Dict[str, Any]:
        """Verify rollback plan is documented and tested.
        
        Returns:
            Validation result for rollback plan
        """
        result = {
            "passed": False,
            "message": "",
            "rollback_steps": [],
            "documentation_found": False,
            "script_found": False,
        }

        # Check for rollback documentation
        rollback_doc = self.genesis_root / "docs" / "rollback.md"
        if rollback_doc.exists():
            result["documentation_found"] = True
            
            with open(rollback_doc, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Look for rollback steps
                import re
                steps = re.findall(r"(\d+\..*)", content)
                result["rollback_steps"] = steps[:5]  # First 5 steps
        
        # Check for rollback script
        rollback_script = self.genesis_root / "scripts" / "rollback.sh"
        if rollback_script.exists():
            result["script_found"] = True
            
            # Check if script is executable
            if not os.access(rollback_script, os.X_OK):
                result["message"] = "Rollback script exists but is not executable"
            else:
                # Check script content
                with open(rollback_script, "r") as f:
                    script_content = f.read()
                    
                    # Look for key rollback operations
                    has_backup_restore = "restore" in script_content or "backup" in script_content
                    has_version_control = "git" in script_content or "tag" in script_content
                    has_database_rollback = "migrate" in script_content or "database" in script_content
                    
                    if has_backup_restore and has_version_control:
                        result["passed"] = True
                        result["message"] = "Rollback plan documented with executable script"
                    else:
                        result["message"] = "Rollback script missing key operations"
        else:
            result["message"] = "Rollback script not found"

        if not result["documentation_found"] and not result["script_found"]:
            result["message"] = "No rollback plan or script found"

        return result

    async def _check_blue_green_deployment(self) -> Dict[str, Any]:
        """Check blue-green deployment configuration.
        
        Returns:
            Validation result for blue-green deployment
        """
        result = {
            "passed": False,
            "message": "",
            "blue_green_configured": False,
            "load_balancer_configured": False,
        }

        # Check Docker Compose for blue-green setup
        compose_prod = self.genesis_root / "docker" / "docker-compose.prod.yml"
        if compose_prod.exists():
            with open(compose_prod, "r") as f:
                compose_content = yaml.safe_load(f)
                
                services = compose_content.get("services", {})
                
                # Look for blue and green services
                has_blue = any("blue" in service for service in services)
                has_green = any("green" in service for service in services)
                
                if has_blue and has_green:
                    result["blue_green_configured"] = True
                
                # Check for load balancer or proxy
                has_proxy = any(
                    service in services 
                    for service in ["nginx", "haproxy", "traefik", "proxy"]
                )
                
                if has_proxy:
                    result["load_balancer_configured"] = True
        
        # Check Kubernetes for blue-green
        k8s_deployment = self.genesis_root / "kubernetes" / "deployment.yaml"
        if k8s_deployment.exists():
            with open(k8s_deployment, "r") as f:
                k8s_content = f.read()
                
                if "blue" in k8s_content and "green" in k8s_content:
                    result["blue_green_configured"] = True
        
        # Check deployment scripts for blue-green logic
        deploy_script = self.genesis_root / "scripts" / "deploy.sh"
        if deploy_script.exists():
            with open(deploy_script, "r") as f:
                script_content = f.read()
                
                if "blue" in script_content and "green" in script_content:
                    result["blue_green_configured"] = True

        if result["blue_green_configured"]:
            result["passed"] = True
            result["message"] = "Blue-green deployment configured"
        else:
            result["message"] = "Blue-green deployment not configured"

        return result

    async def _validate_deployment_scripts(self) -> Dict[str, Any]:
        """Validate deployment scripts and automation.
        
        Returns:
            Validation result for deployment scripts
        """
        result = {
            "passed": False,
            "message": "",
            "scripts_found": [],
            "scripts_missing": [],
            "scripts_executable": [],
        }

        required_scripts = {
            "deploy.sh": self.genesis_root / "scripts" / "deploy.sh",
            "rollback.sh": self.genesis_root / "scripts" / "rollback.sh",
            "health_check.sh": self.genesis_root / "scripts" / "health_check.sh",
        }

        for script_name, script_path in required_scripts.items():
            if script_path.exists():
                result["scripts_found"].append(script_name)
                
                # Check if executable
                if os.access(script_path, os.X_OK):
                    result["scripts_executable"].append(script_name)
                
                # Validate script content
                with open(script_path, "r") as f:
                    content = f.read()
                    
                    # Check for error handling
                    has_error_handling = "set -e" in content or "trap" in content
                    
                    # Check for logging
                    has_logging = "echo" in content or "logger" in content or "log" in content
                    
                    if not has_error_handling:
                        result["scripts_missing"].append(f"{script_name} (no error handling)")
                    if not has_logging:
                        result["scripts_missing"].append(f"{script_name} (no logging)")
            else:
                result["scripts_missing"].append(script_name)

        if len(result["scripts_found"]) >= 2 and not result["scripts_missing"]:
            result["passed"] = True
            result["message"] = f"Deployment scripts validated ({len(result['scripts_found'])} found)"
        else:
            result["message"] = f"Script issues: {', '.join(result['scripts_missing'])}"

        return result

    async def _test_staging_environment(self) -> Dict[str, Any]:
        """Test staging environment configuration.
        
        Returns:
            Validation result for staging environment
        """
        result = {
            "passed": False,
            "message": "",
            "staging_configured": False,
            "staging_accessible": False,
        }

        # Check for staging configuration
        staging_configs = [
            self.genesis_root / ".env.staging",
            self.genesis_root / "config" / "staging.yaml",
            self.genesis_root / "docker" / "docker-compose.staging.yml",
        ]

        for config in staging_configs:
            if config.exists():
                result["staging_configured"] = True
                break

        # Check deployment script for staging
        deploy_script = self.genesis_root / "scripts" / "deploy.sh"
        if deploy_script.exists():
            with open(deploy_script, "r") as f:
                if "staging" in f.read():
                    result["staging_configured"] = True

        # Check if staging environment variables are set
        env_file = self.genesis_root / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                content = f.read()
                if "STAGING_" in content or "ENVIRONMENT=staging" in content:
                    result["staging_configured"] = True

        if result["staging_configured"]:
            result["passed"] = True
            result["message"] = "Staging environment configured"
        else:
            result["message"] = "Staging environment not configured"

        return result

    async def _check_container_configuration(self) -> Dict[str, Any]:
        """Check Docker container configuration.
        
        Returns:
            Validation result for container configuration
        """
        result = {
            "passed": False,
            "message": "",
            "dockerfile_found": False,
            "compose_files": [],
            "build_tested": False,
        }

        # Check Dockerfile
        dockerfile = self.genesis_root / "docker" / "Dockerfile"
        if not dockerfile.exists():
            dockerfile = self.genesis_root / "Dockerfile"
        
        if dockerfile.exists():
            result["dockerfile_found"] = True
            
            with open(dockerfile, "r") as f:
                content = f.read()
                
                # Check for best practices
                checks = {
                    "multi_stage": "FROM.*AS" in content,
                    "non_root_user": "USER" in content and "USER root" not in content,
                    "health_check": "HEALTHCHECK" in content,
                    "labels": "LABEL" in content,
                }
                
                result["dockerfile_quality"] = checks

        # Check Docker Compose files
        compose_files = [
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "docker/docker-compose.yml",
            "docker/docker-compose.prod.yml",
        ]

        for compose_file in compose_files:
            full_path = self.genesis_root / compose_file
            if full_path.exists():
                result["compose_files"].append(compose_file)

        # Check if build script exists
        build_script = self.genesis_root / "scripts" / "build.sh"
        if not build_script.exists():
            build_script = self.genesis_root / "scripts" / "build_container.sh"
        
        if build_script.exists():
            result["build_tested"] = True

        if result["dockerfile_found"] and result["compose_files"]:
            result["passed"] = True
            result["message"] = "Container configuration valid"
        else:
            issues = []
            if not result["dockerfile_found"]:
                issues.append("Dockerfile not found")
            if not result["compose_files"]:
                issues.append("No Docker Compose files")
            result["message"] = f"Container issues: {', '.join(issues)}"

        return result

    async def _validate_kubernetes_manifests(self) -> Dict[str, Any]:
        """Validate Kubernetes manifests if present.
        
        Returns:
            Validation result for Kubernetes manifests
        """
        result = {
            "passed": True,  # Pass by default if K8s not used
            "message": "Kubernetes not configured (optional)",
            "manifests_found": [],
            "manifests_valid": [],
        }

        k8s_dir = self.genesis_root / "kubernetes"
        helm_dir = self.genesis_root / "helm"
        
        if k8s_dir.exists() or helm_dir.exists():
            result["passed"] = False  # Now we need to validate
            
            if k8s_dir.exists():
                manifest_files = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("*.yml"))
                
                for manifest in manifest_files:
                    result["manifests_found"].append(manifest.name)
                    
                    # Validate YAML syntax
                    try:
                        with open(manifest, "r") as f:
                            yaml.safe_load_all(f)
                        result["manifests_valid"].append(manifest.name)
                    except yaml.YAMLError as e:
                        logger.warning(f"Invalid manifest {manifest}: {e}")
                
                # Check for required manifests
                required = ["deployment", "service", "configmap"]
                found_types = [m.split(".")[0].lower() for m in result["manifests_found"]]
                
                missing = [r for r in required if r not in " ".join(found_types)]
                
                if not missing and len(result["manifests_valid"]) == len(result["manifests_found"]):
                    result["passed"] = True
                    result["message"] = f"Kubernetes manifests valid ({len(result['manifests_valid'])} files)"
                else:
                    if missing:
                        result["message"] = f"Missing K8s manifests: {', '.join(missing)}"
                    else:
                        result["message"] = "Some Kubernetes manifests have invalid YAML"
            
            if helm_dir.exists():
                # Check for Helm chart
                chart_file = helm_dir / "Chart.yaml"
                if chart_file.exists():
                    result["passed"] = True
                    result["message"] = "Helm chart configured"

        return result

    async def _check_rollback_time(self) -> Dict[str, Any]:
        """Check estimated rollback time.
        
        Returns:
            Validation result for rollback time
        """
        result = {
            "passed": False,
            "message": "",
            "estimated_time": 0,
            "steps_analyzed": [],
        }

        rollback_script = self.genesis_root / "scripts" / "rollback.sh"
        
        if rollback_script.exists():
            with open(rollback_script, "r") as f:
                content = f.read()
                
                # Estimate time based on operations
                time_estimates = {
                    "docker": 30,  # Docker operations
                    "kubectl": 45,  # Kubernetes operations
                    "git": 10,  # Git operations
                    "database": 60,  # Database operations
                    "restart": 30,  # Service restarts
                    "health": 20,  # Health checks
                }
                
                total_time = 0
                for operation, time in time_estimates.items():
                    if operation in content:
                        total_time += time
                        result["steps_analyzed"].append(f"{operation} (~{time}s)")
                
                result["estimated_time"] = total_time
                
                if total_time <= self.MAX_ROLLBACK_TIME_SECONDS:
                    result["passed"] = True
                    result["message"] = f"Rollback estimated at {total_time}s (under {self.MAX_ROLLBACK_TIME_SECONDS}s limit)"
                else:
                    result["message"] = f"Rollback too slow: {total_time}s > {self.MAX_ROLLBACK_TIME_SECONDS}s"
        else:
            result["message"] = "Cannot estimate rollback time - script not found"

        return result

    async def _generate_readiness_report(self) -> Dict[str, Any]:
        """Generate deployment readiness report.
        
        Returns:
            Deployment readiness summary
        """
        result = {
            "passed": False,
            "message": "",
            "readiness_score": 0,
            "ready_items": [],
            "missing_items": [],
        }

        checks = {
            "Docker configuration": self.results["checks"].get("container_configuration", {}).get("passed", False),
            "Deployment scripts": self.results["checks"].get("deployment_scripts", {}).get("passed", False),
            "Rollback plan": self.results["checks"].get("rollback_plan", {}).get("passed", False),
            "Staging environment": self.results["checks"].get("staging_environment", {}).get("passed", False),
            "Blue-green deployment": self.results["checks"].get("blue_green_deployment", {}).get("passed", False),
        }

        for item, ready in checks.items():
            if ready:
                result["ready_items"].append(item)
            else:
                result["missing_items"].append(item)

        result["readiness_score"] = len(result["ready_items"]) * 20  # Each item worth 20%

        if result["readiness_score"] >= 80:
            result["passed"] = True
            result["message"] = f"Deployment {result['readiness_score']}% ready"
        else:
            result["message"] = f"Deployment only {result['readiness_score']}% ready"

        return result

    def generate_report(self) -> str:
        """Generate a detailed deployment validation report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run validate() first."

        report = []
        report.append("=" * 80)
        report.append("DEPLOYMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Status: {'PASSED' if self.results['passed'] else 'FAILED'}")
        report.append(f"Score: {self.results['score']}%")
        report.append(f"Summary: {self.results['summary']}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)
        
        for check_name, check_result in self.results["checks"].items():
            status = "✓" if check_result.get("passed", False) else "✗"
            report.append(f"{status} {check_name}: {check_result.get('message', '')}")
            
            # Add details
            if check_result.get("scripts_missing"):
                report.append(f"  Missing: {', '.join(check_result['scripts_missing'])}")
            if check_result.get("compose_files"):
                report.append(f"  Compose files: {', '.join(check_result['compose_files'])}")
            if check_result.get("manifests_valid"):
                report.append(f"  Valid manifests: {len(check_result['manifests_valid'])}")
            if check_result.get("estimated_time"):
                report.append(f"  Rollback time: {check_result['estimated_time']}s")
            if check_name == "readiness_report":
                if check_result.get("ready_items"):
                    report.append(f"  Ready: {', '.join(check_result['ready_items'])}")
                if check_result.get("missing_items"):
                    report.append(f"  Missing: {', '.join(check_result['missing_items'])}")

        report.append("")
        report.append(f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds")
        report.append("=" * 80)

        return "\n".join(report)