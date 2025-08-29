"""
Integration tests for GitHub Actions CI/CD pipeline.
Validates workflow syntax and configuration.
"""

import os
import yaml
from pathlib import Path
from decimal import Decimal
import pytest
import structlog

logger = structlog.get_logger(__name__)


class TestGitHubActions:
    """Test GitHub Actions workflows configuration."""
    
    @pytest.fixture
    def workflows_dir(self):
        """Provide workflows directory path."""
        return Path(__file__).parent.parent.parent / ".github" / "workflows"
    
    @pytest.fixture
    def test_workflow(self, workflows_dir):
        """Provide test workflow file path."""
        return workflows_dir / "test.yml"
    
    @pytest.fixture
    def deploy_workflow(self, workflows_dir):
        """Provide deploy workflow file path."""
        return workflows_dir / "deploy.yml"
    
    def test_workflow_files_exist(self, test_workflow, deploy_workflow):
        """Test that workflow files exist."""
        assert test_workflow.exists(), f"Test workflow not found at {test_workflow}"
        assert deploy_workflow.exists(), f"Deploy workflow not found at {deploy_workflow}"
    
    def test_test_workflow_syntax(self, test_workflow):
        """Test that test workflow has valid YAML syntax."""
        with open(test_workflow, 'r') as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "Empty workflow file"
                assert 'name' in config, "Missing workflow name"
                # 'on' can be parsed as True in YAML, check for both
                assert 'on' in config or True in config, "Missing trigger configuration"
                assert 'jobs' in config, "Missing jobs definition"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in test workflow: {e}")
    
    def test_deploy_workflow_syntax(self, deploy_workflow):
        """Test that deploy workflow has valid YAML syntax."""
        with open(deploy_workflow, 'r') as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "Empty workflow file"
                assert 'name' in config, "Missing workflow name"
                # 'on' can be parsed as True in YAML, check for both
                assert 'on' in config or True in config, "Missing trigger configuration"
                assert 'jobs' in config, "Missing jobs definition"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in deploy workflow: {e}")
    
    def test_python_version_consistency(self, test_workflow, deploy_workflow):
        """Test that Python version is consistent across workflows."""
        python_version = '3.11.8'
        
        for workflow_file in [test_workflow, deploy_workflow]:
            with open(workflow_file, 'r') as f:
                content = f.read()
                # Check for Python version in env or direct usage
                assert python_version in content or 'PYTHON_VERSION' in content, \
                    f"Python version {python_version} not specified in {workflow_file.name}"
    
    def test_test_workflow_jobs(self, test_workflow):
        """Test that test workflow has all required jobs."""
        with open(test_workflow, 'r') as f:
            config = yaml.safe_load(f)
            jobs = config.get('jobs', {})
            
            # Required jobs for test pipeline
            required_jobs = [
                'lint',
                'security',
                'unit-tests',
                'integration-tests',
                'docker-build'
            ]
            
            for job in required_jobs:
                assert job in jobs, f"Missing required job: {job}"
                
                # Check job has required fields
                job_config = jobs[job]
                assert 'name' in job_config, f"Job {job} missing name"
                assert 'runs-on' in job_config, f"Job {job} missing runs-on"
                assert 'steps' in job_config, f"Job {job} missing steps"
    
    def test_deploy_workflow_jobs(self, deploy_workflow):
        """Test that deploy workflow has all required jobs."""
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            jobs = config.get('jobs', {})
            
            # Required jobs for deploy pipeline
            required_jobs = [
                'validate-deployment',
                'build-and-push',
                'staging-deployment',
                'production-deployment'
            ]
            
            for job in required_jobs:
                assert job in jobs, f"Missing required job: {job}"
    
    def test_workflow_triggers(self, test_workflow, deploy_workflow):
        """Test workflow trigger configuration."""
        # Test workflow triggers
        with open(test_workflow, 'r') as f:
            config = yaml.safe_load(f)
            # 'on' might be parsed as True
            triggers = config.get('on', config.get(True, {}))
            
            # Should trigger on push and PR
            assert 'push' in triggers or 'pull_request' in triggers, \
                "Test workflow should trigger on push or pull_request"
        
        # Deploy workflow triggers
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            # 'on' might be parsed as True
            triggers = config.get('on', config.get(True, {}))
            
            # Should have manual trigger
            assert 'workflow_dispatch' in triggers, \
                "Deploy workflow should have manual trigger"
            
            # Check for tag trigger for auto-deployment
            if 'push' in triggers:
                push_config = triggers['push']
                assert 'tags' in push_config, \
                    "Deploy workflow should trigger on tags"
    
    def test_caching_configuration(self, test_workflow):
        """Test that caching is properly configured."""
        with open(test_workflow, 'r') as f:
            content = f.read()
            
            # Check for cache action usage
            assert 'actions/cache@' in content, \
                "Should use GitHub Actions cache"
            
            # Check for pip cache
            assert 'cache/pip' in content or 'pip.*cache' in content, \
                "Should cache pip dependencies"
    
    def test_security_scanning(self, test_workflow):
        """Test that security scanning is configured."""
        with open(test_workflow, 'r') as f:
            config = yaml.safe_load(f)
            
            # Check for security job
            assert 'security' in config.get('jobs', {}), \
                "Should have security scanning job"
            
            security_job = config['jobs']['security']
            steps = security_job.get('steps', [])
            
            # Check for security tools
            security_tools = ['pip-audit', 'safety', 'bandit']
            job_content = str(security_job)
            
            for tool in security_tools:
                assert tool in job_content, \
                    f"Security job should use {tool}"
    
    def test_test_coverage_reporting(self, test_workflow):
        """Test that coverage reporting is configured."""
        with open(test_workflow, 'r') as f:
            content = f.read()
            
            # Check for coverage configuration
            assert '--cov' in content, \
                "Should generate coverage reports"
            
            # Check for coverage upload
            assert 'codecov' in content or 'coverage' in content, \
                "Should upload coverage reports"
    
    def test_docker_build_configuration(self, test_workflow, deploy_workflow):
        """Test Docker build configuration in workflows."""
        for workflow_file in [test_workflow, deploy_workflow]:
            with open(workflow_file, 'r') as f:
                content = f.read()
                
                if 'docker' in content.lower():
                    # Check for buildx setup
                    assert 'docker/setup-buildx-action' in content, \
                        f"Should set up Docker Buildx in {workflow_file.name}"
                    
                    # Check for build args
                    assert 'BUILD_DATE' in content, \
                        f"Should pass BUILD_DATE arg in {workflow_file.name}"
                    assert 'VERSION' in content, \
                        f"Should pass VERSION arg in {workflow_file.name}"
    
    def test_deployment_environments(self, deploy_workflow):
        """Test deployment environment configuration."""
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            
            # Check staging deployment
            staging_job = config['jobs'].get('staging-deployment', {})
            if 'environment' in staging_job:
                env_config = staging_job['environment']
                assert 'name' in env_config, \
                    "Staging environment should have name"
                assert env_config['name'] == 'staging', \
                    "Staging environment name should be 'staging'"
            
            # Check production deployment
            prod_job = config['jobs'].get('production-deployment', {})
            if 'environment' in prod_job:
                env_config = prod_job['environment']
                assert 'name' in env_config, \
                    "Production environment should have name"
                assert env_config['name'] == 'production', \
                    "Production environment name should be 'production'"
    
    def test_rollback_mechanism(self, deploy_workflow):
        """Test that rollback is configured."""
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            
            # Check for rollback job
            assert 'rollback' in config.get('jobs', {}), \
                "Should have rollback job"
            
            rollback_job = config['jobs']['rollback']
            
            # Should trigger on failure
            assert 'if' in rollback_job, \
                "Rollback should have conditional trigger"
            assert 'failure()' in rollback_job['if'], \
                "Rollback should trigger on failure"
    
    def test_artifact_handling(self, test_workflow, deploy_workflow):
        """Test artifact upload/download configuration."""
        for workflow_file in [test_workflow, deploy_workflow]:
            with open(workflow_file, 'r') as f:
                content = f.read()
                
                # Check for artifact actions
                if 'artifact' in content:
                    assert 'actions/upload-artifact' in content or \
                           'actions/download-artifact' in content, \
                        f"Should use artifact actions in {workflow_file.name}"
    
    def test_timeout_configuration(self, test_workflow, deploy_workflow):
        """Test that jobs have timeout configuration."""
        for workflow_file in [test_workflow, deploy_workflow]:
            with open(workflow_file, 'r') as f:
                config = yaml.safe_load(f)
                jobs = config.get('jobs', {})
                
                for job_name, job_config in jobs.items():
                    # Skip the all-tests-passed job as it's just a check
                    if job_name == 'all-tests-passed':
                        continue
                    assert 'timeout-minutes' in job_config, \
                        f"Job {job_name} should have timeout-minutes in {workflow_file.name}"
                    
                    timeout = job_config['timeout-minutes']
                    assert isinstance(timeout, int), \
                        f"timeout-minutes should be integer in {job_name}"
                    assert timeout > 0 and timeout <= 360, \
                        f"timeout should be reasonable (1-360 minutes) in {job_name}"
    
    def test_job_dependencies(self, test_workflow, deploy_workflow):
        """Test job dependency configuration."""
        # Test workflow dependencies
        with open(test_workflow, 'r') as f:
            config = yaml.safe_load(f)
            jobs = config.get('jobs', {})
            
            # Integration tests should depend on lint and unit tests
            if 'integration-tests' in jobs:
                integration_job = jobs['integration-tests']
                assert 'needs' in integration_job, \
                    "Integration tests should have dependencies"
                needs = integration_job['needs']
                assert 'lint' in needs or 'unit-tests' in needs, \
                    "Integration tests should depend on lint or unit tests"
        
        # Deploy workflow dependencies
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            jobs = config.get('jobs', {})
            
            # Deployment jobs should depend on validation
            for deploy_job in ['staging-deployment', 'production-deployment']:
                if deploy_job in jobs:
                    job_config = jobs[deploy_job]
                    assert 'needs' in job_config, \
                        f"{deploy_job} should have dependencies"
                    assert 'validate-deployment' in job_config['needs'], \
                        f"{deploy_job} should depend on validation"
    
    def test_secrets_usage(self, deploy_workflow):
        """Test that secrets are properly referenced."""
        with open(deploy_workflow, 'r') as f:
            content = f.read()
            
            # Check for secret references
            assert '${{ secrets.' in content, \
                "Deploy workflow should use secrets"
            
            # Check for required secrets
            required_secrets = [
                'GITHUB_TOKEN',  # For container registry
            ]
            
            for secret in required_secrets:
                assert secret in content, \
                    f"Should reference {secret} secret"
    
    def test_notification_configuration(self, deploy_workflow):
        """Test that deployment notifications are configured."""
        with open(deploy_workflow, 'r') as f:
            config = yaml.safe_load(f)
            
            # Check for notification job
            assert 'notify-deployment' in config.get('jobs', {}), \
                "Should have notification job"
            
            notify_job = config['jobs']['notify-deployment']
            
            # Should always run
            assert 'if' in notify_job, \
                "Notification should have conditional"
            assert 'always()' in notify_job['if'], \
                "Notification should always run"