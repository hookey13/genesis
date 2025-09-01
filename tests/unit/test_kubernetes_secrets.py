"""
Unit tests for Kubernetes secrets and ConfigMap validation.

Tests verify that secrets are properly configured, secured,
and include rotation mechanisms.
"""

import base64
import yaml
import pytest
from pathlib import Path


class TestKubernetesSecrets:
    """Test suite for Kubernetes secrets configuration."""
    
    @pytest.fixture
    def k8s_dir(self):
        """Get the kubernetes directory path."""
        return Path(__file__).parent.parent.parent / "kubernetes"
    
    @pytest.fixture
    def secret_config(self, k8s_dir):
        """Load secret configuration."""
        secret_file = k8s_dir / "secret.yaml"
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                docs = list(yaml.safe_load_all(f))
                return docs
        return []
    
    @pytest.fixture
    def configmap_config(self, k8s_dir):
        """Load ConfigMap configuration."""
        configmap_file = k8s_dir / "configmap.yaml"
        if configmap_file.exists():
            with open(configmap_file, 'r') as f:
                docs = list(yaml.safe_load_all(f))
                return docs
        return []
    
    def test_secret_template_exists(self, k8s_dir):
        """Test that secret template file exists."""
        secret_file = k8s_dir / "secret.yaml"
        assert secret_file.exists(), "Secret template file not found"
        
        # Check that it contains warning about not committing actual secrets
        with open(secret_file, 'r') as f:
            content = f.read()
        
        assert "IMPORTANT: This is a template file" in content, \
            "Secret file should contain warning about template"
        assert "Do not commit actual secrets" in content.lower(), \
            "Secret file should warn against committing secrets"
    
    def test_secret_structure(self, secret_config):
        """Test that secret has correct structure."""
        assert len(secret_config) > 0, "No secrets defined"
        
        # Find main secret
        main_secret = next((d for d in secret_config if 
                          d.get('metadata', {}).get('name') == 'genesis-secrets'), None)
        
        assert main_secret is not None, "Main genesis-secrets not found"
        assert main_secret['kind'] == 'Secret', "Resource should be of kind Secret"
        assert main_secret['type'] == 'Opaque', "Secret should be of type Opaque"
        
        # Check required data keys
        required_keys = [
            'binance-api-key',
            'binance-api-secret',
            'database-url'
        ]
        
        data = main_secret.get('data', {})
        for key in required_keys:
            assert key in data, f"Required key '{key}' not found in secret data"
    
    def test_secret_values_are_base64(self, secret_config):
        """Test that secret values are base64 encoded."""
        main_secret = next((d for d in secret_config if 
                          d.get('metadata', {}).get('name') == 'genesis-secrets'), None)
        
        if main_secret:
            data = main_secret.get('data', {})
            
            for key, value in data.items():
                if value and not value.startswith('#'):
                    try:
                        # Try to decode - should work if valid base64
                        base64.b64decode(value.split()[0])  # Split to remove comments
                    except Exception:
                        pytest.fail(f"Secret value for '{key}' is not valid base64")
    
    def test_tls_secret_structure(self, secret_config):
        """Test TLS secret configuration."""
        tls_secret = next((d for d in secret_config if 
                         d.get('metadata', {}).get('name') == 'genesis-tls'), None)
        
        if tls_secret:
            assert tls_secret['type'] == 'kubernetes.io/tls', \
                "TLS secret should have type kubernetes.io/tls"
            
            data = tls_secret.get('data', {})
            assert 'tls.crt' in data, "TLS certificate not found"
            assert 'tls.key' in data, "TLS private key not found"
    
    def test_configmap_structure(self, configmap_config):
        """Test ConfigMap structure."""
        assert len(configmap_config) > 0, "No ConfigMaps defined"
        
        # Check main config
        main_config = next((d for d in configmap_config if 
                          d.get('metadata', {}).get('name') == 'genesis-config'), None)
        
        assert main_config is not None, "Main genesis-config not found"
        assert main_config['kind'] == 'ConfigMap', "Resource should be of kind ConfigMap"
        
        # Check required configuration keys
        required_keys = [
            'deployment_env',
            'binance_testnet',
            'log_level'
        ]
        
        data = main_config.get('data', {})
        for key in required_keys:
            assert key in data, f"Required config key '{key}' not found"
    
    def test_trading_rules_configmap(self, configmap_config):
        """Test trading rules ConfigMap."""
        trading_rules = next((d for d in configmap_config if 
                            d.get('metadata', {}).get('name') == 'genesis-trading-rules'), None)
        
        assert trading_rules is not None, "Trading rules ConfigMap not found"
        
        data = trading_rules.get('data', {})
        assert 'trading_rules.yaml' in data, "trading_rules.yaml not found in ConfigMap"
        
        # Parse the YAML content
        rules_content = data['trading_rules.yaml']
        rules = yaml.safe_load(rules_content)
        
        # Check tier configurations
        assert 'sniper' in rules, "Sniper tier configuration missing"
        assert 'hunter' in rules, "Hunter tier configuration missing"
        assert 'strategist' in rules, "Strategist tier configuration missing"
        
        # Check required fields for each tier
        for tier in ['sniper', 'hunter', 'strategist']:
            tier_config = rules[tier]
            assert 'max_position_size_usdt' in tier_config, f"{tier}: max_position_size_usdt missing"
            assert 'max_daily_loss_usdt' in tier_config, f"{tier}: max_daily_loss_usdt missing"
            assert 'max_orders_per_minute' in tier_config, f"{tier}: max_orders_per_minute missing"
    
    def test_tier_gates_configmap(self, configmap_config):
        """Test tier gates ConfigMap."""
        tier_gates = next((d for d in configmap_config if 
                         d.get('metadata', {}).get('name') == 'genesis-tier-gates'), None)
        
        assert tier_gates is not None, "Tier gates ConfigMap not found"
        
        data = tier_gates.get('data', {})
        assert 'tier_gates.yaml' in data, "tier_gates.yaml not found in ConfigMap"
        
        # Parse the YAML content
        gates_content = data['tier_gates.yaml']
        gates = yaml.safe_load(gates_content)
        
        # Check gate configurations
        assert 'gates' in gates, "Gates section missing"
        assert 'sniper_to_hunter' in gates['gates'], "Sniper to Hunter gate missing"
        assert 'hunter_to_strategist' in gates['gates'], "Hunter to Strategist gate missing"
        
        # Check demotion triggers
        assert 'demotion_triggers' in gates, "Demotion triggers missing"
        assert 'immediate' in gates['demotion_triggers'], "Immediate demotion triggers missing"
        assert 'daily_review' in gates['demotion_triggers'], "Daily review triggers missing"
    
    def test_secret_rotation_documentation(self, secret_config):
        """Test that secret rotation is documented."""
        # Check for rotation documentation in comments or separate ConfigMap
        rotation_doc = next((d for d in secret_config if 
                           d.get('metadata', {}).get('name') == 'genesis-secret-rotation'), None)
        
        if not rotation_doc:
            # Check if it's in the main secret file
            main_secret = next((d for d in secret_config if 
                              d.get('metadata', {}).get('name') == 'genesis-secrets'), None)
            
            if main_secret:
                annotations = main_secret.get('metadata', {}).get('annotations', {})
                assert 'rotation-policy' in annotations, \
                    "Secret should have rotation policy annotation"
    
    def test_no_hardcoded_secrets(self, k8s_dir):
        """Test that no actual secrets are hardcoded."""
        # Check all YAML files for potential hardcoded secrets
        dangerous_patterns = [
            'password:',
            'secret:',
            'key:',
            'token:'
        ]
        
        for yaml_file in k8s_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                content = f.read().lower()
            
            # Skip template files
            if 'template' in content or 'put-your' in content:
                continue
            
            for pattern in dangerous_patterns:
                if pattern in content:
                    # Check if it's followed by a placeholder or base64 encoded value
                    lines = content.split('\n')
                    for line in lines:
                        if pattern in line and not any(
                            placeholder in line for placeholder in 
                            ['put-your', 'placeholder', 'example', 'base64', 'secret']
                        ):
                            # Additional check: if it looks like a real value
                            value_part = line.split(pattern)[1].strip()
                            if value_part and not value_part.startswith('#'):
                                pytest.fail(f"Potential hardcoded secret in {yaml_file.name}: {line}")
    
    def test_configmap_values_not_sensitive(self, configmap_config):
        """Test that ConfigMaps don't contain sensitive data."""
        sensitive_keywords = [
            'password', 'secret', 'key', 'token', 'credential'
        ]
        
        for configmap in configmap_config:
            if configmap and configmap.get('kind') == 'ConfigMap':
                data = configmap.get('data', {})
                
                for key, value in data.items():
                    key_lower = key.lower()
                    value_lower = str(value).lower() if value else ''
                    
                    for keyword in sensitive_keywords:
                        assert keyword not in key_lower, \
                            f"ConfigMap key '{key}' might contain sensitive data"
                        
                        # Check value but allow documentation
                        if keyword in value_lower and 'example' not in value_lower:
                            # Additional check for actual values vs documentation
                            if not any(doc_word in value_lower for doc_word in 
                                     ['template', 'placeholder', 'put-your', 'description']):
                                pytest.fail(
                                    f"ConfigMap value for '{key}' might contain sensitive data"
                                )
    
    def test_secret_references_in_deployment(self, k8s_dir):
        """Test that deployment correctly references secrets."""
        deployment_file = k8s_dir / "deployment.yaml"
        
        if deployment_file.exists():
            with open(deployment_file, 'r') as f:
                docs = list(yaml.safe_load_all(f))
            
            deployment = next((d for d in docs if d.get('kind') == 'Deployment'), None)
            
            if deployment:
                containers = deployment['spec']['template']['spec']['containers']
                genesis_container = containers[0]
                
                env_vars = genesis_container.get('env', [])
                
                # Check for secret references
                secret_refs = [
                    'BINANCE_API_KEY',
                    'BINANCE_API_SECRET',
                    'DATABASE_URL'
                ]
                
                for ref in secret_refs:
                    env_var = next((e for e in env_vars if e['name'] == ref), None)
                    assert env_var is not None, f"Environment variable {ref} not found"
                    
                    # Check it references a secret
                    assert 'valueFrom' in env_var, f"{ref} should use valueFrom"
                    assert 'secretKeyRef' in env_var['valueFrom'], \
                        f"{ref} should reference a secret"
                    assert env_var['valueFrom']['secretKeyRef']['name'] == 'genesis-secrets', \
                        f"{ref} should reference genesis-secrets"