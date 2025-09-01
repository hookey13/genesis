"""
Integration tests for Kubernetes manifest validation.

Tests verify that Kubernetes manifests are correctly configured
for deployment, scaling, networking, and persistence.
"""

import yaml
import pytest
from pathlib import Path


class TestKubernetesManifests:
    """Test suite for Kubernetes manifest validation."""
    
    @pytest.fixture
    def k8s_dir(self):
        """Get the kubernetes directory path."""
        return Path(__file__).parent.parent.parent / "kubernetes"
    
    @pytest.fixture
    def manifests(self, k8s_dir):
        """Load all Kubernetes manifests."""
        manifests = {}
        for yaml_file in k8s_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                # Handle multi-document YAML files
                docs = list(yaml.safe_load_all(f))
                manifests[yaml_file.stem] = docs
        return manifests
    
    def test_namespace_manifest(self, manifests):
        """Test namespace configuration."""
        assert 'namespace' in manifests, "Missing namespace.yaml"
        
        docs = manifests['namespace']
        namespace_doc = next((d for d in docs if d['kind'] == 'Namespace'), None)
        
        assert namespace_doc is not None, "Namespace resource not found"
        assert namespace_doc['metadata']['name'] == 'genesis', "Incorrect namespace name"
        assert 'labels' in namespace_doc['metadata'], "Missing namespace labels"
        
        # Check for ResourceQuota
        quota_doc = next((d for d in docs if d['kind'] == 'ResourceQuota'), None)
        assert quota_doc is not None, "ResourceQuota not configured"
        
        # Check for LimitRange
        limit_doc = next((d for d in docs if d['kind'] == 'LimitRange'), None)
        assert limit_doc is not None, "LimitRange not configured"
    
    def test_deployment_manifest(self, manifests):
        """Test deployment configuration."""
        assert 'deployment' in manifests, "Missing deployment.yaml"
        
        docs = manifests['deployment']
        deployment_doc = next((d for d in docs if d['kind'] == 'Deployment'), None)
        
        assert deployment_doc is not None, "Deployment resource not found"
        
        # Check basic configuration
        spec = deployment_doc['spec']
        assert 'replicas' in spec, "Missing replicas specification"
        assert 'selector' in spec, "Missing selector"
        assert 'template' in spec, "Missing pod template"
        
        # Check container configuration
        containers = spec['template']['spec']['containers']
        assert len(containers) > 0, "No containers defined"
        
        genesis_container = containers[0]
        assert genesis_container['name'] == 'genesis', "Incorrect container name"
        
        # Check resource limits
        assert 'resources' in genesis_container, "Missing resource specification"
        resources = genesis_container['resources']
        assert 'requests' in resources, "Missing resource requests"
        assert 'limits' in resources, "Missing resource limits"
        assert resources['requests']['cpu'] == '100m', "Incorrect CPU request"
        assert resources['limits']['cpu'] == '500m', "Incorrect CPU limit"
        assert resources['requests']['memory'] == '256Mi', "Incorrect memory request"
        assert resources['limits']['memory'] == '1Gi', "Incorrect memory limit"
        
        # Check health probes
        assert 'livenessProbe' in genesis_container, "Missing liveness probe"
        assert 'readinessProbe' in genesis_container, "Missing readiness probe"
        
        # Check probes use doctor command
        liveness = genesis_container['livenessProbe']
        assert liveness['exec']['command'] == ['python', '-m', 'genesis.cli', 'doctor'], \
            "Liveness probe should use doctor command"
        
        # Check security context
        pod_spec = spec['template']['spec']
        assert 'securityContext' in pod_spec, "Missing pod security context"
        assert pod_spec['securityContext']['runAsNonRoot'] is True, "Should run as non-root"
        assert pod_spec['securityContext']['runAsUser'] == 1000, "Should run as UID 1000"
        
        # Check ServiceAccount
        sa_doc = next((d for d in docs if d['kind'] == 'ServiceAccount'), None)
        assert sa_doc is not None, "ServiceAccount not found"
    
    def test_service_manifest(self, manifests):
        """Test service configuration."""
        assert 'service' in manifests, "Missing service.yaml"
        
        docs = manifests['service']
        
        # Check ClusterIP service
        service_doc = next((d for d in docs if d['kind'] == 'Service' and 
                          d['metadata']['name'] == 'genesis'), None)
        
        assert service_doc is not None, "Service resource not found"
        
        # Check ports
        ports = service_doc['spec']['ports']
        assert len(ports) >= 2, "Should expose at least 2 ports"
        
        api_port = next((p for p in ports if p['name'] == 'api'), None)
        assert api_port is not None, "API port not exposed"
        assert api_port['port'] == 8000, "Incorrect API port"
        
        metrics_port = next((p for p in ports if p['name'] == 'metrics'), None)
        assert metrics_port is not None, "Metrics port not exposed"
        assert metrics_port['port'] == 9090, "Incorrect metrics port"
        
        # Check headless service
        headless_doc = next((d for d in docs if d['kind'] == 'Service' and 
                           'headless' in d['metadata']['name']), None)
        assert headless_doc is not None, "Headless service not found"
        assert headless_doc['spec']['clusterIP'] == 'None', "Headless service should have clusterIP: None"
    
    def test_hpa_manifest(self, manifests):
        """Test HorizontalPodAutoscaler configuration."""
        assert 'hpa' in manifests, "Missing hpa.yaml"
        
        docs = manifests['hpa']
        hpa_doc = docs[0] if docs else None
        
        assert hpa_doc is not None, "HPA resource not found"
        assert hpa_doc['kind'] == 'HorizontalPodAutoscaler', "Incorrect resource kind"
        
        spec = hpa_doc['spec']
        assert spec['minReplicas'] == 1, "Incorrect min replicas"
        assert spec['maxReplicas'] == 3, "Incorrect max replicas"
        
        # Check metrics
        metrics = spec['metrics']
        cpu_metric = next((m for m in metrics if m['resource']['name'] == 'cpu'), None)
        assert cpu_metric is not None, "CPU metric not configured"
        assert cpu_metric['resource']['target']['averageUtilization'] == 70, \
            "CPU target should be 70%"
        
        # Check scaling behavior
        assert 'behavior' in spec, "Scaling behavior not configured"
        assert 'scaleDown' in spec['behavior'], "Scale down behavior not configured"
        assert 'scaleUp' in spec['behavior'], "Scale up behavior not configured"
    
    def test_network_policy_manifest(self, manifests):
        """Test NetworkPolicy configuration."""
        assert 'network-policy' in manifests, "Missing network-policy.yaml"
        
        docs = manifests['network-policy']
        
        # Check main network policy
        main_policy = next((d for d in docs if d['metadata']['name'] == 'genesis-network-policy'), None)
        assert main_policy is not None, "Main network policy not found"
        
        spec = main_policy['spec']
        assert 'policyTypes' in spec, "Policy types not specified"
        assert 'Ingress' in spec['policyTypes'], "Ingress policy not configured"
        assert 'Egress' in spec['policyTypes'], "Egress policy not configured"
        
        # Check ingress rules
        assert 'ingress' in spec, "Ingress rules not configured"
        ingress_rules = spec['ingress']
        assert len(ingress_rules) > 0, "No ingress rules defined"
        
        # Check egress rules
        assert 'egress' in spec, "Egress rules not configured"
        egress_rules = spec['egress']
        
        # Check for DNS egress rule
        dns_rule = next((r for r in egress_rules if any(
            p.get('port') == 53 for p in r.get('ports', [])
        )), None)
        assert dns_rule is not None, "DNS egress rule not configured"
        
        # Check for HTTPS egress rule (Binance API)
        https_rule = next((r for r in egress_rules if any(
            p.get('port') == 443 for p in r.get('ports', [])
        )), None)
        assert https_rule is not None, "HTTPS egress rule not configured for Binance API"
        
        # Check default deny policy
        deny_all = next((d for d in docs if d['metadata']['name'] == 'deny-all-default'), None)
        assert deny_all is not None, "Default deny-all policy not found"
    
    def test_pvc_manifest(self, manifests):
        """Test PersistentVolumeClaim configuration."""
        assert 'pvc' in manifests, "Missing pvc.yaml"
        
        docs = manifests['pvc']
        
        # Check for required PVCs
        required_pvcs = {
            'genesis-data-pvc': '10Gi',
            'genesis-logs-pvc': '5Gi',
            'genesis-state-pvc': '1Gi',
            'genesis-backups-pvc': '20Gi'
        }
        
        for pvc_name, expected_size in required_pvcs.items():
            pvc_doc = next((d for d in docs if d['metadata']['name'] == pvc_name), None)
            assert pvc_doc is not None, f"PVC {pvc_name} not found"
            
            # Check access mode
            assert 'ReadWriteOnce' in pvc_doc['spec']['accessModes'], \
                f"{pvc_name} should have ReadWriteOnce access"
            
            # Check storage size
            storage = pvc_doc['spec']['resources']['requests']['storage']
            assert storage == expected_size, \
                f"{pvc_name} should request {expected_size}, got {storage}"
    
    def test_labels_consistency(self, manifests):
        """Test that labels are consistent across all resources."""
        required_labels = ['app', 'component']
        
        for manifest_name, docs in manifests.items():
            for doc in docs:
                if doc and 'metadata' in doc:
                    labels = doc['metadata'].get('labels', {})
                    
                    # Skip certain resources that might not need all labels
                    if doc['kind'] in ['Namespace', 'ResourceQuota', 'LimitRange']:
                        continue
                    
                    for label in required_labels:
                        assert label in labels, \
                            f"{doc['kind']} in {manifest_name} missing label: {label}"
                    
                    # Check app label value
                    if 'app' in labels:
                        assert labels['app'] == 'genesis', \
                            f"Inconsistent app label in {doc['kind']}: {labels['app']}"
    
    def test_prometheus_annotations(self, manifests):
        """Test Prometheus scraping annotations."""
        # Check deployment
        deployment_docs = manifests.get('deployment', [])
        deployment = next((d for d in deployment_docs if d['kind'] == 'Deployment'), None)
        
        if deployment:
            annotations = deployment['spec']['template']['metadata'].get('annotations', {})
            assert 'prometheus.io/scrape' in annotations, "Missing Prometheus scrape annotation"
            assert annotations['prometheus.io/scrape'] == 'true', "Prometheus scrape should be enabled"
            assert annotations.get('prometheus.io/port') == '9090', "Incorrect Prometheus port"
        
        # Check service
        service_docs = manifests.get('service', [])
        service = next((d for d in service_docs if d['metadata']['name'] == 'genesis'), None)
        
        if service:
            annotations = service['metadata'].get('annotations', {})
            assert 'prometheus.io/scrape' in annotations, "Service missing Prometheus annotation"
    
    def test_service_mesh_readiness(self, manifests):
        """Test service mesh integration annotations."""
        deployment_docs = manifests.get('deployment', [])
        deployment = next((d for d in deployment_docs if d['kind'] == 'Deployment'), None)
        
        if deployment:
            annotations = deployment['spec']['template']['metadata'].get('annotations', {})
            
            # Check Istio annotations
            assert 'sidecar.istio.io/inject' in annotations, "Missing Istio annotation"
            
            # Check Linkerd annotations
            assert 'linkerd.io/inject' in annotations, "Missing Linkerd annotation"
    
    def test_resource_limits_reasonable(self, manifests):
        """Test that resource limits are reasonable for trading application."""
        deployment_docs = manifests.get('deployment', [])
        deployment = next((d for d in deployment_docs if d['kind'] == 'Deployment'), None)
        
        if deployment:
            container = deployment['spec']['template']['spec']['containers'][0]
            resources = container['resources']
            
            # Parse memory values
            request_memory = resources['requests']['memory']
            limit_memory = resources['limits']['memory']
            
            # Basic sanity checks
            assert request_memory.endswith('Mi'), "Memory request should be in Mi"
            assert limit_memory.endswith('Gi') or limit_memory.endswith('Mi'), \
                "Memory limit should be in Gi or Mi"
            
            # Parse CPU values
            request_cpu = resources['requests']['cpu']
            limit_cpu = resources['limits']['cpu']
            
            assert request_cpu.endswith('m'), "CPU request should be in millicores"
            assert limit_cpu.endswith('m'), "CPU limit should be in millicores"
            
            # Check ratios
            request_cpu_value = int(request_cpu.rstrip('m'))
            limit_cpu_value = int(limit_cpu.rstrip('m'))
            
            assert limit_cpu_value >= request_cpu_value, \
                "CPU limit should be >= request"
            assert limit_cpu_value <= request_cpu_value * 10, \
                "CPU limit should not be more than 10x request"