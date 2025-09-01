# Service Mesh Integration Guide

This directory contains configurations and documentation for integrating Genesis Trading System with service mesh technologies.

## Supported Service Meshes

### Istio
- **Version**: 1.19+
- **Features**: Traffic management, security, observability
- **Status**: Ready for integration

### Linkerd
- **Version**: 2.14+
- **Features**: Lightweight, automatic mTLS, observability
- **Status**: Ready for integration

## Enabling Service Mesh

### For Istio

1. **Enable sidecar injection in deployment:**
```yaml
annotations:
  sidecar.istio.io/inject: "true"
```

2. **Apply Istio configurations:**
```bash
kubectl label namespace genesis istio-injection=enabled
kubectl apply -f kubernetes/service-mesh/istio/
```

3. **Using Helm:**
```bash
helm upgrade genesis ./helm/genesis \
  --set serviceMesh.enabled=true \
  --set serviceMesh.type=istio \
  --set serviceMesh.istio.sidecarInject=true
```

### For Linkerd

1. **Enable injection in deployment:**
```yaml
annotations:
  linkerd.io/inject: enabled
```

2. **Apply Linkerd configurations:**
```bash
kubectl annotate namespace genesis linkerd.io/inject=enabled
kubectl apply -f kubernetes/service-mesh/linkerd/
```

3. **Using Helm:**
```bash
helm upgrade genesis ./helm/genesis \
  --set serviceMesh.enabled=true \
  --set serviceMesh.type=linkerd \
  --set serviceMesh.linkerd.inject=enabled
```

## Traffic Management

### Istio VirtualService Example
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genesis
  namespace: genesis
spec:
  hosts:
  - genesis
  http:
  - match:
    - headers:
        tier:
          exact: strategist
    route:
    - destination:
        host: genesis
        subset: v2
  - route:
    - destination:
        host: genesis
        subset: v1
```

### Linkerd Traffic Split Example
```yaml
apiVersion: split.smi-spec.io/v1alpha1
kind: TrafficSplit
metadata:
  name: genesis
  namespace: genesis
spec:
  service: genesis
  backends:
  - service: genesis-v1
    weight: 90
  - service: genesis-v2
    weight: 10
```

## Security Policies

### mTLS Configuration
Both Istio and Linkerd provide automatic mTLS between services.

### Authorization Policies (Istio)
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: genesis-authz
  namespace: genesis
spec:
  selector:
    matchLabels:
      app: genesis
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/monitoring/sa/prometheus"]
    to:
    - operation:
        ports: ["9090"]
```

## Observability

### Metrics
- Both service meshes expose additional metrics
- Prometheus endpoints are automatically discovered
- Grafana dashboards available for both meshes

### Tracing
- Distributed tracing enabled by default
- Jaeger or Zipkin integration supported
- Trace headers automatically propagated

### Service Graph
- Visual representation of service communication
- Available through Kiali (Istio) or Linkerd Dashboard

## Best Practices

1. **Start with canary deployments** - Test service mesh in staging first
2. **Monitor resource usage** - Sidecars add ~100MB memory overhead
3. **Configure circuit breakers** - Prevent cascade failures
4. **Use retry policies carefully** - Avoid retry storms
5. **Enable gradual rollouts** - Use traffic splitting for safe deployments

## Troubleshooting

### Check sidecar injection:
```bash
kubectl get pods -n genesis -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].name}{"\n"}{end}'
```

### Verify mTLS:
```bash
# Istio
istioctl authn tls-check genesis.genesis.svc.cluster.local

# Linkerd
linkerd viz edges -n genesis
```

### Debug traffic routing:
```bash
# Istio
istioctl analyze -n genesis

# Linkerd
linkerd check --proxy -n genesis
```

## Migration Path

1. **Phase 1**: Deploy with service mesh annotations disabled
2. **Phase 2**: Enable in development environment
3. **Phase 3**: Test traffic management features
4. **Phase 4**: Enable in staging with canary deployment
5. **Phase 5**: Production rollout with gradual traffic shift