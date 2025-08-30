# Network Simulation Documentation

## Overview

Network simulation provides controlled testing of distributed system behavior under various network conditions including partitions, latency, packet loss, and asymmetric failures. This framework enables validation of system resilience to real-world network issues before production deployment.

## Purpose

Network simulation addresses critical distributed system concerns:
- Tests split-brain scenario handling
- Validates partition tolerance (CAP theorem)
- Verifies connection recovery mechanisms
- Tests state consistency after network healing
- Validates order reconciliation after splits
- Measures performance under degraded conditions
- Tests timeout and retry logic

## Architecture

### Core Components

#### 1. NetworkSimulator Class
Main simulation engine that:
- Creates network partitions between nodes
- Simulates latency and jitter
- Introduces packet loss
- Creates asymmetric failures
- Tracks network state
- Monitors recovery

#### 2. Network Topology
Represents network structure:
- Nodes (services, databases, clients)
- Connections (bidirectional links)
- Latency maps
- Partition groups
- Routing tables

#### 3. Failure Modes

| Mode | Description | Real-World Scenario |
|------|-------------|-------------------|
| Split-Brain | Network divided into isolated groups | Data center partition |
| Isolated Node | Single node disconnected | Server network failure |
| Asymmetric | One-way communication failure | Firewall misconfiguration |
| Slow Network | High latency on links | WAN congestion |
| Random Drops | Intermittent packet loss | Unreliable connection |

## Configuration

### Basic Setup

```python
from tests.chaos.network_simulator import NetworkSimulator

# Initialize simulator
simulator = NetworkSimulator()

# Create network topology
await simulator.add_node("trading_engine")
await simulator.add_node("risk_manager")
await simulator.add_node("database")

# Create connections
await simulator.connect("trading_engine", "risk_manager")
await simulator.connect("trading_engine", "database")
await simulator.connect("risk_manager", "database")
```

### Advanced Configuration

```python
simulator = NetworkSimulator(
    default_latency_ms=1,
    default_bandwidth_mbps=1000,
    default_packet_loss=0.0,
    enable_logging=True,
    capture_packets=True,
    routing_protocol="shortest_path"
)

# Configure specific links
await simulator.configure_link(
    "trading_engine",
    "database",
    latency_ms=5,
    bandwidth_mbps=100,
    packet_loss=0.001
)
```

## Network Conditions

### 1. Network Partitions

```python
async def test_split_brain():
    """Test split-brain scenario"""
    simulator = NetworkSimulator()
    
    # Create nodes
    nodes_group1 = ["node1", "node2"]
    nodes_group2 = ["node3", "node4"]
    
    # Create partition
    partition_id = await simulator.create_partition(
        group1=nodes_group1,
        group2=nodes_group2,
        duration_seconds=30
    )
    
    # Verify isolation
    for node1 in nodes_group1:
        for node2 in nodes_group2:
            assert not await simulator.is_reachable(node1, node2)
    
    # Wait for healing
    await asyncio.sleep(30)
    
    # Verify recovery
    assert await simulator.is_reachable("node1", "node3")
```

### 2. Latency Simulation

```python
async def test_latency_impact():
    """Test system behavior with network latency"""
    simulator = NetworkSimulator()
    
    # Add progressive latency
    for latency in [10, 50, 100, 200, 500]:
        await simulator.add_latency("client", "server", latency)
        
        # Measure impact
        response_time = await measure_response_time()
        print(f"Latency {latency}ms: Response {response_time}ms")
        
        if response_time > 1000:
            print("System degraded at {latency}ms network latency")
            break
```

### 3. Packet Loss

```python
async def test_packet_loss_resilience():
    """Test system resilience to packet loss"""
    simulator = NetworkSimulator()
    
    # Test different loss rates
    for loss_rate in [0.01, 0.05, 0.1, 0.2]:
        await simulator.set_packet_loss("client", "server", loss_rate)
        
        # Send test messages
        success_rate = await send_test_messages(count=1000)
        
        print(f"Loss rate {loss_rate:.0%}: Success {success_rate:.2%}")
        
        # System should handle up to 5% loss gracefully
        if loss_rate <= 0.05:
            assert success_rate > 0.95
```

### 4. Asymmetric Failures

```python
async def test_asymmetric_partition():
    """Test asymmetric network failures"""
    simulator = NetworkSimulator()
    
    # Create asymmetric failure: A can reach B, but B cannot reach A
    await simulator.create_asymmetric_partition("nodeA", "nodeB")
    
    # Test communication
    assert await simulator.can_send("nodeA", "nodeB", "message")
    assert not await simulator.can_send("nodeB", "nodeA", "response")
    
    # This can cause issues with request-response patterns
    # System should detect and handle appropriately
```

### 5. Network Jitter

```python
async def test_jitter_handling():
    """Test system behavior with network jitter"""
    simulator = NetworkSimulator()
    
    # Add jitter (variable latency)
    await simulator.add_jitter(
        "client",
        "server",
        base_latency_ms=10,
        jitter_ms=50  # ±50ms variation
    )
    
    # Measure impact on streaming data
    latencies = []
    for _ in range(100):
        start = time.time()
        await send_message("client", "server")
        latencies.append((time.time() - start) * 1000)
    
    # Check variance
    std_dev = statistics.stdev(latencies)
    print(f"Latency std dev: {std_dev:.2f}ms")
```

## Usage Examples

### Complete Network Failure Simulation

```python
import asyncio
from tests.chaos.network_simulator import NetworkSimulator

async def simulate_network_failures():
    """Simulate various network failure scenarios"""
    
    simulator = NetworkSimulator()
    
    # Setup topology
    await simulator.create_topology({
        "nodes": ["web", "app", "cache", "db"],
        "connections": [
            ("web", "app"),
            ("app", "cache"),
            ("app", "db"),
            ("cache", "db")
        ]
    })
    
    # Scenario 1: Database isolation
    print("Simulating database isolation...")
    await simulator.isolate_node("db", duration_seconds=30)
    
    # System should handle with cache
    assert await test_read_operations()
    # Writes should fail or queue
    assert not await test_write_operations()
    
    # Scenario 2: Cache failure with db available
    print("Simulating cache failure...")
    await simulator.disconnect("app", "cache")
    
    # System should fallback to database
    assert await test_read_operations()
    assert await test_write_operations()
    
    # Scenario 3: Network degradation
    print("Simulating network degradation...")
    await simulator.degrade_network(
        latency_increase_ms=100,
        packet_loss_increase=0.05
    )
    
    # System should remain functional but slower
    assert await test_system_functionality()

asyncio.run(simulate_network_failures())
```

### Byzantine Failure Testing

```python
async def test_byzantine_failures():
    """Test Byzantine failure scenarios"""
    
    simulator = NetworkSimulator()
    
    # Create cluster
    nodes = [f"node{i}" for i in range(5)]
    for node in nodes:
        await simulator.add_node(node)
    
    # Create Byzantine failure: node sends different data to different peers
    byzantine_node = "node2"
    
    async def byzantine_behavior(sender, receiver, message):
        if receiver == "node0":
            return {"value": 100}
        else:
            return {"value": 200}
    
    await simulator.set_custom_behavior(byzantine_node, byzantine_behavior)
    
    # Test consensus
    consensus_value = await run_consensus_algorithm(nodes)
    
    # System should detect and handle Byzantine node
    assert consensus_value in [100, 200]
    assert byzantine_node in await get_suspected_nodes()
```

### Connection Recovery Testing

```python
async def test_connection_recovery():
    """Test connection recovery mechanisms"""
    
    simulator = NetworkSimulator()
    
    # Setup connection with monitoring
    await simulator.connect("client", "server")
    await simulator.monitor_connection("client", "server")
    
    # Simulate connection failures
    for _ in range(5):
        # Break connection
        await simulator.disconnect("client", "server")
        
        # Measure recovery time
        start = time.time()
        while not await simulator.is_connected("client", "server"):
            await asyncio.sleep(0.1)
        recovery_time = time.time() - start
        
        print(f"Recovery time: {recovery_time:.2f}s")
        
        # Should recover within SLA
        assert recovery_time < 5.0
        
        # Wait before next failure
        await asyncio.sleep(10)
```

## Command-Line Usage

```bash
# Basic network simulation
python -m tests.chaos.network_simulator --duration 3600

# Specific failure scenario
python -m tests.chaos.network_simulator \
    --scenario split-brain \
    --duration 1800 \
    --nodes 5

# Network degradation test
python -m tests.chaos.network_simulator \
    --latency 100 \
    --packet-loss 0.05 \
    --jitter 50 \
    --duration 3600

# Custom topology
python -m tests.chaos.network_simulator \
    --topology topology.yaml \
    --scenario random \
    --failure-rate 0.1
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --duration | 3600 | Simulation duration in seconds |
| --scenario | random | Failure scenario to simulate |
| --nodes | 3 | Number of nodes in topology |
| --latency | 0 | Base latency to add (ms) |
| --packet-loss | 0 | Packet loss rate (0-1) |
| --jitter | 0 | Latency variation (ms) |
| --topology | auto | Topology file or 'auto' |
| --output | network_sim.json | Output file for results |

## Monitoring and Metrics

### Network Metrics

```python
# Get current network state
state = simulator.get_network_state()
print(f"Active nodes: {state['active_nodes']}")
print(f"Partitions: {state['partitions']}")
print(f"Average latency: {state['avg_latency_ms']}ms")
print(f"Packet loss rate: {state['packet_loss_rate']:.2%}")

# Get connection statistics
stats = simulator.get_connection_stats("client", "server")
print(f"Packets sent: {stats['packets_sent']}")
print(f"Packets received: {stats['packets_received']}")
print(f"Packets lost: {stats['packets_lost']}")
print(f"Average RTT: {stats['avg_rtt_ms']}ms")
```

### Partition Detection

```python
# Detect network partitions
partitions = simulator.detect_partitions()
if partitions:
    print(f"Network partitioned into {len(partitions)} groups")
    for i, group in enumerate(partitions):
        print(f"Group {i}: {group}")
```

### Recovery Monitoring

```python
# Monitor partition healing
async def monitor_healing():
    partition_id = await simulator.create_partition(["A"], ["B"])
    
    healing_start = None
    while True:
        if await simulator.is_partitioned():
            print("Network still partitioned")
        else:
            if healing_start is None:
                healing_start = time.time()
                print("Partition healing started")
            elif time.time() - healing_start > 1:
                print("Partition fully healed")
                break
        
        await asyncio.sleep(0.5)
```

## Best Practices

### 1. Realistic Failure Patterns

```python
# Use realistic failure patterns
async def realistic_failures():
    # Correlated failures (e.g., rack failure)
    await simulator.fail_rack(["node1", "node2", "node3"])
    
    # Gradual degradation
    for latency in range(10, 200, 10):
        await simulator.add_latency_all(latency)
        await asyncio.sleep(60)
    
    # Intermittent issues
    await simulator.add_flapping_connection(
        "client", "server",
        up_duration=60,
        down_duration=5
    )
```

### 2. State Validation

```python
# Always validate state after network events
async def validate_after_partition():
    # Create partition
    await simulator.create_partition(["master"], ["replica"])
    
    # Make changes during partition
    await write_to_master(data)
    
    # Heal partition
    await simulator.heal_all_partitions()
    
    # Validate consistency
    master_state = await get_state("master")
    replica_state = await get_state("replica")
    
    # Should eventually converge
    await wait_for_convergence(master_state, replica_state)
```

### 3. Progressive Testing

```python
# Start simple, increase complexity
test_scenarios = [
    # Single node failure
    lambda: simulator.isolate_node("node1"),
    
    # Two-way partition
    lambda: simulator.create_partition(["A"], ["B"]),
    
    # Three-way partition
    lambda: simulator.create_partition(["A"], ["B"], ["C"]),
    
    # Complex asymmetric failure
    lambda: simulator.create_complex_failure_scenario()
]

for scenario in test_scenarios:
    await scenario()
    if not await system_remains_functional():
        break
```

## Troubleshooting

### Detecting Network Issues

```python
# Diagnose network problems
async def diagnose_network():
    diagnostics = await simulator.run_diagnostics()
    
    # Check for partitions
    if diagnostics['has_partitions']:
        print(f"Partitioned nodes: {diagnostics['partitioned_nodes']}")
    
    # Check for high latency links
    for link in diagnostics['high_latency_links']:
        print(f"High latency: {link['from']} -> {link['to']}: {link['latency_ms']}ms")
    
    # Check for lossy links
    for link in diagnostics['lossy_links']:
        print(f"Packet loss: {link['from']} -> {link['to']}: {link['loss_rate']:.2%}")
```

### Recovery Verification

```python
# Verify recovery after failures
async def verify_recovery():
    # Check all nodes reachable
    nodes = simulator.get_all_nodes()
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                assert await simulator.is_reachable(node1, node2)
    
    # Check no residual latency
    for link in simulator.get_all_links():
        latency = await simulator.get_latency(link)
        assert latency == simulator.default_latency_ms
    
    # Check no packet loss
    for link in simulator.get_all_links():
        loss = await simulator.get_packet_loss(link)
        assert loss == 0
```

## Integration Examples

### Docker Network Simulation

```yaml
version: '3.8'

services:
  network-simulator:
    image: genesis/network-simulator
    cap_add:
      - NET_ADMIN  # Required for network manipulation
    environment:
      SIMULATION_MODE: "production"
    command: >
      --topology /config/topology.yaml
      --scenario split-brain
      --duration 3600
    volumes:
      - ./config:/config
      - ./results:/results
```

### Kubernetes Network Policies

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: network-simulation
data:
  simulation.yaml: |
    scenarios:
      - name: pod-isolation
        isolate: ["trading-engine-0"]
        duration: 60
      - name: service-partition
        partition:
          - ["trading-*"]
          - ["risk-*"]
        duration: 120
---
apiVersion: batch/v1
kind: Job
metadata:
  name: network-chaos
spec:
  template:
    spec:
      containers:
      - name: simulator
        image: genesis/network-simulator
        args:
          - --config=/config/simulation.yaml
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: network-simulation
```

### CI/CD Integration

```yaml
name: Network Resilience Test
on: [push, pull_request]

jobs:
  network-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Run network simulation
        run: |
          python -m tests.chaos.network_simulator \
            --scenario all \
            --duration 1800 \
            --failure-rate 0.1
      
      - name: Validate results
        run: |
          python -c "
          import json
          with open('network_sim.json') as f:
              results = json.load(f)
          assert results['recovery_success_rate'] > 0.95
          assert results['data_consistency'] == True
          "
```

## Related Documentation

- [Chaos Engineering](chaos_engineering.md)
- [Load Generator](load_generator.md)
- [Continuous Operation](continuous_operation.md)
- [Disaster Recovery](../operations/disaster_recovery.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*