"""
Unit tests for network partition simulator.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from tests.chaos.network_simulator import NetworkSimulator


class TestNetworkSimulator:
    """Tests for NetworkSimulator class."""
    
    @pytest.mark.asyncio
    async def test_simulator_initialization(self):
        """Test network simulator initialization."""
        simulator = NetworkSimulator()
        
        assert simulator is not None
        assert hasattr(simulator, 'partitions')
        assert hasattr(simulator, 'nodes')
        assert hasattr(simulator, 'latency_map')
    
    @pytest.mark.asyncio
    async def test_create_partition(self):
        """Test creating a network partition."""
        simulator = NetworkSimulator()
        
        # Create a partition between nodes
        partition_id = await simulator.create_partition(['node1', 'node2'], ['node3', 'node4'])
        
        assert partition_id is not None
        assert partition_id in simulator.partitions
    
    @pytest.mark.asyncio
    async def test_heal_partition(self):
        """Test healing a network partition."""
        simulator = NetworkSimulator()
        
        # Create and then heal a partition
        partition_id = await simulator.create_partition(['node1'], ['node2'])
        assert partition_id in simulator.partitions
        
        await simulator.heal_partition(partition_id)
        assert partition_id not in simulator.partitions
    
    @pytest.mark.asyncio
    async def test_add_latency(self):
        """Test adding network latency."""
        simulator = NetworkSimulator()
        
        # Add latency between nodes
        await simulator.add_latency('node1', 'node2', 100)
        
        # Check latency was added
        key = ('node1', 'node2')
        assert key in simulator.latency_map
        assert simulator.latency_map[key] == 100
    
    @pytest.mark.asyncio
    async def test_remove_latency(self):
        """Test removing network latency."""
        simulator = NetworkSimulator()
        
        # Add and then remove latency
        await simulator.add_latency('node1', 'node2', 100)
        await simulator.remove_latency('node1', 'node2')
        
        key = ('node1', 'node2')
        assert key not in simulator.latency_map
    
    @pytest.mark.asyncio
    async def test_simulate_packet_loss(self):
        """Test simulating packet loss."""
        simulator = NetworkSimulator()
        
        # Set packet loss rate
        await simulator.set_packet_loss('node1', 'node2', 0.1)
        
        # Simulate sending packets
        successes = 0
        failures = 0
        
        for _ in range(100):
            result = await simulator.send_packet('node1', 'node2', {'data': 'test'})
            if result:
                successes += 1
            else:
                failures += 1
        
        # With 10% loss rate, we should have some failures
        assert failures > 0
        assert successes > failures  # Most should succeed
    
    @pytest.mark.asyncio
    async def test_is_reachable(self):
        """Test checking if nodes are reachable."""
        simulator = NetworkSimulator()
        
        # Initially all nodes should be reachable
        assert await simulator.is_reachable('node1', 'node2') is True
        
        # Create partition
        partition_id = await simulator.create_partition(['node1'], ['node2'])
        
        # Now they shouldn't be reachable
        assert await simulator.is_reachable('node1', 'node2') is False
        
        # Heal partition
        await simulator.heal_partition(partition_id)
        
        # Should be reachable again
        assert await simulator.is_reachable('node1', 'node2') is True
    
    @pytest.mark.asyncio
    async def test_split_brain_scenario(self):
        """Test split-brain network scenario."""
        simulator = NetworkSimulator()
        
        # Create split-brain: two groups that can't see each other
        nodes_group1 = ['node1', 'node2']
        nodes_group2 = ['node3', 'node4']
        
        partition_id = await simulator.create_partition(nodes_group1, nodes_group2)
        
        # Within group communication should work
        assert await simulator.is_reachable('node1', 'node2') is True
        assert await simulator.is_reachable('node3', 'node4') is True
        
        # Cross-group communication should fail
        assert await simulator.is_reachable('node1', 'node3') is False
        assert await simulator.is_reachable('node2', 'node4') is False
    
    @pytest.mark.asyncio
    async def test_asymmetric_partition(self):
        """Test asymmetric network partition."""
        simulator = NetworkSimulator()
        
        # Create asymmetric partition: node1 can reach node2, but not vice versa
        await simulator.create_asymmetric_partition('node1', 'node2')
        
        # node1 -> node2 should work
        assert await simulator.is_reachable('node1', 'node2') is True
        
        # node2 -> node1 should fail
        assert await simulator.is_reachable('node2', 'node1') is False
    
    @pytest.mark.asyncio
    async def test_network_recovery(self):
        """Test network recovery after failures."""
        simulator = NetworkSimulator()
        
        # Create multiple network issues
        partition1 = await simulator.create_partition(['node1'], ['node2'])
        await simulator.add_latency('node3', 'node4', 200)
        await simulator.set_packet_loss('node5', 'node6', 0.5)
        
        # Clear all network issues
        await simulator.clear_all_issues()
        
        # Everything should be back to normal
        assert await simulator.is_reachable('node1', 'node2') is True
        assert ('node3', 'node4') not in simulator.latency_map
        assert simulator.get_packet_loss('node5', 'node6') == 0
    
    @pytest.mark.asyncio
    async def test_get_network_state(self):
        """Test getting current network state."""
        simulator = NetworkSimulator()
        
        # Set up some network conditions
        await simulator.create_partition(['node1'], ['node2'])
        await simulator.add_latency('node3', 'node4', 100)
        
        state = simulator.get_network_state()
        
        assert 'partitions' in state
        assert 'latencies' in state
        assert len(state['partitions']) > 0
        assert len(state['latencies']) > 0


@pytest.mark.asyncio
async def test_full_network_simulation():
    """Test a complete network simulation scenario."""
    simulator = NetworkSimulator()
    
    # Simulate a complex network scenario
    # 1. Start with healthy network
    assert await simulator.is_reachable('server', 'client') is True
    
    # 2. Add some latency (degraded performance)
    await simulator.add_latency('server', 'client', 50)
    
    # 3. Increase packet loss (connection issues)
    await simulator.set_packet_loss('server', 'client', 0.05)
    
    # 4. Create a partition (network split)
    partition = await simulator.create_partition(['server'], ['client'])
    assert await simulator.is_reachable('server', 'client') is False
    
    # 5. Heal the partition (recovery)
    await simulator.heal_partition(partition)
    assert await simulator.is_reachable('server', 'client') is True
    
    # 6. Clear all issues (full recovery)
    await simulator.clear_all_issues()
    
    # Verify network is healthy
    state = simulator.get_network_state()
    assert len(state.get('partitions', [])) == 0
    assert len(state.get('latencies', {})) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])