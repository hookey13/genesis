"""Integration tests for network partition tolerance."""

import pytest
from tests.chaos.network_simulator import NetworkSimulator, test_split_brain_scenario, test_connection_recovery


@pytest.mark.asyncio
async def test_network_simulator():
    """Test network simulator functionality."""
    simulator = NetworkSimulator()
    
    # Test partition
    assert not simulator.partitioned
    await simulator.create_partition(duration_seconds=1)
    assert not simulator.partitioned  # Should be healed
    
    # Test packet loss
    await simulator.simulate_packet_loss(0.5, duration_seconds=1)
    assert simulator.packet_loss_rate == 0.0  # Should be reset
    
    # Test latency
    await simulator.add_latency(100, duration_seconds=1)
    assert simulator.latency_ms == 0  # Should be reset


@pytest.mark.asyncio
async def test_split_brain_handling():
    """Test split-brain scenario handling."""
    result = await test_split_brain_scenario()
    assert result is True


@pytest.mark.asyncio
async def test_network_recovery():
    """Test connection recovery after network issues."""
    result = await test_connection_recovery()
    assert result is True