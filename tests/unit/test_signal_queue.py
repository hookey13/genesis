"""Unit tests for signal queue management system."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from genesis.core.models import Signal, SignalType
from genesis.core.exceptions import ValidationError
from genesis.engine.signal_queue import (
    SignalQueue, QueuedSignal, SignalStatus, ConflictResolution
)
from genesis.engine.event_bus import EventBus, Event, Priority as EventPriority


class TestQueuedSignal:
    """Test QueuedSignal class."""
    
    def test_priority_score_calculation(self):
        """Test priority score calculation for heap ordering."""
        signal = Signal(
            signal_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        queued = QueuedSignal(signal=signal, priority_score=0)
        
        # Higher priority and confidence should result in lower score (for min-heap)
        expected_score = -(80 + 0.8 * 20)  # -(80 + 16) = -96
        assert queued.priority_score == expected_score
    
    def test_is_expired_with_expiry_time(self):
        """Test expiry check with explicit expiry time."""
        signal = Signal(
            signal_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            expiry_time=datetime.utcnow() - timedelta(minutes=1),  # Already expired
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        queued = QueuedSignal(signal=signal, priority_score=0)
        assert queued.is_expired is True
    
    def test_is_expired_default(self):
        """Test default expiry behavior."""
        signal = Signal(
            signal_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        # Create signal queued 10 minutes ago
        queued = QueuedSignal(signal=signal, priority_score=0)
        queued.queued_at = datetime.utcnow() - timedelta(minutes=10)
        
        assert queued.is_expired is True  # Default expiry is 5 minutes


class TestSignalQueue:
    """Test SignalQueue class."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.save_queued_signal = AsyncMock()
        repo.update_signal_status = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def queue(self, mock_repository, mock_event_bus):
        """Create SignalQueue instance."""
        return SignalQueue(
            repository=mock_repository,
            event_bus=mock_event_bus,
            conflict_resolution=ConflictResolution.HIGHEST_PRIORITY
        )
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        return Signal(
            signal_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.85"),
            priority=75,
            size_recommendation=Decimal("0.5"),
            price_target=Decimal("52000"),
            stop_loss=Decimal("48000"),
            strategy_name="mean_reversion",
            timestamp=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_add_signal(self, queue, sample_signal, mock_repository, mock_event_bus):
        """Test adding a signal to the queue."""
        await queue.add_signal(sample_signal)
        
        assert len(queue._queue) == 1
        assert queue._queue[0].signal == sample_signal
        assert queue._stats["total_queued"] == 1
        
        # Check persistence and event
        mock_repository.save_queued_signal.assert_called_once()
        mock_event_bus.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_signal_with_priority_override(self, queue, sample_signal):
        """Test adding signal with priority override."""
        await queue.add_signal(sample_signal, priority=95)
        
        assert queue._queue[0].signal.priority == 95
    
    @pytest.mark.asyncio
    async def test_add_signal_duplicate(self, queue, sample_signal):
        """Test that duplicate signals are rejected."""
        await queue.add_signal(sample_signal)
        queue._processed_signals.add(sample_signal.signal_id)
        
        # Try adding again
        await queue.add_signal(sample_signal)
        
        # Should still have only one signal
        assert len(queue._queue) == 1
    
    @pytest.mark.asyncio
    async def test_get_next_signal_priority_order(self, queue):
        """Test that signals are returned in priority order."""
        # Create signals with different priorities
        signal1 = Signal(
            signal_id="1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.5"),
            priority=50,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal2 = Signal(
            signal_id="2",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal3 = Signal(
            signal_id="3",
            symbol="SOL/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.9"),
            priority=90,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        # Add in random order
        await queue.add_signal(signal1)
        await queue.add_signal(signal3)
        await queue.add_signal(signal2)
        
        # Should get highest priority first
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "3"  # Highest priority (90)
        
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "2"  # Next priority (80)
        
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "1"  # Lowest priority (50)
    
    @pytest.mark.asyncio
    async def test_get_next_signal_empty_queue(self, queue):
        """Test getting signal from empty queue."""
        signal = await queue.get_next_signal()
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_get_next_signal_expired(self, queue, sample_signal):
        """Test that expired signals are skipped."""
        # Add expired signal
        sample_signal.expiry_time = datetime.utcnow() - timedelta(minutes=1)
        await queue.add_signal(sample_signal)
        
        # Should return None and mark as expired
        signal = await queue.get_next_signal()
        assert signal is None
        assert queue._stats["total_expired"] == 1
    
    @pytest.mark.asyncio
    async def test_conflict_detection_opposite_signals(self, queue):
        """Test conflict detection for opposite signals."""
        # Create conflicting signals
        buy_signal = Signal(
            signal_id="buy",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        sell_signal = Signal(
            signal_id="sell",
            symbol="BTC/USDT",
            signal_type=SignalType.SELL,
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(buy_signal)
        await queue.add_signal(sell_signal)
        
        # Should detect conflict
        assert queue._stats["total_conflicts"] > 0
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_highest_priority(self, queue):
        """Test highest priority conflict resolution."""
        queue.conflict_resolution = ConflictResolution.HIGHEST_PRIORITY
        
        # Create conflicting signals
        signal1 = Signal(
            signal_id="1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal2 = Signal(
            signal_id="2",
            symbol="BTC/USDT",
            signal_type=SignalType.SELL,
            confidence_score=Decimal("0.8"),
            priority=90,  # Higher priority
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(signal1)
        await queue.add_signal(signal2)
        
        # Higher priority signal should win
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "2"
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_highest_confidence(self, queue):
        """Test highest confidence conflict resolution."""
        queue.conflict_resolution = ConflictResolution.HIGHEST_CONFIDENCE
        
        # Create conflicting signals
        signal1 = Signal(
            signal_id="1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.9"),  # Higher confidence
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal2 = Signal(
            signal_id="2",
            symbol="BTC/USDT",
            signal_type=SignalType.SELL,
            confidence_score=Decimal("0.7"),
            priority=90,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(signal1)
        await queue.add_signal(signal2)
        
        # Higher confidence signal should win
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "1"
    
    @pytest.mark.asyncio
    async def test_get_pending_signals_by_symbol(self, queue):
        """Test getting pending signals filtered by symbol."""
        # Add signals for different symbols
        signal1 = Signal(
            signal_id="1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal2 = Signal(
            signal_id="2",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(signal1)
        await queue.add_signal(signal2)
        
        btc_signals = await queue.get_pending_signals("BTC/USDT")
        assert len(btc_signals) == 1
        assert btc_signals[0].signal_id == "1"
        
        eth_signals = await queue.get_pending_signals("ETH/USDT")
        assert len(eth_signals) == 1
        assert eth_signals[0].signal_id == "2"
    
    @pytest.mark.asyncio
    async def test_cancel_signal(self, queue, sample_signal):
        """Test cancelling a signal."""
        await queue.add_signal(sample_signal)
        assert len(queue._queue) == 1
        
        # Cancel the signal
        cancelled = await queue.cancel_signal(sample_signal.signal_id)
        assert cancelled is True
        assert len(queue._queue) == 0
        assert queue._stats["total_rejected"] == 1
    
    @pytest.mark.asyncio
    async def test_cancel_signal_not_found(self, queue):
        """Test cancelling non-existent signal."""
        cancelled = await queue.cancel_signal("nonexistent")
        assert cancelled is False
    
    @pytest.mark.asyncio
    async def test_clear_expired(self, queue):
        """Test clearing expired signals."""
        # Add mixed signals
        fresh_signal = Signal(
            signal_id="fresh",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            expiry_time=datetime.utcnow() + timedelta(minutes=10),
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        expired_signal = Signal(
            signal_id="expired",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            expiry_time=datetime.utcnow() - timedelta(minutes=1),
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(fresh_signal)
        await queue.add_signal(expired_signal)
        
        assert len(queue._queue) == 2
        
        # Clear expired
        expired_count = await queue.clear_expired()
        assert expired_count == 1
        assert len(queue._queue) == 1
        assert queue._queue[0].signal.signal_id == "fresh"
    
    @pytest.mark.asyncio
    async def test_queue_size_limit(self, queue):
        """Test queue size limit enforcement."""
        queue.MAX_QUEUE_SIZE = 5
        
        # Add signals up to limit
        for i in range(5):
            signal = Signal(
                signal_id=str(i),
                symbol="BTC/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.8"),
                priority=80,
                strategy_name="test",
                timestamp=datetime.utcnow()
            )
            await queue.add_signal(signal)
        
        assert len(queue._queue) == 5
        
        # Try to add one more
        extra_signal = Signal(
            signal_id="extra",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        with pytest.raises(ValidationError, match="Signal queue full"):
            await queue.add_signal(extra_signal)
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, queue):
        """Test getting queue statistics."""
        # Add some signals
        signal1 = Signal(
            signal_id="1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        signal2 = Signal(
            signal_id="2",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )
        
        await queue.add_signal(signal1)
        await queue.add_signal(signal2)
        
        stats = await queue.get_queue_stats()
        
        assert stats["queue_size"] == 2
        assert stats["by_symbol"]["BTC/USDT"] == 1
        assert stats["by_symbol"]["ETH/USDT"] == 1
        assert stats["total_queued"] == 2
        assert stats["oldest_signal_age"] is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, queue):
        """Test thread safety with concurrent operations."""
        async def add_signals():
            for i in range(10):
                signal = Signal(
                    signal_id=f"add_{i}",
                    symbol=f"PAIR{i % 3}/USDT",
                    signal_type=SignalType.BUY,
                    confidence_score=Decimal("0.8"),
                    priority=50 + i,
                    strategy_name="test",
                    timestamp=datetime.utcnow()
                )
                await queue.add_signal(signal)
                await asyncio.sleep(0.01)
        
        async def process_signals():
            processed = []
            for _ in range(5):
                signal = await queue.get_next_signal()
                if signal:
                    processed.append(signal)
                await asyncio.sleep(0.015)
            return processed
        
        # Run concurrently
        results = await asyncio.gather(
            add_signals(),
            process_signals(),
            process_signals()
        )
        
        # Should have processed signals without errors
        processed_count = sum(len(r) for r in results[1:] if r)
        assert processed_count > 0
        assert processed_count <= 10