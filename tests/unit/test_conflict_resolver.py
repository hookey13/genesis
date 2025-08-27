"""Unit tests for Conflict Resolver"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from genesis.engine.conflict_resolver import (
    ConflictResolver,
    ResolutionMethod,
    SignalConflict,
    ConflictType
)
from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create mock event bus"""
    event_bus = Mock(spec=EventBus)
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def resolver(mock_event_bus):
    """Create conflict resolver instance"""
    return ConflictResolver(mock_event_bus)


@pytest.fixture
def sample_signals():
    """Create sample trading signals"""
    return [
        {
            "strategy_id": "momentum",
            "symbol": "BTC/USDT",
            "action": "buy",
            "quantity": Decimal("1"),
            "price": Decimal("50000"),
            "priority": 2,
            "confidence": Decimal("0.8"),
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "strategy_id": "mean_reversion",
            "symbol": "BTC/USDT",
            "action": "sell",
            "quantity": Decimal("0.5"),
            "price": Decimal("50100"),
            "priority": 3,
            "confidence": Decimal("0.7"),
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "strategy_id": "arbitrage",
            "symbol": "ETH/USDT",
            "action": "buy",
            "quantity": Decimal("10"),
            "price": Decimal("3000"),
            "priority": 1,
            "confidence": Decimal("0.9"),
            "timestamp": datetime.now(timezone.utc)
        }
    ]


class TestConflictResolver:
    """Test conflict resolution functionality"""
    
    def test_initialization(self, resolver):
        """Test resolver initialization"""
        assert resolver.resolution_method == ResolutionMethod.PRIORITY
        assert resolver.conflict_history == []
        assert resolver.resolution_rules == {}
        assert resolver.veto_strategies == set()
    
    def test_set_resolution_method(self, resolver):
        """Test setting resolution method"""
        resolver.set_resolution_method(ResolutionMethod.VOTING)
        assert resolver.resolution_method == ResolutionMethod.VOTING
        
        resolver.set_resolution_method(ResolutionMethod.CONFIDENCE)
        assert resolver.resolution_method == ResolutionMethod.CONFIDENCE
    
    def test_add_resolution_rule(self, resolver):
        """Test adding custom resolution rules"""
        def custom_rule(signals):
            return signals[0] if signals else None
        
        resolver.add_resolution_rule("custom", custom_rule)
        assert "custom" in resolver.resolution_rules
        assert resolver.resolution_rules["custom"] == custom_rule
    
    def test_add_veto_strategy(self, resolver):
        """Test adding veto strategies"""
        resolver.add_veto_strategy("risk_manager")
        assert "risk_manager" in resolver.veto_strategies
        
        resolver.add_veto_strategy("compliance")
        assert len(resolver.veto_strategies) == 2
    
    def test_remove_veto_strategy(self, resolver):
        """Test removing veto strategies"""
        resolver.add_veto_strategy("risk_manager")
        resolver.remove_veto_strategy("risk_manager")
        assert "risk_manager" not in resolver.veto_strategies
    
    def test_detect_conflicts_same_symbol_opposite_action(self, resolver, sample_signals):
        """Test detecting conflicts for same symbol, opposite actions"""
        conflicts = resolver._detect_conflicts(sample_signals[:2])
        
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.OPPOSING_DIRECTION
        assert conflict.symbol == "BTC/USDT"
        assert len(conflict.conflicting_signals) == 2
    
    def test_detect_conflicts_no_conflict(self, resolver, sample_signals):
        """Test no conflicts detected for different symbols"""
        # Only signals for different symbols
        signals = [sample_signals[0], sample_signals[2]]
        conflicts = resolver._detect_conflicts(signals)
        
        assert len(conflicts) == 0
    
    def test_detect_conflicts_same_direction(self, resolver):
        """Test detecting conflicts for same direction but different params"""
        signals = [
            {
                "strategy_id": "momentum",
                "symbol": "BTC/USDT",
                "action": "buy",
                "quantity": Decimal("1"),
                "price": Decimal("50000")
            },
            {
                "strategy_id": "trend_following",
                "symbol": "BTC/USDT",
                "action": "buy",
                "quantity": Decimal("2"),
                "price": Decimal("49900")
            }
        ]
        
        conflicts = resolver._detect_conflicts(signals)
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.QUANTITY_MISMATCH
    
    @pytest.mark.asyncio
    async def test_resolve_by_priority(self, resolver, sample_signals):
        """Test resolution by priority (higher wins)"""
        resolver.set_resolution_method(ResolutionMethod.PRIORITY)
        
        resolved = await resolver.resolve(sample_signals[:2])
        
        # Higher priority (3) should win
        assert len(resolved) == 1
        assert resolved[0]["strategy_id"] == "mean_reversion"
        assert resolved[0]["priority"] == 3
    
    @pytest.mark.asyncio
    async def test_resolve_by_confidence(self, resolver, sample_signals):
        """Test resolution by confidence (higher wins)"""
        resolver.set_resolution_method(ResolutionMethod.CONFIDENCE)
        
        resolved = await resolver.resolve(sample_signals[:2])
        
        # Higher confidence (0.8) should win
        assert len(resolved) == 1
        assert resolved[0]["strategy_id"] == "momentum"
        assert resolved[0]["confidence"] == Decimal("0.8")
    
    @pytest.mark.asyncio
    async def test_resolve_by_voting(self, resolver):
        """Test resolution by voting (majority wins)"""
        resolver.set_resolution_method(ResolutionMethod.VOTING)
        
        signals = [
            {"strategy_id": "s1", "symbol": "BTC/USDT", "action": "buy", "votes": 2},
            {"strategy_id": "s2", "symbol": "BTC/USDT", "action": "sell", "votes": 1},
            {"strategy_id": "s3", "symbol": "BTC/USDT", "action": "buy", "votes": 2}
        ]
        
        resolved = await resolver.resolve(signals)
        
        # Buy action has more votes (4 vs 1)
        assert len(resolved) > 0
        assert resolved[0]["action"] == "buy"
    
    @pytest.mark.asyncio
    async def test_resolve_by_consensus(self, resolver):
        """Test resolution by consensus (all must agree)"""
        resolver.set_resolution_method(ResolutionMethod.CONSENSUS)
        
        # All agree on buy
        signals = [
            {"strategy_id": "s1", "symbol": "BTC/USDT", "action": "buy"},
            {"strategy_id": "s2", "symbol": "BTC/USDT", "action": "buy"},
            {"strategy_id": "s3", "symbol": "BTC/USDT", "action": "buy"}
        ]
        
        resolved = await resolver.resolve(signals)
        assert len(resolved) == 3  # All signals pass
        
        # Disagreement
        signals[1]["action"] = "sell"
        resolved = await resolver.resolve(signals)
        assert len(resolved) == 0  # No consensus, no signals pass
    
    @pytest.mark.asyncio
    async def test_resolve_with_veto(self, resolver, sample_signals):
        """Test resolution with veto strategies"""
        resolver.add_veto_strategy("risk_manager")
        
        # Add veto signal
        veto_signal = {
            "strategy_id": "risk_manager",
            "symbol": "BTC/USDT",
            "action": "veto",
            "reason": "Risk limit exceeded"
        }
        
        signals = sample_signals[:2] + [veto_signal]
        resolved = await resolver.resolve(signals)
        
        # Veto should block BTC/USDT signals
        assert len(resolved) == 0
    
    @pytest.mark.asyncio
    async def test_resolve_with_custom_rule(self, resolver, sample_signals):
        """Test resolution with custom rule"""
        def largest_quantity_wins(conflicting_signals):
            """Custom rule: largest quantity wins"""
            return max(conflicting_signals, key=lambda s: s.get("quantity", 0))
        
        resolver.add_resolution_rule("largest_quantity", largest_quantity_wins)
        resolver.resolution_method = ResolutionMethod.CUSTOM
        resolver.custom_rule_name = "largest_quantity"
        
        resolved = await resolver.resolve(sample_signals[:2])
        
        # momentum has quantity=1, mean_reversion has quantity=0.5
        assert len(resolved) == 1
        assert resolved[0]["strategy_id"] == "momentum"
        assert resolved[0]["quantity"] == Decimal("1")
    
    @pytest.mark.asyncio
    async def test_resolve_multiple_conflicts(self, resolver):
        """Test resolving multiple independent conflicts"""
        signals = [
            {"strategy_id": "s1", "symbol": "BTC/USDT", "action": "buy", "priority": 1},
            {"strategy_id": "s2", "symbol": "BTC/USDT", "action": "sell", "priority": 2},
            {"strategy_id": "s3", "symbol": "ETH/USDT", "action": "buy", "priority": 1},
            {"strategy_id": "s4", "symbol": "ETH/USDT", "action": "sell", "priority": 3}
        ]
        
        resolver.set_resolution_method(ResolutionMethod.PRIORITY)
        resolved = await resolver.resolve(signals)
        
        # Should resolve both conflicts independently
        assert len(resolved) == 2
        btc_signal = next(s for s in resolved if s["symbol"] == "BTC/USDT")
        eth_signal = next(s for s in resolved if s["symbol"] == "ETH/USDT")
        
        assert btc_signal["strategy_id"] == "s2"  # Higher priority
        assert eth_signal["strategy_id"] == "s4"  # Higher priority
    
    @pytest.mark.asyncio
    async def test_conflict_event_published(self, resolver, mock_event_bus, sample_signals):
        """Test that conflict events are published"""
        await resolver.resolve(sample_signals[:2])
        
        # Should publish conflict event
        mock_event_bus.publish.assert_called()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.STRATEGY_CONFLICT
        assert "conflict_type" in event.data
        assert "resolution_method" in event.data
    
    def test_conflict_history_tracking(self, resolver, sample_signals):
        """Test conflict history is tracked"""
        resolver._detect_conflicts(sample_signals[:2])
        
        # History should be updated
        assert len(resolver.conflict_history) > 0
        
        # Test history limit
        for _ in range(1005):
            resolver._detect_conflicts(sample_signals[:2])
        
        assert len(resolver.conflict_history) <= 1000
    
    def test_get_conflict_statistics(self, resolver):
        """Test conflict statistics calculation"""
        # Add some conflict history
        for _ in range(10):
            resolver.conflict_history.append(SignalConflict(
                conflict_type=ConflictType.OPPOSITE_DIRECTION,
                symbol="BTC/USDT",
                conflicting_signals=[],
                timestamp=datetime.now(timezone.utc)
            ))
        
        for _ in range(5):
            resolver.conflict_history.append(SignalConflict(
                conflict_type=ConflictType.QUANTITY_MISMATCH,
                symbol="ETH/USDT",
                conflicting_signals=[],
                timestamp=datetime.now(timezone.utc)
            ))
        
        stats = resolver.get_conflict_statistics()
        
        assert stats["total_conflicts"] == 15
        assert stats["by_type"][ConflictType.OPPOSITE_DIRECTION] == 10
        assert stats["by_type"][ConflictType.QUANTITY_MISMATCH] == 5
        assert "BTC/USDT" in stats["by_symbol"]
        assert stats["by_symbol"]["BTC/USDT"] == 10
    
    @pytest.mark.asyncio
    async def test_resolve_empty_signals(self, resolver):
        """Test resolving empty signal list"""
        resolved = await resolver.resolve([])
        assert resolved == []
    
    @pytest.mark.asyncio
    async def test_resolve_single_signal(self, resolver, sample_signals):
        """Test resolving single signal (no conflict)"""
        resolved = await resolver.resolve([sample_signals[0]])
        assert len(resolved) == 1
        assert resolved[0] == sample_signals[0]
    
    def test_conflict_type_detection(self, resolver):
        """Test different conflict type detection"""
        # Opposite direction
        signals = [
            {"symbol": "BTC/USDT", "action": "buy"},
            {"symbol": "BTC/USDT", "action": "sell"}
        ]
        conflicts = resolver._detect_conflicts(signals)
        assert conflicts[0].conflict_type == ConflictType.OPPOSITE_DIRECTION
        
        # Quantity mismatch
        signals = [
            {"symbol": "BTC/USDT", "action": "buy", "quantity": Decimal("1")},
            {"symbol": "BTC/USDT", "action": "buy", "quantity": Decimal("2")}
        ]
        conflicts = resolver._detect_conflicts(signals)
        assert conflicts[0].conflict_type == ConflictType.QUANTITY_MISMATCH
        
        # Price divergence
        signals = [
            {"symbol": "BTC/USDT", "action": "buy", "price": Decimal("50000")},
            {"symbol": "BTC/USDT", "action": "buy", "price": Decimal("51000")}
        ]
        conflicts = resolver._detect_conflicts(signals)
        assert conflicts[0].conflict_type == ConflictType.PRICE_DIVERGENCE
    
    @pytest.mark.asyncio
    async def test_resolve_with_timestamp_priority(self, resolver):
        """Test resolution using timestamp (most recent wins)"""
        now = datetime.now(timezone.utc)
        signals = [
            {
                "strategy_id": "old",
                "symbol": "BTC/USDT",
                "action": "buy",
                "timestamp": now.replace(hour=now.hour - 1)
            },
            {
                "strategy_id": "new",
                "symbol": "BTC/USDT",
                "action": "sell",
                "timestamp": now
            }
        ]
        
        resolver.set_resolution_method(ResolutionMethod.TIMESTAMP)
        resolved = await resolver.resolve(signals)
        
        assert len(resolved) == 1
        assert resolved[0]["strategy_id"] == "new"
    
    def test_serialize_resolver_state(self, resolver):
        """Test serialization of resolver state"""
        resolver.set_resolution_method(ResolutionMethod.CONFIDENCE)
        resolver.add_veto_strategy("risk_manager")
        
        # Add some history
        resolver.conflict_history.append(SignalConflict(
            conflict_type=ConflictType.OPPOSITE_DIRECTION,
            symbol="BTC/USDT",
            conflicting_signals=[],
            timestamp=datetime.now(timezone.utc)
        ))
        
        state = resolver.to_dict()
        
        assert state["resolution_method"] == "confidence"
        assert "risk_manager" in state["veto_strategies"]
        assert len(state["conflict_history"]) == 1
    
    def test_load_resolver_state(self, resolver):
        """Test loading resolver state from dict"""
        state = {
            "resolution_method": "voting",
            "veto_strategies": ["risk_manager", "compliance"],
            "conflict_history": []
        }
        
        resolver.from_dict(state)
        
        assert resolver.resolution_method == ResolutionMethod.VOTING
        assert "risk_manager" in resolver.veto_strategies
        assert "compliance" in resolver.veto_strategies