"""
Conflict resolution system for competing strategy signals.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ResolutionMethod(Enum):
    """Methods for resolving conflicts between signals."""
    PRIORITY = "priority"  # Use signal priority
    CONFIDENCE = "confidence"  # Use signal confidence
    VOTING = "voting"  # Multiple strategies vote
    VETO = "veto"  # Allow veto power
    CUSTOM = "custom"  # Custom resolution logic
    CONSENSUS = "consensus"  # All must agree


class ConflictType(Enum):
    """Types of conflicts between signals."""
    OPPOSING_DIRECTION = "opposing_direction"  # Buy vs Sell
    SAME_ASSET = "same_asset"  # Multiple signals for same asset
    CAPITAL_CONSTRAINT = "capital_constraint"  # Insufficient capital
    RISK_LIMIT = "risk_limit"  # Would exceed risk limits
    CORRELATION = "correlation"  # Would increase correlation
    QUANTITY_MISMATCH = "quantity_mismatch"  # Different quantities for same asset


@dataclass
class SignalConflict:
    """Represents a conflict between trading signals."""
    conflict_id: str
    conflict_type: ConflictType
    primary_signal: dict[str, Any]
    conflicting_signals: list[dict[str, Any]]
    timestamp: datetime
    resolved: bool = False
    resolution: dict[str, Any] | None = None
    resolution_method: ResolutionMethod | None = None


class ConflictResolver:
    """Resolve conflicts between competing strategy signals."""

    def __init__(self, event_bus=None):
        """Initialize conflict resolver.

        Args:
            event_bus: Event bus for publishing conflict events
        """
        self.event_bus = event_bus
        self.resolution_method = ResolutionMethod.PRIORITY
        self.conflict_history: list[SignalConflict] = []
        self.resolution_rules: dict[str, Any] = {}
        self.veto_strategies: set = set()
        self.custom_rule_name: str | None = None

    def set_resolution_method(self, method: ResolutionMethod) -> None:
        """Set the resolution method.

        Args:
            method: Resolution method to use
        """
        self.resolution_method = method
        logger.info("Resolution method changed", method=method.value)

    def add_resolution_rule(self, name: str, rule: Any) -> None:
        """Add a custom resolution rule.

        Args:
            name: Rule name
            rule: Rule function or configuration
        """
        self.resolution_rules[name] = rule
        logger.info("Resolution rule added", rule_name=name)

    def add_veto_strategy(self, strategy_id: str) -> None:
        """Add a strategy with veto power.

        Args:
            strategy_id: Strategy identifier
        """
        self.veto_strategies.add(strategy_id)
        logger.info("Veto strategy added", strategy_id=strategy_id)

    def remove_veto_strategy(self, strategy_id: str) -> None:
        """Remove a strategy's veto power.

        Args:
            strategy_id: Strategy identifier
        """
        self.veto_strategies.discard(strategy_id)
        logger.info("Veto strategy removed", strategy_id=strategy_id)

    def _detect_conflicts(self, signals: list[dict[str, Any]]) -> list[SignalConflict]:
        """Detect conflicts among signals.

        Args:
            signals: List of trading signals

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Group signals by symbol
        by_symbol = {}
        for signal in signals:
            symbol = signal.get("symbol")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(signal)

        # Check for conflicts within each symbol
        for symbol, symbol_signals in by_symbol.items():
            if len(symbol_signals) > 1:
                # Check for opposing directions
                actions = {s.get("action") for s in symbol_signals}
                if "buy" in actions and "sell" in actions:
                    conflict = SignalConflict(
                        conflict_id=f"conflict_{datetime.now().timestamp()}",
                        conflict_type=ConflictType.OPPOSING_DIRECTION,
                        primary_signal=symbol_signals[0],
                        conflicting_signals=symbol_signals,
                        timestamp=datetime.now(),
                        resolved=False
                    )
                    conflict.symbol = symbol  # Add symbol attribute
                    conflicts.append(conflict)
                    self.conflict_history.append(conflict)
                elif len(actions) == 1:  # Same direction
                    # Check for quantity mismatch
                    quantities = {s.get("quantity") for s in symbol_signals}
                    if len(quantities) > 1:
                        conflict = SignalConflict(
                            conflict_id=f"conflict_{datetime.now().timestamp()}",
                            conflict_type=ConflictType.QUANTITY_MISMATCH,
                            primary_signal=symbol_signals[0],
                            conflicting_signals=symbol_signals,
                            timestamp=datetime.now(),
                            resolved=False
                        )
                        conflict.symbol = symbol  # Add symbol attribute
                        conflicts.append(conflict)
                        self.conflict_history.append(conflict)

        # Limit history size
        if len(self.conflict_history) > 1000:
            self.conflict_history = self.conflict_history[-1000:]

        return conflicts

    async def resolve(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Resolve conflicts among signals.

        Args:
            signals: List of all signals to resolve

        Returns:
            List of resolved signals
        """
        if not signals:
            return []

        # Detect conflicts
        conflicts = self._detect_conflicts(signals)

        # If no conflicts, return all signals
        if not conflicts:
            return signals

        resolved_signals = []

        # Group signals by symbol for resolution
        by_symbol = {}
        for signal in signals:
            symbol = signal.get("symbol")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(signal)

        # Resolve conflicts for each symbol
        for symbol, symbol_signals in by_symbol.items():
            if len(symbol_signals) == 1:
                resolved_signals.append(symbol_signals[0])
                continue

            resolved = await self._resolve_symbol_conflicts(symbol_signals)
            if resolved:
                if isinstance(resolved, list):
                    resolved_signals.extend(resolved)
                else:
                    resolved_signals.append(resolved)

        # Publish conflict event if event bus available
        if self.event_bus and conflicts:
            from genesis.core.events import Event, EventType
            event = Event(
                event_type=EventType.STRATEGY_CONFLICT,
                event_data={
                    "conflict_type": conflicts[0].conflict_type.value,
                    "resolution_method": self.resolution_method.value,
                    "num_conflicts": len(conflicts)
                }
            )
            await self.event_bus.publish(event)

        return resolved_signals

    async def _resolve_symbol_conflicts(self, signals: list[dict[str, Any]]) -> Any | None:
        """
        Resolve conflicts for signals of the same symbol.

        Args:
            signals: List of conflicting signals for same symbol

        Returns:
            Resolved signal(s) or None if rejected
        """
        # Check for veto
        for signal in signals:
            if signal.get("strategy_id") in self.veto_strategies:
                if signal.get("action") == "veto":
                    logger.info(
                        "Signals vetoed",
                        vetoing_strategy=signal.get("strategy_id"),
                        reason=signal.get("reason", "No reason provided")
                    )
                    return None

        # Resolve based on method
        if self.resolution_method == ResolutionMethod.PRIORITY:
            # Sort by priority (higher number = higher priority)
            signals.sort(key=lambda s: -s.get("priority", 0))
            winning_signal = signals[0]

        elif self.resolution_method == ResolutionMethod.CONFIDENCE:
            # Sort by confidence (higher confidence wins)
            signals.sort(key=lambda s: -float(s.get("confidence", 0)))
            winning_signal = signals[0]

        elif self.resolution_method == ResolutionMethod.VOTING:
            # Count votes for each action
            vote_counts = {}
            for signal in signals:
                action = signal.get("action")
                votes = signal.get("votes", 1)
                if action not in vote_counts:
                    vote_counts[action] = []
                vote_counts[action].append((signal, votes))

            # Find action with most votes
            max_votes = 0
            winning_action = None
            for action, signal_votes in vote_counts.items():
                total_votes = sum(v for _, v in signal_votes)
                if total_votes > max_votes:
                    max_votes = total_votes
                    winning_action = action

            # Return first signal with winning action
            if winning_action:
                winning_signal = next(s for s in signals if s.get("action") == winning_action)
            else:
                winning_signal = signals[0]

        elif self.resolution_method == ResolutionMethod.CONSENSUS:
            # All must have same action
            actions = {s.get("action") for s in signals}
            if len(actions) == 1:
                # Consensus achieved, return all signals
                return signals
            else:
                # No consensus
                return []

        elif self.resolution_method == ResolutionMethod.CUSTOM:
            # Use custom rule
            if self.custom_rule_name and self.custom_rule_name in self.resolution_rules:
                rule = self.resolution_rules[self.custom_rule_name]
                winning_signal = rule(signals)
            else:
                winning_signal = signals[0]

        else:
            # Default to first signal
            winning_signal = signals[0]

        # Log resolution
        if winning_signal:
            logger.info(
                "Conflict resolved",
                resolution_method=self.resolution_method.value,
                winner=winning_signal.get("strategy_id") if isinstance(winning_signal, dict) else "multiple",
                num_conflicts=len(signals)
            )

        return winning_signal

    def get_conflict_statistics(self) -> dict[str, Any]:
        """Get statistics about conflict history.

        Returns:
            Dictionary with conflict statistics
        """
        if not self.conflict_history:
            return {
                "total_conflicts": 0,
                "by_type": {},
                "resolution_success_rate": 0
            }

        by_type = {}
        resolved_count = 0

        for conflict in self.conflict_history:
            # Count by type
            type_name = conflict.conflict_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count resolved
            if conflict.resolved:
                resolved_count += 1

        return {
            "total_conflicts": len(self.conflict_history),
            "by_type": by_type,
            "resolution_success_rate": resolved_count / len(self.conflict_history) if self.conflict_history else 0
        }
