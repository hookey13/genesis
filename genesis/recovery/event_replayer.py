"""Event replayer for reconstructing state from event stream."""

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class EventReplayer:
    """Replays events to reconstruct application state."""
    
    # Event handlers mapping
    EVENT_HANDLERS = {
        "POSITION_OPENED": "_handle_position_opened",
        "POSITION_UPDATED": "_handle_position_updated",
        "POSITION_CLOSED": "_handle_position_closed",
        "ORDER_PLACED": "_handle_order_placed",
        "ORDER_FILLED": "_handle_order_filled",
        "ORDER_CANCELLED": "_handle_order_cancelled",
        "BALANCE_UPDATED": "_handle_balance_updated",
        "TIER_CHANGED": "_handle_tier_changed",
        "RISK_LIMIT_SET": "_handle_risk_limit_set",
        "CHECKPOINT": "_handle_checkpoint"
    }
    
    def replay_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Replay a single event.
        
        Args:
            conn: Database connection
            event_type: Type of event
            aggregate_id: Aggregate ID
            event_data: Event data dictionary
        """
        handler_name = self.EVENT_HANDLERS.get(event_type)
        
        if not handler_name:
            logger.warning(f"No handler for event type: {event_type}")
            return
        
        handler = getattr(self, handler_name, None)
        if handler:
            handler(conn, aggregate_id, event_data)
        else:
            logger.error(f"Handler method not found: {handler_name}")
    
    def replay_events(
        self,
        conn: sqlite3.Connection,
        events: List[sqlite3.Row],
        target_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Replay multiple events sequentially.
        
        Args:
            conn: Database connection
            events: List of event rows
            target_state: Optional target state for validation
            
        Returns:
            Reconstructed state
        """
        state = {
            "positions": {},
            "orders": {},
            "balances": {},
            "tier": "sniper",
            "risk_limits": {},
            "checkpoints": []
        }
        
        for event in events:
            event_data = json.loads(event["event_data"])
            self.replay_event(
                conn,
                event["event_type"],
                event["aggregate_id"],
                event_data
            )
            
            # Update state tracking
            if event["event_type"].startswith("POSITION_"):
                self._update_position_state(state, event["aggregate_id"], event_data)
            elif event["event_type"].startswith("ORDER_"):
                self._update_order_state(state, event["aggregate_id"], event_data)
            elif event["event_type"] == "BALANCE_UPDATED":
                state["balances"][event_data.get("asset", "USDT")] = event_data.get("balance")
            elif event["event_type"] == "TIER_CHANGED":
                state["tier"] = event_data.get("new_tier")
        
        # Validate against target state if provided
        if target_state:
            validation_errors = self._validate_state(state, target_state)
            if validation_errors:
                logger.warning("State validation errors", errors=validation_errors)
        
        return state
    
    def _handle_position_opened(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle position opened event."""
        conn.execute("""
            INSERT OR REPLACE INTO positions (
                position_id, symbol, side, entry_price, quantity,
                status, opened_at, strategy_id
            ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
        """, (
            aggregate_id,
            event_data.get("symbol"),
            event_data.get("side"),
            event_data.get("entry_price"),
            event_data.get("quantity"),
            event_data.get("opened_at", datetime.utcnow().isoformat()),
            event_data.get("strategy_id")
        ))
    
    def _handle_position_updated(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle position updated event."""
        updates = []
        params = []
        
        for field in ["quantity", "unrealized_pnl", "realized_pnl", "current_price"]:
            if field in event_data:
                updates.append(f"{field} = ?")
                params.append(event_data[field])
        
        if updates:
            params.append(aggregate_id)
            conn.execute(
                f"UPDATE positions SET {', '.join(updates)} WHERE position_id = ?",
                params
            )
    
    def _handle_position_closed(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle position closed event."""
        conn.execute("""
            UPDATE positions 
            SET status = 'closed',
                exit_price = ?,
                realized_pnl = ?,
                closed_at = ?
            WHERE position_id = ?
        """, (
            event_data.get("exit_price"),
            event_data.get("realized_pnl"),
            event_data.get("closed_at", datetime.utcnow().isoformat()),
            aggregate_id
        ))
    
    def _handle_order_placed(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle order placed event."""
        conn.execute("""
            INSERT OR REPLACE INTO orders (
                order_id, client_order_id, symbol, side, order_type,
                quantity, price, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?)
        """, (
            aggregate_id,
            event_data.get("client_order_id"),
            event_data.get("symbol"),
            event_data.get("side"),
            event_data.get("order_type"),
            event_data.get("quantity"),
            event_data.get("price"),
            event_data.get("created_at", datetime.utcnow().isoformat())
        ))
    
    def _handle_order_filled(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle order filled event."""
        conn.execute("""
            UPDATE orders
            SET status = ?,
                filled_quantity = ?,
                executed_price = ?,
                filled_at = ?
            WHERE order_id = ?
        """, (
            "filled" if event_data.get("is_fully_filled") else "partially_filled",
            event_data.get("filled_quantity"),
            event_data.get("executed_price"),
            event_data.get("filled_at", datetime.utcnow().isoformat()),
            aggregate_id
        ))
    
    def _handle_order_cancelled(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle order cancelled event."""
        conn.execute("""
            UPDATE orders
            SET status = 'cancelled',
                cancelled_at = ?
            WHERE order_id = ?
        """, (
            event_data.get("cancelled_at", datetime.utcnow().isoformat()),
            aggregate_id
        ))
    
    def _handle_balance_updated(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle balance updated event."""
        conn.execute("""
            INSERT OR REPLACE INTO balances (
                asset, free_balance, locked_balance, total_balance,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            event_data.get("asset", "USDT"),
            event_data.get("free_balance"),
            event_data.get("locked_balance"),
            event_data.get("total_balance"),
            event_data.get("updated_at", datetime.utcnow().isoformat())
        ))
    
    def _handle_tier_changed(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle tier changed event."""
        conn.execute("""
            INSERT INTO tier_history (
                old_tier, new_tier, reason, changed_at
            ) VALUES (?, ?, ?, ?)
        """, (
            event_data.get("old_tier"),
            event_data.get("new_tier"),
            event_data.get("reason"),
            event_data.get("changed_at", datetime.utcnow().isoformat())
        ))
        
        # Update current tier
        conn.execute("""
            INSERT OR REPLACE INTO system_state (
                key, value, updated_at
            ) VALUES ('current_tier', ?, ?)
        """, (
            event_data.get("new_tier"),
            datetime.utcnow().isoformat()
        ))
    
    def _handle_risk_limit_set(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle risk limit set event."""
        conn.execute("""
            INSERT OR REPLACE INTO risk_limits (
                limit_type, limit_value, tier, updated_at
            ) VALUES (?, ?, ?, ?)
        """, (
            event_data.get("limit_type"),
            event_data.get("limit_value"),
            event_data.get("tier"),
            event_data.get("updated_at", datetime.utcnow().isoformat())
        ))
    
    def _handle_checkpoint(
        self,
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle checkpoint event."""
        # Checkpoint events mark consistent state points
        conn.execute("""
            INSERT INTO checkpoints (
                checkpoint_id, state_hash, created_at
            ) VALUES (?, ?, ?)
        """, (
            aggregate_id,
            event_data.get("state_hash"),
            event_data.get("created_at", datetime.utcnow().isoformat())
        ))
    
    def _update_position_state(
        self,
        state: Dict[str, Any],
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Update position state tracking."""
        if aggregate_id not in state["positions"]:
            state["positions"][aggregate_id] = {}
        
        state["positions"][aggregate_id].update(event_data)
    
    def _update_order_state(
        self,
        state: Dict[str, Any],
        aggregate_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Update order state tracking."""
        if aggregate_id not in state["orders"]:
            state["orders"][aggregate_id] = {}
        
        state["orders"][aggregate_id].update(event_data)
    
    def _validate_state(
        self,
        reconstructed_state: Dict[str, Any],
        target_state: Dict[str, Any]
    ) -> List[str]:
        """Validate reconstructed state against target.
        
        Args:
            reconstructed_state: Reconstructed state
            target_state: Expected target state
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate positions
        for pos_id, target_pos in target_state.get("positions", {}).items():
            if pos_id not in reconstructed_state["positions"]:
                errors.append(f"Missing position: {pos_id}")
            else:
                recon_pos = reconstructed_state["positions"][pos_id]
                for field in ["quantity", "entry_price", "status"]:
                    if field in target_pos and target_pos[field] != recon_pos.get(field):
                        errors.append(
                            f"Position {pos_id} field {field} mismatch: "
                            f"expected {target_pos[field]}, got {recon_pos.get(field)}"
                        )
        
        # Validate balances
        for asset, target_balance in target_state.get("balances", {}).items():
            if asset not in reconstructed_state["balances"]:
                errors.append(f"Missing balance for {asset}")
            elif abs(Decimal(str(target_balance)) - Decimal(str(reconstructed_state["balances"][asset]))) > Decimal("0.01"):
                errors.append(
                    f"Balance mismatch for {asset}: "
                    f"expected {target_balance}, got {reconstructed_state['balances'][asset]}"
                )
        
        return errors