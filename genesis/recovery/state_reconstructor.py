"""State reconstructor for validating recovered database state."""

import json
import sqlite3
from decimal import Decimal
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


class StateReconstructor:
    """Reconstructs and validates application state from database."""
    
    def validate_state(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Validate database state consistency.
        
        Args:
            conn: Database connection
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check positions consistency
        position_validation = self._validate_positions(conn)
        errors.extend(position_validation["errors"])
        warnings.extend(position_validation["warnings"])
        
        # Check orders consistency
        order_validation = self._validate_orders(conn)
        errors.extend(order_validation["errors"])
        warnings.extend(order_validation["warnings"])
        
        # Check balances consistency
        balance_validation = self._validate_balances(conn)
        errors.extend(balance_validation["errors"])
        warnings.extend(balance_validation["warnings"])
        
        # Check event stream integrity
        event_validation = self._validate_event_stream(conn)
        errors.extend(event_validation["errors"])
        warnings.extend(event_validation["warnings"])
        
        return {
            "is_consistent": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validations_performed": [
                "positions",
                "orders",
                "balances",
                "event_stream"
            ]
        }
    
    def reconstruct_state(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Reconstruct complete application state.
        
        Args:
            conn: Database connection
            
        Returns:
            Reconstructed state
        """
        state = {
            "positions": self._reconstruct_positions(conn),
            "orders": self._reconstruct_orders(conn),
            "balances": self._reconstruct_balances(conn),
            "tier": self._get_current_tier(conn),
            "risk_limits": self._reconstruct_risk_limits(conn),
            "system_state": self._reconstruct_system_state(conn)
        }
        
        # Calculate aggregate metrics
        state["metrics"] = self._calculate_metrics(state)
        
        return state
    
    def _validate_positions(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """Validate position consistency."""
        errors = []
        warnings = []
        
        cursor = conn.execute("""
            SELECT position_id, symbol, side, entry_price, quantity,
                   status, realized_pnl, unrealized_pnl
            FROM positions
            WHERE status = 'open'
        """)
        
        for row in cursor:
            # Check for negative quantities
            if row["quantity"] and Decimal(str(row["quantity"])) < 0:
                errors.append(f"Position {row['position_id']} has negative quantity")
            
            # Check for missing entry price
            if not row["entry_price"]:
                errors.append(f"Position {row['position_id']} missing entry price")
            
            # Check for invalid side
            if row["side"] not in ["buy", "sell", "long", "short"]:
                errors.append(f"Position {row['position_id']} has invalid side: {row['side']}")
            
            # Warn about large unrealized losses
            if row["unrealized_pnl"] and Decimal(str(row["unrealized_pnl"])) < -1000:
                warnings.append(
                    f"Position {row['position_id']} has large unrealized loss: {row['unrealized_pnl']}"
                )
        
        # Check for orphaned positions (no corresponding orders)
        cursor = conn.execute("""
            SELECT p.position_id
            FROM positions p
            LEFT JOIN orders o ON p.position_id = o.position_id
            WHERE p.status = 'open' AND o.order_id IS NULL
        """)
        
        orphaned = cursor.fetchall()
        if orphaned:
            for row in orphaned:
                warnings.append(f"Position {row[0]} has no corresponding orders")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_orders(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """Validate order consistency."""
        errors = []
        warnings = []
        
        cursor = conn.execute("""
            SELECT order_id, client_order_id, status, quantity, 
                   filled_quantity, price, executed_price
            FROM orders
            WHERE status IN ('new', 'partially_filled')
        """)
        
        for row in cursor:
            # Check for duplicate client order IDs
            dup_cursor = conn.execute("""
                SELECT COUNT(*) FROM orders 
                WHERE client_order_id = ? AND order_id != ?
            """, (row["client_order_id"], row["order_id"]))
            
            if dup_cursor.fetchone()[0] > 0:
                errors.append(f"Duplicate client_order_id: {row['client_order_id']}")
            
            # Check filled quantity consistency
            if row["filled_quantity"]:
                filled = Decimal(str(row["filled_quantity"]))
                total = Decimal(str(row["quantity"]))
                
                if filled > total:
                    errors.append(
                        f"Order {row['order_id']} filled quantity exceeds total quantity"
                    )
                
                if row["status"] == "new" and filled > 0:
                    warnings.append(
                        f"Order {row['order_id']} has status 'new' but has fills"
                    )
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_balances(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """Validate balance consistency."""
        errors = []
        warnings = []
        
        cursor = conn.execute("""
            SELECT asset, free_balance, locked_balance, total_balance
            FROM balances
        """)
        
        for row in cursor:
            free = Decimal(str(row["free_balance"]))
            locked = Decimal(str(row["locked_balance"]))
            total = Decimal(str(row["total_balance"]))
            
            # Check balance equation
            if abs((free + locked) - total) > Decimal("0.01"):
                errors.append(
                    f"Balance mismatch for {row['asset']}: "
                    f"free({free}) + locked({locked}) != total({total})"
                )
            
            # Check for negative balances
            if free < 0 or locked < 0 or total < 0:
                errors.append(f"Negative balance detected for {row['asset']}")
            
            # Warn about low balances
            if row["asset"] == "USDT" and total < 100:
                warnings.append(f"Low USDT balance: {total}")
        
        # Check position exposure vs balance
        cursor = conn.execute("""
            SELECT SUM(quantity * entry_price) as total_exposure
            FROM positions
            WHERE status = 'open' AND side IN ('buy', 'long')
        """)
        
        exposure = cursor.fetchone()["total_exposure"]
        if exposure:
            # Get USDT balance
            cursor = conn.execute("""
                SELECT total_balance FROM balances WHERE asset = 'USDT'
            """)
            usdt_row = cursor.fetchone()
            
            if usdt_row:
                usdt_balance = Decimal(str(usdt_row["total_balance"]))
                exposure_decimal = Decimal(str(exposure))
                
                if exposure_decimal > usdt_balance * 10:  # 10x leverage warning
                    warnings.append(
                        f"High leverage detected: exposure({exposure_decimal}) > 10x balance({usdt_balance})"
                    )
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_event_stream(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """Validate event stream integrity."""
        errors = []
        warnings = []
        
        # Check for sequence gaps
        cursor = conn.execute("""
            SELECT sequence_number, 
                   LAG(sequence_number) OVER (ORDER BY sequence_number) as prev_seq
            FROM events
        """)
        
        gaps = []
        for row in cursor:
            if row["prev_seq"] is not None:
                expected = row["prev_seq"] + 1
                if row["sequence_number"] != expected:
                    gaps.append((row["prev_seq"], row["sequence_number"]))
        
        if gaps:
            for prev, curr in gaps[:5]:  # Show first 5 gaps
                errors.append(f"Sequence gap: {prev} -> {curr}")
            
            if len(gaps) > 5:
                errors.append(f"... and {len(gaps) - 5} more sequence gaps")
        
        # Check for event ordering
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM (
                SELECT created_at,
                       LAG(created_at) OVER (ORDER BY sequence_number) as prev_time
                FROM events
            )
            WHERE prev_time > created_at
        """)
        
        out_of_order = cursor.fetchone()["count"]
        if out_of_order > 0:
            warnings.append(f"Found {out_of_order} events with out-of-order timestamps")
        
        return {"errors": errors, "warnings": warnings}
    
    def _reconstruct_positions(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Reconstruct position state."""
        positions = {}
        
        cursor = conn.execute("""
            SELECT * FROM positions WHERE status = 'open'
        """)
        
        for row in cursor:
            positions[row["position_id"]] = dict(row)
        
        return positions
    
    def _reconstruct_orders(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Reconstruct order state."""
        orders = {}
        
        cursor = conn.execute("""
            SELECT * FROM orders 
            WHERE status IN ('new', 'partially_filled')
        """)
        
        for row in cursor:
            orders[row["order_id"]] = dict(row)
        
        return orders
    
    def _reconstruct_balances(self, conn: sqlite3.Connection) -> Dict[str, Decimal]:
        """Reconstruct balance state."""
        balances = {}
        
        cursor = conn.execute("""
            SELECT asset, total_balance FROM balances
        """)
        
        for row in cursor:
            balances[row["asset"]] = Decimal(str(row["total_balance"]))
        
        return balances
    
    def _get_current_tier(self, conn: sqlite3.Connection) -> str:
        """Get current tier from system state."""
        cursor = conn.execute("""
            SELECT value FROM system_state WHERE key = 'current_tier'
        """)
        
        row = cursor.fetchone()
        return row["value"] if row else "sniper"
    
    def _reconstruct_risk_limits(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Reconstruct risk limits."""
        limits = {}
        
        cursor = conn.execute("""
            SELECT limit_type, limit_value, tier
            FROM risk_limits
            WHERE tier = (SELECT value FROM system_state WHERE key = 'current_tier')
        """)
        
        for row in cursor:
            limits[row["limit_type"]] = {
                "value": Decimal(str(row["limit_value"])),
                "tier": row["tier"]
            }
        
        return limits
    
    def _reconstruct_system_state(self, conn: sqlite3.Connection) -> Dict[str, str]:
        """Reconstruct system state."""
        state = {}
        
        cursor = conn.execute("SELECT key, value FROM system_state")
        
        for row in cursor:
            state[row["key"]] = row["value"]
        
        return state
    
    def _calculate_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics from state."""
        metrics = {
            "open_positions": len(state["positions"]),
            "pending_orders": len(state["orders"]),
            "total_exposure": Decimal("0"),
            "total_pnl": Decimal("0"),
            "asset_count": len(state["balances"])
        }
        
        # Calculate total exposure and PnL
        for position in state["positions"].values():
            if position.get("quantity") and position.get("entry_price"):
                exposure = Decimal(str(position["quantity"])) * Decimal(str(position["entry_price"]))
                metrics["total_exposure"] += exposure
            
            if position.get("unrealized_pnl"):
                metrics["total_pnl"] += Decimal(str(position["unrealized_pnl"]))
            
            if position.get("realized_pnl"):
                metrics["total_pnl"] += Decimal(str(position["realized_pnl"]))
        
        return metrics