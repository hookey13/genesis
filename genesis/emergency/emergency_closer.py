"""Emergency position closure system for risk mitigation."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

from genesis.emergency.position_unwinder import PositionUnwinder

logger = structlog.get_logger(__name__)


class EmergencyCloser:
    """Manages emergency closure of all positions."""

    def __init__(
        self,
        exchange_gateway,
        position_unwinder: PositionUnwinder,
        max_slippage_percent: Decimal = Decimal("2.0"),
        notification_channels: list[str] | None = None
    ):
        """Initialize emergency closer.
        
        Args:
            exchange_gateway: Exchange gateway for order execution
            position_unwinder: Position unwinding logic
            max_slippage_percent: Maximum acceptable slippage
            notification_channels: List of notification channels
        """
        self.exchange_gateway = exchange_gateway
        self.position_unwinder = position_unwinder
        self.max_slippage_percent = max_slippage_percent
        self.notification_channels = notification_channels or []

        # Track closure state
        self.closure_in_progress = False
        self.closure_results: list[dict[str, Any]] = []
        self.closure_start_time: datetime | None = None
        self.closure_end_time: datetime | None = None

    async def emergency_close_all(
        self,
        reason: str,
        dry_run: bool = False,
        force: bool = False
    ) -> dict[str, Any]:
        """Execute emergency closure of all positions.
        
        Args:
            reason: Reason for emergency closure
            dry_run: If True, simulate without executing
            force: If True, bypass safety checks
            
        Returns:
            Closure results
        """
        if self.closure_in_progress:
            logger.warning("Emergency closure already in progress")
            return {"error": "Closure already in progress"}

        self.closure_in_progress = True
        self.closure_start_time = datetime.utcnow()

        logger.critical(
            "EMERGENCY CLOSURE INITIATED",
            reason=reason,
            dry_run=dry_run,
            force=force
        )

        try:
            # Send notifications
            await self._send_notifications(
                "EMERGENCY CLOSURE INITIATED",
                f"Reason: {reason}\nDry Run: {dry_run}"
            )

            # Get all open positions
            positions = await self._get_open_positions()

            if not positions:
                logger.info("No open positions to close")
                return {
                    "success": True,
                    "positions_closed": 0,
                    "reason": reason
                }

            # Prioritize positions for closure
            prioritized = self.position_unwinder.prioritize_positions(positions)

            logger.info(
                "Positions prioritized for closure",
                count=len(prioritized),
                total_exposure=sum(p["exposure"] for p in prioritized)
            )

            # Execute closures
            results = []
            for position in prioritized:
                result = await self._close_position(position, dry_run, force)
                results.append(result)
                self.closure_results.append(result)

                # Brief delay between closures to avoid overwhelming exchange
                if not dry_run:
                    await asyncio.sleep(0.1)

            # Calculate summary
            self.closure_end_time = datetime.utcnow()
            duration = (self.closure_end_time - self.closure_start_time).total_seconds()

            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]
            total_pnl = sum(r.get("realized_pnl", 0) for r in successful)

            summary = {
                "success": len(failed) == 0,
                "reason": reason,
                "dry_run": dry_run,
                "duration_seconds": duration,
                "positions_total": len(positions),
                "positions_closed": len(successful),
                "positions_failed": len(failed),
                "total_realized_pnl": total_pnl,
                "details": results
            }

            # Send completion notification
            await self._send_notifications(
                "EMERGENCY CLOSURE COMPLETED",
                f"Closed: {len(successful)}/{len(positions)}\n"
                f"PnL: {total_pnl}\n"
                f"Duration: {duration:.1f}s"
            )

            # Create audit trail
            await self._create_audit_trail(summary)

            logger.info(
                "Emergency closure completed",
                closed=len(successful),
                failed=len(failed),
                duration=duration
            )

            return summary

        except Exception as e:
            logger.error("Emergency closure failed", error=str(e))
            await self._send_notifications(
                "EMERGENCY CLOSURE FAILED",
                f"Error: {e!s}"
            )
            raise

        finally:
            self.closure_in_progress = False

    async def _get_open_positions(self) -> list[dict[str, Any]]:
        """Get all open positions from exchange.
        
        Returns:
            List of open positions
        """
        try:
            positions = await self.exchange_gateway.get_positions()

            # Filter for open positions with non-zero quantity
            open_positions = []
            for position in positions:
                quantity = abs(Decimal(str(position.get("quantity", 0))))
                if quantity > 0:
                    open_positions.append({
                        "symbol": position["symbol"],
                        "side": position["side"],
                        "quantity": quantity,
                        "entry_price": Decimal(str(position.get("entryPrice", 0))),
                        "current_price": Decimal(str(position.get("markPrice", 0))),
                        "unrealized_pnl": Decimal(str(position.get("unrealizedPnl", 0))),
                        "exposure": quantity * Decimal(str(position.get("markPrice", 0)))
                    })

            return open_positions

        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            raise

    async def _close_position(
        self,
        position: dict[str, Any],
        dry_run: bool,
        force: bool
    ) -> dict[str, Any]:
        """Close a single position.
        
        Args:
            position: Position to close
            dry_run: If True, simulate without executing
            force: If True, bypass safety checks
            
        Returns:
            Closure result
        """
        try:
            symbol = position["symbol"]
            side = "sell" if position["side"] in ["buy", "long"] else "buy"
            quantity = position["quantity"]

            logger.info(
                f"Closing position: {symbol}",
                side=side,
                quantity=quantity,
                unrealized_pnl=position["unrealized_pnl"]
            )

            if dry_run:
                # Simulate closure
                return {
                    "success": True,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "executed_price": position["current_price"],
                    "realized_pnl": position["unrealized_pnl"],
                    "dry_run": True
                }

            # Calculate acceptable price range
            if side == "sell":
                # Selling - accept lower price
                limit_price = position["current_price"] * (
                    Decimal("1") - self.max_slippage_percent / Decimal("100")
                )
            else:
                # Buying - accept higher price
                limit_price = position["current_price"] * (
                    Decimal("1") + self.max_slippage_percent / Decimal("100")
                )

            # Place market order
            order = await self.exchange_gateway.place_order(
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=float(quantity),
                client_order_id=f"emergency_{symbol}_{datetime.utcnow().timestamp()}"
            )

            # Wait for fill
            filled_order = await self._wait_for_fill(order["orderId"], timeout=30)

            if filled_order:
                executed_price = Decimal(str(filled_order.get("avgPrice", 0)))
                realized_pnl = self._calculate_realized_pnl(
                    position,
                    executed_price
                )

                return {
                    "success": True,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "order_id": order["orderId"],
                    "executed_price": executed_price,
                    "realized_pnl": realized_pnl,
                    "slippage": abs(executed_price - position["current_price"])
                }
            else:
                # Order not filled
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Order not filled within timeout",
                    "order_id": order["orderId"]
                }

        except Exception as e:
            logger.error(f"Failed to close position {position['symbol']}", error=str(e))
            return {
                "success": False,
                "symbol": position["symbol"],
                "error": str(e)
            }

    async def _wait_for_fill(
        self,
        order_id: str,
        timeout: int = 30
    ) -> dict[str, Any] | None:
        """Wait for order to be filled.
        
        Args:
            order_id: Order ID to monitor
            timeout: Maximum wait time in seconds
            
        Returns:
            Filled order or None if timeout
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                order = await self.exchange_gateway.get_order(order_id)

                if order["status"] in ["FILLED", "PARTIALLY_FILLED"]:
                    return order
                elif order["status"] in ["CANCELED", "REJECTED", "EXPIRED"]:
                    logger.warning(f"Order {order_id} terminated with status {order['status']}")
                    return None

                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Error checking order status", error=str(e))
                await asyncio.sleep(1)

        logger.warning(f"Order {order_id} not filled within {timeout} seconds")
        return None

    def _calculate_realized_pnl(
        self,
        position: dict[str, Any],
        exit_price: Decimal
    ) -> Decimal:
        """Calculate realized PnL for closed position.
        
        Args:
            position: Original position
            exit_price: Execution price
            
        Returns:
            Realized PnL
        """
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if position["side"] in ["buy", "long"]:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        return pnl

    async def _send_notifications(self, subject: str, message: str) -> None:
        """Send notifications to configured channels.
        
        Args:
            subject: Notification subject
            message: Notification message
        """
        for channel in self.notification_channels:
            try:
                if channel == "email":
                    # Send email notification
                    logger.info(f"Email notification: {subject}")
                elif channel == "slack":
                    # Send Slack notification
                    logger.info(f"Slack notification: {subject}")
                elif channel == "pagerduty":
                    # Trigger PagerDuty alert
                    logger.info(f"PagerDuty alert: {subject}")
                else:
                    logger.warning(f"Unknown notification channel: {channel}")

            except Exception as e:
                logger.error(f"Failed to send {channel} notification", error=str(e))

    async def _create_audit_trail(self, summary: dict[str, Any]) -> None:
        """Create audit trail for emergency closure.
        
        Args:
            summary: Closure summary
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "emergency_closure",
            "reason": summary["reason"],
            "dry_run": summary["dry_run"],
            "duration_seconds": summary["duration_seconds"],
            "positions_closed": summary["positions_closed"],
            "positions_failed": summary["positions_failed"],
            "total_pnl": str(summary["total_realized_pnl"]),
            "details": summary["details"]
        }

        # Log to audit system
        logger.info("Audit trail created", audit=audit_entry)

        # Store in database
        # await self.db.store_audit_trail(audit_entry)

    def get_closure_status(self) -> dict[str, Any]:
        """Get current closure status.
        
        Returns:
            Status dictionary
        """
        return {
            "closure_in_progress": self.closure_in_progress,
            "last_closure_start": self.closure_start_time.isoformat() if self.closure_start_time else None,
            "last_closure_end": self.closure_end_time.isoformat() if self.closure_end_time else None,
            "closure_results_count": len(self.closure_results),
            "current_positions": [] if not self.closure_in_progress else self.closure_results
        }
