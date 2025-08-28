"""
Month-end reconciliation system for Project GENESIS.

Provides automated balance verification, position reconciliation,
and discrepancy alerts for institutional-grade accounting accuracy.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog

from genesis.core.constants import TradingTier
from genesis.core.models import Account, Position
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import BinanceGateway
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class ReconciliationEngine:
    """Automated reconciliation system for trading accounts."""

    def __init__(
        self,
        repository: Repository,
        event_bus: EventBus,
        gateway: BinanceGateway,
    ):
        """Initialize ReconciliationEngine."""
        self.repository = repository
        self.event_bus = event_bus
        self.gateway = gateway
        self._reconciliation_history: List[Dict[str, Any]] = []
        logger.info("reconciliation_engine_initialized")

    @requires_tier(TradingTier.STRATEGIST)
    async def perform_balance_reconciliation(
        self, account_id: str
    ) -> Dict[str, Any]:
        """
        Verify account balance against exchange.
        
        Args:
            account_id: Account to reconcile
            
        Returns:
            Reconciliation result with discrepancies
        """
        try:
            # Get database balance
            account = await self.repository.get_account(account_id)
            if not account:
                raise ValueError(f"Account {account_id} not found")
            
            db_balance = account.balance_usdt
            
            # Get exchange balance
            exchange_info = await self.gateway.get_account_info()
            exchange_balance = Decimal("0")
            
            for balance in exchange_info.get("balances", []):
                if balance["asset"] == "USDT":
                    exchange_balance = Decimal(str(balance["free"]))
                    break
            
            # Calculate discrepancy
            discrepancy = db_balance - exchange_balance
            is_reconciled = abs(discrepancy) < Decimal("0.01")  # 1 cent tolerance
            
            result = {
                "reconciliation_id": str(uuid4()),
                "account_id": account_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "balance",
                "database_balance": str(db_balance),
                "exchange_balance": str(exchange_balance),
                "discrepancy": str(discrepancy),
                "is_reconciled": is_reconciled,
                "action_required": not is_reconciled,
            }
            
            # Store reconciliation result
            self._reconciliation_history.append(result)
            await self.repository.save_reconciliation_result(result)
            
            # If discrepancy found, publish alert
            if not is_reconciled:
                await self.event_bus.publish("reconciliation.discrepancy_found", {
                    "account_id": account_id,
                    "type": "balance",
                    "discrepancy": str(discrepancy),
                    "database_value": str(db_balance),
                    "exchange_value": str(exchange_balance),
                })
                
                logger.warning(
                    "balance_discrepancy_found",
                    account_id=account_id,
                    discrepancy=str(discrepancy),
                )
            else:
                logger.info(
                    "balance_reconciled",
                    account_id=account_id,
                    balance=str(db_balance),
                )
            
            return result
            
        except Exception as e:
            logger.error("balance_reconciliation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def perform_position_reconciliation(
        self, account_id: str
    ) -> Dict[str, Any]:
        """
        Reconcile positions with exchange.
        
        Args:
            account_id: Account to reconcile
            
        Returns:
            Reconciliation result with position discrepancies
        """
        try:
            # Get database positions
            db_positions = await self.repository.get_positions_by_account(account_id)
            
            # Get exchange positions
            exchange_positions = await self.gateway.get_open_positions()
            
            # Create position maps for comparison
            db_position_map = {p.symbol: p for p in db_positions}
            exchange_position_map = {}
            
            for ex_pos in exchange_positions:
                symbol = ex_pos["symbol"]
                quantity = Decimal(str(ex_pos["positionAmt"]))
                if quantity != 0:
                    exchange_position_map[symbol] = {
                        "symbol": symbol,
                        "quantity": abs(quantity),
                        "side": "LONG" if quantity > 0 else "SHORT",
                        "notional": Decimal(str(ex_pos.get("notional", 0))),
                    }
            
            # Find discrepancies
            discrepancies = []
            
            # Check positions in DB but not on exchange
            for symbol, db_pos in db_position_map.items():
                if symbol not in exchange_position_map:
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "missing_on_exchange",
                        "database_quantity": str(db_pos.quantity),
                        "exchange_quantity": "0",
                    })
            
            # Check positions on exchange but not in DB
            for symbol, ex_pos in exchange_position_map.items():
                if symbol not in db_position_map:
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "missing_in_database",
                        "database_quantity": "0",
                        "exchange_quantity": str(ex_pos["quantity"]),
                    })
                else:
                    # Check quantity mismatch
                    db_pos = db_position_map[symbol]
                    quantity_diff = abs(db_pos.quantity - ex_pos["quantity"])
                    
                    if quantity_diff > Decimal("0.00001"):  # Small tolerance
                        discrepancies.append({
                            "symbol": symbol,
                            "type": "quantity_mismatch",
                            "database_quantity": str(db_pos.quantity),
                            "exchange_quantity": str(ex_pos["quantity"]),
                            "difference": str(quantity_diff),
                        })
            
            is_reconciled = len(discrepancies) == 0
            
            result = {
                "reconciliation_id": str(uuid4()),
                "account_id": account_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "positions",
                "database_position_count": len(db_positions),
                "exchange_position_count": len(exchange_position_map),
                "discrepancies": discrepancies,
                "is_reconciled": is_reconciled,
                "action_required": not is_reconciled,
            }
            
            # Store result
            self._reconciliation_history.append(result)
            await self.repository.save_reconciliation_result(result)
            
            # Alert if discrepancies
            if discrepancies:
                await self.event_bus.publish("reconciliation.position_discrepancies", {
                    "account_id": account_id,
                    "discrepancy_count": len(discrepancies),
                    "discrepancies": discrepancies,
                })
                
                logger.warning(
                    "position_discrepancies_found",
                    account_id=account_id,
                    discrepancy_count=len(discrepancies),
                )
            else:
                logger.info(
                    "positions_reconciled",
                    account_id=account_id,
                    position_count=len(db_positions),
                )
            
            return result
            
        except Exception as e:
            logger.error("position_reconciliation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def generate_reconciliation_report(
        self, account_id: str, period_end: datetime = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive reconciliation report.
        
        Args:
            account_id: Account to report on
            period_end: End of period (defaults to now)
            
        Returns:
            Reconciliation report
        """
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        
        period_start = period_end.replace(day=1, hour=0, minute=0, second=0)
        
        # Perform reconciliations
        balance_result = await self.perform_balance_reconciliation(account_id)
        position_result = await self.perform_position_reconciliation(account_id)
        
        # Get historical reconciliations for period
        historical = [
            r for r in self._reconciliation_history
            if r["account_id"] == account_id
            and datetime.fromisoformat(r["timestamp"]) >= period_start
            and datetime.fromisoformat(r["timestamp"]) <= period_end
        ]
        
        # Calculate statistics
        total_reconciliations = len(historical)
        successful_reconciliations = sum(1 for r in historical if r["is_reconciled"])
        success_rate = (
            successful_reconciliations / total_reconciliations 
            if total_reconciliations > 0 else 0
        )
        
        report = {
            "report_id": str(uuid4()),
            "account_id": account_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_status": {
                "balance": balance_result,
                "positions": position_result,
            },
            "period_statistics": {
                "total_reconciliations": total_reconciliations,
                "successful": successful_reconciliations,
                "failed": total_reconciliations - successful_reconciliations,
                "success_rate": success_rate,
            },
            "requires_action": (
                not balance_result["is_reconciled"] or 
                not position_result["is_reconciled"]
            ),
        }
        
        # Store report
        await self.repository.save_reconciliation_report(report)
        
        # Publish report generated event
        await self.event_bus.publish("reconciliation.report_generated", {
            "account_id": account_id,
            "report_id": report["report_id"],
            "period_end": period_end.isoformat(),
            "requires_action": report["requires_action"],
        })
        
        logger.info(
            "reconciliation_report_generated",
            account_id=account_id,
            success_rate=success_rate,
            requires_action=report["requires_action"],
        )
        
        return report

    @requires_tier(TradingTier.STRATEGIST)
    async def schedule_monthly_reconciliation(self, account_id: str):
        """
        Schedule automated monthly reconciliation.
        
        Args:
            account_id: Account to schedule reconciliation for
        """
        # This would be handled by a scheduler in production
        # For now, just perform immediate reconciliation
        report = await self.generate_reconciliation_report(account_id)
        
        logger.info(
            "monthly_reconciliation_scheduled",
            account_id=account_id,
            next_run=datetime.now(timezone.utc).replace(
                month=datetime.now(timezone.utc).month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0
            ).isoformat(),
        )
        
        return report

    @requires_tier(TradingTier.STRATEGIST)
    async def resolve_discrepancy(
        self,
        account_id: str,
        reconciliation_id: str,
        resolution_action: str,
    ) -> bool:
        """
        Resolve a reconciliation discrepancy.
        
        Args:
            account_id: Account with discrepancy
            reconciliation_id: ID of reconciliation with discrepancy
            resolution_action: Action to resolve (use_exchange, use_database, manual)
            
        Returns:
            True if resolved successfully
        """
        try:
            # Find the reconciliation
            reconciliation = None
            for r in self._reconciliation_history:
                if r["reconciliation_id"] == reconciliation_id:
                    reconciliation = r
                    break
            
            if not reconciliation:
                raise ValueError(f"Reconciliation {reconciliation_id} not found")
            
            if reconciliation["type"] == "balance":
                if resolution_action == "use_exchange":
                    # Update database to match exchange
                    account = await self.repository.get_account(account_id)
                    account.balance_usdt = Decimal(reconciliation["exchange_balance"])
                    await self.repository.update_account(account)
                    
                elif resolution_action == "use_database":
                    # Would need exchange API to update exchange balance
                    # This is typically not possible, so log for manual action
                    logger.warning(
                        "manual_exchange_update_required",
                        account_id=account_id,
                        target_balance=reconciliation["database_balance"],
                    )
                
                elif resolution_action == "manual":
                    # Mark for manual resolution
                    reconciliation["manual_resolution_required"] = True
            
            # Mark as resolved
            reconciliation["resolved"] = True
            reconciliation["resolution_action"] = resolution_action
            reconciliation["resolved_at"] = datetime.now(timezone.utc).isoformat()
            
            # Publish resolution event
            await self.event_bus.publish("reconciliation.discrepancy_resolved", {
                "account_id": account_id,
                "reconciliation_id": reconciliation_id,
                "resolution_action": resolution_action,
            })
            
            logger.info(
                "discrepancy_resolved",
                account_id=account_id,
                reconciliation_id=reconciliation_id,
                action=resolution_action,
            )
            
            return True
            
        except Exception as e:
            logger.error("discrepancy_resolution_failed", error=str(e))
            return False