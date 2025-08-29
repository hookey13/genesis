"""
Order reconciliation system for maintaining consistency between local and exchange state.

Periodically syncs order state with the exchange to detect and resolve discrepancies,
handling orphaned orders (local but not on exchange) and zombie orders (on exchange
but not local).
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Set

import structlog

from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderResponse
from genesis.core.events import Event, EventType, EventPriority
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class DiscrepancyType(str, Enum):
    """Types of order discrepancies."""
    
    ORPHANED = "orphaned"  # Local order not found on exchange
    ZOMBIE = "zombie"  # Exchange order not found locally
    STATUS_MISMATCH = "status_mismatch"  # Different status
    QUANTITY_MISMATCH = "quantity_mismatch"  # Different filled quantity
    PRICE_MISMATCH = "price_mismatch"  # Different price


@dataclass
class OrderDiscrepancy:
    """Represents a discrepancy between local and exchange order state."""
    
    order_id: str
    symbol: str
    discrepancy_type: DiscrepancyType
    local_state: Optional[Dict] = None
    exchange_state: Optional[Dict] = None
    detected_at: datetime = None
    resolved: bool = False
    resolution: Optional[str] = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()


class OrderReconciliation:
    """
    Order reconciliation system for maintaining state consistency.
    
    Performs periodic and on-demand synchronization between local order
    tracking and exchange state, detecting and resolving discrepancies.
    """
    
    def __init__(
        self,
        gateway: BinanceGateway,
        event_bus: Optional[EventBus] = None,
        sync_interval: int = 300,  # 5 minutes default
        on_reconnection: bool = True,
        on_demand: bool = True,
    ):
        """
        Initialize the reconciliation system.
        
        Args:
            gateway: Binance gateway for API access
            event_bus: Event bus for notifications
            sync_interval: Seconds between periodic syncs (default 300)
            on_reconnection: Trigger sync on reconnection
            on_demand: Allow manual sync triggers
        """
        self.gateway = gateway
        self.event_bus = event_bus
        self.sync_interval = sync_interval
        self.on_reconnection = on_reconnection
        self.on_demand = on_demand
        
        # Local order tracking (would come from database in production)
        self.local_orders: Dict[str, Dict] = {}
        
        # Discrepancy tracking
        self.discrepancies: List[OrderDiscrepancy] = []
        self.resolved_discrepancies: List[OrderDiscrepancy] = []
        
        # Sync state
        self.last_sync_time: Optional[datetime] = None
        self.sync_in_progress = False
        self.sync_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.total_syncs = 0
        self.total_discrepancies_found = 0
        self.total_discrepancies_resolved = 0
        self.sync_errors = 0
        
        logger.info(
            "OrderReconciliation initialized",
            sync_interval=sync_interval,
            on_reconnection=on_reconnection,
        )
    
    async def start(self) -> None:
        """Start the reconciliation system."""
        if self.running:
            logger.warning("Reconciliation already running")
            return
        
        self.running = True
        self.sync_task = asyncio.create_task(self._periodic_sync_loop())
        
        # Perform initial sync
        await self.sync_orders()
        
        logger.info("Order reconciliation started")
    
    async def stop(self) -> None:
        """Stop the reconciliation system."""
        self.running = False
        
        if self.sync_task and not self.sync_task.done():
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info(
            "Order reconciliation stopped",
            total_syncs=self.total_syncs,
            discrepancies_found=self.total_discrepancies_found,
            discrepancies_resolved=self.total_discrepancies_resolved,
        )
    
    async def _periodic_sync_loop(self) -> None:
        """Periodic synchronization loop."""
        while self.running:
            try:
                await asyncio.sleep(self.sync_interval)
                
                if self.running:
                    await self.sync_orders()
                    
            except Exception as e:
                logger.error("Error in periodic sync", error=str(e))
                self.sync_errors += 1
    
    async def sync_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Perform order synchronization.
        
        Args:
            symbol: Optional symbol to sync (None for all)
            
        Returns:
            Sync results dictionary
        """
        if self.sync_in_progress:
            logger.warning("Sync already in progress")
            return {"status": "already_in_progress"}
        
        self.sync_in_progress = True
        start_time = time.time()
        
        try:
            logger.info("Starting order sync", symbol=symbol)
            
            # Fetch exchange orders
            exchange_orders = await self._fetch_exchange_orders(symbol)
            
            # Fetch local orders
            local_orders = self._fetch_local_orders(symbol)
            
            # Detect discrepancies
            discrepancies = self._detect_discrepancies(local_orders, exchange_orders)
            
            # Resolve discrepancies
            resolved = await self._resolve_discrepancies(discrepancies)
            
            # Update statistics
            self.total_syncs += 1
            self.total_discrepancies_found += len(discrepancies)
            self.total_discrepancies_resolved += resolved
            self.last_sync_time = datetime.now()
            
            # Publish sync event
            if self.event_bus:
                await self._publish_sync_event(discrepancies, resolved)
            
            sync_time = time.time() - start_time
            
            logger.info(
                "Order sync completed",
                sync_time=f"{sync_time:.2f}s",
                local_orders=len(local_orders),
                exchange_orders=len(exchange_orders),
                discrepancies=len(discrepancies),
                resolved=resolved,
            )
            
            return {
                "status": "completed",
                "sync_time": sync_time,
                "local_orders": len(local_orders),
                "exchange_orders": len(exchange_orders),
                "discrepancies": len(discrepancies),
                "resolved": resolved,
            }
            
        except Exception as e:
            logger.error("Order sync failed", error=str(e))
            self.sync_errors += 1
            return {"status": "failed", "error": str(e)}
            
        finally:
            self.sync_in_progress = False
    
    async def _fetch_exchange_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch open orders from the exchange.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of exchange orders
        """
        try:
            orders = await self.gateway.get_open_orders(symbol)
            
            # Convert to dict format for comparison
            exchange_orders = []
            for order in orders:
                exchange_orders.append({
                    "order_id": order.order_id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "type": order.type,
                    "status": order.status,
                    "price": order.price,
                    "quantity": order.quantity,
                    "filled_quantity": order.filled_quantity,
                    "created_at": order.created_at,
                    "updated_at": order.updated_at,
                })
            
            return exchange_orders
            
        except Exception as e:
            logger.error("Failed to fetch exchange orders", error=str(e))
            raise
    
    def _fetch_local_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Fetch open orders from local tracking.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of local orders
        """
        # In production, this would query the database
        # For now, return orders from in-memory tracking
        local_orders = []
        
        for order_id, order in self.local_orders.items():
            # Filter by symbol if specified
            if symbol and order.get("symbol") != symbol:
                continue
            
            # Only include open orders
            if order.get("status") in ["open", "partially_filled", "pending"]:
                local_orders.append(order)
        
        return local_orders
    
    def _detect_discrepancies(
        self,
        local_orders: List[Dict],
        exchange_orders: List[Dict],
    ) -> List[OrderDiscrepancy]:
        """
        Detect discrepancies between local and exchange orders.
        
        Args:
            local_orders: Local order list
            exchange_orders: Exchange order list
            
        Returns:
            List of detected discrepancies
        """
        discrepancies = []
        
        # Create lookup maps
        local_by_id = {o["order_id"]: o for o in local_orders}
        exchange_by_id = {o["order_id"]: o for o in exchange_orders}
        
        # Check for local client order IDs too
        local_by_client_id = {
            o["client_order_id"]: o 
            for o in local_orders 
            if o.get("client_order_id")
        }
        exchange_by_client_id = {
            o["client_order_id"]: o 
            for o in exchange_orders 
            if o.get("client_order_id")
        }
        
        # Find orphaned orders (local but not on exchange)
        for order_id, local_order in local_by_id.items():
            if order_id not in exchange_by_id:
                # Check if it exists by client order ID
                client_id = local_order.get("client_order_id")
                if not client_id or client_id not in exchange_by_client_id:
                    discrepancies.append(
                        OrderDiscrepancy(
                            order_id=order_id,
                            symbol=local_order["symbol"],
                            discrepancy_type=DiscrepancyType.ORPHANED,
                            local_state=local_order,
                            exchange_state=None,
                        )
                    )
        
        # Find zombie orders (on exchange but not local)
        for order_id, exchange_order in exchange_by_id.items():
            if order_id not in local_by_id:
                # Check if we have it by client order ID
                client_id = exchange_order.get("client_order_id")
                if not client_id or client_id not in local_by_client_id:
                    discrepancies.append(
                        OrderDiscrepancy(
                            order_id=order_id,
                            symbol=exchange_order["symbol"],
                            discrepancy_type=DiscrepancyType.ZOMBIE,
                            local_state=None,
                            exchange_state=exchange_order,
                        )
                    )
        
        # Check for mismatches in orders that exist both locally and on exchange
        for order_id in set(local_by_id.keys()) & set(exchange_by_id.keys()):
            local_order = local_by_id[order_id]
            exchange_order = exchange_by_id[order_id]
            
            # Check status mismatch
            if local_order["status"] != exchange_order["status"]:
                discrepancies.append(
                    OrderDiscrepancy(
                        order_id=order_id,
                        symbol=local_order["symbol"],
                        discrepancy_type=DiscrepancyType.STATUS_MISMATCH,
                        local_state=local_order,
                        exchange_state=exchange_order,
                    )
                )
            
            # Check filled quantity mismatch
            local_filled = local_order.get("filled_quantity", 0)
            exchange_filled = exchange_order.get("filled_quantity", 0)
            if abs(float(local_filled or 0) - float(exchange_filled or 0)) > 0.00000001:
                discrepancies.append(
                    OrderDiscrepancy(
                        order_id=order_id,
                        symbol=local_order["symbol"],
                        discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                        local_state=local_order,
                        exchange_state=exchange_order,
                    )
                )
        
        return discrepancies
    
    async def _resolve_discrepancies(
        self,
        discrepancies: List[OrderDiscrepancy],
    ) -> int:
        """
        Attempt to resolve detected discrepancies.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            Number of discrepancies resolved
        """
        resolved_count = 0
        
        for discrepancy in discrepancies:
            try:
                if discrepancy.discrepancy_type == DiscrepancyType.ORPHANED:
                    # Local order not on exchange - mark as failed/cancelled
                    await self._resolve_orphaned_order(discrepancy)
                    
                elif discrepancy.discrepancy_type == DiscrepancyType.ZOMBIE:
                    # Exchange order not local - add to local tracking
                    await self._resolve_zombie_order(discrepancy)
                    
                elif discrepancy.discrepancy_type == DiscrepancyType.STATUS_MISMATCH:
                    # Update local status to match exchange
                    await self._resolve_status_mismatch(discrepancy)
                    
                elif discrepancy.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH:
                    # Update local filled quantity to match exchange
                    await self._resolve_quantity_mismatch(discrepancy)
                
                discrepancy.resolved = True
                resolved_count += 1
                self.resolved_discrepancies.append(discrepancy)
                
                logger.info(
                    "Discrepancy resolved",
                    order_id=discrepancy.order_id,
                    type=discrepancy.discrepancy_type,
                    resolution=discrepancy.resolution,
                )
                
            except Exception as e:
                logger.error(
                    "Failed to resolve discrepancy",
                    order_id=discrepancy.order_id,
                    type=discrepancy.discrepancy_type,
                    error=str(e),
                )
                self.discrepancies.append(discrepancy)
        
        return resolved_count
    
    async def _resolve_orphaned_order(self, discrepancy: OrderDiscrepancy) -> None:
        """
        Resolve an orphaned order (local but not on exchange).
        
        Args:
            discrepancy: Orphaned order discrepancy
        """
        order_id = discrepancy.order_id
        
        # Update local order status to cancelled/failed
        if order_id in self.local_orders:
            self.local_orders[order_id]["status"] = "cancelled"
            self.local_orders[order_id]["cancelled_reason"] = "Not found on exchange"
            self.local_orders[order_id]["reconciled_at"] = datetime.now()
        
        discrepancy.resolution = "Marked as cancelled locally"
        
        # Publish event
        if self.event_bus:
            event = Event(
                event_type=EventType.ORDER_CANCELLED,
                aggregate_id=order_id,
                event_data={
                    "order_id": order_id,
                    "symbol": discrepancy.symbol,
                    "reason": "Orphaned order - not found on exchange",
                    "reconciliation": True,
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.HIGH)
    
    async def _resolve_zombie_order(self, discrepancy: OrderDiscrepancy) -> None:
        """
        Resolve a zombie order (on exchange but not local).
        
        Args:
            discrepancy: Zombie order discrepancy
        """
        order_id = discrepancy.order_id
        exchange_order = discrepancy.exchange_state
        
        # Add to local tracking
        self.local_orders[order_id] = exchange_order.copy()
        self.local_orders[order_id]["recovered"] = True
        self.local_orders[order_id]["recovered_at"] = datetime.now()
        
        discrepancy.resolution = "Added to local tracking"
        
        # Publish event
        if self.event_bus:
            event = Event(
                event_type=EventType.ORDER_RECOVERED,
                aggregate_id=order_id,
                event_data={
                    "order_id": order_id,
                    "symbol": discrepancy.symbol,
                    "order": exchange_order,
                    "reason": "Zombie order - found on exchange but not local",
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.HIGH)
    
    async def _resolve_status_mismatch(self, discrepancy: OrderDiscrepancy) -> None:
        """
        Resolve a status mismatch between local and exchange.
        
        Args:
            discrepancy: Status mismatch discrepancy
        """
        order_id = discrepancy.order_id
        exchange_status = discrepancy.exchange_state["status"]
        
        # Update local status to match exchange
        if order_id in self.local_orders:
            old_status = self.local_orders[order_id]["status"]
            self.local_orders[order_id]["status"] = exchange_status
            self.local_orders[order_id]["status_synced_at"] = datetime.now()
            
            discrepancy.resolution = f"Updated status from {old_status} to {exchange_status}"
    
    async def _resolve_quantity_mismatch(self, discrepancy: OrderDiscrepancy) -> None:
        """
        Resolve a filled quantity mismatch.
        
        Args:
            discrepancy: Quantity mismatch discrepancy
        """
        order_id = discrepancy.order_id
        exchange_filled = discrepancy.exchange_state["filled_quantity"]
        
        # Update local filled quantity to match exchange
        if order_id in self.local_orders:
            old_filled = self.local_orders[order_id].get("filled_quantity", 0)
            self.local_orders[order_id]["filled_quantity"] = exchange_filled
            self.local_orders[order_id]["quantity_synced_at"] = datetime.now()
            
            discrepancy.resolution = f"Updated filled from {old_filled} to {exchange_filled}"
    
    async def _publish_sync_event(
        self,
        discrepancies: List[OrderDiscrepancy],
        resolved: int,
    ) -> None:
        """
        Publish sync completion event.
        
        Args:
            discrepancies: List of found discrepancies
            resolved: Number resolved
        """
        event = Event(
            event_type=EventType.ORDER_SYNC_COMPLETED,
            aggregate_id="ORDER_RECONCILIATION",
            event_data={
                "timestamp": datetime.now().isoformat(),
                "discrepancies_found": len(discrepancies),
                "discrepancies_resolved": resolved,
                "discrepancy_types": {
                    dtype: sum(1 for d in discrepancies if d.discrepancy_type == dtype)
                    for dtype in DiscrepancyType
                },
                "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            },
        )
        await self.event_bus.publish(event, priority=EventPriority.NORMAL)
    
    def add_local_order(self, order: Dict) -> None:
        """
        Add an order to local tracking.
        
        Args:
            order: Order dictionary
        """
        order_id = order.get("order_id") or order.get("id")
        if order_id:
            self.local_orders[order_id] = order
            logger.debug("Added order to local tracking", order_id=order_id)
    
    def remove_local_order(self, order_id: str) -> None:
        """
        Remove an order from local tracking.
        
        Args:
            order_id: Order ID to remove
        """
        if order_id in self.local_orders:
            del self.local_orders[order_id]
            logger.debug("Removed order from local tracking", order_id=order_id)
    
    async def trigger_reconnection_sync(self) -> None:
        """Trigger sync after reconnection if configured."""
        if self.on_reconnection:
            logger.info("Triggering post-reconnection order sync")
            await self.sync_orders()
    
    async def trigger_manual_sync(self, symbol: Optional[str] = None) -> Dict:
        """
        Manually trigger order synchronization.
        
        Args:
            symbol: Optional symbol to sync
            
        Returns:
            Sync results
        """
        if not self.on_demand:
            logger.warning("Manual sync not enabled")
            return {"status": "disabled"}
        
        logger.info("Manual sync triggered", symbol=symbol)
        return await self.sync_orders(symbol)
    
    def get_statistics(self) -> Dict:
        """
        Get reconciliation statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "running": self.running,
            "sync_interval": self.sync_interval,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "sync_in_progress": self.sync_in_progress,
            "total_syncs": self.total_syncs,
            "sync_errors": self.sync_errors,
            "error_rate": self.sync_errors / self.total_syncs if self.total_syncs > 0 else 0,
            "discrepancies": {
                "total_found": self.total_discrepancies_found,
                "total_resolved": self.total_discrepancies_resolved,
                "pending": len(self.discrepancies),
                "resolution_rate": (
                    self.total_discrepancies_resolved / self.total_discrepancies_found
                    if self.total_discrepancies_found > 0
                    else 0
                ),
            },
            "local_orders": len(self.local_orders),
        }