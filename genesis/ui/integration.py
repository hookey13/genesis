from typing import Optional

"""Integration layer connecting UI to Genesis core components."""

from decimal import Decimal

import structlog

from genesis.core.account_manager import AccountManager
from genesis.core.models import PositionSide
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.risk_engine import RiskEngine
from genesis.exchange.gateway import BinanceGateway
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionWidget

logger = structlog.get_logger(__name__)


class UIIntegration:
    """Connects UI widgets to core Genesis components."""

    def __init__(
        self,
        account_manager: Optional[AccountManager] = None,
        risk_engine: Optional[RiskEngine] = None,
        order_executor: Optional[MarketOrderExecutor] = None,
        gateway: Optional[BinanceGateway] = None
    ):
        """
        Initialize UI integration.

        Args:
            account_manager: Account manager for balance data
            risk_engine: Risk engine for position data
            order_executor: Order executor for command execution
            gateway: Exchange gateway for connection status
        """
        self.account_manager = account_manager
        self.risk_engine = risk_engine
        self.order_executor = order_executor
        self.gateway = gateway

        # Widget references
        self.pnl_widget: Optional[PnLWidget] = None
        self.position_widget: Optional[PositionWidget] = None

    def connect_widgets(
        self,
        pnl_widget: PnLWidget,
        position_widget: PositionWidget
    ) -> None:
        """
        Connect UI widgets to integration layer.

        Args:
            pnl_widget: P&L display widget
            position_widget: Position display widget
        """
        self.pnl_widget = pnl_widget
        self.position_widget = position_widget

    async def update_pnl_data(self) -> None:
        """Update P&L widget with latest data from components."""
        if not self.pnl_widget:
            return

        try:
            # Get account data
            if self.account_manager:
                account = self.account_manager.account
                self.pnl_widget.account_balance = account.balance_usdt

                # Get session P&L
                if self.risk_engine and self.risk_engine.session:
                    session = self.risk_engine.session
                    self.pnl_widget.daily_pnl = session.realized_pnl

                    # Calculate current P&L if position exists
                    if self.risk_engine.position:
                        unrealized_pnl = self.risk_engine.calculate_unrealized_pnl()
                        self.pnl_widget.current_pnl = unrealized_pnl
                    else:
                        self.pnl_widget.current_pnl = Decimal("0.00")

        except Exception as e:
            logger.error("Error updating P&L data", error=str(e))

    async def update_position_data(self) -> None:
        """Update position widget with latest data from components."""
        if not self.position_widget:
            return

        try:
            if self.risk_engine and self.risk_engine.position:
                position = self.risk_engine.position

                # Update position widget
                self.position_widget.has_position = True
                self.position_widget.symbol = position.symbol
                self.position_widget.side = position.side.value
                self.position_widget.quantity = position.quantity
                self.position_widget.entry_price = position.entry_price

                # Get current price from gateway if available
                if self.gateway:
                    try:
                        ticker = await self.gateway.get_ticker(position.symbol)
                        current_price = ticker.get("last", position.entry_price)
                        self.position_widget.current_price = Decimal(str(current_price))
                    except Exception:
                        # Use entry price if can't get current
                        self.position_widget.current_price = position.entry_price

                # Set stop loss if available
                self.position_widget.stop_loss = position.stop_loss

                # Calculate unrealized P&L
                unrealized_pnl = self.risk_engine.calculate_unrealized_pnl()
                self.position_widget.unrealized_pnl = unrealized_pnl
            else:
                # No position
                self.position_widget.has_position = False

        except Exception as e:
            logger.error("Error updating position data", error=str(e))

    async def execute_buy_command(self, amount_usdt: Decimal) -> dict:
        """
        Execute buy command through order executor.

        Args:
            amount_usdt: Amount in USDT to buy

        Returns:
            Command result dictionary
        """
        if not self.order_executor:
            return {
                "success": False,
                "message": "Order executor not connected"
            }

        try:
            # Calculate position size
            if self.risk_engine:
                quantity = self.risk_engine.calculate_position_size(
                    amount_usdt,
                    Decimal("2")  # 2% stop loss default
                )
            else:
                # Simple calculation without risk engine
                # Assumes BTC price around $40k for example
                quantity = amount_usdt / Decimal("40000")

            # Execute market buy
            order = await self.order_executor.execute_market_order(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=quantity
            )

            return {
                "success": True,
                "message": f"Buy order executed: {quantity:.8f} BTC",
                "order": order
            }

        except Exception as e:
            logger.error("Buy command failed", error=str(e))
            return {
                "success": False,
                "message": f"Buy failed: {e!s}"
            }

    async def execute_sell_command(self, amount_usdt: Decimal) -> dict:
        """
        Execute sell command through order executor.

        Args:
            amount_usdt: Amount in USDT to sell

        Returns:
            Command result dictionary
        """
        if not self.order_executor:
            return {
                "success": False,
                "message": "Order executor not connected"
            }

        try:
            # Check if we have a position to sell
            if not self.risk_engine or not self.risk_engine.position:
                return {
                    "success": False,
                    "message": "No position to sell"
                }

            position = self.risk_engine.position

            # Calculate quantity to sell
            # For now, sell entire position if amount matches
            quantity = position.quantity

            # Execute market sell
            order = await self.order_executor.execute_market_order(
                symbol=position.symbol,
                side=PositionSide.SHORT,
                quantity=quantity
            )

            return {
                "success": True,
                "message": f"Sell order executed: {quantity:.8f} BTC",
                "order": order
            }

        except Exception as e:
            logger.error("Sell command failed", error=str(e))
            return {
                "success": False,
                "message": f"Sell failed: {e!s}"
            }

    async def cancel_all_orders(self) -> dict:
        """
        Cancel all open orders.

        Returns:
            Command result dictionary
        """
        if not self.order_executor:
            return {
                "success": False,
                "message": "Order executor not connected"
            }

        try:
            cancelled = await self.order_executor.cancel_all_orders()

            return {
                "success": True,
                "message": f"Cancelled {cancelled} orders",
                "count": cancelled
            }

        except Exception as e:
            logger.error("Cancel all orders failed", error=str(e))
            return {
                "success": False,
                "message": f"Cancel failed: {e!s}"
            }

    def get_connection_status(self) -> str:
        """
        Get connection status to exchange.

        Returns:
            Status string
        """
        if self.gateway and self.gateway.connected:
            return "Connected"
        else:
            return "Disconnected"

    def get_system_status(self) -> dict:
        """
        Get overall system status.

        Returns:
            Status dictionary
        """
        status = {
            "exchange": self.get_connection_status(),
            "trading": "Active" if self.order_executor else "Inactive",
            "tier": "SNIPER",
            "daily_limit": "$25"
        }

        if self.account_manager:
            status["balance"] = f"${self.account_manager.account.balance_usdt:.2f}"

        if self.risk_engine and self.risk_engine.session:
            session = self.risk_engine.session
            status["daily_pnl"] = f"${session.realized_pnl:.2f}"
            status["trades_today"] = session.trade_count

        return status
