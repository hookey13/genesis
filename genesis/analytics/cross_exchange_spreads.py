"""
Cross-Exchange Spread Analyzer

Foundation for comparing spreads across multiple exchanges.
Currently supports Binance only, but designed for multi-exchange expansion.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class Exchange(str, Enum):
    """Supported exchanges."""

    BINANCE = "BINANCE"
    # Future exchanges
    COINBASE = "COINBASE"  # Placeholder
    KRAKEN = "KRAKEN"  # Placeholder
    FTX = "FTX"  # Placeholder


@dataclass
class ExchangeSpreadData:
    """Spread data for a specific exchange."""

    exchange: Exchange
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    spread_bps: Decimal
    bid_volume: Decimal
    ask_volume: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    fee_bps: Decimal = field(default=Decimal("10"))  # Default 0.1% fee

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / Decimal("2")


@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity."""

    symbol: str
    buy_exchange: Exchange
    sell_exchange: Exchange
    buy_price: Decimal
    sell_price: Decimal
    spread_bps: Decimal
    profit_bps: Decimal  # After fees
    buy_volume: Decimal
    sell_volume: Decimal
    max_volume: Decimal
    estimated_profit_usdt: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ExchangeGatewayProtocol(Protocol):
    """Protocol for exchange gateways."""

    async def get_order_book(self, symbol: str) -> dict:
        """Get order book for a symbol."""
        ...

    async def get_ticker(self, symbol: str) -> dict:
        """Get ticker data for a symbol."""
        ...

    async def get_trading_fee(self, symbol: str) -> Decimal:
        """Get trading fee for a symbol."""
        ...


class CrossExchangeSpreadAnalyzer:
    """
    Analyzer for comparing spreads and identifying arbitrage opportunities
    across multiple exchanges.
    """

    def __init__(self):
        """Initialize cross-exchange spread analyzer."""
        self.exchange_gateways: dict[Exchange, ExchangeGatewayProtocol] = {}
        self.spread_data: dict[str, dict[Exchange, ExchangeSpreadData]] = {}
        self.arbitrage_opportunities: list[ArbitrageOpportunity] = []
        self._logger = logger.bind(component="CrossExchangeSpreadAnalyzer")

        # Monitoring state
        self._monitoring = False
        self._monitoring_task: asyncio.Task | None = None

    def register_exchange(
        self, exchange: Exchange, gateway: ExchangeGatewayProtocol
    ) -> None:
        """
        Register an exchange gateway.

        Args:
            exchange: Exchange identifier
            gateway: Gateway implementation
        """
        self.exchange_gateways[exchange] = gateway
        self._logger.info(f"Registered exchange: {exchange.value}")

    async def fetch_spread_data(
        self, symbol: str, exchange: Exchange
    ) -> ExchangeSpreadData | None:
        """
        Fetch spread data for a symbol from an exchange.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange to fetch from

        Returns:
            ExchangeSpreadData or None if error
        """
        gateway = self.exchange_gateways.get(exchange)
        if not gateway:
            self._logger.warning(f"No gateway registered for {exchange.value}")
            return None

        try:
            # Fetch order book
            order_book = await gateway.get_order_book(symbol)

            if (
                not order_book
                or not order_book.get("bids")
                or not order_book.get("asks")
            ):
                return None

            # Extract best bid/ask
            best_bid = Decimal(str(order_book["bids"][0][0]))
            best_ask = Decimal(str(order_book["asks"][0][0]))
            bid_volume = Decimal(str(order_book["bids"][0][1]))
            ask_volume = Decimal(str(order_book["asks"][0][1]))

            # Calculate spread
            mid_price = (best_bid + best_ask) / Decimal("2")
            spread_bps = ((best_ask - best_bid) / mid_price) * Decimal("10000")

            # Get trading fee
            fee_bps = await gateway.get_trading_fee(symbol)

            return ExchangeSpreadData(
                exchange=exchange,
                symbol=symbol,
                bid_price=best_bid,
                ask_price=best_ask,
                spread_bps=spread_bps,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                fee_bps=fee_bps,
            )

        except Exception as e:
            self._logger.error(
                "Failed to fetch spread data",
                exchange=exchange.value,
                symbol=symbol,
                error=str(e),
            )
            return None

    async def compare_spreads(self, symbol: str) -> dict[Exchange, ExchangeSpreadData]:
        """
        Compare spreads for a symbol across all registered exchanges.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary of exchange to spread data
        """
        tasks = []
        exchanges = []

        for exchange in self.exchange_gateways.keys():
            tasks.append(self.fetch_spread_data(symbol, exchange))
            exchanges.append(exchange)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        spread_comparison = {}
        for exchange, result in zip(exchanges, results, strict=False):
            if isinstance(result, ExchangeSpreadData):
                spread_comparison[exchange] = result

        # Cache the data
        if symbol not in self.spread_data:
            self.spread_data[symbol] = {}
        self.spread_data[symbol].update(spread_comparison)

        return spread_comparison

    def identify_arbitrage(
        self, symbol: str, min_profit_bps: Decimal = Decimal("20")
    ) -> ArbitrageOpportunity | None:
        """
        Identify arbitrage opportunity for a symbol.

        Args:
            symbol: Trading pair symbol
            min_profit_bps: Minimum profit threshold in basis points

        Returns:
            ArbitrageOpportunity or None if no opportunity
        """
        if symbol not in self.spread_data or len(self.spread_data[symbol]) < 2:
            return None

        spreads = self.spread_data[symbol]
        best_opportunity = None
        max_profit = Decimal("0")

        # Compare all exchange pairs
        exchanges = list(spreads.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                data1, data2 = spreads[ex1], spreads[ex2]

                # Check buy from ex1, sell to ex2
                if data1.ask_price < data2.bid_price:
                    spread_bps = (
                        (data2.bid_price - data1.ask_price) / data1.ask_price
                    ) * Decimal("10000")
                    profit_bps = spread_bps - data1.fee_bps - data2.fee_bps

                    if profit_bps > min_profit_bps and profit_bps > max_profit:
                        max_volume = min(data1.ask_volume, data2.bid_volume)
                        estimated_profit = (
                            (profit_bps / Decimal("10000"))
                            * max_volume
                            * data1.ask_price
                        )

                        best_opportunity = ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=ex1,
                            sell_exchange=ex2,
                            buy_price=data1.ask_price,
                            sell_price=data2.bid_price,
                            spread_bps=spread_bps,
                            profit_bps=profit_bps,
                            buy_volume=data1.ask_volume,
                            sell_volume=data2.bid_volume,
                            max_volume=max_volume,
                            estimated_profit_usdt=estimated_profit,
                        )
                        max_profit = profit_bps

                # Check buy from ex2, sell to ex1
                if data2.ask_price < data1.bid_price:
                    spread_bps = (
                        (data1.bid_price - data2.ask_price) / data2.ask_price
                    ) * Decimal("10000")
                    profit_bps = spread_bps - data2.fee_bps - data1.fee_bps

                    if profit_bps > min_profit_bps and profit_bps > max_profit:
                        max_volume = min(data2.ask_volume, data1.bid_volume)
                        estimated_profit = (
                            (profit_bps / Decimal("10000"))
                            * max_volume
                            * data2.ask_price
                        )

                        best_opportunity = ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=ex2,
                            sell_exchange=ex1,
                            buy_price=data2.ask_price,
                            sell_price=data1.bid_price,
                            spread_bps=spread_bps,
                            profit_bps=profit_bps,
                            buy_volume=data2.ask_volume,
                            sell_volume=data1.bid_volume,
                            max_volume=max_volume,
                            estimated_profit_usdt=estimated_profit,
                        )
                        max_profit = profit_bps

        if best_opportunity:
            self.arbitrage_opportunities.append(best_opportunity)
            self._logger.info(
                "Arbitrage opportunity found",
                symbol=symbol,
                buy_exchange=best_opportunity.buy_exchange.value,
                sell_exchange=best_opportunity.sell_exchange.value,
                profit_bps=float(best_opportunity.profit_bps),
                estimated_profit=float(best_opportunity.estimated_profit_usdt),
            )

        return best_opportunity

    async def start_monitoring(
        self, symbols: list[str], interval_seconds: int = 5
    ) -> None:
        """
        Start monitoring spreads across exchanges.

        Args:
            symbols: List of symbols to monitor
            interval_seconds: Update interval in seconds
        """
        if self._monitoring:
            self._logger.warning("Monitoring already started")
            return

        self._monitoring = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(symbols, interval_seconds)
        )
        self._logger.info(f"Started monitoring {len(symbols)} symbols")

    async def stop_monitoring(self) -> None:
        """Stop monitoring spreads."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped monitoring")

    async def _monitoring_loop(self, symbols: list[str], interval_seconds: int) -> None:
        """
        Background monitoring loop.

        Args:
            symbols: Symbols to monitor
            interval_seconds: Update interval
        """
        while self._monitoring:
            try:
                # Update spreads for all symbols
                tasks = [self.compare_spreads(symbol) for symbol in symbols]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Check for arbitrage opportunities
                for symbol in symbols:
                    self.identify_arbitrage(symbol)

                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)

    def get_best_exchange_for_buy(self, symbol: str) -> Exchange | None:
        """
        Get exchange with best price for buying.

        Args:
            symbol: Trading pair symbol

        Returns:
            Exchange with lowest ask price
        """
        if symbol not in self.spread_data:
            return None

        best_exchange = None
        best_price = Decimal("999999999")

        for exchange, data in self.spread_data[symbol].items():
            if data.ask_price < best_price:
                best_price = data.ask_price
                best_exchange = exchange

        return best_exchange

    def get_best_exchange_for_sell(self, symbol: str) -> Exchange | None:
        """
        Get exchange with best price for selling.

        Args:
            symbol: Trading pair symbol

        Returns:
            Exchange with highest bid price
        """
        if symbol not in self.spread_data:
            return None

        best_exchange = None
        best_price = Decimal("0")

        for exchange, data in self.spread_data[symbol].items():
            if data.bid_price > best_price:
                best_price = data.bid_price
                best_exchange = exchange

        return best_exchange

    def get_spread_differential(self, symbol: str) -> Decimal | None:
        """
        Calculate maximum spread differential across exchanges.

        Args:
            symbol: Trading pair symbol

        Returns:
            Maximum spread differential in basis points
        """
        if symbol not in self.spread_data or len(self.spread_data[symbol]) < 2:
            return None

        spreads = [data.spread_bps for data in self.spread_data[symbol].values()]
        return max(spreads) - min(spreads)

    def get_statistics(self) -> dict:
        """Get analyzer statistics."""
        return {
            "registered_exchanges": len(self.exchange_gateways),
            "monitored_symbols": len(self.spread_data),
            "total_arbitrage_opportunities": len(self.arbitrage_opportunities),
            "recent_opportunities": (
                self.arbitrage_opportunities[-10:]
                if self.arbitrage_opportunities
                else []
            ),
            "monitoring_active": self._monitoring,
        }

    def clear_opportunities(self) -> None:
        """Clear stored arbitrage opportunities."""
        self.arbitrage_opportunities.clear()
        self._logger.info("Cleared arbitrage opportunities")
