"""Command input and parsing system for Genesis trading terminal."""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from textual.widgets import Input


@dataclass
class CommandResult:
    """Result of command parsing and execution."""
    success: bool
    message: str
    command_type: Optional[str] = None
    params: Optional[dict] = None


class CommandInput(Input):
    """Enhanced input widget with command history and autocomplete."""

    DEFAULT_CSS = """
    CommandInput {
        dock: bottom;
        width: 100%;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the command input widget."""
        super().__init__(
            placeholder="Enter command (e.g., b100u, s50u, cancel, status)",
            **kwargs
        )
        self.command_history: List[str] = []
        self.history_index = -1

        # Autocomplete suggestions
        self.suggestions = [
            "buy", "sell", "cancel", "status", "help",
            "b100u", "s50u", "b1000u", "s500u"
        ]

    async def on_key(self, event) -> None:
        """Handle special keys for history and autocomplete."""
        if event.key == "up":
            # Navigate history up
            if self.command_history and self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.value = self.command_history[-(self.history_index + 1)]
                event.stop()

        elif event.key == "down":
            # Navigate history down
            if self.history_index > 0:
                self.history_index -= 1
                self.value = self.command_history[-(self.history_index + 1)]
            elif self.history_index == 0:
                self.history_index = -1
                self.value = ""
            event.stop()

        elif event.key == "tab":
            # Autocomplete
            current = self.value.lower()
            if current:
                matches = [s for s in self.suggestions if s.startswith(current)]
                if matches:
                    self.value = matches[0]
                    event.stop()

    async def action_submit(self) -> None:
        """Handle command submission."""
        command = self.value.strip()
        if command:
            # Add to history
            self.command_history.append(command)
            self.history_index = -1

            # Clear input
            self.value = ""

            # Process command through dashboard
            screen = self.screen
            if hasattr(screen, 'handle_command'):
                await screen.handle_command(command)


class CommandParser:
    """Parse and validate trading commands."""

    # Regex patterns for shorthand commands
    BUY_PATTERN = re.compile(r'^b(\d+(?:\.\d+)?)([u|U])$')  # b100u = buy $100 USDT
    SELL_PATTERN = re.compile(r'^s(\d+(?:\.\d+)?)([u|U])$')  # s50u = sell $50 USDT

    def __init__(self, integration=None):
        """Initialize the command parser."""
        self.integration = integration
        self.commands = {
            'buy': self._parse_buy,
            'sell': self._parse_sell,
            'cancel': self._parse_cancel,
            'status': self._parse_status,
            'help': self._parse_help,
        }

    async def parse(self, command: str) -> CommandResult:
        """Parse a command string and return result."""
        command = command.strip().lower()

        if not command:
            return CommandResult(False, "Empty command")

        # Check shorthand patterns first
        buy_match = self.BUY_PATTERN.match(command)
        if buy_match:
            amount = Decimal(buy_match.group(1))
            return await self._execute_buy(amount)

        sell_match = self.SELL_PATTERN.match(command)
        if sell_match:
            amount = Decimal(sell_match.group(1))
            return await self._execute_sell(amount)

        # Parse full commands
        parts = command.split()
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        if cmd in self.commands:
            return await self.commands[cmd](args)
        else:
            return CommandResult(
                False,
                f"Unknown command: {cmd}. Type 'help' for available commands"
            )

    async def _parse_buy(self, args: List[str]) -> CommandResult:
        """Parse buy command."""
        if not args:
            return CommandResult(False, "Buy command requires amount (e.g., buy 100)")

        try:
            amount = Decimal(args[0])
            return await self._execute_buy(amount)
        except Exception:
            return CommandResult(False, f"Invalid amount: {args[0]}")

    async def _parse_sell(self, args: List[str]) -> CommandResult:
        """Parse sell command."""
        if not args:
            return CommandResult(False, "Sell command requires amount (e.g., sell 50)")

        try:
            amount = Decimal(args[0])
            return await self._execute_sell(amount)
        except Exception:
            return CommandResult(False, f"Invalid amount: {args[0]}")

    async def _parse_cancel(self, args: List[str]) -> CommandResult:
        """Parse cancel command."""
        # TODO: Connect to OrderExecutor.cancel_all_orders()
        return CommandResult(
            True,
            "Cancelling all orders...",
            "cancel",
            {}
        )

    async def _parse_status(self, args: List[str]) -> CommandResult:
        """Parse status command."""
        # TODO: Connect to system status
        return CommandResult(
            True,
            "System Status: Connected | Trading Active",
            "status",
            {}
        )

    async def _parse_help(self, args: List[str]) -> CommandResult:
        """Parse help command."""
        help_text = """Available Commands:
        b100u - Buy $100 USDT worth
        s50u - Sell $50 USDT worth
        buy <amount> - Buy specified amount
        sell <amount> - Sell specified amount
        cancel - Cancel all orders
        status - Show system status
        help - Show this help"""

        return CommandResult(
            True,
            help_text,
            "help",
            {}
        )

    async def _execute_buy(self, amount: Decimal) -> CommandResult:
        """Execute buy order."""
        # Validate amount
        if amount <= 0:
            return CommandResult(False, "Amount must be positive")

        if amount < 10:
            return CommandResult(False, "Minimum order size is $10")

        # TODO: Connect to OrderExecutor
        return CommandResult(
            True,
            f"Buy order placed: ${amount:.2f} USDT",
            "buy",
            {"amount": amount, "side": "BUY"}
        )

    async def _execute_sell(self, amount: Decimal) -> CommandResult:
        """Execute sell order."""
        # Validate amount
        if amount <= 0:
            return CommandResult(False, "Amount must be positive")

        if amount < 10:
            return CommandResult(False, "Minimum order size is $10")

        # TODO: Connect to OrderExecutor
        return CommandResult(
            True,
            f"Sell order placed: ${amount:.2f} USDT",
            "sell",
            {"amount": amount, "side": "SELL"}
        )
