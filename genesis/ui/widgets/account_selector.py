"""
Account selector widget for Project GENESIS UI.

Provides UI component for switching between multiple trading accounts
with hierarchy display and permission indicators.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from rich.align import Align
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Select, Static

from genesis.core.constants import TradingTier
from genesis.core.models import Account, AccountType


class AccountSwitched(Message):
    """Message emitted when account is switched."""
    
    def __init__(self, account_id: str, account: Account):
        super().__init__()
        self.account_id = account_id
        self.account = account


class AccountSelectorWidget(Widget):
    """Widget for selecting and switching between accounts."""

    BINDINGS = [
        Binding("a", "switch_account", "Switch Account"),
        Binding("r", "refresh", "Refresh"),
    ]

    # Reactive properties
    selected_account_id = reactive(None)
    accounts = reactive([])
    total_balance = reactive(Decimal("0"))
    
    def __init__(self, accounts: List[Account] = None, **kwargs):
        """Initialize account selector widget."""
        super().__init__(**kwargs)
        if accounts:
            self.update_accounts(accounts)

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical(id="account-selector-container"):
            yield Static(id="account-display")
            yield Select(
                id="account-select",
                options=[],
                prompt="Select Account",
            )

    def update_accounts(self, accounts: List[Account]) -> None:
        """Update the list of available accounts."""
        self.accounts = accounts
        
        # Calculate total balance
        self.total_balance = sum(a.balance_usdt for a in accounts)
        
        # Update select options
        options = []
        for account in accounts:
            label = self._format_account_label(account)
            options.append((label, account.account_id))
        
        # Update the select widget
        select = self.query_one("#account-select", Select)
        select.set_options(options)
        
        # Select first account if none selected
        if not self.selected_account_id and accounts:
            self.selected_account_id = accounts[0].account_id
        
        self.refresh()

    def _format_account_label(self, account: Account) -> str:
        """Format account label for display."""
        # Base label
        label = f"{account.account_type.value}"
        
        # Add tier
        label += f" [{account.tier.value}]"
        
        # Add balance
        label += f" ${account.balance_usdt:,.2f}"
        
        # Add parent indicator for sub-accounts
        if account.parent_account_id:
            label += " (SUB)"
        
        # Add permission indicators
        perms = []
        if account.permissions.get("trading"):
            perms.append("T")
        if account.permissions.get("withdrawals"):
            perms.append("W")
        if perms:
            label += f" [{'/'.join(perms)}]"
        
        return label

    def render(self) -> RenderableType:
        """Render the account selector display."""
        # Create accounts table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Type", style="white")
        table.add_column("Tier", style="yellow")
        table.add_column("Balance", justify="right", style="green")
        table.add_column("Permissions", style="cyan")
        table.add_column("Status", justify="center")
        
        # Group accounts by hierarchy
        master_accounts = [a for a in self.accounts if a.account_type == AccountType.MASTER]
        
        for master in master_accounts:
            # Add master account
            is_selected = master.account_id == self.selected_account_id
            row_style = "bold white on blue" if is_selected else "white"
            
            perms = self._format_permissions(master.permissions)
            status = "🟢 Active" if is_selected else ""
            
            table.add_row(
                "MASTER",
                master.tier.value,
                f"${master.balance_usdt:,.2f}",
                perms,
                status,
                style=row_style,
            )
            
            # Add sub-accounts
            sub_accounts = [
                a for a in self.accounts 
                if a.parent_account_id == master.account_id
            ]
            
            for sub in sub_accounts:
                is_selected = sub.account_id == self.selected_account_id
                row_style = "bold white on blue" if is_selected else "dim white"
                
                perms = self._format_permissions(sub.permissions)
                status = "🟢 Active" if is_selected else ""
                
                table.add_row(
                    "  └─ SUB",
                    sub.tier.value,
                    f"${sub.balance_usdt:,.2f}",
                    perms,
                    status,
                    style=row_style,
                )
        
        # Add paper trading accounts
        paper_accounts = [a for a in self.accounts if a.account_type == AccountType.PAPER]
        for paper in paper_accounts:
            is_selected = paper.account_id == self.selected_account_id
            row_style = "bold white on blue" if is_selected else "yellow"
            
            perms = self._format_permissions(paper.permissions)
            status = "🟢 Active" if is_selected else ""
            
            table.add_row(
                "PAPER",
                paper.tier.value,
                f"${paper.balance_usdt:,.2f}",
                perms,
                status,
                style=row_style,
            )
        
        # Add totals row
        table.add_row("", "", "", "", "", style="dim white")
        table.add_row(
            "TOTAL",
            "",
            f"${self.total_balance:,.2f}",
            "",
            "",
            style="bold green",
        )
        
        # Create panel
        selected_account = self._get_selected_account()
        if selected_account:
            title = f"📊 Accounts - Active: {selected_account.account_type.value}"
        else:
            title = "📊 Accounts"
        
        panel = Panel(
            Align.center(table),
            title=title,
            title_align="center",
            border_style="cyan",
            subtitle=f"Total Accounts: {len(self.accounts)}",
            subtitle_align="right",
        )
        
        return panel

    def _format_permissions(self, permissions: dict) -> str:
        """Format permissions for display."""
        if not permissions:
            return "None"
        
        perms = []
        if permissions.get("trading"):
            perms.append("Trade")
        if permissions.get("withdrawals"):
            perms.append("Withdraw")
        if permissions.get("admin"):
            perms.append("Admin")
        
        return ", ".join(perms) if perms else "View Only"

    def _get_selected_account(self) -> Optional[Account]:
        """Get the currently selected account."""
        if not self.selected_account_id:
            return None
        
        for account in self.accounts:
            if account.account_id == self.selected_account_id:
                return account
        
        return None

    async def action_switch_account(self) -> None:
        """Action to switch active account."""
        select = self.query_one("#account-select", Select)
        if select.value:
            self.selected_account_id = select.value
            
            # Get the selected account
            selected = self._get_selected_account()
            if selected:
                # Emit account switched message
                self.post_message(AccountSwitched(select.value, selected))
            
            self.refresh()

    async def action_refresh(self) -> None:
        """Refresh account list."""
        # This would typically fetch updated account data
        self.refresh()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget change."""
        if event.value:
            self.selected_account_id = event.value
            
            # Get the selected account
            selected = self._get_selected_account()
            if selected:
                # Emit account switched message
                self.post_message(AccountSwitched(event.value, selected))
            
            self.refresh()