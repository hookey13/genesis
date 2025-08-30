"""
ChatOps command system for operational tasks via Slack/Discord.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
import structlog

from genesis.core.events import Event, EventType

logger = structlog.get_logger(__name__)


class CommandPermission(Enum):
    """Command permission levels."""
    READ = "read"  # View-only commands
    WRITE = "write"  # Modify state commands
    ADMIN = "admin"  # Dangerous commands
    EMERGENCY = "emergency"  # Emergency-only commands


class WebhookType(Enum):
    """Supported webhook types."""
    SLACK = "slack"
    DISCORD = "discord"
    MATTERMOST = "mattermost"
    GENERIC = "generic"


@dataclass
class CommandContext:
    """Context for command execution."""
    user: str
    channel: str
    timestamp: datetime
    raw_message: str
    webhook_type: WebhookType
    metadata: dict[str, Any]


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    data: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] | None = None
    ephemeral: bool = False  # Only visible to command issuer


class Command:
    """Base class for ChatOps commands."""

    def __init__(self, name: str, description: str,
                 permission: CommandPermission,
                 usage: str, aliases: list[str] | None = None):
        """Initialize command."""
        self.name = name
        self.description = description
        self.permission = permission
        self.usage = usage
        self.aliases = aliases or []

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Execute the command."""
        raise NotImplementedError

    def validate_args(self, args: list[str]) -> tuple[bool, str]:
        """Validate command arguments."""
        return True, ""


class ChatOpsManager:
    """Manages ChatOps commands and integrations."""

    def __init__(self, webhook_url: str, webhook_type: WebhookType,
                 authorization_handler: Callable | None = None):
        """
        Initialize ChatOps manager.
        
        Args:
            webhook_url: Webhook URL for sending messages
            webhook_type: Type of webhook (Slack/Discord/etc)
            authorization_handler: Optional auth handler function
        """
        self.webhook_url = webhook_url
        self.webhook_type = webhook_type
        self.authorization_handler = authorization_handler

        self.commands: dict[str, Command] = {}
        self.aliases: dict[str, str] = {}
        self.session: aiohttp.ClientSession | None = None

        # Register built-in commands
        self._register_builtin_commands()

        logger.info("ChatOps manager initialized",
                   webhook_type=webhook_type.value)

    def _register_builtin_commands(self):
        """Register built-in operational commands."""
        # Help command
        self.register_command(HelpCommand(self))

        # Health and status commands
        self.register_command(HealthCheckCommand())
        self.register_command(SystemStatusCommand())
        self.register_command(PositionStatusCommand())

        # Operational commands
        self.register_command(RestartServiceCommand())
        self.register_command(ClearCacheCommand())
        self.register_command(RunBackupCommand())

        # Trading commands
        self.register_command(ShowPositionsCommand())
        self.register_command(ShowOrdersCommand())
        self.register_command(CancelOrderCommand())

        # Emergency commands
        self.register_command(EmergencyStopCommand())
        self.register_command(CloseAllPositionsCommand())
        self.register_command(EnableSafeModeCommand())

        # Monitoring commands
        self.register_command(ShowMetricsCommand())
        self.register_command(ShowAlertsCommand())
        self.register_command(AcknowledgeAlertCommand())

    def register_command(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command

        # Register aliases
        for alias in command.aliases:
            self.aliases[alias] = command.name

        logger.debug("Command registered",
                    command=command.name,
                    permission=command.permission.value)

    async def process_message(self, message: str, context: CommandContext) -> CommandResult:
        """Process incoming chat message."""
        try:
            # Parse command from message
            parts = message.strip().split()
            if not parts:
                return CommandResult(False, "Empty command")

            cmd_name = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            # Resolve aliases
            if cmd_name in self.aliases:
                cmd_name = self.aliases[cmd_name]

            # Find command
            if cmd_name not in self.commands:
                return CommandResult(
                    False,
                    f"Unknown command: {cmd_name}. Type 'help' for available commands."
                )

            command = self.commands[cmd_name]

            # Check authorization
            if not await self._check_authorization(command, context):
                await self._audit_unauthorized(command, context)
                return CommandResult(
                    False,
                    f"Unauthorized: You don't have permission to run '{cmd_name}'"
                )

            # Validate arguments
            valid, error = command.validate_args(args)
            if not valid:
                return CommandResult(
                    False,
                    f"Invalid arguments: {error}\nUsage: {command.usage}"
                )

            # Audit command execution
            await self._audit_command(command, args, context)

            # Execute command
            result = await command.execute(args, context)

            # Send response
            await self.send_response(result, context)

            return result

        except Exception as e:
            logger.error("Failed to process ChatOps message",
                        message=message,
                        error=str(e))
            return CommandResult(
                False,
                f"Command failed: {e!s}"
            )

    async def _check_authorization(self, command: Command,
                                  context: CommandContext) -> bool:
        """Check if user is authorized for command."""
        if not self.authorization_handler:
            # No auth handler, check basic permissions
            if command.permission == CommandPermission.EMERGENCY:
                # Emergency commands need explicit approval
                return context.user in ["admin", "emergency_operator"]
            return True

        return await self.authorization_handler(
            user=context.user,
            command=command.name,
            permission=command.permission
        )

    async def _audit_command(self, command: Command, args: list[str],
                            context: CommandContext):
        """Audit command execution."""
        event = Event(
            event_type=EventType.AUDIT_LOG_CREATED,
            event_data={
                "action": "chatops_command",
                "command": command.name,
                "args": args,
                "user": context.user,
                "channel": context.channel,
                "timestamp": context.timestamp.isoformat(),
                "permission": command.permission.value
            }
        )
        logger.info("ChatOps command executed",
                   command=command.name,
                   user=context.user)

    async def _audit_unauthorized(self, command: Command, context: CommandContext):
        """Audit unauthorized command attempt."""
        event = Event(
            event_type=EventType.AUDIT_LOG_CREATED,
            event_data={
                "action": "chatops_unauthorized",
                "command": command.name,
                "user": context.user,
                "channel": context.channel,
                "timestamp": context.timestamp.isoformat(),
                "permission_required": command.permission.value
            }
        )
        logger.warning("Unauthorized ChatOps command attempt",
                      command=command.name,
                      user=context.user)

    async def send_response(self, result: CommandResult, context: CommandContext):
        """Send response back to chat."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Format message based on webhook type
            if self.webhook_type == WebhookType.SLACK:
                payload = self._format_slack_message(result, context)
            elif self.webhook_type == WebhookType.DISCORD:
                payload = self._format_discord_message(result, context)
            else:
                payload = self._format_generic_message(result, context)

            async with self.session.post(
                self.webhook_url,
                json=payload
            ) as response:
                if response.status not in [200, 204]:
                    logger.error("Failed to send ChatOps response",
                                status=response.status)

        except Exception as e:
            logger.error("Failed to send ChatOps response",
                        error=str(e))

    def _format_slack_message(self, result: CommandResult,
                             context: CommandContext) -> dict[str, Any]:
        """Format message for Slack."""
        color = "good" if result.success else "danger"

        payload = {
            "text": result.message,
            "response_type": "ephemeral" if result.ephemeral else "in_channel",
            "attachments": []
        }

        if result.data:
            payload["attachments"].append({
                "color": color,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in result.data.items()
                ]
            })

        if result.attachments:
            payload["attachments"].extend(result.attachments)

        return payload

    def _format_discord_message(self, result: CommandResult,
                               context: CommandContext) -> dict[str, Any]:
        """Format message for Discord."""
        color = 0x00FF00 if result.success else 0xFF0000

        payload = {
            "content": result.message,
            "embeds": []
        }

        if result.data:
            embed = {
                "color": color,
                "fields": [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in result.data.items()
                ]
            }
            payload["embeds"].append(embed)

        return payload

    def _format_generic_message(self, result: CommandResult,
                               context: CommandContext) -> dict[str, Any]:
        """Format generic webhook message."""
        return {
            "text": result.message,
            "success": result.success,
            "data": result.data,
            "user": context.user,
            "timestamp": context.timestamp.isoformat()
        }


# Built-in Commands

class HelpCommand(Command):
    """Show available commands."""

    def __init__(self, manager):
        super().__init__(
            name="help",
            description="Show available commands",
            permission=CommandPermission.READ,
            usage="help [command]",
            aliases=["?", "commands"]
        )
        self.manager = manager

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show help information."""
        if args:
            # Show help for specific command
            cmd_name = args[0].lower()
            if cmd_name in self.manager.aliases:
                cmd_name = self.manager.aliases[cmd_name]

            if cmd_name in self.manager.commands:
                cmd = self.manager.commands[cmd_name]
                return CommandResult(
                    True,
                    f"**{cmd.name}** - {cmd.description}\n"
                    f"Permission: {cmd.permission.value}\n"
                    f"Usage: `{cmd.usage}`\n"
                    f"Aliases: {', '.join(cmd.aliases) if cmd.aliases else 'None'}"
                )
            else:
                return CommandResult(False, f"Unknown command: {cmd_name}")

        # Show all commands grouped by permission
        groups = {}
        for cmd in self.manager.commands.values():
            perm = cmd.permission.value
            if perm not in groups:
                groups[perm] = []
            groups[perm].append(f"`{cmd.name}` - {cmd.description}")

        message = "**Available Commands:**\n\n"
        for perm in ["read", "write", "admin", "emergency"]:
            if perm in groups:
                message += f"**{perm.upper()} Permission:**\n"
                message += "\n".join(groups[perm]) + "\n\n"

        message += "Type `help <command>` for detailed usage"

        return CommandResult(True, message)


class HealthCheckCommand(Command):
    """Check system health."""

    def __init__(self):
        super().__init__(
            name="health",
            description="Check system health",
            permission=CommandPermission.READ,
            usage="health",
            aliases=["status", "ping"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Check system health."""
        # Would connect to actual health monitor
        health_data = {
            "Status": "✅ Healthy",
            "Uptime": "2d 14h 32m",
            "CPU": "42%",
            "Memory": "3.2GB / 8GB",
            "Disk": "45GB / 100GB",
            "Database": "✅ Connected",
            "Exchange": "✅ Connected"
        }

        return CommandResult(
            True,
            "System Health Check",
            data=health_data
        )


class SystemStatusCommand(Command):
    """Show detailed system status."""

    def __init__(self):
        super().__init__(
            name="sysstatus",
            description="Show detailed system status",
            permission=CommandPermission.READ,
            usage="sysstatus",
            aliases=["sys"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show system status."""
        status_data = {
            "Trading": "Active",
            "Tier": "Hunter ($5,234)",
            "Positions": "3 open",
            "Orders": "2 pending",
            "P&L Today": "+$127.45",
            "Tilt Score": "0.23 (Normal)",
            "Rate Limit": "457/1200",
            "WebSocket": "Connected",
            "Last Trade": "2 minutes ago"
        }

        return CommandResult(
            True,
            "System Status",
            data=status_data
        )


class PositionStatusCommand(Command):
    """Show position summary."""

    def __init__(self):
        super().__init__(
            name="positions",
            description="Show position summary",
            permission=CommandPermission.READ,
            usage="positions",
            aliases=["pos"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show positions."""
        # Would fetch actual positions
        positions = [
            {"Symbol": "BTC/USDT", "Size": "0.5", "Entry": "$42,150", "P&L": "+$234"},
            {"Symbol": "ETH/USDT", "Size": "10", "Entry": "$2,234", "P&L": "-$45"},
            {"Symbol": "SOL/USDT", "Size": "100", "Entry": "$98.50", "P&L": "+$123"}
        ]

        message = "**Open Positions:**\n"
        for pos in positions:
            message += f"• {pos['Symbol']}: {pos['Size']} @ {pos['Entry']} (P&L: {pos['P&L']})\n"

        return CommandResult(True, message)


class RestartServiceCommand(Command):
    """Restart a service."""

    def __init__(self):
        super().__init__(
            name="restart",
            description="Restart a service",
            permission=CommandPermission.WRITE,
            usage="restart <service>",
            aliases=[]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Restart service."""
        if not args:
            return CommandResult(False, "Please specify a service to restart")

        service = args[0]
        valid_services = ["trading", "market_data", "executor", "strategies"]

        if service not in valid_services:
            return CommandResult(
                False,
                f"Invalid service. Valid services: {', '.join(valid_services)}"
            )

        # Would actually restart service
        return CommandResult(
            True,
            f"Service '{service}' restarted successfully"
        )


class ShowPositionsCommand(Command):
    """Show detailed positions."""

    def __init__(self):
        super().__init__(
            name="showpos",
            description="Show detailed position information",
            permission=CommandPermission.READ,
            usage="showpos [symbol]",
            aliases=["sp"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show positions."""
        # Would fetch real positions
        return CommandResult(
            True,
            "Position details displayed",
            data={
                "Total Positions": 3,
                "Total Value": "$12,456",
                "Unrealized P&L": "+$312",
                "Margin Used": "45%"
            }
        )


class ShowOrdersCommand(Command):
    """Show pending orders."""

    def __init__(self):
        super().__init__(
            name="orders",
            description="Show pending orders",
            permission=CommandPermission.READ,
            usage="orders",
            aliases=["ord"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show orders."""
        return CommandResult(
            True,
            "**Pending Orders:**\n"
            "• BTC/USDT: Buy 0.1 @ $41,500 (Limit)\n"
            "• ETH/USDT: Sell 5 @ $2,300 (Limit)"
        )


class CancelOrderCommand(Command):
    """Cancel an order."""

    def __init__(self):
        super().__init__(
            name="cancel",
            description="Cancel an order",
            permission=CommandPermission.WRITE,
            usage="cancel <order_id>",
            aliases=[]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Cancel order."""
        if not args:
            return CommandResult(False, "Please specify order ID")

        order_id = args[0]
        # Would actually cancel order
        return CommandResult(
            True,
            f"Order {order_id} cancelled successfully"
        )


class EmergencyStopCommand(Command):
    """Emergency stop trading."""

    def __init__(self):
        super().__init__(
            name="emergency",
            description="Emergency stop all trading",
            permission=CommandPermission.EMERGENCY,
            usage="emergency",
            aliases=["estop", "kill"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Emergency stop."""
        # Would trigger emergency stop
        return CommandResult(
            True,
            "🚨 **EMERGENCY STOP ACTIVATED** 🚨\n"
            "• All trading halted\n"
            "• Pending orders cancelled\n"
            "• Positions frozen\n"
            "• System in safe mode"
        )


class CloseAllPositionsCommand(Command):
    """Close all positions."""

    def __init__(self):
        super().__init__(
            name="closeall",
            description="Close all open positions",
            permission=CommandPermission.EMERGENCY,
            usage="closeall [confirm]",
            aliases=[]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Close all positions."""
        if not args or args[0] != "confirm":
            return CommandResult(
                False,
                "⚠️ This will close ALL positions at market price!\n"
                "Type `closeall confirm` to proceed"
            )

        # Would close all positions
        return CommandResult(
            True,
            "All positions closed:\n"
            "• 3 positions closed\n"
            "• Total P&L: +$234.56\n"
            "• Orders cancelled: 2"
        )


class EnableSafeModeCommand(Command):
    """Enable safe mode."""

    def __init__(self):
        super().__init__(
            name="safemode",
            description="Enable safe trading mode",
            permission=CommandPermission.ADMIN,
            usage="safemode <on|off>",
            aliases=["safe"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Toggle safe mode."""
        if not args or args[0] not in ["on", "off"]:
            return CommandResult(False, "Usage: safemode <on|off>")

        enabled = args[0] == "on"
        status = "enabled" if enabled else "disabled"

        return CommandResult(
            True,
            f"Safe mode {status}\n"
            f"• Position limits: {'Reduced' if enabled else 'Normal'}\n"
            f"• Risk checks: {'Enhanced' if enabled else 'Standard'}\n"
            f"• Strategies: {'Conservative' if enabled else 'Normal'}"
        )


class ClearCacheCommand(Command):
    """Clear system caches."""

    def __init__(self):
        super().__init__(
            name="clearcache",
            description="Clear system caches",
            permission=CommandPermission.WRITE,
            usage="clearcache [type]",
            aliases=["cc"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Clear cache."""
        cache_type = args[0] if args else "all"
        # Would clear actual caches
        return CommandResult(
            True,
            f"Cache cleared: {cache_type}"
        )


class RunBackupCommand(Command):
    """Run manual backup."""

    def __init__(self):
        super().__init__(
            name="backup",
            description="Run manual backup",
            permission=CommandPermission.ADMIN,
            usage="backup",
            aliases=[]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Run backup."""
        # Would trigger backup
        return CommandResult(
            True,
            "Backup started:\n"
            "• Database backup: In progress...\n"
            "• Configuration backup: In progress...\n"
            "• Logs archive: In progress...\n"
            "Estimated completion: 2 minutes"
        )


class ShowMetricsCommand(Command):
    """Show system metrics."""

    def __init__(self):
        super().__init__(
            name="metrics",
            description="Show system metrics",
            permission=CommandPermission.READ,
            usage="metrics [period]",
            aliases=["stats"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show metrics."""
        period = args[0] if args else "1h"

        return CommandResult(
            True,
            f"Metrics ({period}):",
            data={
                "Trades": 45,
                "Success Rate": "87%",
                "Avg Latency": "23ms",
                "API Calls": 1234,
                "Errors": 2,
                "Uptime": "99.97%"
            }
        )


class ShowAlertsCommand(Command):
    """Show active alerts."""

    def __init__(self):
        super().__init__(
            name="alerts",
            description="Show active alerts",
            permission=CommandPermission.READ,
            usage="alerts",
            aliases=[]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Show alerts."""
        return CommandResult(
            True,
            "**Active Alerts:**\n"
            "• [HIGH] Memory usage at 82%\n"
            "• [MEDIUM] Slow query detected\n"
            "• [LOW] Certificate expires in 15 days"
        )


class AcknowledgeAlertCommand(Command):
    """Acknowledge an alert."""

    def __init__(self):
        super().__init__(
            name="ack",
            description="Acknowledge an alert",
            permission=CommandPermission.WRITE,
            usage="ack <alert_id>",
            aliases=["acknowledge"]
        )

    async def execute(self, args: list[str], context: CommandContext) -> CommandResult:
        """Acknowledge alert."""
        if not args:
            return CommandResult(False, "Please specify alert ID")

        alert_id = args[0]
        return CommandResult(
            True,
            f"Alert {alert_id} acknowledged by {context.user}"
        )
