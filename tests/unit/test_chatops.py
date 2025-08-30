"""
Unit tests for ChatOps command system.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from genesis.operations.chatops import (
    ChatOpsManager, Command, CommandContext, CommandResult,
    CommandPermission, WebhookType, HelpCommand, HealthCheckCommand,
    EmergencyStopCommand, ShowPositionsCommand
)


@pytest.fixture
def chatops_manager():
    """Create ChatOps manager."""
    return ChatOpsManager(
        webhook_url="https://hooks.example.com/webhook",
        webhook_type=WebhookType.SLACK
    )


@pytest.fixture
def command_context():
    """Create command context."""
    return CommandContext(
        user="john.doe",
        channel="#operations",
        timestamp=datetime.utcnow(),
        raw_message="health",
        webhook_type=WebhookType.SLACK,
        metadata={}
    )


class TestCommand:
    """Test Command base class."""
    
    def test_command_creation(self):
        """Test command creation."""
        cmd = Command(
            name="test",
            description="Test command",
            permission=CommandPermission.READ,
            usage="test [args]",
            aliases=["t"]
        )
        
        assert cmd.name == "test"
        assert cmd.permission == CommandPermission.READ
        assert "t" in cmd.aliases
    
    def test_validate_args_default(self):
        """Test default argument validation."""
        cmd = Command(
            name="test",
            description="Test",
            permission=CommandPermission.READ,
            usage="test"
        )
        
        valid, error = cmd.validate_args([])
        assert valid
        assert error == ""


class TestCommandContext:
    """Test CommandContext."""
    
    def test_context_creation(self):
        """Test context creation."""
        ctx = CommandContext(
            user="alice",
            channel="#general",
            timestamp=datetime.utcnow(),
            raw_message="help",
            webhook_type=WebhookType.DISCORD,
            metadata={"key": "value"}
        )
        
        assert ctx.user == "alice"
        assert ctx.channel == "#general"
        assert ctx.webhook_type == WebhookType.DISCORD
        assert ctx.metadata["key"] == "value"


class TestCommandResult:
    """Test CommandResult."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = CommandResult(
            success=True,
            message="Command executed",
            data={"status": "ok"},
            ephemeral=True
        )
        
        assert result.success
        assert result.message == "Command executed"
        assert result.data["status"] == "ok"
        assert result.ephemeral


class TestChatOpsManager:
    """Test ChatOpsManager."""
    
    def test_manager_initialization(self, chatops_manager):
        """Test manager initialization."""
        assert chatops_manager.webhook_type == WebhookType.SLACK
        assert "help" in chatops_manager.commands
        assert "health" in chatops_manager.commands
        assert "emergency" in chatops_manager.commands
    
    def test_register_command(self, chatops_manager):
        """Test command registration."""
        cmd = Command(
            name="custom",
            description="Custom command",
            permission=CommandPermission.READ,
            usage="custom",
            aliases=["c"]
        )
        
        chatops_manager.register_command(cmd)
        
        assert "custom" in chatops_manager.commands
        assert chatops_manager.aliases["c"] == "custom"
    
    @pytest.mark.asyncio
    async def test_process_help_command(self, chatops_manager, command_context):
        """Test processing help command."""
        result = await chatops_manager.process_message("help", command_context)
        
        assert result.success
        assert "Available Commands" in result.message
    
    @pytest.mark.asyncio
    async def test_process_unknown_command(self, chatops_manager, command_context):
        """Test processing unknown command."""
        result = await chatops_manager.process_message("unknown", command_context)
        
        assert not result.success
        assert "Unknown command" in result.message
    
    @pytest.mark.asyncio
    async def test_process_command_with_args(self, chatops_manager, command_context):
        """Test processing command with arguments."""
        result = await chatops_manager.process_message("help health", command_context)
        
        assert result.success
        assert "health" in result.message
        assert "Check system health" in result.message
    
    @pytest.mark.asyncio
    async def test_alias_resolution(self, chatops_manager, command_context):
        """Test command alias resolution."""
        # "?" is alias for "help"
        result = await chatops_manager.process_message("?", command_context)
        
        assert result.success
        assert "Available Commands" in result.message
    
    @pytest.mark.asyncio
    async def test_authorization_check(self):
        """Test authorization checking."""
        async def auth_handler(user, command, permission):
            # Only allow admin for emergency commands
            if permission == CommandPermission.EMERGENCY:
                return user == "admin"
            return True
        
        manager = ChatOpsManager(
            webhook_url="https://example.com",
            webhook_type=WebhookType.SLACK,
            authorization_handler=auth_handler
        )
        
        # Non-admin trying emergency command
        context = CommandContext(
            user="regular_user",
            channel="#ops",
            timestamp=datetime.utcnow(),
            raw_message="emergency",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        result = await manager.process_message("emergency", context)
        assert not result.success
        assert "Unauthorized" in result.message
        
        # Admin trying emergency command
        context.user = "admin"
        result = await manager.process_message("emergency", context)
        assert result.success
    
    def test_format_slack_message(self, chatops_manager):
        """Test Slack message formatting."""
        result = CommandResult(
            success=True,
            message="Test message",
            data={"key": "value"},
            ephemeral=True
        )
        
        context = CommandContext(
            user="test",
            channel="#test",
            timestamp=datetime.utcnow(),
            raw_message="test",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        payload = chatops_manager._format_slack_message(result, context)
        
        assert payload["text"] == "Test message"
        assert payload["response_type"] == "ephemeral"
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["color"] == "good"
    
    def test_format_discord_message(self, chatops_manager):
        """Test Discord message formatting."""
        result = CommandResult(
            success=False,
            message="Error message",
            data={"error": "details"}
        )
        
        context = CommandContext(
            user="test",
            channel="#test",
            timestamp=datetime.utcnow(),
            raw_message="test",
            webhook_type=WebhookType.DISCORD,
            metadata={}
        )
        
        payload = chatops_manager._format_discord_message(result, context)
        
        assert payload["content"] == "Error message"
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["color"] == 0xFF0000  # Red for error


class TestBuiltinCommands:
    """Test built-in commands."""
    
    @pytest.mark.asyncio
    async def test_help_command(self, chatops_manager):
        """Test help command."""
        cmd = chatops_manager.commands["help"]
        context = CommandContext(
            user="test",
            channel="#test",
            timestamp=datetime.utcnow(),
            raw_message="help",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        # Help without args
        result = await cmd.execute([], context)
        assert result.success
        assert "Available Commands" in result.message
        
        # Help with specific command
        result = await cmd.execute(["health"], context)
        assert result.success
        assert "health" in result.message
        assert "Check system health" in result.message
    
    @pytest.mark.asyncio
    async def test_health_command(self):
        """Test health check command."""
        cmd = HealthCheckCommand()
        context = CommandContext(
            user="test",
            channel="#test",
            timestamp=datetime.utcnow(),
            raw_message="health",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        result = await cmd.execute([], context)
        
        assert result.success
        assert "System Health Check" in result.message
        assert result.data is not None
        assert "Status" in result.data
        assert "CPU" in result.data
        assert "Memory" in result.data
    
    @pytest.mark.asyncio
    async def test_positions_command(self):
        """Test positions command."""
        cmd = ShowPositionsCommand()
        context = CommandContext(
            user="test",
            channel="#test",
            timestamp=datetime.utcnow(),
            raw_message="showpos",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        result = await cmd.execute([], context)
        
        assert result.success
        assert result.data is not None
        assert "Total Positions" in result.data
    
    @pytest.mark.asyncio
    async def test_emergency_command(self):
        """Test emergency stop command."""
        cmd = EmergencyStopCommand()
        context = CommandContext(
            user="admin",
            channel="#ops",
            timestamp=datetime.utcnow(),
            raw_message="emergency",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        result = await cmd.execute([], context)
        
        assert result.success
        assert "EMERGENCY STOP ACTIVATED" in result.message
        assert "trading halted" in result.message


class TestCommandPermissions:
    """Test command permission levels."""
    
    def test_permission_levels(self):
        """Test different permission levels."""
        read_cmd = Command(
            name="read",
            description="Read command",
            permission=CommandPermission.READ,
            usage="read"
        )
        
        write_cmd = Command(
            name="write",
            description="Write command",
            permission=CommandPermission.WRITE,
            usage="write"
        )
        
        admin_cmd = Command(
            name="admin",
            description="Admin command",
            permission=CommandPermission.ADMIN,
            usage="admin"
        )
        
        emergency_cmd = Command(
            name="emergency",
            description="Emergency command",
            permission=CommandPermission.EMERGENCY,
            usage="emergency"
        )
        
        assert read_cmd.permission == CommandPermission.READ
        assert write_cmd.permission == CommandPermission.WRITE
        assert admin_cmd.permission == CommandPermission.ADMIN
        assert emergency_cmd.permission == CommandPermission.EMERGENCY


class TestWebhookTypes:
    """Test webhook type support."""
    
    def test_webhook_types(self):
        """Test different webhook types."""
        slack_manager = ChatOpsManager(
            webhook_url="https://slack.com",
            webhook_type=WebhookType.SLACK
        )
        
        discord_manager = ChatOpsManager(
            webhook_url="https://discord.com",
            webhook_type=WebhookType.DISCORD
        )
        
        mattermost_manager = ChatOpsManager(
            webhook_url="https://mattermost.com",
            webhook_type=WebhookType.MATTERMOST
        )
        
        assert slack_manager.webhook_type == WebhookType.SLACK
        assert discord_manager.webhook_type == WebhookType.DISCORD
        assert mattermost_manager.webhook_type == WebhookType.MATTERMOST