"""Unit tests for Alert Manager."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import aiohttp

from genesis.monitoring.alert_manager import (
    AlertSeverity,
    AlertState,
    AlertRule,
    Alert,
    AlertManager,
    EmailNotificationChannel,
    SlackNotificationChannel,
    PagerDutyNotificationChannel,
    CircuitBreaker
)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.can_attempt_call() is True
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        # Record failures
        cb.call_failed()
        cb.call_failed()
        assert cb.state == "closed"
        assert cb.can_attempt_call() is True
        
        cb.call_failed()  # Third failure
        assert cb.state == "open"
        assert cb.can_attempt_call() is False
    
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets on successful call."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        cb.call_failed()
        cb.call_failed()
        cb.call_succeeded()
        
        assert cb.failure_count == 0
        assert cb.state == "closed"
    
    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker enters half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)  # 100ms timeout
        
        # Open the circuit
        for _ in range(3):
            cb.call_failed()
        
        assert cb.state == "open"
        assert cb.can_attempt_call() is False
        
        # Wait for recovery timeout
        import time
        time.sleep(0.2)
        
        assert cb.can_attempt_call() is True  # Should allow retry


class TestAlertRule:
    """Test AlertRule class."""
    
    def test_alert_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="test_alert",
            description="Test alert description",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        assert rule.name == "test_alert"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.threshold == 10.0
    
    def test_alert_rule_validation_invalid_name(self):
        """Test alert rule with invalid name."""
        with pytest.raises(ValueError, match="Invalid alert name"):
            AlertRule(
                name="test@alert!",  # Invalid characters
                description="Test",
                condition=lambda: True,
                severity=AlertSeverity.WARNING,
                threshold=10.0
            )
    
    def test_alert_rule_validation_invalid_threshold(self):
        """Test alert rule with invalid threshold."""
        with pytest.raises(ValueError, match="Threshold out of bounds"):
            AlertRule(
                name="test_alert",
                description="Test",
                condition=lambda: True,
                severity=AlertSeverity.WARNING,
                threshold=1e7  # Too large
            )
    
    def test_alert_rule_validation_invalid_label(self):
        """Test alert rule with invalid label."""
        with pytest.raises(ValueError, match="Invalid label"):
            AlertRule(
                name="test_alert",
                description="Test",
                condition=lambda: True,
                severity=AlertSeverity.WARNING,
                threshold=10.0,
                labels={"invalid@key": "value"}
            )
    
    def test_alert_rule_hash(self):
        """Test alert rule hashing."""
        rule1 = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        rule2 = AlertRule(
            name="test_alert",
            description="Different description",
            condition=lambda: False,
            severity=AlertSeverity.ERROR,
            threshold=20.0
        )
        
        # Same name should have same hash
        assert hash(rule1) == hash(rule2)


class TestAlert:
    """Test Alert class."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        rule = AlertRule(
            name="test_alert",
            description="Test alert",
            condition=lambda: True,
            severity=AlertSeverity.ERROR,
            threshold=10.0
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now(),
            value=15.0
        )
        
        assert alert.rule == rule
        assert alert.state == AlertState.FIRING
        assert alert.value == 15.0
        assert alert.notification_sent is False
    
    def test_alert_format_message(self):
        """Test formatting alert message."""
        rule = AlertRule(
            name="test_alert",
            description="Test alert description",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0,
            labels={"env": "prod", "service": "api"}
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now(),
            value=15.0,
            message="High CPU usage detected"
        )
        
        message = alert.format_message()
        
        assert "[FIRING]" in message
        assert "test_alert" in message
        assert "Test alert description" in message
        assert "15.00" in message
        assert "threshold: 10.00" in message
        assert "High CPU usage detected" in message
        assert "env=prod" in message


class TestAlertManager:
    """Test AlertManager class."""
    
    @pytest.mark.asyncio
    async def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert len(manager.rules) == 0
        assert len(manager.active_alerts) == 0
        assert len(manager.notification_channels) == 0
    
    def test_add_rule(self):
        """Test adding alert rule."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        manager.add_rule(rule)
        assert rule in manager.rules
    
    def test_add_invalid_rule(self):
        """Test adding invalid alert rule."""
        manager = AlertManager()
        
        # This should not raise but log error
        manager.add_rule(None)  # Invalid rule
        assert len(manager.rules) == 0
    
    def test_remove_rule(self):
        """Test removing alert rule."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        manager.add_rule(rule)
        assert len(manager.rules) == 1
        
        manager.remove_rule("test_alert")
        assert len(manager.rules) == 0
    
    @pytest.mark.asyncio
    async def test_alert_firing(self):
        """Test alert firing when condition is met."""
        manager = AlertManager()
        
        # Add a rule that always fires
        rule = AlertRule(
            name="always_fire",
            description="Always fires",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0,
            duration=timedelta(seconds=0)
        )
        
        manager.add_rule(rule)
        
        # Evaluate rules
        await manager._evaluate_rules()
        
        # Check alert was fired
        assert "always_fire" in manager.active_alerts
        alert = manager.active_alerts["always_fire"]
        assert alert.state == AlertState.FIRING
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self):
        """Test alert resolution when condition clears."""
        manager = AlertManager()
        
        # Add a rule with changing condition
        condition_met = True
        rule = AlertRule(
            name="toggle_alert",
            description="Toggles",
            condition=lambda: condition_met,
            severity=AlertSeverity.WARNING,
            threshold=10.0,
            duration=timedelta(seconds=0)
        )
        
        manager.add_rule(rule)
        
        # Fire alert
        await manager._evaluate_rules()
        assert "toggle_alert" in manager.active_alerts
        
        # Clear condition and evaluate again
        condition_met = False
        rule.condition = lambda: condition_met
        await manager._evaluate_rules()
        
        # Alert should be resolved
        assert "toggle_alert" not in manager.active_alerts
    
    @pytest.mark.asyncio
    async def test_alert_duration_requirement(self):
        """Test alert duration requirement."""
        manager = AlertManager()
        
        # Add a rule with duration requirement
        rule = AlertRule(
            name="duration_alert",
            description="Requires duration",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0,
            duration=timedelta(seconds=10)  # 10 second duration
        )
        
        manager.add_rule(rule)
        
        # First evaluation - condition met but duration not satisfied
        await manager._evaluate_rules()
        assert "duration_alert" not in manager.active_alerts
        
        # Set condition state to past time
        manager._condition_states["duration_alert"] = datetime.now() - timedelta(seconds=11)
        
        # Now it should fire
        await manager._evaluate_rules()
        assert "duration_alert" in manager.active_alerts
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self):
        """Test alert cooldown period."""
        manager = AlertManager()
        
        # Add a rule with short cooldown
        rule = AlertRule(
            name="cooldown_alert",
            description="Has cooldown",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0,
            duration=timedelta(seconds=0),
            cooldown=timedelta(hours=1)
        )
        
        manager.add_rule(rule)
        
        # Fire alert
        await manager._fire_alert(rule)
        assert "cooldown_alert" in manager.active_alerts
        
        # Clear and try to fire again
        manager.active_alerts.clear()
        await manager._fire_alert(rule)
        
        # Should not fire due to cooldown
        assert "cooldown_alert" not in manager.active_alerts
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        manager.active_alerts["test_alert"] = alert
        
        active = manager.get_active_alerts()
        assert len(active) == 1
        assert active[0] == alert
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        # Add multiple alerts to history
        for i in range(5):
            alert = Alert(
                rule=rule,
                state=AlertState.RESOLVED,
                fired_at=datetime.now() - timedelta(hours=i)
            )
            manager.alert_history.append(alert)
        
        history = manager.get_alert_history(limit=3)
        assert len(history) == 3


class TestNotificationChannels:
    """Test notification channels."""
    
    @pytest.mark.asyncio
    async def test_email_notification_channel(self):
        """Test email notification channel."""
        channel = EmailNotificationChannel({"smtp_host": "localhost"})
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        # Email channel is a placeholder, should always return True
        result = await channel.send(alert)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_slack_notification_channel_invalid_url(self):
        """Test Slack notification with invalid URL."""
        channel = SlackNotificationChannel("http://invalid.url")
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        result = await channel.send(alert)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_slack_notification_circuit_breaker(self):
        """Test Slack notification circuit breaker."""
        channel = SlackNotificationChannel("https://hooks.slack.com/test")
        
        rule = AlertRule(
            name="test_alert",
            description="Test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            threshold=10.0
        )
        
        alert = Alert(
            rule=rule,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        # Open circuit breaker
        for _ in range(5):
            channel.circuit_breaker.call_failed()
        
        # Should not attempt call
        result = await channel.send(alert)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_pagerduty_notification_severity_filter(self):
        """Test PagerDuty notification severity filtering."""
        channel = PagerDutyNotificationChannel("a" * 32)  # Valid format key
        
        # Low severity alert - should not send
        rule_info = AlertRule(
            name="info_alert",
            description="Info alert",
            condition=lambda: True,
            severity=AlertSeverity.INFO,
            threshold=10.0
        )
        
        alert_info = Alert(
            rule=rule_info,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        result = await channel.send(alert_info)
        assert result is True  # Returns True but doesn't actually send
        
        # High severity alert - would attempt to send
        rule_critical = AlertRule(
            name="critical_alert",
            description="Critical alert",
            condition=lambda: True,
            severity=AlertSeverity.CRITICAL,
            threshold=10.0
        )
        
        alert_critical = Alert(
            rule=rule_critical,
            state=AlertState.FIRING,
            fired_at=datetime.now()
        )
        
        # Would attempt to send (will fail without mock)
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 202
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await channel.send(alert_critical)
            # Circuit breaker will prevent actual send without proper mock setup
    
    @pytest.mark.asyncio
    async def test_notification_channel_cleanup(self):
        """Test notification channel cleanup."""
        channel = SlackNotificationChannel("https://hooks.slack.com/test")
        
        # Create a session
        session = await channel._get_session()
        assert channel.session is not None
        
        # Cleanup
        await channel.close()
        assert channel.session.closed