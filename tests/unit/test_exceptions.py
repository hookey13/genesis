"""
Unit tests for custom exception hierarchy.
"""

from decimal import Decimal
from typing import Dict, Any

import pytest

from genesis.core.exceptions import (
    BaseError,
    DomainError,
    TradingError,
    OrderError,
    RiskLimitExceeded,
    TierViolation,
    TiltInterventionRequired,
    InsufficientBalance,
    NetworkError,
    ConnectionTimeout,
    RateLimitError,
    DatabaseError,
    DatabaseLocked,
    TransactionRollback,
    SystemError,
    MemoryError,
    ConfigurationMissing,
    OrderRejected,
    OrderNotFound,
    InvalidOrderParameters,
    ValidationError,
    map_binance_error_to_domain,
)


class TestBaseError:
    """Test BaseError base class."""
    
    def test_base_error_initialization(self):
        """Test BaseError initializes correctly."""
        error = BaseError("Test error", code="TEST-001", details={"key": "value"})
        
        assert str(error) == "Test error"
        assert error.code == "TEST-001"
        assert error.details == {"key": "value"}
    
    def test_base_error_auto_code_generation(self):
        """Test automatic error code generation."""
        error = BaseError("Test error")
        
        assert error.code.startswith("GENESIS-")
        assert len(error.code) == 12  # GENESIS-XXXX
    
    def test_base_error_empty_details(self):
        """Test BaseError with no details."""
        error = BaseError("Test error")
        
        assert error.details == {}


class TestExceptionHierarchy:
    """Test exception hierarchy relationships."""
    
    def test_domain_error_inheritance(self):
        """Test DomainError inherits from BaseError."""
        error = DomainError("Domain error")
        
        assert isinstance(error, BaseError)
        assert isinstance(error, DomainError)
    
    def test_trading_error_inheritance(self):
        """Test TradingError inherits from DomainError."""
        error = TradingError("Trading error")
        
        assert isinstance(error, BaseError)
        assert isinstance(error, DomainError)
        assert isinstance(error, TradingError)
    
    def test_order_error_inheritance(self):
        """Test OrderError inherits from TradingError."""
        error = OrderError("Order error", order_id="ORD-123")
        
        assert isinstance(error, BaseError)
        assert isinstance(error, DomainError)
        assert isinstance(error, TradingError)
        assert isinstance(error, OrderError)
        assert error.order_id == "ORD-123"
        assert error.details["order_id"] == "ORD-123"


class TestBusinessExceptions:
    """Test business logic exceptions."""
    
    def test_risk_limit_exceeded(self):
        """Test RiskLimitExceeded exception."""
        error = RiskLimitExceeded(
            "Position too large",
            limit_type="position_size",
            current_value=Decimal("1000"),
            limit_value=Decimal("500"),
        )
        
        assert str(error) == "Position too large"
        assert error.code == "GENESIS-5701"
        assert error.limit_type == "position_size"
        assert error.current_value == Decimal("1000")
        assert error.limit_value == Decimal("500")
        assert error.details["limit_type"] == "position_size"
        assert error.details["current_value"] == "1000"
        assert error.details["limit_value"] == "500"
    
    def test_tier_violation(self):
        """Test TierViolation exception."""
        error = TierViolation(
            "Feature not available",
            required_tier="HUNTER",
            current_tier="SNIPER",
        )
        
        assert str(error) == "Feature not available"
        assert error.code == "GENESIS-5702"
        assert error.required_tier == "HUNTER"
        assert error.current_tier == "SNIPER"
        assert error.details["required_tier"] == "HUNTER"
        assert error.details["current_tier"] == "SNIPER"
    
    def test_tilt_intervention_required(self):
        """Test TiltInterventionRequired exception."""
        indicators = {
            "click_speed": 0.8,
            "cancel_rate": 0.7,
            "revenge_trading": 0.9,
        }
        
        error = TiltInterventionRequired(
            "Tilt detected",
            tilt_score=0.85,
            threshold=0.75,
            indicators=indicators,
        )
        
        assert str(error) == "Tilt detected"
        assert error.code == "GENESIS-5903"
        assert error.tilt_score == 0.85
        assert error.threshold == 0.75
        assert error.indicators == indicators
        assert error.details["tilt_score"] == 0.85
        assert error.details["threshold"] == 0.75
        assert error.details["indicators"] == indicators
    
    def test_insufficient_balance(self):
        """Test InsufficientBalance exception."""
        error = InsufficientBalance(
            "Not enough funds",
            required_amount=Decimal("1000"),
            available_amount=Decimal("500"),
        )
        
        assert str(error) == "Not enough funds"
        assert error.required_amount == Decimal("1000")
        assert error.available_amount == Decimal("500")
        assert isinstance(error, TradingError)


class TestNetworkExceptions:
    """Test network-related exceptions."""
    
    def test_connection_timeout(self):
        """Test ConnectionTimeout exception."""
        error = ConnectionTimeout("Request timed out", timeout_seconds=30.0)
        
        assert str(error) == "Request timed out"
        assert error.code == "GENESIS-2501"
        assert error.timeout_seconds == 30.0
        assert error.details["timeout_seconds"] == 30.0
        assert isinstance(error, NetworkError)
    
    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after_seconds=60,
            endpoint="/api/v3/order",
        )
        
        assert str(error) == "Rate limit exceeded"
        assert error.code == "GENESIS-2502"
        assert error.retry_after_seconds == 60
        assert error.endpoint == "/api/v3/order"
        assert error.details["retry_after_seconds"] == 60
        assert error.details["endpoint"] == "/api/v3/order"
    
    def test_rate_limit_error_minimal(self):
        """Test RateLimitError with minimal parameters."""
        error = RateLimitError("Rate limit exceeded")
        
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after_seconds is None
        assert error.endpoint is None


class TestDatabaseExceptions:
    """Test database-related exceptions."""
    
    def test_database_locked(self):
        """Test DatabaseLocked exception."""
        error = DatabaseLocked("Database is locked", table="orders")
        
        assert str(error) == "Database is locked"
        assert error.code == "GENESIS-3501"
        assert error.table == "orders"
        assert error.details["table"] == "orders"
        assert isinstance(error, DatabaseError)
    
    def test_database_locked_no_table(self):
        """Test DatabaseLocked without table specification."""
        error = DatabaseLocked("Database is locked")
        
        assert error.table is None
        assert error.details["table"] is None
    
    def test_transaction_rollback(self):
        """Test TransactionRollback exception."""
        error = TransactionRollback(
            "Transaction failed",
            reason="Constraint violation",
        )
        
        assert str(error) == "Transaction failed"
        assert error.code == "GENESIS-3701"
        assert error.reason == "Constraint violation"
        assert error.details["reason"] == "Constraint violation"


class TestSystemExceptions:
    """Test system-level exceptions."""
    
    def test_memory_error(self):
        """Test MemoryError exception."""
        error = MemoryError(
            "Memory limit exceeded",
            memory_usage_mb=1024.5,
            limit_mb=512.0,
        )
        
        assert str(error) == "Memory limit exceeded"
        assert error.code == "GENESIS-6901"
        assert error.memory_usage_mb == 1024.5
        assert error.limit_mb == 512.0
        assert error.details["memory_usage_mb"] == 1024.5
        assert error.details["limit_mb"] == 512.0
        assert isinstance(error, SystemError)
    
    def test_configuration_missing(self):
        """Test ConfigurationMissing exception."""
        error = ConfigurationMissing(
            "Required config not found",
            config_key="BINANCE_API_KEY",
        )
        
        assert str(error) == "Required config not found"
        assert error.code == "GENESIS-6701"
        assert error.config_key == "BINANCE_API_KEY"
        assert error.details["config_key"] == "BINANCE_API_KEY"


class TestOrderExceptions:
    """Test order-specific exceptions."""
    
    def test_order_rejected(self):
        """Test OrderRejected exception."""
        error = OrderRejected(
            "Order was rejected",
            order_id="ORD-456",
            reason="INSUFFICIENT_BALANCE",
        )
        
        assert str(error) == "Order was rejected"
        assert error.code == "GENESIS-1701"
        assert error.order_id == "ORD-456"
        assert error.reason == "INSUFFICIENT_BALANCE"
        assert error.details["order_id"] == "ORD-456"
        assert error.details["reason"] == "INSUFFICIENT_BALANCE"
        assert isinstance(error, OrderError)
    
    def test_order_rejected_minimal(self):
        """Test OrderRejected with minimal parameters."""
        error = OrderRejected("Order was rejected")
        
        assert error.order_id is None
        assert error.reason is None
        assert "reason" not in error.details
    
    def test_order_not_found(self):
        """Test OrderNotFound exception."""
        error = OrderNotFound("Order not found", order_id="ORD-789")
        
        assert str(error) == "Order not found"
        assert error.code == "GENESIS-1502"
        assert error.order_id == "ORD-789"
        assert error.details["order_id"] == "ORD-789"
    
    def test_invalid_order_parameters(self):
        """Test InvalidOrderParameters exception."""
        invalid_params = {
            "quantity": "invalid",
            "price": "-100",
        }
        
        error = InvalidOrderParameters(
            "Invalid parameters",
            order_id="ORD-999",
            invalid_params=invalid_params,
        )
        
        assert str(error) == "Invalid parameters"
        assert error.code == "GENESIS-4501"
        assert error.order_id == "ORD-999"
        assert error.invalid_params == invalid_params
        assert error.details["order_id"] == "ORD-999"
        assert error.details["invalid_params"] == invalid_params


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_with_field(self):
        """Test ValidationError with field specification."""
        error = ValidationError("Invalid value", field="quantity")
        
        assert str(error) == "Validation error for quantity: Invalid value"
        assert error.field == "quantity"
    
    def test_validation_error_without_field(self):
        """Test ValidationError without field specification."""
        error = ValidationError("General validation error")
        
        assert str(error) == "General validation error"
        assert error.field is None


class TestBinanceErrorMapping:
    """Test Binance error code mapping."""
    
    def test_map_unknown_error(self):
        """Test mapping Binance unknown error."""
        error = map_binance_error_to_domain(-1000, "Unknown issue")
        
        assert isinstance(error, NetworkError)
        assert error.code == "GENESIS-1901"
        assert "Unknown error" in str(error)
    
    def test_map_rate_limit_error(self):
        """Test mapping Binance rate limit error."""
        error = map_binance_error_to_domain(-1003, "Too many requests")
        
        assert isinstance(error, RateLimitError)
        assert error.retry_after_seconds == 60
        assert "Too many requests" in str(error)
    
    def test_map_timeout_error(self):
        """Test mapping Binance timeout error."""
        error = map_binance_error_to_domain(-1007, "Timeout waiting for response")
        
        assert isinstance(error, ConnectionTimeout)
        assert error.timeout_seconds == 30
        assert "Timeout" in str(error)
    
    def test_map_unauthorized_error(self):
        """Test mapping Binance unauthorized error."""
        error = map_binance_error_to_domain(-1002, "Unauthorized request")
        
        assert isinstance(error, ValidationError)
        assert error.field == "api_key"
        assert "Unauthorized" in str(error)
    
    def test_map_order_rejected_insufficient_balance(self):
        """Test mapping Binance insufficient balance error."""
        error = map_binance_error_to_domain(-2010, "Account has insufficient balance")
        
        assert isinstance(error, OrderRejected)
        assert error.reason == "INSUFFICIENT_BALANCE"
        assert "New order rejected" in str(error)
    
    def test_map_order_not_found(self):
        """Test mapping Binance order not found error."""
        error = map_binance_error_to_domain(-2013, "Order does not exist")
        
        assert isinstance(error, OrderNotFound)
        assert error.order_id == "unknown"
        assert "Order does not exist" in str(error)
    
    def test_map_invalid_symbol(self):
        """Test mapping Binance invalid symbol error."""
        error = map_binance_error_to_domain(-1121, "Invalid symbol")
        
        assert isinstance(error, InvalidOrderParameters)
        assert error.invalid_params == {"symbol": "invalid"}
        assert "Invalid symbol" in str(error)
    
    def test_map_invalid_quantity(self):
        """Test mapping Binance invalid quantity error."""
        error = map_binance_error_to_domain(-1013, "Invalid quantity")
        
        assert isinstance(error, InvalidOrderParameters)
        assert error.invalid_params == {"quantity": "invalid"}
        assert "Invalid quantity" in str(error)
    
    def test_map_unmapped_error_code(self):
        """Test mapping unmapped Binance error code."""
        from genesis.core.exceptions import ExchangeError
        
        error = map_binance_error_to_domain(-9999, "Some new error")
        
        assert isinstance(error, ExchangeError)
        assert "Binance error -9999" in str(error)
        assert "Some new error" in str(error)
    
    def test_map_validation_errors(self):
        """Test mapping various Binance validation errors."""
        validation_codes = [
            (-1100, "Illegal characters"),
            (-1101, "Too many parameters"),
            (-1102, "Mandatory parameter missing"),
            (-1103, "Unknown parameter"),
        ]
        
        for code, message in validation_codes:
            error = map_binance_error_to_domain(code, message)
            assert isinstance(error, ValidationError)
            assert error.field == "request"
            assert message in str(error)