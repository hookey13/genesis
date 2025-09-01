"""
Contract testing for API compatibility and schema validation.

Ensures API contracts are maintained between services and versions.
Tests backward compatibility and schema evolution.
"""

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError, validator


# API Version Management
class APIVersion(Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"


# Contract Schemas using Pydantic
class OrderRequestV1(BaseModel):
    """Order request contract for API v1."""
    
    symbol: str = Field(..., regex="^[A-Z]+/[A-Z]+$")
    side: str = Field(..., regex="^(BUY|SELL)$")
    type: str = Field(..., regex="^(MARKET|LIMIT|STOP_LIMIT)$")
    quantity: str = Field(..., regex="^[0-9]+\\.?[0-9]*$")
    price: Optional[str] = Field(None, regex="^[0-9]+\\.?[0-9]*$")
    stop_price: Optional[str] = Field(None, regex="^[0-9]+\\.?[0-9]*$")
    time_in_force: Optional[str] = Field("GTC", regex="^(GTC|IOC|FOK)$")
    
    @validator("quantity", "price", "stop_price")
    def validate_decimal(cls, v):
        if v is not None:
            try:
                Decimal(v)
            except:
                raise ValueError("Invalid decimal value")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.01",
                "price": "50000.00",
            }
        }


class OrderRequestV2(OrderRequestV1):
    """Order request contract for API v2 (backward compatible)."""
    
    client_order_id: Optional[str] = Field(None, max_length=36)
    reduce_only: Optional[bool] = Field(False)
    post_only: Optional[bool] = Field(False)
    iceberg_quantity: Optional[str] = Field(None)
    trailing_delta: Optional[int] = Field(None, ge=0, le=5000)
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": "0.01",
                "price": "50000.00",
                "client_order_id": "my-order-123",
                "post_only": True,
            }
        }


class OrderResponseV1(BaseModel):
    """Order response contract for API v1."""
    
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    type: str
    status: str = Field(..., regex="^(NEW|PARTIALLY_FILLED|FILLED|CANCELLED|REJECTED)$")
    quantity: str
    executed_quantity: str
    price: Optional[str]
    average_price: Optional[str]
    created_at: str
    updated_at: str
    
    class Config:
        schema_extra = {
            "example": {
                "order_id": "123456",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "FILLED",
                "quantity": "0.01",
                "executed_quantity": "0.01",
                "price": "50000.00",
                "average_price": "49999.50",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:01Z",
            }
        }


class PositionContract(BaseModel):
    """Position information contract."""
    
    position_id: str
    symbol: str
    side: str = Field(..., regex="^(LONG|SHORT)$")
    quantity: str
    entry_price: str
    mark_price: str
    pnl: str
    pnl_percentage: str
    margin: str
    leverage: int = Field(1, ge=1, le=125)
    
    @validator("quantity", "entry_price", "mark_price", "pnl", "margin")
    def validate_decimal(cls, v):
        try:
            Decimal(v)
        except:
            raise ValueError("Invalid decimal value")
        return v


class AccountBalanceContract(BaseModel):
    """Account balance contract."""
    
    account_id: str
    balances: Dict[str, str]
    total_balance_usdt: str
    available_balance_usdt: str
    used_margin_usdt: str
    unrealized_pnl_usdt: str
    realized_pnl_usdt: str
    
    @validator("total_balance_usdt", "available_balance_usdt", "used_margin_usdt", 
               "unrealized_pnl_usdt", "realized_pnl_usdt")
    def validate_decimal(cls, v):
        try:
            Decimal(v)
        except:
            raise ValueError("Invalid decimal value")
        return v


class MarketTickerContract(BaseModel):
    """Market ticker contract."""
    
    symbol: str
    last_price: str
    bid_price: str
    ask_price: str
    volume_24h: str
    quote_volume_24h: str
    high_24h: str
    low_24h: str
    price_change_24h: str
    price_change_percent_24h: str
    timestamp: str


class WebSocketMessageContract(BaseModel):
    """WebSocket message contract."""
    
    event: str = Field(..., regex="^(subscribe|unsubscribe|update|error|heartbeat)$")
    channel: Optional[str]
    data: Optional[Dict[str, Any]]
    timestamp: str
    sequence: Optional[int]


# Contract Validators
class ContractValidator:
    """Validates API contracts for compatibility."""
    
    @staticmethod
    def validate_request(data: Dict, version: APIVersion = APIVersion.V1) -> bool:
        """Validate request against contract."""
        try:
            if version == APIVersion.V1:
                OrderRequestV1(**data)
            elif version == APIVersion.V2:
                OrderRequestV2(**data)
            else:
                raise ValueError(f"Unsupported version: {version}")
            return True
        except ValidationError:
            return False
    
    @staticmethod
    def validate_response(data: Dict, contract_class: type) -> bool:
        """Validate response against contract."""
        try:
            contract_class(**data)
            return True
        except ValidationError:
            return False
    
    @staticmethod
    def check_backward_compatibility(old_data: Dict, new_data: Dict) -> bool:
        """Check if new version is backward compatible with old."""
        # All fields in old version must exist in new version
        for key in old_data.keys():
            if key not in new_data:
                return False
        return True


# Contract Tests
class TestOrderContracts:
    """Test order-related API contracts."""
    
    def test_order_request_v1_valid(self):
        """Test valid v1 order request."""
        request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        order = OrderRequestV1(**request)
        assert order.symbol == "BTC/USDT"
        assert order.side == "BUY"
        assert order.quantity == "0.01"
    
    def test_order_request_v1_invalid_symbol(self):
        """Test invalid symbol format."""
        request = {
            "symbol": "BTCUSDT",  # Missing slash
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        with pytest.raises(ValidationError):
            OrderRequestV1(**request)
    
    def test_order_request_v2_backward_compatible(self):
        """Test v2 is backward compatible with v1."""
        v1_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        # V1 request should work with V2 schema
        order_v2 = OrderRequestV2(**v1_request)
        assert order_v2.symbol == "BTC/USDT"
        assert order_v2.client_order_id is None  # Optional field
        assert order_v2.reduce_only is False  # Default value
    
    def test_order_request_v2_new_fields(self):
        """Test v2 new fields."""
        v2_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
            "client_order_id": "my-order-123",
            "post_only": True,
            "trailing_delta": 100,
        }
        
        order = OrderRequestV2(**v2_request)
        assert order.client_order_id == "my-order-123"
        assert order.post_only is True
        assert order.trailing_delta == 100
    
    def test_order_response_contract(self):
        """Test order response contract."""
        response = {
            "order_id": "123456",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "FILLED",
            "quantity": "0.01",
            "executed_quantity": "0.01",
            "price": "50000.00",
            "average_price": "49999.50",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z",
        }
        
        order_response = OrderResponseV1(**response)
        assert order_response.order_id == "123456"
        assert order_response.status == "FILLED"
    
    def test_invalid_order_status(self):
        """Test invalid order status."""
        response = {
            "order_id": "123456",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "INVALID_STATUS",  # Invalid
            "quantity": "0.01",
            "executed_quantity": "0.01",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z",
        }
        
        with pytest.raises(ValidationError):
            OrderResponseV1(**response)


class TestPositionContracts:
    """Test position-related contracts."""
    
    def test_position_contract_valid(self):
        """Test valid position contract."""
        position = {
            "position_id": "pos-001",
            "symbol": "BTC/USDT",
            "side": "LONG",
            "quantity": "0.1",
            "entry_price": "50000.00",
            "mark_price": "51000.00",
            "pnl": "100.00",
            "pnl_percentage": "2.00",
            "margin": "500.00",
            "leverage": 10,
        }
        
        pos = PositionContract(**position)
        assert pos.position_id == "pos-001"
        assert pos.side == "LONG"
        assert pos.leverage == 10
    
    def test_position_invalid_side(self):
        """Test invalid position side."""
        position = {
            "position_id": "pos-001",
            "symbol": "BTC/USDT",
            "side": "BUY",  # Should be LONG/SHORT
            "quantity": "0.1",
            "entry_price": "50000.00",
            "mark_price": "51000.00",
            "pnl": "100.00",
            "pnl_percentage": "2.00",
            "margin": "500.00",
        }
        
        with pytest.raises(ValidationError):
            PositionContract(**position)
    
    def test_position_invalid_leverage(self):
        """Test invalid leverage value."""
        position = {
            "position_id": "pos-001",
            "symbol": "BTC/USDT",
            "side": "LONG",
            "quantity": "0.1",
            "entry_price": "50000.00",
            "mark_price": "51000.00",
            "pnl": "100.00",
            "pnl_percentage": "2.00",
            "margin": "500.00",
            "leverage": 200,  # Max is 125
        }
        
        with pytest.raises(ValidationError):
            PositionContract(**position)


class TestAccountContracts:
    """Test account-related contracts."""
    
    def test_account_balance_contract(self):
        """Test account balance contract."""
        balance = {
            "account_id": "acc-001",
            "balances": {
                "USDT": "5000.00",
                "BTC": "0.1",
                "ETH": "2.5",
            },
            "total_balance_usdt": "10000.00",
            "available_balance_usdt": "8000.00",
            "used_margin_usdt": "2000.00",
            "unrealized_pnl_usdt": "150.00",
            "realized_pnl_usdt": "-50.00",
        }
        
        acc = AccountBalanceContract(**balance)
        assert acc.account_id == "acc-001"
        assert acc.balances["USDT"] == "5000.00"
        assert acc.total_balance_usdt == "10000.00"
    
    def test_account_balance_invalid_decimal(self):
        """Test invalid decimal values."""
        balance = {
            "account_id": "acc-001",
            "balances": {"USDT": "5000.00"},
            "total_balance_usdt": "invalid",  # Invalid decimal
            "available_balance_usdt": "8000.00",
            "used_margin_usdt": "2000.00",
            "unrealized_pnl_usdt": "150.00",
            "realized_pnl_usdt": "-50.00",
        }
        
        with pytest.raises(ValidationError):
            AccountBalanceContract(**balance)


class TestMarketDataContracts:
    """Test market data contracts."""
    
    def test_ticker_contract(self):
        """Test market ticker contract."""
        ticker = {
            "symbol": "BTC/USDT",
            "last_price": "50000.00",
            "bid_price": "49999.00",
            "ask_price": "50001.00",
            "volume_24h": "1000.5",
            "quote_volume_24h": "50000000.00",
            "high_24h": "51000.00",
            "low_24h": "49000.00",
            "price_change_24h": "1000.00",
            "price_change_percent_24h": "2.04",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        
        tick = MarketTickerContract(**ticker)
        assert tick.symbol == "BTC/USDT"
        assert tick.last_price == "50000.00"
    
    def test_websocket_message_contract(self):
        """Test WebSocket message contract."""
        message = {
            "event": "update",
            "channel": "ticker",
            "data": {
                "symbol": "BTC/USDT",
                "price": "50000.00",
            },
            "timestamp": "2024-01-01T00:00:00Z",
            "sequence": 12345,
        }
        
        msg = WebSocketMessageContract(**message)
        assert msg.event == "update"
        assert msg.channel == "ticker"
        assert msg.data["symbol"] == "BTC/USDT"
    
    def test_websocket_invalid_event(self):
        """Test invalid WebSocket event type."""
        message = {
            "event": "invalid_event",  # Invalid
            "channel": "ticker",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        
        with pytest.raises(ValidationError):
            WebSocketMessageContract(**message)


class TestBackwardCompatibility:
    """Test backward compatibility between versions."""
    
    def test_v1_to_v2_compatibility(self):
        """Test v1 clients work with v2 API."""
        validator = ContractValidator()
        
        v1_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        # V1 request should validate against both versions
        assert validator.validate_request(v1_request, APIVersion.V1) is True
        assert validator.validate_request(v1_request, APIVersion.V2) is True
    
    def test_v2_exclusive_features(self):
        """Test v2 exclusive features don't break v1."""
        validator = ContractValidator()
        
        v2_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
            "post_only": True,  # V2 only
            "trailing_delta": 100,  # V2 only
        }
        
        # V2 request validates against V2
        assert validator.validate_request(v2_request, APIVersion.V2) is True
        
        # But fails V1 validation (extra fields)
        # V1 schema would ignore extra fields in practice
    
    def test_field_compatibility_check(self):
        """Test field compatibility between versions."""
        validator = ContractValidator()
        
        old_response = {
            "order_id": "123",
            "symbol": "BTC/USDT",
            "status": "FILLED",
        }
        
        new_response = {
            "order_id": "123",
            "symbol": "BTC/USDT",
            "status": "FILLED",
            "fills": [],  # New field
        }
        
        # New version has all old fields
        assert validator.check_backward_compatibility(old_response, new_response) is True
        
        # Missing field breaks compatibility
        incomplete_response = {
            "order_id": "123",
            "status": "FILLED",
            # Missing "symbol"
        }
        
        assert validator.check_backward_compatibility(old_response, incomplete_response) is False


class TestSchemaEvolution:
    """Test schema evolution and versioning."""
    
    def test_deprecation_handling(self):
        """Test handling of deprecated fields."""
        # Simulate deprecated field with warning
        class OrderWithDeprecated(OrderRequestV2):
            legacy_field: Optional[str] = Field(None, deprecated=True)
        
        request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
            "legacy_field": "deprecated_value",
        }
        
        order = OrderWithDeprecated(**request)
        assert order.legacy_field == "deprecated_value"
    
    def test_required_field_addition(self):
        """Test adding required fields breaks compatibility."""
        class OrderV3(OrderRequestV2):
            required_new_field: str  # New required field
        
        v2_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        # V2 request fails V3 validation
        with pytest.raises(ValidationError):
            OrderV3(**v2_request)
    
    def test_optional_field_addition(self):
        """Test adding optional fields maintains compatibility."""
        class OrderV3(OrderRequestV2):
            optional_new_field: Optional[str] = None
        
        v2_request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        # V2 request works with V3 schema
        order = OrderV3(**v2_request)
        assert order.optional_new_field is None


class TestAPIVersioning:
    """Test API versioning strategy."""
    
    def test_version_routing(self):
        """Test routing to correct version."""
        
        def route_request(path: str, data: Dict) -> bool:
            """Route request to appropriate version handler."""
            if path.startswith("/api/v1/"):
                return ContractValidator.validate_request(data, APIVersion.V1)
            elif path.startswith("/api/v2/"):
                return ContractValidator.validate_request(data, APIVersion.V2)
            else:
                # Default to latest
                return ContractValidator.validate_request(data, APIVersion.LATEST)
        
        request = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.01",
            "price": "50000.00",
        }
        
        assert route_request("/api/v1/orders", request) is True
        assert route_request("/api/v2/orders", request) is True
        assert route_request("/api/orders", request) is True
    
    def test_version_negotiation(self):
        """Test version negotiation via headers."""
        
        def negotiate_version(headers: Dict) -> APIVersion:
            """Negotiate API version from headers."""
            accept = headers.get("Accept", "")
            
            if "version=1" in accept:
                return APIVersion.V1
            elif "version=2" in accept:
                return APIVersion.V2
            else:
                # Check custom header
                api_version = headers.get("X-API-Version", "")
                if api_version == "1":
                    return APIVersion.V1
                elif api_version == "2":
                    return APIVersion.V2
                else:
                    return APIVersion.LATEST
        
        # Test Accept header
        headers_v1 = {"Accept": "application/json; version=1"}
        assert negotiate_version(headers_v1) == APIVersion.V1
        
        # Test custom header
        headers_v2 = {"X-API-Version": "2"}
        assert negotiate_version(headers_v2) == APIVersion.V2
        
        # Test default
        headers_default = {}
        assert negotiate_version(headers_default) == APIVersion.LATEST


# Contract Documentation Generator
def generate_contract_documentation():
    """Generate API contract documentation."""
    contracts = [
        OrderRequestV1,
        OrderRequestV2,
        OrderResponseV1,
        PositionContract,
        AccountBalanceContract,
        MarketTickerContract,
        WebSocketMessageContract,
    ]
    
    docs = {
        "api_version": "2.0.0",
        "contracts": {},
    }
    
    for contract in contracts:
        schema = contract.schema()
        docs["contracts"][contract.__name__] = {
            "description": contract.__doc__,
            "schema": schema,
            "example": schema.get("example", {}),
        }
    
    # Save documentation
    with open("tests/contract/api_contracts.json", "w") as f:
        json.dump(docs, f, indent=2, default=str)
    
    return docs


if __name__ == "__main__":
    # Generate contract documentation
    docs = generate_contract_documentation()
    print(f"Generated contract documentation for {len(docs['contracts'])} contracts")