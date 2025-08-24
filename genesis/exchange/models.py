"""
Exchange data models for Project GENESIS.

This module contains all Pydantic models used for exchange interactions,
including request/response validation and data structures.
"""

from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo


class OrderRequest(BaseModel):
    """Validated order placement request."""
    
    symbol: str = Field(..., description="Trading pair (e.g., BTC/USDT)")
    side: str = Field(..., pattern="^(buy|sell)$")
    type: str = Field(..., pattern="^(market|limit|stop_limit)$")
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = Field(None, gt=0)
    stop_price: Optional[Decimal] = Field(None, gt=0)
    client_order_id: Optional[str] = Field(None, min_length=1, max_length=36)
    
    @field_validator("quantity", "price", "stop_price", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v
    
    @model_validator(mode='after')
    def validate_price_for_limit(self):
        """Ensure price is provided for limit orders."""
        if self.type == "limit" and self.price is None:
            raise ValueError("Price required for limit orders")
        return self


class OrderResponse(BaseModel):
    """Order placement/query response."""
    
    order_id: str = Field(..., description="Exchange order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str
    side: str
    type: str
    price: Optional[Decimal]
    quantity: Decimal
    filled_quantity: Decimal = Field(default=Decimal("0"))
    status: str = Field(..., description="Order status")
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @field_validator("price", "quantity", "filled_quantity", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v


class MarketTicker(BaseModel):
    """24hr market ticker data."""
    
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    volume_24h: Decimal
    quote_volume_24h: Decimal
    price_change_percent: Decimal
    high_24h: Decimal
    low_24h: Decimal
    
    @field_validator("last_price", "bid_price", "ask_price", "volume_24h", 
                    "quote_volume_24h", "price_change_percent", "high_24h", "low_24h", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class OrderBook(BaseModel):
    """Order book snapshot."""
    
    symbol: str
    bids: List[List[Decimal]]  # [[price, quantity], ...]
    asks: List[List[Decimal]]  # [[price, quantity], ...]
    timestamp: datetime
    
    @field_validator("bids", "asks", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert all values to Decimal."""
        return [[Decimal(str(price)), Decimal(str(qty))] for price, qty in v]


class AccountBalance(BaseModel):
    """Account balance information."""
    
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal
    
    @field_validator("free", "locked", "total", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class KlineData(BaseModel):
    """Candlestick/Kline data."""
    
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int
    
    @field_validator("open", "high", "low", "close", "volume", "quote_volume", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))