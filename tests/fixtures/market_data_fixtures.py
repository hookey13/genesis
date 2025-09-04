"""Market data fixtures for testing."""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import json


def generate_orderbook(symbol: str, mid_price: Decimal, spread: Decimal = Decimal("10"), levels: int = 10):
    """Generate a realistic order book."""
    bids = []
    asks = []
    
    for i in range(levels):
        bid_price = mid_price - spread * (i + 1)
        ask_price = mid_price + spread * (i + 1)
        
        bid_size = Decimal(str(random.uniform(0.1, 10.0)))
        ask_size = Decimal(str(random.uniform(0.1, 10.0)))
        
        bids.append({
            "price": bid_price,
            "quantity": bid_size,
            "count": random.randint(1, 10)
        })
        
        asks.append({
            "price": ask_price,
            "quantity": ask_size,
            "count": random.randint(1, 10)
        })
    
    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "timestamp": datetime.now().timestamp()
    }


def generate_trade_stream(symbol: str, base_price: Decimal, duration_minutes: int = 60):
    """Generate a stream of trades."""
    trades = []
    current_price = base_price
    start_time = datetime.now() - timedelta(minutes=duration_minutes)
    
    for i in range(duration_minutes * 20):  # ~20 trades per minute
        # Random walk
        price_change = Decimal(str(random.uniform(-0.001, 0.001)))
        current_price = current_price * (Decimal("1") + price_change)
        
        trade = {
            "symbol": symbol,
            "price": current_price,
            "quantity": Decimal(str(random.uniform(0.001, 1.0))),
            "side": random.choice(["buy", "sell"]),
            "timestamp": (start_time + timedelta(seconds=i * 3)).timestamp(),
            "trade_id": f"trade_{i}"
        }
        
        trades.append(trade)
    
    return trades


def generate_ohlcv_candles(symbol: str, base_price: Decimal, periods: int = 100):
    """Generate OHLCV candlestick data."""
    candles = []
    current_price = base_price
    start_time = datetime.now() - timedelta(minutes=periods)
    
    for i in range(periods):
        open_price = current_price
        high_price = current_price * Decimal("1.002")
        low_price = current_price * Decimal("0.998")
        close_price = current_price * (Decimal("1") + Decimal(str(random.uniform(-0.001, 0.001))))
        volume = Decimal(str(random.uniform(100, 1000)))
        
        candle = {
            "symbol": symbol,
            "timestamp": (start_time + timedelta(minutes=i)).timestamp(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        
        candles.append(candle)
        current_price = close_price
    
    return candles


def generate_market_depth_snapshot():
    """Generate a complete market depth snapshot."""
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    base_prices = {
        "BTC/USDT": Decimal("50000"),
        "ETH/USDT": Decimal("3000"),
        "BNB/USDT": Decimal("400"),
        "SOL/USDT": Decimal("100"),
        "ADA/USDT": Decimal("0.5")
    }
    
    snapshot = {}
    
    for symbol in symbols:
        snapshot[symbol] = {
            "orderbook": generate_orderbook(symbol, base_prices[symbol]),
            "last_trade": {
                "price": base_prices[symbol],
                "quantity": Decimal("0.1"),
                "timestamp": datetime.now().timestamp()
            },
            "ticker": {
                "bid": base_prices[symbol] - Decimal("10"),
                "ask": base_prices[symbol] + Decimal("10"),
                "last": base_prices[symbol],
                "volume_24h": Decimal("10000"),
                "high_24h": base_prices[symbol] * Decimal("1.05"),
                "low_24h": base_prices[symbol] * Decimal("0.95"),
                "change_24h": Decimal("0.02")
            }
        }
    
    return snapshot


def generate_volatile_market_data(symbol: str, base_price: Decimal, volatility: Decimal = Decimal("0.1")):
    """Generate highly volatile market data for stress testing."""
    data_points = []
    current_price = base_price
    
    for i in range(1000):
        # Large random moves
        volatility_factor = Decimal(str(random.uniform(-float(volatility), float(volatility))))
        current_price = current_price * (Decimal("1") + volatility_factor)
        
        # Ensure price doesn't go negative
        if current_price < Decimal("1"):
            current_price = Decimal("1")
        
        data_point = {
            "symbol": symbol,
            "price": current_price,
            "bid": current_price - Decimal("1"),
            "ask": current_price + Decimal("1"),
            "volume": Decimal(str(random.uniform(0, 1000))),
            "timestamp": datetime.now().timestamp() + i
        }
        
        data_points.append(data_point)
    
    return data_points


def generate_correlated_pairs_data():
    """Generate data for correlated trading pairs."""
    base_btc = Decimal("50000")
    correlation_factor = Decimal("0.8")
    
    data = {}
    
    # BTC as base
    data["BTC/USDT"] = {
        "price": base_btc,
        "change": Decimal("0")
    }
    
    # Correlated pairs
    btc_change = Decimal(str(random.uniform(-0.05, 0.05)))
    
    data["ETH/USDT"] = {
        "price": Decimal("3000") * (Decimal("1") + btc_change * correlation_factor),
        "change": btc_change * correlation_factor
    }
    
    data["BCH/USDT"] = {
        "price": Decimal("500") * (Decimal("1") + btc_change * Decimal("0.9")),
        "change": btc_change * Decimal("0.9")
    }
    
    data["LTC/USDT"] = {
        "price": Decimal("150") * (Decimal("1") + btc_change * Decimal("0.7")),
        "change": btc_change * Decimal("0.7")
    }
    
    return data


def generate_arbitrage_opportunity():
    """Generate market data with arbitrage opportunity."""
    base_price = Decimal("50000")
    spread = Decimal("50")  # $50 arbitrage opportunity
    
    return {
        "exchange_a": {
            "BTC/USDT": {
                "bid": base_price - spread,
                "ask": base_price - spread + Decimal("10"),
                "last": base_price - spread + Decimal("5")
            }
        },
        "exchange_b": {
            "BTC/USDT": {
                "bid": base_price + spread - Decimal("10"),
                "ask": base_price + spread,
                "last": base_price + spread - Decimal("5")
            }
        },
        "opportunity": {
            "buy_exchange": "exchange_a",
            "sell_exchange": "exchange_b",
            "profit": spread - Decimal("20"),  # Minus fees
            "profit_pct": (spread - Decimal("20")) / base_price
        }
    }


def generate_market_crash_scenario():
    """Generate market crash scenario data."""
    prices = []
    base_price = Decimal("50000")
    
    # Normal market
    for i in range(100):
        price = base_price + Decimal(str(random.uniform(-100, 100)))
        prices.append({
            "timestamp": i,
            "price": price,
            "volume": Decimal("100")
        })
    
    # Crash begins
    crash_price = base_price
    for i in range(100, 150):
        crash_price = crash_price * Decimal("0.98")  # 2% drop each tick
        prices.append({
            "timestamp": i,
            "price": crash_price,
            "volume": Decimal("1000")  # High volume during crash
        })
    
    # Recovery
    for i in range(150, 200):
        crash_price = crash_price * Decimal("1.01")  # 1% recovery each tick
        prices.append({
            "timestamp": i,
            "price": crash_price,
            "volume": Decimal("500")
        })
    
    return prices


def load_historical_data(filepath: str = None):
    """Load historical market data from file."""
    if filepath:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # Return sample data if no file provided
    return {
        "BTC/USDT": generate_ohlcv_candles("BTC/USDT", Decimal("50000"), 1440),  # 24 hours
        "ETH/USDT": generate_ohlcv_candles("ETH/USDT", Decimal("3000"), 1440)
    }