"""
Technical Indicators Library for Trading Strategies.

This module provides a collection of technical analysis indicators
used by various trading strategies in the GENESIS system.
"""

from collections import deque
from decimal import Decimal
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


def calculate_sma(prices: List[Decimal], period: int) -> Decimal:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: List of price values
        period: Number of periods for the average
        
    Returns:
        Simple moving average value
    """
    if not prices or len(prices) < period:
        return Decimal("0")
    
    recent_prices = prices[-period:] if len(prices) > period else prices
    return sum(recent_prices) / len(recent_prices)


def calculate_ema(
    current_price: Decimal,
    previous_ema: Optional[Decimal],
    period: int
) -> Decimal:
    """
    Calculate Exponential Moving Average.
    
    Args:
        current_price: Current price value
        previous_ema: Previous EMA value (None for first calculation)
        period: Number of periods for the EMA
        
    Returns:
        Exponential moving average value
    """
    multiplier = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
    
    if previous_ema is None:
        # First EMA is same as current price
        return current_price
    
    ema = (current_price * multiplier) + (previous_ema * (Decimal("1") - multiplier))
    return ema


def calculate_atr(
    highs: List[Decimal],
    lows: List[Decimal],
    closes: List[Decimal],
    period: int = 14,
    previous_atr: Optional[Decimal] = None
) -> Decimal:
    """
    Calculate Average True Range.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (default 14)
        previous_atr: Previous ATR for smoothing
        
    Returns:
        Average True Range value
    """
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return Decimal("0")
    
    # Calculate True Ranges
    true_ranges = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)
    
    if not true_ranges:
        return Decimal("0")
    
    # Calculate ATR
    if previous_atr is None:
        # Initial ATR is simple average
        atr = sum(true_ranges[-period:]) / min(len(true_ranges), period)
    else:
        # Smoothed ATR
        current_tr = true_ranges[-1]
        atr = ((previous_atr * (period - 1)) + current_tr) / period
    
    return atr


def calculate_adx(
    highs: List[Decimal],
    lows: List[Decimal],
    closes: List[Decimal],
    period: int = 14
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate Average Directional Index with +DI and -DI.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ADX period (default 14)
        
    Returns:
        Tuple of (ADX, +DI, -DI) values
    """
    if len(highs) < period + 1 or len(lows) < period + 1:
        return Decimal("0"), Decimal("0"), Decimal("0")
    
    # Calculate directional movements
    plus_dm_list = []
    minus_dm_list = []
    tr_list = []
    
    for i in range(1, len(highs)):
        # Directional movement
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        
        plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
        minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")
        
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        
        # True Range
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        true_range = max(high_low, high_close, low_close)
        tr_list.append(true_range)
    
    if len(tr_list) < period:
        return Decimal("0"), Decimal("0"), Decimal("0")
    
    # Smooth the DMs and TR
    smooth_plus_dm = sum(plus_dm_list[-period:]) / period
    smooth_minus_dm = sum(minus_dm_list[-period:]) / period
    smooth_tr = sum(tr_list[-period:]) / period
    
    # Calculate DI values
    plus_di = Decimal("0")
    minus_di = Decimal("0")
    
    if smooth_tr > 0:
        plus_di = (smooth_plus_dm / smooth_tr) * Decimal("100")
        minus_di = (smooth_minus_dm / smooth_tr) * Decimal("100")
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    if di_sum > 0:
        dx = abs(plus_di - minus_di) / di_sum * Decimal("100")
    else:
        dx = Decimal("0")
    
    # For simplicity, using DX as ADX (should be smoothed over time)
    adx = dx
    
    return adx, plus_di, minus_di


def calculate_volume_average(volumes: List[Decimal], period: int = 20) -> Decimal:
    """
    Calculate average volume over a period.
    
    Args:
        volumes: List of volume values
        period: Number of periods for the average
        
    Returns:
        Average volume value
    """
    if not volumes:
        return Decimal("0")
    
    recent_volumes = volumes[-period:] if len(volumes) > period else volumes
    return sum(recent_volumes) / len(recent_volumes)


def calculate_rsi(prices: List[Decimal], period: int = 14) -> Decimal:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: List of price values
        period: RSI period (default 14)
        
    Returns:
        RSI value between 0 and 100
    """
    if len(prices) < period + 1:
        return Decimal("50")  # Neutral RSI
    
    # Calculate price changes
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))
    
    # Calculate average gain and loss
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return Decimal("100")  # Maximum RSI
    
    rs = avg_gain / avg_loss
    rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
    
    return rsi


def calculate_bollinger_bands(
    prices: List[Decimal],
    period: int = 20,
    std_dev_multiplier: Decimal = Decimal("2")
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: List of price values
        period: Period for moving average (default 20)
        std_dev_multiplier: Standard deviation multiplier (default 2)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        return Decimal("0"), Decimal("0"), Decimal("0")
    
    # Calculate middle band (SMA)
    middle_band = calculate_sma(prices, period)
    
    # Calculate standard deviation
    recent_prices = prices[-period:]
    squared_diffs = [(price - middle_band) ** 2 for price in recent_prices]
    variance = sum(squared_diffs) / len(squared_diffs)
    std_dev = variance.sqrt()
    
    # Calculate bands
    upper_band = middle_band + (std_dev * std_dev_multiplier)
    lower_band = middle_band - (std_dev * std_dev_multiplier)
    
    return upper_band, middle_band, lower_band


def calculate_macd(
    prices: List[Decimal],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of price values
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(prices) < slow_period:
        return Decimal("0"), Decimal("0"), Decimal("0")
    
    # Calculate EMAs
    fast_ema = Decimal("0")
    slow_ema = Decimal("0")
    
    # Initialize EMAs with SMA
    if len(prices) >= fast_period:
        fast_ema = calculate_sma(prices[:fast_period], fast_period)
        for price in prices[fast_period:]:
            fast_ema = calculate_ema(price, fast_ema, fast_period)
    
    if len(prices) >= slow_period:
        slow_ema = calculate_sma(prices[:slow_period], slow_period)
        for price in prices[slow_period:]:
            slow_ema = calculate_ema(price, slow_ema, slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # For simplicity, signal line is set to 0 (would need history of MACD values)
    signal_line = Decimal("0")
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(
    highs: List[Decimal],
    lows: List[Decimal],
    closes: List[Decimal],
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[Decimal, Decimal]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: Lookback period (default 14)
        smooth_k: K line smoothing period (default 3)
        smooth_d: D line smoothing period (default 3)
        
    Returns:
        Tuple of (%K, %D) values
    """
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return Decimal("50"), Decimal("50")  # Neutral values
    
    # Calculate %K
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    current_close = closes[-1]
    
    highest_high = max(recent_highs)
    lowest_low = min(recent_lows)
    
    if highest_high == lowest_low:
        k_percent = Decimal("50")
    else:
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * Decimal("100")
    
    # For simplicity, %D is same as %K (would need history for proper smoothing)
    d_percent = k_percent
    
    return k_percent, d_percent


def calculate_pivot_points(
    high: Decimal,
    low: Decimal,
    close: Decimal
) -> dict[str, Decimal]:
    """
    Calculate pivot points for support and resistance levels.
    
    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close
        
    Returns:
        Dictionary with pivot point levels
    """
    pivot = (high + low + close) / Decimal("3")
    
    r1 = (Decimal("2") * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + Decimal("2") * (pivot - low)
    
    s1 = (Decimal("2") * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - Decimal("2") * (high - pivot)
    
    return {
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "s1": s1,
        "s2": s2,
        "s3": s3
    }


def calculate_vwap(
    prices: List[Decimal],
    volumes: List[Decimal]
) -> Decimal:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        prices: List of price values
        volumes: List of volume values
        
    Returns:
        VWAP value
    """
    if not prices or not volumes or len(prices) != len(volumes):
        return Decimal("0")
    
    total_value = sum(p * v for p, v in zip(prices, volumes))
    total_volume = sum(volumes)
    
    if total_volume == 0:
        return Decimal("0")
    
    return total_value / total_volume


def calculate_obv(
    closes: List[Decimal],
    volumes: List[Decimal]
) -> Decimal:
    """
    Calculate On-Balance Volume.
    
    Args:
        closes: List of close prices
        volumes: List of volume values
        
    Returns:
        OBV value
    """
    if len(closes) < 2 or len(volumes) < 2:
        return Decimal("0")
    
    obv = Decimal("0")
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        # If prices are equal, OBV stays the same
    
    return obv


def detect_support_resistance(
    highs: List[Decimal],
    lows: List[Decimal],
    lookback: int = 20,
    min_touches: int = 2
) -> Tuple[List[Decimal], List[Decimal]]:
    """
    Detect support and resistance levels.
    
    Args:
        highs: List of high prices
        lows: List of low prices
        lookback: Number of periods to look back
        min_touches: Minimum touches to confirm level
        
    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    if len(highs) < lookback or len(lows) < lookback:
        return [], []
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Simple implementation: find local extremes
    resistance_levels = []
    support_levels = []
    
    # Find resistance levels (local highs)
    for i in range(1, len(recent_highs) - 1):
        if recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i + 1]:
            resistance_levels.append(recent_highs[i])
    
    # Find support levels (local lows)
    for i in range(1, len(recent_lows) - 1):
        if recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i + 1]:
            support_levels.append(recent_lows[i])
    
    # Remove duplicates and sort
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    support_levels = sorted(list(set(support_levels)))
    
    return support_levels, resistance_levels