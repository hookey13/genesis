"""
Unit tests for technical indicators module.

Tests all indicator calculations for accuracy and edge cases.
"""

from decimal import Decimal

import pytest

from genesis.analytics import technical_indicators as ti


class TestSimpleIndicators:
    """Tests for simple technical indicators."""
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        prices = [Decimal(str(i)) for i in [100, 102, 101, 103, 105]]
        
        # Test 3-period SMA
        sma = ti.calculate_sma(prices, 3)
        expected = (Decimal("101") + Decimal("103") + Decimal("105")) / 3
        assert sma == expected
        
        # Test with period longer than data
        sma = ti.calculate_sma(prices, 10)
        expected = sum(prices) / len(prices)
        assert sma == expected
    
    def test_sma_empty_data(self):
        """Test SMA with empty data."""
        sma = ti.calculate_sma([], 5)
        assert sma == Decimal("0")
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        # First EMA (no previous)
        ema = ti.calculate_ema(Decimal("100"), None, 10)
        assert ema == Decimal("100")
        
        # Subsequent EMA
        ema = ti.calculate_ema(Decimal("105"), Decimal("100"), 10)
        multiplier = Decimal("2") / (Decimal("10") + Decimal("1"))
        expected = (Decimal("105") * multiplier) + (Decimal("100") * (Decimal("1") - multiplier))
        assert abs(ema - expected) < Decimal("0.01")
    
    def test_volume_average(self):
        """Test volume average calculation."""
        volumes = [Decimal(str(v)) for v in [1000, 1200, 800, 1100, 900]]
        
        avg = ti.calculate_volume_average(volumes, 3)
        expected = (Decimal("800") + Decimal("1100") + Decimal("900")) / 3
        assert abs(avg - expected) < Decimal("0.01")
    
    def test_volume_average_empty(self):
        """Test volume average with empty data."""
        avg = ti.calculate_volume_average([], 5)
        assert avg == Decimal("0")


class TestATRCalculation:
    """Tests for Average True Range calculation."""
    
    def test_atr_basic(self):
        """Test basic ATR calculation."""
        highs = [Decimal(str(h)) for h in [102, 103, 101, 104, 105]]
        lows = [Decimal(str(l)) for l in [98, 99, 97, 100, 101]]
        closes = [Decimal(str(c)) for c in [100, 101, 99, 102, 103]]
        
        atr = ti.calculate_atr(highs, lows, closes, period=3)
        assert atr > Decimal("0")
        assert atr < Decimal("10")  # Reasonable range
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        highs = [Decimal("100")]
        lows = [Decimal("99")]
        closes = [Decimal("99.5")]
        
        atr = ti.calculate_atr(highs, lows, closes)
        assert atr == Decimal("0")
    
    def test_atr_with_previous(self):
        """Test ATR calculation with previous value (smoothing)."""
        highs = [Decimal(str(h)) for h in [102, 103, 101, 104, 105]]
        lows = [Decimal(str(l)) for l in [98, 99, 97, 100, 101]]
        closes = [Decimal(str(c)) for c in [100, 101, 99, 102, 103]]
        
        # First ATR
        atr1 = ti.calculate_atr(highs[:-1], lows[:-1], closes[:-1], period=3)
        
        # Second ATR with smoothing
        atr2 = ti.calculate_atr(highs, lows, closes, period=3, previous_atr=atr1)
        
        # Should be smoothed (different from fresh calculation)
        atr_fresh = ti.calculate_atr(highs, lows, closes, period=3)
        assert atr2 != atr_fresh


class TestADXCalculation:
    """Tests for Average Directional Index calculation."""
    
    def test_adx_trending_market(self):
        """Test ADX in trending market."""
        # Create uptrend data
        highs = [Decimal(str(100 + i * 2)) for i in range(20)]
        lows = [Decimal(str(98 + i * 2)) for i in range(20)]
        closes = [Decimal(str(99 + i * 2)) for i in range(20)]
        
        adx, plus_di, minus_di = ti.calculate_adx(highs, lows, closes)
        
        # In uptrend, +DI should be > -DI
        assert plus_di > minus_di
        assert adx > Decimal("0")
    
    def test_adx_ranging_market(self):
        """Test ADX in ranging market."""
        # Create ranging data
        highs = []
        lows = []
        closes = []
        
        for i in range(20):
            price = Decimal("100") + (Decimal(str(i % 3)) - Decimal("1"))
            highs.append(price + Decimal("1"))
            lows.append(price - Decimal("1"))
            closes.append(price)
        
        adx, plus_di, minus_di = ti.calculate_adx(highs, lows, closes)
        
        # In ranging market, ADX should be low
        assert adx < Decimal("25")
    
    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        highs = [Decimal("100"), Decimal("101")]
        lows = [Decimal("99"), Decimal("100")]
        closes = [Decimal("99.5"), Decimal("100.5")]
        
        adx, plus_di, minus_di = ti.calculate_adx(highs, lows, closes)
        
        assert adx == Decimal("0")
        assert plus_di == Decimal("0")
        assert minus_di == Decimal("0")


class TestRSICalculation:
    """Tests for Relative Strength Index calculation."""
    
    def test_rsi_overbought(self):
        """Test RSI in overbought conditions."""
        # Create strong uptrend
        prices = [Decimal(str(100 + i * 2)) for i in range(20)]
        
        rsi = ti.calculate_rsi(prices)
        assert rsi > Decimal("70")  # Overbought
    
    def test_rsi_oversold(self):
        """Test RSI in oversold conditions."""
        # Create strong downtrend
        prices = [Decimal(str(100 - i * 2)) for i in range(20)]
        
        rsi = ti.calculate_rsi(prices)
        assert rsi < Decimal("30")  # Oversold
    
    def test_rsi_neutral(self):
        """Test RSI in neutral conditions."""
        # Create sideways movement
        prices = []
        for i in range(20):
            prices.append(Decimal("100") + (Decimal(str(i % 3)) - Decimal("1")))
        
        rsi = ti.calculate_rsi(prices)
        assert Decimal("40") < rsi < Decimal("60")  # Neutral range
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = [Decimal("100"), Decimal("101")]
        
        rsi = ti.calculate_rsi(prices)
        assert rsi == Decimal("50")  # Default neutral


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        prices = [Decimal(str(p)) for p in [100, 102, 98, 101, 99, 103, 97, 102, 100, 101]]
        
        upper, middle, lower = ti.calculate_bollinger_bands(prices, period=5)
        
        # Middle band should be SMA
        assert middle == ti.calculate_sma(prices, 5)
        
        # Upper band > middle > lower
        assert upper > middle
        assert middle > lower
    
    def test_bollinger_bands_tight(self):
        """Test Bollinger Bands with low volatility."""
        # Low volatility data
        prices = [Decimal("100") + Decimal(str(i % 2)) * Decimal("0.1") for i in range(20)]
        
        upper, middle, lower = ti.calculate_bollinger_bands(prices)
        
        # Bands should be tight
        band_width = upper - lower
        assert band_width < Decimal("5")
    
    def test_bollinger_bands_wide(self):
        """Test Bollinger Bands with high volatility."""
        # High volatility data
        prices = [Decimal("100") + Decimal(str(i % 5)) * Decimal("5") for i in range(20)]
        
        upper, middle, lower = ti.calculate_bollinger_bands(prices)
        
        # Bands should be wide
        band_width = upper - lower
        assert band_width > Decimal("5")
    
    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        prices = [Decimal("100"), Decimal("101")]
        
        upper, middle, lower = ti.calculate_bollinger_bands(prices, period=5)
        
        assert upper == Decimal("0")
        assert middle == Decimal("0")
        assert lower == Decimal("0")


class TestMACDCalculation:
    """Tests for MACD calculation."""
    
    def test_macd_basic(self):
        """Test basic MACD calculation."""
        # Create trend data
        prices = [Decimal(str(100 + i)) for i in range(30)]
        
        macd_line, signal_line, histogram = ti.calculate_macd(prices)
        
        # MACD should be positive in uptrend
        assert macd_line > Decimal("0")
        assert histogram == macd_line  # Since signal_line is 0 in simple implementation
    
    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = [Decimal("100") for i in range(10)]
        
        macd_line, signal_line, histogram = ti.calculate_macd(prices)
        
        assert macd_line == Decimal("0")
        assert signal_line == Decimal("0")
        assert histogram == Decimal("0")


class TestStochasticOscillator:
    """Tests for Stochastic Oscillator calculation."""
    
    def test_stochastic_overbought(self):
        """Test Stochastic in overbought zone."""
        # Price at upper range
        highs = [Decimal(str(100 + i)) for i in range(14)]
        lows = [Decimal(str(98 + i)) for i in range(14)]
        closes = [Decimal(str(99 + i)) for i in range(14)]
        closes[-1] = highs[-1]  # Close at high
        
        k_percent, d_percent = ti.calculate_stochastic(highs, lows, closes)
        
        assert k_percent > Decimal("80")  # Overbought
    
    def test_stochastic_oversold(self):
        """Test Stochastic in oversold zone."""
        # Price at lower range
        highs = [Decimal(str(102 - i * 0.5)) for i in range(14)]
        lows = [Decimal(str(98 - i * 0.5)) for i in range(14)]
        closes = [Decimal(str(100 - i * 0.5)) for i in range(14)]
        closes[-1] = lows[-1]  # Close at low
        
        k_percent, d_percent = ti.calculate_stochastic(highs, lows, closes)
        
        assert k_percent < Decimal("20")  # Oversold
    
    def test_stochastic_neutral(self):
        """Test Stochastic in neutral zone."""
        # Price in middle of range
        highs = [Decimal("102") for _ in range(14)]
        lows = [Decimal("98") for _ in range(14)]
        closes = [Decimal("100") for _ in range(14)]
        
        k_percent, d_percent = ti.calculate_stochastic(highs, lows, closes)
        
        assert Decimal("40") < k_percent < Decimal("60")  # Neutral


class TestPivotPoints:
    """Tests for Pivot Points calculation."""
    
    def test_pivot_points_basic(self):
        """Test basic pivot points calculation."""
        high = Decimal("105")
        low = Decimal("95")
        close = Decimal("100")
        
        levels = ti.calculate_pivot_points(high, low, close)
        
        # Check pivot calculation
        expected_pivot = (high + low + close) / Decimal("3")
        assert levels["pivot"] == expected_pivot
        
        # Check resistance levels are above pivot
        assert levels["r1"] > levels["pivot"]
        assert levels["r2"] > levels["r1"]
        assert levels["r3"] > levels["r2"]
        
        # Check support levels are below pivot
        assert levels["s1"] < levels["pivot"]
        assert levels["s2"] < levels["s1"]
        assert levels["s3"] < levels["s2"]


class TestVWAPCalculation:
    """Tests for Volume Weighted Average Price calculation."""
    
    def test_vwap_basic(self):
        """Test basic VWAP calculation."""
        prices = [Decimal("100"), Decimal("101"), Decimal("102")]
        volumes = [Decimal("1000"), Decimal("2000"), Decimal("1500")]
        
        vwap = ti.calculate_vwap(prices, volumes)
        
        expected = (
            Decimal("100") * Decimal("1000") +
            Decimal("101") * Decimal("2000") +
            Decimal("102") * Decimal("1500")
        ) / Decimal("4500")
        
        assert abs(vwap - expected) < Decimal("0.01")
    
    def test_vwap_zero_volume(self):
        """Test VWAP with zero volume."""
        prices = [Decimal("100"), Decimal("101")]
        volumes = [Decimal("0"), Decimal("0")]
        
        vwap = ti.calculate_vwap(prices, volumes)
        assert vwap == Decimal("0")
    
    def test_vwap_mismatched_lengths(self):
        """Test VWAP with mismatched data lengths."""
        prices = [Decimal("100"), Decimal("101")]
        volumes = [Decimal("1000")]
        
        vwap = ti.calculate_vwap(prices, volumes)
        assert vwap == Decimal("0")


class TestOBVCalculation:
    """Tests for On-Balance Volume calculation."""
    
    def test_obv_uptrend(self):
        """Test OBV in uptrend."""
        closes = [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103")]
        volumes = [Decimal("1000"), Decimal("1200"), Decimal("1100"), Decimal("1300")]
        
        obv = ti.calculate_obv(closes, volumes)
        
        # OBV should be positive in uptrend
        expected = Decimal("1200") + Decimal("1100") + Decimal("1300")
        assert obv == expected
    
    def test_obv_downtrend(self):
        """Test OBV in downtrend."""
        closes = [Decimal("103"), Decimal("102"), Decimal("101"), Decimal("100")]
        volumes = [Decimal("1000"), Decimal("1200"), Decimal("1100"), Decimal("1300")]
        
        obv = ti.calculate_obv(closes, volumes)
        
        # OBV should be negative in downtrend
        expected = -Decimal("1200") - Decimal("1100") - Decimal("1300")
        assert obv == expected
    
    def test_obv_mixed(self):
        """Test OBV with mixed price action."""
        closes = [Decimal("100"), Decimal("101"), Decimal("100"), Decimal("102")]
        volumes = [Decimal("1000"), Decimal("1200"), Decimal("1100"), Decimal("1300")]
        
        obv = ti.calculate_obv(closes, volumes)
        
        # +1200 (up) -1100 (down) +1300 (up)
        expected = Decimal("1200") - Decimal("1100") + Decimal("1300")
        assert obv == expected


class TestSupportResistance:
    """Tests for support and resistance detection."""
    
    def test_support_resistance_detection(self):
        """Test detection of support and resistance levels."""
        # Create data with clear levels
        highs = []
        lows = []
        
        # Create pattern with resistance at 105 and support at 95
        for i in range(20):
            if i % 5 == 0:
                highs.append(Decimal("105"))  # Resistance
                lows.append(Decimal("95"))     # Support
            else:
                highs.append(Decimal("102"))
                lows.append(Decimal("98"))
        
        support_levels, resistance_levels = ti.detect_support_resistance(highs, lows)
        
        # Should detect levels
        assert len(support_levels) > 0
        assert len(resistance_levels) > 0
        
        # Check if key levels are detected
        assert any(abs(s - Decimal("95")) < Decimal("1") for s in support_levels)
        assert any(abs(r - Decimal("105")) < Decimal("1") for r in resistance_levels)
    
    def test_support_resistance_insufficient_data(self):
        """Test support/resistance with insufficient data."""
        highs = [Decimal("100"), Decimal("101")]
        lows = [Decimal("99"), Decimal("100")]
        
        support_levels, resistance_levels = ti.detect_support_resistance(highs, lows, lookback=10)
        
        assert len(support_levels) == 0
        assert len(resistance_levels) == 0