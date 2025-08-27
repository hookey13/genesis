"""Unit tests for VolumeAnalyzer."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

from genesis.analytics.volume_analyzer import (
    VolumeAnalyzer, VolumeProfile, VolumePrediction
)
from genesis.core.models import Symbol
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway


@pytest.fixture
def mock_exchange_gateway():
    """Create a mock exchange gateway."""
    gateway = Mock(spec=ExchangeGateway)
    gateway.get_klines = AsyncMock()
    return gateway


@pytest.fixture
def volume_analyzer(mock_exchange_gateway):
    """Create a VolumeAnalyzer instance with mock dependencies."""
    return VolumeAnalyzer(mock_exchange_gateway)


@pytest.fixture
def sample_klines_data():
    """Generate sample klines data for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    data = []
    
    for day in range(30):
        for hour in range(0, 24, 1):  # Hourly data
            timestamp = base_time + timedelta(days=day, hours=hour)
            volume = 1000 + (hour * 100) + np.random.randint(-200, 200)  # Volume pattern
            data.append([
                int(timestamp.timestamp() * 1000),  # timestamp in ms
                100.0,  # open
                101.0,  # high
                99.0,   # low
                100.5,  # close
                volume,  # volume
                0, 0, 0, 0, 0, 0  # other fields
            ])
    
    return data


class TestVolumeProfile:
    """Test VolumeProfile dataclass."""
    
    def test_get_bucket_percentage(self):
        """Test bucket percentage calculation."""
        profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('10000'),
            time_buckets={
                0: Decimal('1000'),
                30: Decimal('2000'),
                60: Decimal('3000'),
                90: Decimal('4000')
            }
        )
        
        # Test exact bucket
        assert profile.get_bucket_percentage(0) == Decimal('0.1')  # 1000/10000
        assert profile.get_bucket_percentage(30) == Decimal('0.2')  # 2000/10000
        
        # Test in-between values (should round down to bucket)
        assert profile.get_bucket_percentage(45) == Decimal('0.2')  # Uses 30-min bucket
        assert profile.get_bucket_percentage(75) == Decimal('0.3')  # Uses 60-min bucket
    
    def test_get_bucket_percentage_zero_volume(self):
        """Test bucket percentage with zero total volume."""
        profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('0'),
            time_buckets={}
        )
        
        assert profile.get_bucket_percentage(0) == Decimal('0')
    
    def test_cumulative_percentage(self):
        """Test cumulative percentage retrieval."""
        profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('10000'),
            time_buckets={},
            cumulative_percentages={
                0: Decimal('0.1'),
                30: Decimal('0.3'),
                60: Decimal('0.6'),
                90: Decimal('1.0')
            }
        )
        
        assert profile.get_cumulative_percentage(0) == Decimal('0.1')
        assert profile.get_cumulative_percentage(60) == Decimal('0.6')
        assert profile.get_cumulative_percentage(90) == Decimal('1.0')


class TestVolumeAnalyzer:
    """Test VolumeAnalyzer class."""
    
    @pytest.mark.asyncio
    async def test_fetch_historical_volume(self, volume_analyzer, mock_exchange_gateway, sample_klines_data):
        """Test fetching historical volume data."""
        mock_exchange_gateway.get_klines.return_value = sample_klines_data
        
        symbol = Symbol('BTC/USDT')
        df = await volume_analyzer.fetch_historical_volume(symbol, days=30, interval='30m')
        
        # Verify API call
        mock_exchange_gateway.get_klines.assert_called_once()
        call_args = mock_exchange_gateway.get_klines.call_args
        assert call_args.kwargs['symbol'] == 'BTC/USDT'
        assert call_args.kwargs['interval'] == '30m'
        assert call_args.kwargs['limit'] == 1000
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'volume' in df.columns
        assert 'minute_of_day' in df.columns
        assert 'date' in df.columns
        
        # Verify data processing
        assert len(df) == len(sample_klines_data)
        assert all(isinstance(v, Decimal) for v in df['volume'])
    
    @pytest.mark.asyncio
    async def test_calculate_typical_profile(self, volume_analyzer, mock_exchange_gateway):
        """Test calculation of typical volume profile."""
        # Create mock historical data
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=48, freq='30min', tz='UTC'),
            'volume': [Decimal('1000') for _ in range(48)],
            'minute_of_day': [i * 30 for i in range(48)],
            'date': pd.date_range(start='2024-01-01', periods=48, freq='30min', tz='UTC').date
        })
        
        with patch.object(volume_analyzer, 'fetch_historical_volume', return_value=mock_df):
            symbol = Symbol('BTC/USDT')
            profile = await volume_analyzer.calculate_typical_profile(symbol, lookback_days=1)
            
            # Verify profile structure
            assert isinstance(profile, VolumeProfile)
            assert profile.symbol == symbol
            assert profile.total_volume > 0
            assert len(profile.time_buckets) > 0
            assert len(profile.cumulative_percentages) > 0
            
            # Verify cumulative percentages are increasing
            prev_pct = Decimal('0')
            for minute in sorted(profile.cumulative_percentages.keys()):
                current_pct = profile.cumulative_percentages[minute]
                assert current_pct >= prev_pct
                prev_pct = current_pct
            
            # Last cumulative should be 1.0 or close to it
            last_cumulative = list(profile.cumulative_percentages.values())[-1]
            assert last_cumulative >= Decimal('0.99')
    
    @pytest.mark.asyncio
    async def test_calculate_typical_profile_caching(self, volume_analyzer, mock_exchange_gateway):
        """Test that typical profile is cached."""
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=48, freq='30min', tz='UTC'),
            'volume': [Decimal('1000') for _ in range(48)],
            'minute_of_day': [i * 30 for i in range(48)],
            'date': pd.date_range(start='2024-01-01', periods=48, freq='30min', tz='UTC').date
        })
        
        with patch.object(volume_analyzer, 'fetch_historical_volume', return_value=mock_df) as mock_fetch:
            symbol = Symbol('BTC/USDT')
            
            # First call should fetch data
            profile1 = await volume_analyzer.calculate_typical_profile(symbol, lookback_days=30)
            assert mock_fetch.call_count == 1
            
            # Second call should use cache
            profile2 = await volume_analyzer.calculate_typical_profile(symbol, lookback_days=30)
            assert mock_fetch.call_count == 1  # No additional fetch
            
            # Should return same profile
            assert profile1.date == profile2.date
            assert profile1.total_volume == profile2.total_volume
    
    @pytest.mark.asyncio
    async def test_predict_intraday_volume(self, volume_analyzer):
        """Test intraday volume prediction."""
        # Create mock profile
        mock_profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('48000'),
            time_buckets={i * 30: Decimal('1000') for i in range(48)},
            cumulative_percentages={}
        )
        
        # Mock recent data
        mock_recent_df = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc)],
            'volume': [Decimal('1200')],  # 20% higher than typical
            'minute_of_day': [0],
            'date': [datetime.now(timezone.utc).date()]
        })
        
        with patch.object(volume_analyzer, 'calculate_typical_profile', return_value=mock_profile):
            with patch.object(volume_analyzer, 'fetch_historical_volume', return_value=mock_recent_df):
                symbol = Symbol('BTC/USDT')
                current_time = datetime.now(timezone.utc).replace(hour=12, minute=0)
                
                prediction = await volume_analyzer.predict_intraday_volume(
                    symbol, current_time, horizon_hours=4
                )
                
                # Verify prediction structure
                assert isinstance(prediction, VolumePrediction)
                assert prediction.symbol == symbol
                assert len(prediction.predicted_buckets) > 0
                assert len(prediction.confidence_scores) == len(prediction.predicted_buckets)
                assert prediction.total_predicted > 0
                assert Decimal('0') < prediction.model_accuracy <= Decimal('1')
                
                # Verify confidence decreases with time
                confidences = list(prediction.confidence_scores.values())
                for i in range(1, len(confidences)):
                    assert confidences[i] <= confidences[i-1]
    
    @pytest.mark.asyncio
    async def test_get_optimal_participation_rate(self, volume_analyzer):
        """Test optimal participation rate calculation."""
        # Create mock prediction
        prediction = VolumePrediction(
            symbol=Symbol('BTC/USDT'),
            prediction_time=datetime.now(timezone.utc),
            predicted_buckets={
                0: Decimal('1000'),
                30: Decimal('2000'),
                60: Decimal('3000')
            },
            confidence_scores={
                0: Decimal('0.9'),
                30: Decimal('0.8'),
                60: Decimal('0.7')
            },
            total_predicted=Decimal('6000'),
            model_accuracy=Decimal('0.85')
        )
        
        target_volume = Decimal('600')  # 10% of total predicted
        max_participation = Decimal('0.15')
        
        rates = volume_analyzer.get_optimal_participation_rate(
            target_volume, prediction, max_participation
        )
        
        # Verify rates structure
        assert isinstance(rates, dict)
        assert len(rates) == len(prediction.predicted_buckets)
        
        # Verify all rates are within limits
        for bucket, rate in rates.items():
            assert Decimal('0') <= rate <= max_participation
        
        # Verify confidence adjustment
        assert rates[0] > rates[60]  # Higher confidence bucket should have higher rate
    
    @pytest.mark.asyncio
    async def test_analyze_volume_spike(self, volume_analyzer):
        """Test volume spike detection."""
        # Create mock profile with typical volume
        mock_profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('48000'),
            time_buckets={
                0: Decimal('1000'),
                30: Decimal('1000'),
                60: Decimal('1000')
            },
            cumulative_percentages={}
        )
        
        with patch.object(volume_analyzer, 'calculate_typical_profile', return_value=mock_profile):
            symbol = Symbol('BTC/USDT')
            
            # Test normal volume - no spike
            is_spike, deviation = await volume_analyzer.analyze_volume_spike(
                symbol, Decimal('1100'), 30
            )
            assert is_spike is False
            assert deviation == Decimal('1.1')
            
            # Test spike volume - 3x normal
            is_spike, deviation = await volume_analyzer.analyze_volume_spike(
                symbol, Decimal('3000'), 30
            )
            assert is_spike is True
            assert deviation == Decimal('3.0')
    
    @pytest.mark.asyncio
    async def test_analyze_volume_spike_no_typical_data(self, volume_analyzer):
        """Test volume spike detection with no typical data."""
        mock_profile = VolumeProfile(
            symbol=Symbol('BTC/USDT'),
            date=datetime.now(timezone.utc),
            total_volume=Decimal('0'),
            time_buckets={},
            cumulative_percentages={}
        )
        
        with patch.object(volume_analyzer, 'calculate_typical_profile', return_value=mock_profile):
            symbol = Symbol('BTC/USDT')
            
            is_spike, deviation = await volume_analyzer.analyze_volume_spike(
                symbol, Decimal('1000'), 30
            )
            
            assert is_spike is False
            assert deviation == Decimal('1.0')
    
    @pytest.mark.asyncio
    async def test_error_handling_fetch_historical(self, volume_analyzer, mock_exchange_gateway):
        """Test error handling in fetch_historical_volume."""
        mock_exchange_gateway.get_klines.side_effect = Exception("API Error")
        
        symbol = Symbol('BTC/USDT')
        
        with pytest.raises(Exception) as exc_info:
            await volume_analyzer.fetch_historical_volume(symbol, days=30)
        
        assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_error_handling_calculate_profile(self, volume_analyzer):
        """Test error handling in calculate_typical_profile."""
        with patch.object(volume_analyzer, 'fetch_historical_volume', side_effect=Exception("Data Error")):
            symbol = Symbol('BTC/USDT')
            
            with pytest.raises(Exception) as exc_info:
                await volume_analyzer.calculate_typical_profile(symbol)
            
            assert "Data Error" in str(exc_info.value)
    
    def test_volume_prediction_confidence_bounds(self):
        """Test that confidence scores are properly bounded."""
        prediction = VolumePrediction(
            symbol=Symbol('BTC/USDT'),
            prediction_time=datetime.now(timezone.utc),
            predicted_buckets={0: Decimal('1000')},
            confidence_scores={0: Decimal('0.8')},
            total_predicted=Decimal('1000'),
            model_accuracy=Decimal('0.85')
        )
        
        # Verify confidence is between 0 and 1
        for confidence in prediction.confidence_scores.values():
            assert Decimal('0') <= confidence <= Decimal('1')
        
        # Verify model accuracy is between 0 and 1
        assert Decimal('0') <= prediction.model_accuracy <= Decimal('1')