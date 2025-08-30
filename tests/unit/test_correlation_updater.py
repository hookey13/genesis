"""Unit tests for correlation matrix auto-update system."""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import json

from genesis.operations.correlation_updater import (
    CorrelationUpdater,
    CorrelationMatrix,
    CorrelationValidation,
    HotReloadManager
)


class TestCorrelationMatrix:
    """Test correlation matrix data structure."""
    
    def test_matrix_initialization(self):
        """Test correlation matrix initialization."""
        pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        matrix = CorrelationMatrix(pairs=pairs)
        
        assert matrix.pairs == pairs
        assert matrix.matrix.shape == (3, 3)
        assert np.all(np.diag(matrix.matrix) == 1.0)  # Diagonal should be 1
    
    def test_matrix_symmetry(self):
        """Test correlation matrix symmetry."""
        pairs = ["BTC/USDT", "ETH/USDT"]
        matrix = CorrelationMatrix(pairs=pairs)
        
        matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.85)
        
        assert matrix.get_correlation("BTC/USDT", "ETH/USDT") == 0.85
        assert matrix.get_correlation("ETH/USDT", "BTC/USDT") == 0.85
    
    def test_matrix_validation(self):
        """Test correlation matrix validation."""
        pairs = ["BTC/USDT", "ETH/USDT"]
        matrix = CorrelationMatrix(pairs=pairs)
        
        # Valid correlation
        assert matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.5)
        
        # Invalid correlations
        assert not matrix.set_correlation("BTC/USDT", "ETH/USDT", 1.5)  # > 1
        assert not matrix.set_correlation("BTC/USDT", "ETH/USDT", -1.5)  # < -1
    
    def test_matrix_serialization(self):
        """Test matrix serialization to JSON."""
        pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        matrix = CorrelationMatrix(pairs=pairs)
        
        matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.85)
        matrix.set_correlation("BTC/USDT", "BNB/USDT", 0.65)
        matrix.set_correlation("ETH/USDT", "BNB/USDT", 0.75)
        
        data = matrix.to_dict()
        assert "pairs" in data
        assert "matrix" in data
        assert "timestamp" in data
        
        # Ensure JSON serializable
        json_str = json.dumps(data, default=str)
        assert json_str


class TestCorrelationValidation:
    """Test correlation validation logic."""
    
    def test_validate_against_historical(self):
        """Test validation against historical patterns."""
        validator = CorrelationValidation()
        
        historical_corr = 0.85
        new_corr = 0.80
        
        # Small drift should be acceptable
        is_valid = validator.validate_drift(
            historical_corr, new_corr, max_drift=0.1
        )
        assert is_valid
        
        # Large drift should be rejected
        new_corr = 0.30
        is_valid = validator.validate_drift(
            historical_corr, new_corr, max_drift=0.1
        )
        assert not is_valid
    
    def test_validate_matrix_properties(self):
        """Test validation of matrix mathematical properties."""
        validator = CorrelationValidation()
        
        # Valid matrix (positive semi-definite)
        valid_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
        assert validator.is_positive_semidefinite(valid_matrix)
        
        # Invalid matrix (not positive semi-definite)
        invalid_matrix = np.array([
            [1.0, 0.95, 0.95],
            [0.95, 1.0, -0.95],
            [0.95, -0.95, 1.0]
        ])
        # This might not be positive semi-definite
        eigenvalues = np.linalg.eigvals(invalid_matrix)
        if np.any(eigenvalues < -1e-10):
            assert not validator.is_positive_semidefinite(invalid_matrix)


class TestCorrelationUpdater:
    """Test main correlation updater system."""
    
    @pytest.mark.asyncio
    async def test_updater_initialization(self):
        """Test correlation updater initialization."""
        with patch('genesis.operations.correlation_updater.get_market_data'):
            updater = CorrelationUpdater()
            
            assert updater.update_window_days == 30
            assert updater.min_data_points == 100
            assert updater.max_drift == 0.15
            assert updater.current_matrix is None
    
    @pytest.mark.asyncio
    async def test_fetch_market_data(self):
        """Test fetching market data for correlation calculation."""
        with patch('genesis.operations.correlation_updater.ccxt.binance') as mock_exchange:
            # Mock market data
            mock_ohlcv = [
                [1609459200000, 30000, 30500, 29500, 30200, 1000],  # BTC
                [1609459260000, 30200, 30600, 30000, 30400, 1100],
            ]
            mock_exchange.return_value.fetch_ohlcv.return_value = mock_ohlcv
            
            updater = CorrelationUpdater()
            data = await updater.fetch_market_data(
                symbol="BTC/USDT",
                timeframe="1h",
                limit=100
            )
            
            assert len(data) > 0
            assert "close" in data.columns
    
    @pytest.mark.asyncio
    async def test_calculate_correlations(self):
        """Test correlation calculation from market data."""
        updater = CorrelationUpdater()
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Correlated price movements
        btc_prices = 30000 + np.random.randn(100) * 1000
        eth_prices = 2000 + btc_prices / 15 + np.random.randn(100) * 50  # Correlated to BTC
        bnb_prices = 300 + np.random.randn(100) * 10  # Less correlated
        
        price_data = pd.DataFrame({
            'BTC/USDT': btc_prices,
            'ETH/USDT': eth_prices,
            'BNB/USDT': bnb_prices
        }, index=dates)
        
        correlation_matrix = await updater.calculate_correlations(price_data)
        
        # Check correlations
        btc_eth_corr = correlation_matrix.loc['BTC/USDT', 'ETH/USDT']
        assert 0.5 < btc_eth_corr < 1.0  # Should be positively correlated
        
        # Diagonal should be 1
        assert all(correlation_matrix.values[i, i] == 1.0 for i in range(len(correlation_matrix)))
    
    @pytest.mark.asyncio
    async def test_gradual_rollout(self):
        """Test A/B testing gradual rollout of new correlations."""
        with patch('genesis.operations.correlation_updater.random.random') as mock_random:
            updater = CorrelationUpdater()
            
            old_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
            new_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
            new_matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.9)
            
            # Test 50% rollout
            updater.rollout_percentage = 0.5
            
            # Should use new matrix 50% of the time
            mock_random.return_value = 0.3  # < 0.5
            matrix = await updater.select_matrix_for_request(old_matrix, new_matrix)
            assert matrix == new_matrix
            
            mock_random.return_value = 0.7  # > 0.5
            matrix = await updater.select_matrix_for_request(old_matrix, new_matrix)
            assert matrix == old_matrix
    
    @pytest.mark.asyncio
    async def test_hot_reload(self):
        """Test hot reload without service interruption."""
        with patch('genesis.operations.correlation_updater.get_active_positions') as mock_positions:
            mock_positions.return_value = [
                MagicMock(symbol="BTC/USDT", size=1.0),
                MagicMock(symbol="ETH/USDT", size=10.0)
            ]
            
            updater = CorrelationUpdater()
            
            # Set initial matrix
            initial_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
            initial_matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.8)
            updater.current_matrix = initial_matrix
            
            # Create new matrix
            new_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
            new_matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.85)
            
            # Perform hot reload
            success = await updater.hot_reload_matrix(new_matrix)
            assert success
            
            # Verify positions weren't affected
            mock_positions.assert_called()
            assert updater.current_matrix == new_matrix
    
    @pytest.mark.asyncio
    async def test_correlation_drift_monitoring(self):
        """Test monitoring and alerting for correlation drift."""
        with patch('genesis.operations.correlation_updater.send_alert') as mock_alert:
            updater = CorrelationUpdater()
            
            # Set baseline correlation
            updater.baseline_correlations = {
                ("BTC/USDT", "ETH/USDT"): 0.85,
                ("BTC/USDT", "BNB/USDT"): 0.65
            }
            
            # Check normal drift
            await updater.check_correlation_drift("BTC/USDT", "ETH/USDT", 0.82)
            assert not mock_alert.called
            
            # Check excessive drift
            await updater.check_correlation_drift("BTC/USDT", "ETH/USDT", 0.40)
            assert mock_alert.called
            
            alert_data = mock_alert.call_args[0][0]
            assert "correlation drift" in alert_data.message.lower()
            assert alert_data.severity == "HIGH"
    
    @pytest.mark.asyncio
    async def test_validation_before_update(self):
        """Test validation checks before updating correlations."""
        with patch('genesis.operations.correlation_updater.fetch_market_data') as mock_fetch:
            # Insufficient data
            mock_fetch.return_value = pd.DataFrame({'close': [1, 2, 3]})  # Too few points
            
            updater = CorrelationUpdater()
            updater.min_data_points = 100
            
            result = await updater.update_correlations()
            assert not result  # Should fail due to insufficient data
    
    @pytest.mark.asyncio
    async def test_rollback_on_validation_failure(self):
        """Test rollback when new correlations fail validation."""
        updater = CorrelationUpdater()
        
        # Set current valid matrix
        current = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
        current.set_correlation("BTC/USDT", "ETH/USDT", 0.85)
        updater.current_matrix = current
        
        # Try to update with invalid matrix
        with patch.object(updater, 'validate_new_matrix', return_value=False):
            new_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
            new_matrix.set_correlation("BTC/USDT", "ETH/USDT", -0.99)  # Suspicious value
            
            success = await updater.apply_new_matrix(new_matrix)
            assert not success
            assert updater.current_matrix == current  # Should keep old matrix
    
    @pytest.mark.asyncio  
    async def test_persistence_to_database(self):
        """Test saving correlation matrix to database."""
        with patch('genesis.operations.correlation_updater.get_db_session') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value = mock_session
            
            updater = CorrelationUpdater()
            
            matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
            matrix.set_correlation("BTC/USDT", "ETH/USDT", 0.85)
            
            await updater.save_matrix_to_db(matrix)
            
            # Verify database operations
            assert mock_session.add.called
            assert mock_session.commit.called


class TestHotReloadManager:
    """Test hot reload manager for zero-downtime updates."""
    
    @pytest.mark.asyncio
    async def test_atomic_swap(self):
        """Test atomic swap of correlation matrix."""
        manager = HotReloadManager()
        
        old_matrix = CorrelationMatrix(pairs=["BTC/USDT"])
        new_matrix = CorrelationMatrix(pairs=["BTC/USDT", "ETH/USDT"])
        
        # Perform atomic swap
        manager.current = old_matrix
        await manager.atomic_swap(new_matrix)
        
        assert manager.current == new_matrix
        assert manager.previous == old_matrix
    
    @pytest.mark.asyncio
    async def test_version_tracking(self):
        """Test version tracking for correlation matrices."""
        manager = HotReloadManager()
        
        matrices = []
        for i in range(5):
            matrix = CorrelationMatrix(pairs=["BTC/USDT"])
            matrix.version = i + 1
            matrices.append(matrix)
            await manager.update_with_version(matrix)
        
        assert manager.current.version == 5
        assert len(manager.version_history) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Test thread-safe access during hot reload."""
        manager = HotReloadManager()
        
        initial = CorrelationMatrix(pairs=["BTC/USDT"])
        manager.current = initial
        
        async def reader():
            """Simulate reading correlation."""
            for _ in range(100):
                matrix = manager.current
                assert matrix is not None
                await asyncio.sleep(0.001)
        
        async def updater():
            """Simulate updating correlation."""
            for i in range(10):
                new_matrix = CorrelationMatrix(pairs=["BTC/USDT", f"ETH{i}/USDT"])
                await manager.atomic_swap(new_matrix)
                await asyncio.sleep(0.01)
        
        # Run concurrent operations
        await asyncio.gather(
            reader(),
            reader(),
            reader(),
            updater()
        )
        
        # Should complete without errors
        assert manager.current is not None