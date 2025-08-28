"""Unit tests for pair correlation monitoring system."""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from genesis.analytics.pair_correlation_monitor import (
    CorrelationAlert,
    CorrelationMonitor,
    PriceHistory,
)
from genesis.core.models import Position, PositionSide, PriceData


class TestPriceHistory:
    """Test PriceHistory class."""

    def test_add_price(self):
        """Test adding prices to history."""
        history = PriceHistory(symbol="BTC/USDT")

        price1 = Decimal("50000")
        time1 = datetime.utcnow()
        history.add_price(price1, time1)

        assert len(history.prices) == 1
        assert history.prices[0] == price1
        assert history.timestamps[0] == time1

    def test_get_returns_insufficient_data(self):
        """Test returns calculation with insufficient data."""
        history = PriceHistory(symbol="BTC/USDT")

        # Add only one price
        history.add_price(Decimal("50000"), datetime.utcnow())

        returns = history.get_returns(60)
        assert len(returns) == 0

    def test_get_returns_calculation(self):
        """Test returns calculation."""
        history = PriceHistory(symbol="BTC/USDT")

        # Add price series
        base_time = datetime.utcnow()
        prices = [
            Decimal("50000"),
            Decimal("51000"),
            Decimal("50500"),
            Decimal("52000"),
        ]

        for i, price in enumerate(prices):
            history.add_price(price, base_time + timedelta(minutes=i))

        returns = history.get_returns(60)
        assert len(returns) == 3  # n-1 returns for n prices

        # First return should be log(51000/50000)
        expected_first = np.log(51000 / 50000)
        assert abs(returns[0] - expected_first) < 0.0001

    def test_get_returns_time_window(self):
        """Test returns calculation respects time window."""
        history = PriceHistory(symbol="BTC/USDT")

        base_time = datetime.utcnow()

        # Add old prices (outside window)
        history.add_price(Decimal("40000"), base_time - timedelta(minutes=120))
        history.add_price(Decimal("41000"), base_time - timedelta(minutes=115))

        # Add recent prices (within window)
        history.add_price(Decimal("50000"), base_time - timedelta(minutes=30))
        history.add_price(Decimal("51000"), base_time - timedelta(minutes=20))
        history.add_price(Decimal("52000"), base_time - timedelta(minutes=10))

        returns = history.get_returns(60)
        assert len(returns) == 2  # Only recent prices counted


class TestCorrelationMonitor:
    """Test CorrelationMonitor class."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.get_price_history = AsyncMock(return_value=[])
        repo.store_correlation = AsyncMock()
        return repo

    @pytest.fixture
    def monitor(self, mock_repository):
        """Create CorrelationMonitor instance."""
        return CorrelationMonitor(repository=mock_repository)

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return [
            Position(
                position_id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("50000"),
                side=PositionSide.LONG,
            ),
            Position(
                position_id=str(uuid.uuid4()),
                symbol="ETH/USDT",
                quantity=Decimal("10"),
                dollar_value=Decimal("30000"),
                side=PositionSide.LONG,
            ),
            Position(
                position_id=str(uuid.uuid4()),
                symbol="SOL/USDT",
                quantity=Decimal("100"),
                dollar_value=Decimal("10000"),
                side=PositionSide.LONG,
            ),
        ]

    @pytest.mark.asyncio
    async def test_update_price(self, monitor):
        """Test price update functionality."""
        symbol = "BTC/USDT"
        price = Decimal("50000")
        timestamp = datetime.utcnow()

        await monitor.update_price(symbol, price, timestamp)

        assert symbol in monitor._price_histories
        assert len(monitor._price_histories[symbol].prices) == 1
        assert monitor._price_histories[symbol].prices[0] == price

    @pytest.mark.asyncio
    async def test_calculate_pair_correlation_no_data(self, monitor):
        """Test correlation calculation with no price data."""
        correlation = await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")
        assert correlation == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_pair_correlation_insufficient_data(self, monitor):
        """Test correlation with insufficient data points."""
        # Add less than MIN_DATA_POINTS prices
        base_time = datetime.utcnow()
        for i in range(10):  # Less than 20 required
            await monitor.update_price(
                "BTC/USDT",
                Decimal("50000") + Decimal(i * 100),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") + Decimal(i * 10),
                base_time + timedelta(minutes=i),
            )

        correlation = await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")
        assert correlation == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_pair_correlation_perfect_positive(self, monitor):
        """Test correlation calculation with perfectly correlated data."""
        base_time = datetime.utcnow()

        # Create perfectly correlated price movements
        for i in range(30):
            btc_price = Decimal("50000") * (Decimal("1") + Decimal(i) / Decimal("100"))
            eth_price = Decimal("3000") * (Decimal("1") + Decimal(i) / Decimal("100"))

            await monitor.update_price(
                "BTC/USDT", btc_price, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT", eth_price, base_time + timedelta(minutes=i)
            )

        correlation = await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")

        # Should be close to 1.0 (perfect positive correlation)
        assert correlation > Decimal("0.95")

    @pytest.mark.asyncio
    async def test_calculate_pair_correlation_negative(self, monitor):
        """Test correlation calculation with negative correlation."""
        base_time = datetime.utcnow()

        # Create negatively correlated price movements
        for i in range(30):
            btc_price = Decimal("50000") * (Decimal("1") + Decimal(i) / Decimal("100"))
            eth_price = Decimal("3000") * (
                Decimal("1") - Decimal(i) / Decimal("200")
            )  # Opposite direction

            await monitor.update_price(
                "BTC/USDT", btc_price, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT", eth_price, base_time + timedelta(minutes=i)
            )

        correlation = await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")

        # Should be negative
        assert correlation < Decimal("0")

    @pytest.mark.asyncio
    async def test_correlation_caching(self, monitor):
        """Test that correlations are cached."""
        base_time = datetime.utcnow()

        # Add sufficient data
        for i in range(30):
            await monitor.update_price(
                "BTC/USDT",
                Decimal("50000") + Decimal(i * 100),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") + Decimal(i * 10),
                base_time + timedelta(minutes=i),
            )

        # First calculation
        correlation1 = await monitor.calculate_pair_correlation(
            "BTC/USDT", "ETH/USDT", 60
        )

        # Second calculation (should use cache)
        with patch.object(monitor, "_store_correlation") as mock_store:
            correlation2 = await monitor.calculate_pair_correlation(
                "BTC/USDT", "ETH/USDT", 60
            )
            # Should not store again due to cache
            mock_store.assert_not_called()

        assert correlation1 == correlation2

    @pytest.mark.asyncio
    async def test_correlation_alert_warning(self, monitor):
        """Test warning alert generation."""
        base_time = datetime.utcnow()

        # Create data that will trigger warning (>0.6 correlation)
        for i in range(30):
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price(
                "BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") * (factor * Decimal("0.95")),
                base_time + timedelta(minutes=i),
            )

        await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")

        alerts = await monitor.get_recent_alerts()
        assert len(alerts) > 0
        assert alerts[0].severity == "WARNING" or alerts[0].severity == "CRITICAL"

    @pytest.mark.asyncio
    async def test_correlation_alert_critical(self, monitor):
        """Test critical alert generation."""
        base_time = datetime.utcnow()

        # Create data with very high correlation (>0.8)
        for i in range(30):
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price(
                "BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT", Decimal("3000") * factor, base_time + timedelta(minutes=i)
            )

        await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")

        alerts = await monitor.get_recent_alerts()
        assert len(alerts) > 0
        # With perfect correlation, should be critical
        assert any(alert.severity == "CRITICAL" for alert in alerts)

    @pytest.mark.asyncio
    async def test_get_correlation_matrix(self, monitor):
        """Test correlation matrix generation."""
        base_time = datetime.utcnow()

        # Add data for three symbols
        for i in range(30):
            await monitor.update_price(
                "BTC/USDT",
                Decimal("50000") + Decimal(i * 100),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") + Decimal(i * 10),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "SOL/USDT",
                Decimal("100") - Decimal(i),
                base_time + timedelta(minutes=i),
            )

        matrix = await monitor.get_correlation_matrix(
            ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )

        assert matrix.shape == (3, 3)
        # Diagonal should be 1.0
        assert np.allclose(np.diag(matrix), 1.0)
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)

    @pytest.mark.asyncio
    async def test_get_highly_correlated_pairs(self, monitor):
        """Test finding highly correlated pairs."""
        base_time = datetime.utcnow()

        # Create correlated BTC/ETH and uncorrelated SOL
        for i in range(30):
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price(
                "BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT", Decimal("3000") * factor, base_time + timedelta(minutes=i)
            )
            # SOL with different pattern
            sol_factor = Decimal("1") + Decimal(i % 10) / Decimal("100")
            await monitor.update_price(
                "SOL/USDT",
                Decimal("100") * sol_factor,
                base_time + timedelta(minutes=i),
            )

        high_corr_pairs = await monitor.get_highly_correlated_pairs()

        # BTC/ETH should be highly correlated
        assert len(high_corr_pairs) > 0
        assert any(
            (pair[0] == "BTC/USDT" and pair[1] == "ETH/USDT")
            or (pair[0] == "ETH/USDT" and pair[1] == "BTC/USDT")
            for pair in high_corr_pairs
        )

    @pytest.mark.asyncio
    async def test_analyze_portfolio_correlation(self, monitor, sample_positions):
        """Test portfolio correlation analysis."""
        base_time = datetime.utcnow()

        # Add price data
        for i in range(30):
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price(
                "BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") * factor * Decimal("0.9"),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "SOL/USDT",
                Decimal("100") * (Decimal("1") - Decimal(i) / Decimal("200")),
                base_time + timedelta(minutes=i),
            )

        analysis = await monitor.analyze_portfolio_correlation(sample_positions)

        assert "max_correlation" in analysis
        assert "avg_correlation" in analysis
        assert "correlation_pairs" in analysis
        assert "risk_level" in analysis
        assert analysis["positions_analyzed"] == 3

    @pytest.mark.asyncio
    async def test_analyze_portfolio_correlation_high_risk(self, monitor):
        """Test portfolio analysis with high correlation risk."""
        base_time = datetime.utcnow()

        # Create highly correlated positions
        positions = [
            Position(
                position_id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("50000"),
            ),
            Position(
                position_id=str(uuid.uuid4()),
                symbol="ETH/USDT",
                quantity=Decimal("10"),
                dollar_value=Decimal("30000"),
            ),
        ]

        # Add highly correlated data
        for i in range(30):
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price(
                "BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i)
            )
            await monitor.update_price(
                "ETH/USDT", Decimal("3000") * factor, base_time + timedelta(minutes=i)
            )

        analysis = await monitor.analyze_portfolio_correlation(positions)

        # Should detect critical risk
        assert analysis["risk_level"] == "CRITICAL"
        assert analysis["max_correlation"] > Decimal("0.8")

    @pytest.mark.asyncio
    async def test_clear_alerts(self, monitor):
        """Test clearing alerts."""
        # Create an alert
        alert = CorrelationAlert(
            alert_id=str(uuid.uuid4()),
            symbol1="BTC/USDT",
            symbol2="ETH/USDT",
            correlation=Decimal("0.85"),
            threshold=Decimal("0.8"),
            timestamp=datetime.utcnow(),
            message="Test alert",
            severity="CRITICAL",
        )
        monitor._alerts.append(alert)

        assert len(monitor._alerts) == 1

        await monitor.clear_alerts()

        assert len(monitor._alerts) == 0

    @pytest.mark.asyncio
    async def test_load_price_history_from_repository(self, monitor, mock_repository):
        """Test loading price history from repository."""
        # Mock repository data
        price_data = [
            PriceData(
                symbol="BTC/USDT",
                price=Decimal("50000"),
                timestamp=datetime.utcnow() - timedelta(minutes=i),
            )
            for i in range(30)
        ]
        mock_repository.get_price_history.return_value = price_data

        await monitor._load_price_history("BTC/USDT", 60)

        assert "BTC/USDT" in monitor._price_histories
        assert len(monitor._price_histories["BTC/USDT"].prices) == 30

    @pytest.mark.asyncio
    async def test_concurrent_correlation_calculations(self, monitor):
        """Test thread safety with concurrent operations."""
        base_time = datetime.utcnow()

        # Add price data
        for i in range(30):
            await monitor.update_price(
                "BTC/USDT",
                Decimal("50000") + Decimal(i * 100),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "ETH/USDT",
                Decimal("3000") + Decimal(i * 10),
                base_time + timedelta(minutes=i),
            )
            await monitor.update_price(
                "SOL/USDT",
                Decimal("100") + Decimal(i),
                base_time + timedelta(minutes=i),
            )

        # Run multiple correlations concurrently
        tasks = [
            monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT"),
            monitor.calculate_pair_correlation("BTC/USDT", "SOL/USDT"),
            monitor.calculate_pair_correlation("ETH/USDT", "SOL/USDT"),
            monitor.get_correlation_matrix(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete without errors
        assert len(results) == 4
        assert all(result is not None for result in results[:3])
        assert results[3].shape == (3, 3)  # Matrix shape

    @pytest.mark.asyncio
    async def test_alert_limit(self, monitor):
        """Test that alerts are limited to prevent memory issues."""
        # Add many alerts
        for i in range(150):
            alert = CorrelationAlert(
                alert_id=str(uuid.uuid4()),
                symbol1=f"PAIR{i}",
                symbol2=f"PAIR{i+1}",
                correlation=Decimal("0.7"),
                threshold=Decimal("0.6"),
                timestamp=datetime.utcnow(),
                message=f"Alert {i}",
                severity="WARNING",
            )
            monitor._alerts.append(alert)

        # Trigger cleanup in _check_correlation_alert
        await monitor._check_correlation_alert("TEST1", "TEST2", Decimal("0.65"))

        # Should keep only last 100 alerts
        assert len(monitor._alerts) <= 100
