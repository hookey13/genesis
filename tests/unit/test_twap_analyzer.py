"""Unit tests for TWAP analyzer."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.twap_analyzer import TwapAnalyzer, TwapReport
from genesis.engine.executor.base import OrderSide


class TestTwapAnalyzer:
    """Test TWAP analyzer functionality."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        return repo

    @pytest.fixture
    def analyzer(self, mock_repository):
        """Create TWAP analyzer instance."""
        return TwapAnalyzer(repository=mock_repository)

    @pytest.fixture
    def sample_execution_data(self):
        """Create sample execution data."""
        return {
            'execution_id': 'test-exec-123',
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'total_quantity': '10.0',
            'executed_quantity': '9.5',
            'duration_minutes': 15,
            'slice_count': 10,
            'arrival_price': '50000',
            'early_completion': False,
            'started_at': datetime.now() - timedelta(minutes=15),
            'completed_at': datetime.now(),
            'remaining_quantity': '0.5'
        }

    @pytest.fixture
    def sample_slice_history(self):
        """Create sample slice history."""
        slices = []
        base_price = 50000
        for i in range(10):
            slices.append({
                'slice_number': i + 1,
                'executed_quantity': '0.95' if i < 10 else '0',
                'execution_price': str(base_price + i * 10),
                'market_price': str(base_price + i * 8),
                'slippage_bps': str(5 + i),
                'participation_rate': str(7 + (i % 3)),
                'volume_at_execution': '100',
                'status': 'EXECUTED' if i < 10 else 'PENDING',
                'executed_at': datetime.now() - timedelta(minutes=15 - i)
            })
        return slices

    def test_calculate_price_metrics(self, analyzer, sample_execution_data, sample_slice_history):
        """Test price metrics calculation."""
        metrics = analyzer._calculate_price_metrics(sample_execution_data, sample_slice_history)

        assert 'arrival_price' in metrics
        assert metrics['arrival_price'] == Decimal('50000')
        assert 'average_execution_price' in metrics
        assert metrics['average_execution_price'] > 0
        assert 'twap_price' in metrics
        assert metrics['twap_price'] > 0
        assert 'best_slice_price' in metrics
        assert 'worst_slice_price' in metrics
        assert metrics['best_slice_price'] <= metrics['worst_slice_price']

    def test_calculate_performance_metrics(self, analyzer, sample_execution_data, sample_slice_history):
        """Test performance metrics calculation."""
        price_metrics = {
            'arrival_price': Decimal('50000'),
            'average_execution_price': Decimal('50050'),
            'twap_price': Decimal('50045'),
            'vwap_market_price': Decimal('50040'),
            'best_possible_price': Decimal('50000'),
            'worst_slice_price': Decimal('50090'),
            'best_slice_price': Decimal('50000')
        }

        metrics = analyzer._calculate_performance_metrics(
            sample_execution_data,
            sample_slice_history,
            price_metrics
        )

        assert 'implementation_shortfall' in metrics
        assert 'twap_effectiveness' in metrics
        assert 'slippage_bps' in metrics
        assert 'max_slice_slippage_bps' in metrics
        assert 'total_slippage_cost' in metrics
        assert metrics['slippage_bps'] >= 0
        assert metrics['max_slice_slippage_bps'] >= metrics['slippage_bps']

    @pytest.mark.asyncio
    async def test_estimate_market_impact(self, analyzer, sample_execution_data, sample_slice_history):
        """Test market impact estimation."""
        impact = await analyzer._estimate_market_impact(sample_execution_data, sample_slice_history)

        assert 'estimated_impact_bps' in impact
        assert 'temporary_impact_bps' in impact
        assert 'permanent_impact_bps' in impact
        assert impact['estimated_impact_bps'] >= 0
        assert impact['temporary_impact_bps'] + impact['permanent_impact_bps'] == impact['estimated_impact_bps']

    def test_calculate_volume_metrics(self, analyzer, sample_slice_history):
        """Test volume metrics calculation."""
        metrics = analyzer._calculate_volume_metrics(sample_slice_history)

        assert 'avg_participation_rate' in metrics
        assert 'max_participation_rate' in metrics
        assert 'min_participation_rate' in metrics
        assert 'volume_weighted_participation' in metrics
        assert metrics['min_participation_rate'] <= metrics['avg_participation_rate'] <= metrics['max_participation_rate']

    def test_calculate_timing_effectiveness(self, analyzer, sample_execution_data, sample_slice_history):
        """Test timing effectiveness calculation."""
        price_metrics = {
            'arrival_price': Decimal('50000'),
            'average_execution_price': Decimal('50050'),
            'twap_price': Decimal('50045'),
            'vwap_market_price': Decimal('50040'),
            'best_possible_price': Decimal('50000'),
            'worst_slice_price': Decimal('50090'),
            'best_slice_price': Decimal('50000')
        }

        metrics = analyzer._calculate_timing_effectiveness(
            sample_execution_data,
            sample_slice_history,
            price_metrics
        )

        assert 'timing_score' in metrics
        assert 0 <= metrics['timing_score'] <= 100
        assert metrics['early_completion_benefit'] is None  # No early completion

        # Test with early completion
        sample_execution_data['early_completion'] = True
        metrics = analyzer._calculate_timing_effectiveness(
            sample_execution_data,
            sample_slice_history,
            price_metrics
        )
        assert metrics['early_completion_benefit'] is not None

    def test_calculate_risk_metrics(self, analyzer, sample_slice_history):
        """Test risk metrics calculation."""
        volume_metrics = {
            'avg_participation_rate': Decimal('8'),
            'max_participation_rate': Decimal('12'),
            'min_participation_rate': Decimal('5'),
            'volume_weighted_participation': Decimal('7.5')
        }

        metrics = analyzer._calculate_risk_metrics(sample_slice_history, volume_metrics)

        assert 'execution_risk_score' in metrics
        assert 'concentration_risk' in metrics
        assert 0 <= metrics['execution_risk_score'] <= 100
        assert metrics['concentration_risk'] >= 0

    def test_generate_recommendations(self, analyzer, sample_execution_data):
        """Test recommendation generation."""
        performance_metrics = {
            'implementation_shortfall': Decimal('25'),  # 25 bps
            'twap_effectiveness': Decimal('85'),
            'slippage_bps': Decimal('15'),
            'max_slice_slippage_bps': Decimal('25'),
            'total_slippage_cost': Decimal('100')
        }

        volume_metrics = {
            'avg_participation_rate': Decimal('8'),
            'max_participation_rate': Decimal('12'),
            'min_participation_rate': Decimal('5'),
            'volume_weighted_participation': Decimal('7.5')
        }

        risk_metrics = {
            'execution_risk_score': Decimal('45'),
            'concentration_risk': Decimal('20')
        }

        recommendations = analyzer._generate_recommendations(
            sample_execution_data,
            performance_metrics,
            volume_metrics,
            risk_metrics
        )

        assert 'optimal_slice_count' in recommendations
        assert 'recommended_duration' in recommendations
        assert 'suggested_participation_rate' in recommendations
        assert 'improvement_opportunities' in recommendations
        assert isinstance(recommendations['improvement_opportunities'], list)

    def test_generate_recommendations_high_slippage(self, analyzer, sample_execution_data):
        """Test recommendations for high slippage scenario."""
        performance_metrics = {
            'implementation_shortfall': Decimal('75'),  # High slippage
            'twap_effectiveness': Decimal('65'),
            'slippage_bps': Decimal('35'),  # High
            'max_slice_slippage_bps': Decimal('50'),
            'total_slippage_cost': Decimal('500')
        }

        volume_metrics = {
            'avg_participation_rate': Decimal('12'),  # High participation
            'max_participation_rate': Decimal('18'),  # Too high
            'min_participation_rate': Decimal('8'),
            'volume_weighted_participation': Decimal('11')
        }

        risk_metrics = {
            'execution_risk_score': Decimal('75'),  # High risk
            'concentration_risk': Decimal('40')  # High concentration
        }

        recommendations = analyzer._generate_recommendations(
            sample_execution_data,
            performance_metrics,
            volume_metrics,
            risk_metrics
        )

        # Should recommend longer duration and more slices
        assert recommendations['recommended_duration'] > sample_execution_data['duration_minutes']
        assert recommendations['optimal_slice_count'] > sample_execution_data['slice_count']
        assert recommendations['suggested_participation_rate'] < volume_metrics['avg_participation_rate']
        assert len(recommendations['improvement_opportunities']) > 3  # Multiple issues

    @pytest.mark.asyncio
    async def test_generate_execution_report(self, analyzer, mock_repository, sample_execution_data, sample_slice_history):
        """Test full report generation."""
        mock_repository.get_twap_execution = AsyncMock(return_value=sample_execution_data)
        mock_repository.get_twap_slices = AsyncMock(return_value=sample_slice_history)
        mock_repository.save_twap_analysis = AsyncMock()

        report = await analyzer.generate_execution_report('test-exec-123')

        assert isinstance(report, TwapReport)
        assert report.execution_id == 'test-exec-123'
        assert report.symbol == 'BTC/USDT'
        assert report.side == OrderSide.BUY
        assert report.total_quantity == Decimal('10.0')
        assert report.slice_count == 10
        assert report.arrival_price == Decimal('50000')
        assert report.timing_score >= 0
        assert report.execution_risk_score >= 0
        assert len(report.improvement_opportunities) >= 0

        # Verify repository calls
        mock_repository.get_twap_execution.assert_called_once_with('test-exec-123')
        mock_repository.get_twap_slices.assert_called_once_with('test-exec-123')
        mock_repository.save_twap_analysis.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_execution_report_not_found(self, analyzer, mock_repository):
        """Test report generation with missing execution."""
        mock_repository.get_twap_execution = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Execution .* not found"):
            await analyzer.generate_execution_report('non-existent')

    @pytest.mark.asyncio
    async def test_generate_execution_report_no_slices(self, analyzer, mock_repository, sample_execution_data):
        """Test report generation with no slice history."""
        mock_repository.get_twap_execution = AsyncMock(return_value=sample_execution_data)
        mock_repository.get_twap_slices = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="No slice history"):
            await analyzer.generate_execution_report('test-exec-123')

    @pytest.mark.asyncio
    async def test_compare_executions(self, analyzer, mock_repository):
        """Test comparing multiple executions."""
        # Mock two different executions with varying performance
        execution1 = {
            'execution_id': 'exec-1',
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'total_quantity': '10.0',
            'executed_quantity': '10.0',
            'duration_minutes': 15,
            'slice_count': 10,
            'arrival_price': '50000',
            'early_completion': False,
            'started_at': datetime.now() - timedelta(minutes=30),
            'completed_at': datetime.now() - timedelta(minutes=15),
            'remaining_quantity': '0'
        }

        execution2 = {
            'execution_id': 'exec-2',
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'total_quantity': '10.0',
            'executed_quantity': '10.0',
            'duration_minutes': 20,
            'slice_count': 15,
            'arrival_price': '50000',
            'early_completion': False,
            'started_at': datetime.now() - timedelta(minutes=20),
            'completed_at': datetime.now(),
            'remaining_quantity': '0'
        }

        slices1 = [
            {
                'slice_number': i + 1,
                'executed_quantity': '1.0',
                'execution_price': str(50000 + i * 5),  # Better prices
                'market_price': str(50000 + i * 4),
                'slippage_bps': str(3 + i % 2),
                'participation_rate': str(6),
                'volume_at_execution': '100',
                'status': 'EXECUTED',
                'executed_at': datetime.now()
            }
            for i in range(10)
        ]

        slices2 = [
            {
                'slice_number': i + 1,
                'executed_quantity': '0.67',
                'execution_price': str(50000 + i * 10),  # Worse prices
                'market_price': str(50000 + i * 8),
                'slippage_bps': str(5 + i % 3),
                'participation_rate': str(8),
                'volume_at_execution': '100',
                'status': 'EXECUTED',
                'executed_at': datetime.now()
            }
            for i in range(15)
        ]

        mock_repository.get_twap_execution = AsyncMock(side_effect=[execution1, execution2])
        mock_repository.get_twap_slices = AsyncMock(side_effect=[slices1, slices2])
        mock_repository.save_twap_analysis = AsyncMock()

        comparison = await analyzer.compare_executions(['exec-1', 'exec-2'])

        assert 'execution_ids' in comparison
        assert comparison['execution_ids'] == ['exec-1', 'exec-2']
        assert 'best_implementation_shortfall' in comparison
        assert 'worst_implementation_shortfall' in comparison
        assert 'avg_implementation_shortfall' in comparison
        assert 'best_execution' in comparison
        assert 'worst_execution' in comparison
        assert comparison['best_execution'] in ['exec-1', 'exec-2']

    @pytest.mark.asyncio
    async def test_compare_executions_insufficient_data(self, analyzer, mock_repository):
        """Test comparison with insufficient executions."""
        mock_repository.get_twap_execution = AsyncMock(return_value=None)

        comparison = await analyzer.compare_executions(['exec-1'])

        assert 'error' in comparison
        assert 'at least 2' in comparison['error']
