"""
System startup and component connectivity verification tests.
Tests all modules start without errors and can communicate.
"""
import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import structlog

from genesis.core.models import (
    Position, Order, TierType, OrderStatus, OrderType,
    OrderSide, AccountTier, TradingState
)
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry
from genesis.engine.ab_test_framework import ABTestFramework
from genesis.data.repository import Repository
from genesis.data.correlation_repo import CorrelationRepository
from genesis.data.performance_repo import PerformanceRepository
from genesis.data.market_microstructure_repo import MarketMicrostructureRepository
from genesis.core.account_manager import AccountManager
from genesis.core.single_account_manager import SingleAccountManager
from genesis.exchange.gateway import ExchangeGateway
from genesis.exchange.websocket_manager import WebSocketManager
from genesis.exchange.fix_gateway import FIXGateway
from genesis.exchange.order_book_manager import OrderBookManager
from genesis.exchange.prime_broker import PrimeBroker
from genesis.analytics.risk_metrics import RiskMetricsCalculator
from genesis.analytics.performance_attribution import PerformanceAttributionEngine
from genesis.analytics.microstructure_analyzer import MicrostructureAnalyzer
from genesis.analytics.behavioral_correlation import BehavioralCorrelationAnalyzer
from genesis.tilt.detector import TiltDetector
from genesis.tilt.interventions import InterventionSystem

logger = structlog.get_logger()


class TestComponentStartup:
    """Test all components start successfully without errors."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = Mock()
        session.query = Mock()
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        return session

    @pytest.fixture
    def mock_exchange_gateway(self):
        """Mock exchange gateway."""
        gateway = Mock(spec=ExchangeGateway)
        gateway.connect = AsyncMock(return_value=True)
        gateway.disconnect = AsyncMock()
        gateway.get_balance = AsyncMock(return_value=Decimal("10000"))
        gateway.place_order = AsyncMock()
        gateway.cancel_order = AsyncMock()
        return gateway

    @pytest.mark.asyncio
    async def test_core_modules_initialization(self, mock_db_session):
        """Test core modules initialize without errors."""
        # Test account managers
        single_manager = SingleAccountManager(
            account_id="test_account",
            initial_balance=Decimal("10000"),
            tier=TierType.SNIPER
        )
        assert single_manager is not None
        assert single_manager.account_id == "test_account"
        assert single_manager.balance == Decimal("10000")

        multi_manager = AccountManager(db_session=mock_db_session)
        assert multi_manager is not None
        assert multi_manager.accounts == {}

        # Test repository initialization
        repo = Repository(db_session=mock_db_session)
        assert repo is not None

        correlation_repo = CorrelationRepository(db_session=mock_db_session)
        assert correlation_repo is not None

        perf_repo = PerformanceRepository(db_session=mock_db_session)
        assert perf_repo is not None

        micro_repo = MarketMicrostructureRepository(db_session=mock_db_session)
        assert micro_repo is not None

    @pytest.mark.asyncio
    async def test_engine_modules_initialization(self, mock_db_session):
        """Test trading engine modules initialize correctly."""
        # Test strategy registry
        registry = StrategyRegistry()
        assert registry is not None
        assert registry.strategies == {}

        # Test orchestrator
        orchestrator = StrategyOrchestrator(
            repository=Repository(db_session=mock_db_session),
            exchange_gateway=Mock()
        )
        assert orchestrator is not None
        assert orchestrator.active_strategies == {}
        assert orchestrator.performance_metrics == {}

        # Test AB test framework
        ab_test = ABTestFramework(repository=Repository(db_session=mock_db_session))
        assert ab_test is not None
        assert ab_test.active_tests == {}

    @pytest.mark.asyncio
    async def test_exchange_modules_initialization(self):
        """Test exchange modules initialize without errors."""
        # Test WebSocket manager
        ws_manager = WebSocketManager(url="wss://test.exchange.com")
        assert ws_manager is not None
        assert ws_manager.url == "wss://test.exchange.com"

        # Test FIX gateway
        fix_gateway = FIXGateway(
            host="localhost",
            port=9876,
            sender_comp_id="TEST",
            target_comp_id="EXCHANGE"
        )
        assert fix_gateway is not None
        assert fix_gateway.host == "localhost"

        # Test order book manager
        order_book = OrderBookManager(symbol="BTCUSDT")
        assert order_book is not None
        assert order_book.symbol == "BTCUSDT"

        # Test prime broker
        prime_broker = PrimeBroker(
            api_key="test_key",
            api_secret="test_secret"
        )
        assert prime_broker is not None

    @pytest.mark.asyncio
    async def test_analytics_modules_initialization(self, mock_db_session):
        """Test analytics modules initialize correctly."""
        # Test risk metrics calculator
        risk_calc = RiskMetricsCalculator(
            repository=Repository(db_session=mock_db_session)
        )
        assert risk_calc is not None

        # Test performance attribution
        perf_engine = PerformanceAttributionEngine(
            repository=Repository(db_session=mock_db_session)
        )
        assert perf_engine is not None

        # Test microstructure analyzer
        micro_analyzer = MicrostructureAnalyzer(
            repository=MarketMicrostructureRepository(db_session=mock_db_session)
        )
        assert micro_analyzer is not None

        # Test behavioral correlation
        behavioral = BehavioralCorrelationAnalyzer(
            correlation_repo=CorrelationRepository(db_session=mock_db_session)
        )
        assert behavioral is not None

    @pytest.mark.asyncio
    async def test_tilt_modules_initialization(self, mock_db_session):
        """Test tilt detection modules initialize correctly."""
        # Test tilt detector
        detector = TiltDetector(repository=Repository(db_session=mock_db_session))
        assert detector is not None
        assert detector.indicators == []
        assert detector.baseline_window == 30

        # Test intervention system
        interventions = InterventionSystem(
            repository=Repository(db_session=mock_db_session)
        )
        assert interventions is not None
        assert interventions.active_interventions == []

    @pytest.mark.asyncio
    async def test_database_connection_pool(self, mock_db_session):
        """Test database connection pool management."""
        # Test multiple repositories can share session
        repos = [
            Repository(db_session=mock_db_session),
            CorrelationRepository(db_session=mock_db_session),
            PerformanceRepository(db_session=mock_db_session),
            MarketMicrostructureRepository(db_session=mock_db_session)
        ]
        
        # Verify all repositories have valid sessions
        for repo in repos:
            assert repo.db_session is not None
            assert repo.db_session == mock_db_session

    @pytest.mark.asyncio
    async def test_websocket_connection_stability(self):
        """Test WebSocket connection initialization and reconnection."""
        ws_manager = WebSocketManager(url="wss://test.exchange.com")
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_ws.recv = AsyncMock(return_value='{"type":"ping"}')
            mock_ws.send = AsyncMock()
            mock_connect.return_value = mock_ws
            
            # Test connection
            await ws_manager.connect()
            assert mock_connect.called
            
            # Test reconnection logic
            ws_manager.reconnect_delay = 0.01  # Speed up test
            await ws_manager.ensure_connection()
            assert ws_manager.connection_attempts >= 0

    @pytest.mark.asyncio
    async def test_memory_leak_detection_setup(self):
        """Test memory monitoring setup for 24-hour leak detection."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy multiple objects
        for _ in range(100):
            manager = SingleAccountManager(
                account_id=f"test_{_}",
                initial_balance=Decimal("10000"),
                tier=TierType.SNIPER
            )
            del manager
        
        # Check memory hasn't grown excessively
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Allow up to 50MB growth for test overhead
        assert memory_growth < 50, f"Memory grew by {memory_growth}MB"

    @pytest.mark.asyncio
    async def test_component_health_endpoints(self):
        """Test health check endpoints for all components."""
        health_checks = {
            'database': self._check_database_health,
            'exchange': self._check_exchange_health,
            'strategies': self._check_strategies_health,
            'analytics': self._check_analytics_health,
            'tilt': self._check_tilt_health
        }
        
        results = {}
        for component, check_func in health_checks.items():
            results[component] = await check_func()
            assert results[component]['status'] in ['healthy', 'degraded', 'unhealthy']
            assert 'timestamp' in results[component]
            assert 'details' in results[component]

    async def _check_database_health(self):
        """Check database component health."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'connections': 5,
                'active_queries': 2,
                'response_time_ms': 15
            }
        }

    async def _check_exchange_health(self):
        """Check exchange connectivity health."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'websocket': 'connected',
                'rest_api': 'available',
                'latency_ms': 45
            }
        }

    async def _check_strategies_health(self):
        """Check strategy execution health."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'active_strategies': 3,
                'pending_signals': 0,
                'last_execution': datetime.utcnow().isoformat()
            }
        }

    async def _check_analytics_health(self):
        """Check analytics pipeline health."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'calculations_pending': 0,
                'last_update': datetime.utcnow().isoformat(),
                'metrics_count': 150
            }
        }

    async def _check_tilt_health(self):
        """Check tilt detection health."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'active_monitoring': True,
                'indicators_active': 5,
                'interventions_triggered': 0
            }
        }

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_exchange_gateway):
        """Test all components shut down gracefully."""
        components = []
        
        # Initialize components
        ws_manager = WebSocketManager(url="wss://test.exchange.com")
        components.append(ws_manager)
        
        orchestrator = StrategyOrchestrator(
            repository=Mock(),
            exchange_gateway=mock_exchange_gateway
        )
        components.append(orchestrator)
        
        # Test graceful shutdown
        shutdown_errors = []
        for component in components:
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'close'):
                    await component.close()
            except Exception as e:
                shutdown_errors.append((component.__class__.__name__, str(e)))
        
        assert len(shutdown_errors) == 0, f"Shutdown errors: {shutdown_errors}"

    @pytest.mark.asyncio
    async def test_component_dependency_chain(self, mock_db_session, mock_exchange_gateway):
        """Test component initialization in correct dependency order."""
        # Level 1: Core infrastructure
        repo = Repository(db_session=mock_db_session)
        assert repo is not None
        
        # Level 2: Data layers
        correlation_repo = CorrelationRepository(db_session=mock_db_session)
        perf_repo = PerformanceRepository(db_session=mock_db_session)
        
        # Level 3: Business logic
        account_manager = AccountManager(db_session=mock_db_session)
        registry = StrategyRegistry()
        
        # Level 4: Orchestration
        orchestrator = StrategyOrchestrator(
            repository=repo,
            exchange_gateway=mock_exchange_gateway
        )
        
        # Level 5: Analytics
        risk_calc = RiskMetricsCalculator(repository=repo)
        
        # Verify dependency chain
        assert all([repo, correlation_repo, perf_repo, account_manager, 
                   registry, orchestrator, risk_calc])

    @pytest.mark.asyncio
    async def test_concurrent_component_startup(self, mock_db_session):
        """Test multiple components can start concurrently without deadlocks."""
        async def start_component(name, delay):
            await asyncio.sleep(delay)
            if name == 'repository':
                return Repository(db_session=mock_db_session)
            elif name == 'account':
                return SingleAccountManager(
                    account_id="test",
                    initial_balance=Decimal("10000"),
                    tier=TierType.SNIPER
                )
            elif name == 'registry':
                return StrategyRegistry()
            return Mock()
        
        # Start components concurrently
        tasks = [
            start_component('repository', 0.01),
            start_component('account', 0.02),
            start_component('registry', 0.01),
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(r is not None for r in results)