"""Integration tests for multi-account institutional workflows - End-to-end validation."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from genesis.core.models import (
    Account, AccountType, Position, Order, Tier, 
    OrderSide, OrderType, TaxMethod
)
from genesis.core.account_manager import AccountManager
from genesis.analytics.compliance_reporter import ComplianceReporter
from genesis.analytics.risk_dashboard import RiskDashboard
from genesis.analytics.reconciliation import ReconciliationEngine
from genesis.analytics.tax_optimizer import TaxOptimizer
from genesis.exchange.fix_gateway import FIXGateway
from genesis.exchange.prime_broker import PrimeBrokerAdapter
from genesis.utils.disaster_recovery import DisasterRecovery
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import ExchangeGateway


class TestMultiAccountInstitutionalWorkflows:
    """End-to-end integration tests for institutional features."""
    
    @pytest.fixture
    async def setup_infrastructure(self):
        """Setup complete infrastructure for integration testing."""
        # Mock components
        repo = Mock(spec=Repository)
        repo.get_account = AsyncMock()
        repo.get_accounts_by_parent = AsyncMock()
        repo.save_account = AsyncMock()
        repo.get_positions_by_account = AsyncMock()
        repo.save_position = AsyncMock()
        repo.get_orders_by_account = AsyncMock()
        repo.save_order = AsyncMock()
        repo.save_event = AsyncMock()
        
        exchange = Mock(spec=ExchangeGateway)
        exchange.fetch_balance = AsyncMock()
        exchange.fetch_positions = AsyncMock()
        exchange.place_order = AsyncMock()
        exchange.cancel_order = AsyncMock()
        
        event_bus = EventBus()
        
        # Create components
        account_manager = AccountManager(repository=repo)
        compliance_reporter = ComplianceReporter(repository=repo)
        risk_dashboard = RiskDashboard(repository=repo, event_bus=event_bus)
        reconciliation_engine = ReconciliationEngine(repository=repo, exchange=exchange)
        tax_optimizer = TaxOptimizer(repository=repo)
        fix_gateway = FIXGateway(config={'sender_comp_id': 'GENESIS', 'target_comp_id': 'BROKER'})
        prime_broker = PrimeBrokerAdapter(config={'broker_type': 'GOLDMAN_SACHS'})
        disaster_recovery = DisasterRecovery(repository=repo, exchange=exchange, event_bus=event_bus)
        
        return {
            'repo': repo,
            'exchange': exchange,
            'event_bus': event_bus,
            'account_manager': account_manager,
            'compliance_reporter': compliance_reporter,
            'risk_dashboard': risk_dashboard,
            'reconciliation_engine': reconciliation_engine,
            'tax_optimizer': tax_optimizer,
            'fix_gateway': fix_gateway,
            'prime_broker': prime_broker,
            'disaster_recovery': disaster_recovery
        }
    
    @pytest.fixture
    def master_account(self):
        """Create master account with sub-accounts."""
        return Account(
            account_id="MASTER_001",
            account_type=AccountType.MASTER,
            tier=Tier.STRATEGIST,
            balance_usdt=Decimal("1000000.00"),
            permissions={
                'multi_account': True,
                'prime_broker': True,
                'fix_protocol': True,
                'compliance_reporting': True
            }
        )
    
    @pytest.fixture
    def sub_accounts(self, master_account):
        """Create sub-accounts under master."""
        return [
            Account(
                account_id="SUB_001",
                parent_account_id=master_account.account_id,
                account_type=AccountType.SUB,
                tier=Tier.STRATEGIST,
                balance_usdt=Decimal("300000.00"),
                compliance_settings={'jurisdiction': 'US', 'reporting': 'MiFID_II'}
            ),
            Account(
                account_id="SUB_002",
                parent_account_id=master_account.account_id,
                account_type=AccountType.SUB,
                tier=Tier.STRATEGIST,
                balance_usdt=Decimal("400000.00"),
                compliance_settings={'jurisdiction': 'EU', 'reporting': 'EMIR'}
            ),
            Account(
                account_id="PAPER_001",
                parent_account_id=master_account.account_id,
                account_type=AccountType.PAPER,
                tier=Tier.STRATEGIST,
                balance_usdt=Decimal("100000.00"),
                compliance_settings={'jurisdiction': 'TEST', 'reporting': 'NONE'}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_multi_account_creation_and_hierarchy(self, setup_infrastructure, master_account, sub_accounts):
        """Test creation and management of account hierarchy."""
        infra = await setup_infrastructure
        account_manager = infra['account_manager']
        repo = infra['repo']
        
        # Create master account
        master = await account_manager.create_account(master_account)
        assert master.account_type == AccountType.MASTER
        assert master.tier == Tier.STRATEGIST
        
        # Create sub-accounts
        created_subs = []
        for sub in sub_accounts:
            created_sub = await account_manager.create_sub_account(
                parent_id=master.account_id,
                account_data=sub
            )
            created_subs.append(created_sub)
            assert created_sub.parent_account_id == master.account_id
        
        # Verify hierarchy
        hierarchy = await account_manager.get_account_hierarchy(master.account_id)
        assert hierarchy['master'] == master.account_id
        assert len(hierarchy['sub_accounts']) == 3
        assert 'PAPER_001' in hierarchy['sub_accounts']
    
    @pytest.mark.asyncio
    async def test_multi_account_position_isolation(self, setup_infrastructure, sub_accounts):
        """Test that positions are properly isolated between accounts."""
        infra = await setup_infrastructure
        repo = infra['repo']
        exchange = infra['exchange']
        
        # Setup positions for different accounts
        positions_sub1 = [
            Position(
                position_id="pos_sub1_1",
                account_id="SUB_001",
                symbol="BTC/USDT",
                size=Decimal("5.0"),
                entry_price=Decimal("45000.00")
            )
        ]
        
        positions_sub2 = [
            Position(
                position_id="pos_sub2_1",
                account_id="SUB_002",
                symbol="ETH/USDT",
                size=Decimal("50.0"),
                entry_price=Decimal("3000.00")
            )
        ]
        
        repo.get_positions_by_account.side_effect = lambda acc_id: {
            "SUB_001": positions_sub1,
            "SUB_002": positions_sub2,
            "PAPER_001": []
        }.get(acc_id, [])
        
        # Verify isolation
        sub1_positions = await repo.get_positions_by_account("SUB_001")
        sub2_positions = await repo.get_positions_by_account("SUB_002")
        paper_positions = await repo.get_positions_by_account("PAPER_001")
        
        assert len(sub1_positions) == 1
        assert sub1_positions[0].symbol == "BTC/USDT"
        assert len(sub2_positions) == 1
        assert sub2_positions[0].symbol == "ETH/USDT"
        assert len(paper_positions) == 0
    
    @pytest.mark.asyncio
    async def test_compliance_reporting_workflow(self, setup_infrastructure, sub_accounts):
        """Test end-to-end compliance reporting across multiple accounts."""
        infra = await setup_infrastructure
        compliance_reporter = infra['compliance_reporter']
        repo = infra['repo']
        
        # Setup trade history
        trades = [
            {
                'trade_id': 'T001',
                'account_id': 'SUB_001',
                'symbol': 'BTC/USDT',
                'side': 'BUY',
                'quantity': Decimal('1.0'),
                'price': Decimal('45000.00'),
                'timestamp': datetime.now(timezone.utc) - timedelta(days=1)
            },
            {
                'trade_id': 'T002',
                'account_id': 'SUB_002',
                'symbol': 'ETH/USDT',
                'side': 'SELL',
                'quantity': Decimal('10.0'),
                'price': Decimal('3100.00'),
                'timestamp': datetime.now(timezone.utc)
            }
        ]
        
        repo.get_trades_for_period = AsyncMock(return_value=trades)
        
        # Generate MiFID II report for SUB_001
        mifid_report = await compliance_reporter.generate_report(
            account_id='SUB_001',
            report_type='MiFID_II',
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc)
        )
        
        assert mifid_report['report_type'] == 'MiFID_II'
        assert mifid_report['account_id'] == 'SUB_001'
        assert len(mifid_report['transactions']) >= 0
        assert 'lei_code' in mifid_report
        
        # Generate EMIR report for SUB_002
        emir_report = await compliance_reporter.generate_report(
            account_id='SUB_002',
            report_type='EMIR',
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc)
        )
        
        assert emir_report['report_type'] == 'EMIR'
        assert emir_report['account_id'] == 'SUB_002'
    
    @pytest.mark.asyncio
    async def test_risk_metrics_aggregation(self, setup_infrastructure, master_account, sub_accounts):
        """Test VaR and CVaR calculation across account portfolio."""
        infra = await setup_infrastructure
        risk_dashboard = infra['risk_dashboard']
        repo = infra['repo']
        
        # Setup portfolio data
        portfolio_data = {
            'SUB_001': {
                'positions': [
                    {'symbol': 'BTC/USDT', 'value': Decimal('225000.00'), 'size': Decimal('5.0')}
                ],
                'historical_returns': [0.02, -0.01, 0.03, -0.02, 0.01] * 20  # 100 data points
            },
            'SUB_002': {
                'positions': [
                    {'symbol': 'ETH/USDT', 'value': Decimal('155000.00'), 'size': Decimal('50.0')}
                ],
                'historical_returns': [0.03, -0.02, 0.04, -0.03, 0.02] * 20
            }
        }
        
        # Calculate portfolio VaR at 95% confidence
        var_95 = await risk_dashboard.calculate_portfolio_var(
            account_ids=['SUB_001', 'SUB_002'],
            confidence_level=0.95,
            portfolio_data=portfolio_data
        )
        
        assert isinstance(var_95, Decimal)
        assert var_95 > Decimal('0')
        
        # Calculate CVaR
        cvar_95 = await risk_dashboard.calculate_portfolio_cvar(
            account_ids=['SUB_001', 'SUB_002'],
            confidence_level=0.95,
            portfolio_data=portfolio_data
        )
        
        assert isinstance(cvar_95, Decimal)
        assert cvar_95 >= var_95  # CVaR should be >= VaR
    
    @pytest.mark.asyncio
    async def test_month_end_reconciliation_workflow(self, setup_infrastructure, sub_accounts):
        """Test automated month-end reconciliation across all accounts."""
        infra = await setup_infrastructure
        reconciliation_engine = infra['reconciliation_engine']
        repo = infra['repo']
        exchange = infra['exchange']
        
        # Setup balances
        db_balances = {
            'SUB_001': Decimal('300000.00'),
            'SUB_002': Decimal('400000.00'),
            'PAPER_001': Decimal('100000.00')
        }
        
        exchange_balances = {
            'SUB_001': {'USDT': {'total': 299995.50}},  # Small discrepancy
            'SUB_002': {'USDT': {'total': 400000.00}},  # Match
            'PAPER_001': {'USDT': {'total': 100000.00}}  # Match (paper trading)
        }
        
        repo.get_all_accounts.return_value = sub_accounts
        repo.get_account_balance = AsyncMock(side_effect=lambda acc_id: db_balances.get(acc_id))
        exchange.fetch_balance = AsyncMock(side_effect=lambda acc_id: exchange_balances.get(acc_id))
        
        # Run month-end reconciliation
        reports = await reconciliation_engine.run_month_end_reconciliation()
        
        assert len(reports) == 3
        
        # Check SUB_001 has discrepancy
        sub1_report = next(r for r in reports if r.account_id == 'SUB_001')
        assert sub1_report.status == 'DISCREPANCY'
        assert sub1_report.total_discrepancy == Decimal('4.50')
        
        # Check SUB_002 matches
        sub2_report = next(r for r in reports if r.account_id == 'SUB_002')
        assert sub2_report.status == 'MATCHED'
    
    @pytest.mark.asyncio
    async def test_tax_optimization_workflow(self, setup_infrastructure):
        """Test tax-aware position closing across accounts."""
        infra = await setup_infrastructure
        tax_optimizer = infra['tax_optimizer']
        repo = infra['repo']
        
        # Setup tax lots
        tax_lots = [
            {
                'lot_id': 'LOT_001',
                'account_id': 'SUB_001',
                'symbol': 'BTC/USDT',
                'quantity': Decimal('2.0'),
                'cost_basis': Decimal('80000.00'),
                'acquisition_date': datetime.now(timezone.utc) - timedelta(days=400),
                'current_price': Decimal('45000.00')
            },
            {
                'lot_id': 'LOT_002',
                'account_id': 'SUB_001',
                'symbol': 'BTC/USDT',
                'quantity': Decimal('3.0'),
                'cost_basis': Decimal('135000.00'),
                'acquisition_date': datetime.now(timezone.utc) - timedelta(days=100),
                'current_price': Decimal('45000.00')
            }
        ]
        
        repo.get_tax_lots_by_account = AsyncMock(return_value=tax_lots)
        
        # Optimize for HIFO method
        optimal_lots = await tax_optimizer.select_lots_for_sale(
            account_id='SUB_001',
            symbol='BTC/USDT',
            quantity=Decimal('3.0'),
            method=TaxMethod.HIFO
        )
        
        assert len(optimal_lots) > 0
        assert optimal_lots[0]['lot_id'] == 'LOT_002'  # Higher cost basis selected first
        
        # Calculate tax impact
        tax_impact = await tax_optimizer.calculate_tax_impact(
            lots_to_sell=optimal_lots,
            sale_price=Decimal('45000.00')
        )
        
        assert 'capital_gain' in tax_impact
        assert 'tax_liability' in tax_impact
        assert isinstance(tax_impact['capital_gain'], Decimal)
    
    @pytest.mark.asyncio
    async def test_fix_prime_broker_order_routing(self, setup_infrastructure):
        """Test order routing through FIX gateway to prime broker."""
        infra = await setup_infrastructure
        fix_gateway = infra['fix_gateway']
        prime_broker = infra['prime_broker']
        
        # Create order
        order = Order(
            order_id='ORD_001',
            client_order_id='CLIENT_001',
            account_id='SUB_001',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('44500.00')
        )
        
        # Convert to FIX message
        fix_message = fix_gateway.order_to_fix(order)
        assert fix_message.msg_type == 'D'  # New Order Single
        assert fix_message.get_field('55') == 'BTC/USDT'  # Symbol
        
        # Route through prime broker
        with patch.object(prime_broker, 'send_order') as mock_send:
            mock_send.return_value = AsyncMock(return_value={
                'status': 'FILLED',
                'avg_price': Decimal('44499.50'),
                'fills': [
                    {'venue': 'BINANCE', 'quantity': Decimal('0.6'), 'price': Decimal('44499.00')},
                    {'venue': 'COINBASE', 'quantity': Decimal('0.4'), 'price': Decimal('44500.00')}
                ]
            })
            
            execution = await prime_broker.route_order_smart(order)
            
            assert execution['status'] == 'FILLED'
            assert len(execution['fills']) == 2
            assert execution['avg_price'] == Decimal('44499.50')
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_workflow(self, setup_infrastructure, sub_accounts):
        """Test complete disaster recovery workflow."""
        infra = await setup_infrastructure
        disaster_recovery = infra['disaster_recovery']
        repo = infra['repo']
        exchange = infra['exchange']
        
        # Simulate system failure detection
        with patch.object(disaster_recovery, 'detect_failure', return_value=True):
            # Setup open positions and orders
            open_orders = [
                Order(order_id='ORD_001', status='OPEN', account_id='SUB_001'),
                Order(order_id='ORD_002', status='OPEN', account_id='SUB_002')
            ]
            
            open_positions = [
                Position(position_id='POS_001', account_id='SUB_001', size=Decimal('1.0')),
                Position(position_id='POS_002', account_id='SUB_002', size=Decimal('10.0'))
            ]
            
            exchange.get_open_orders.return_value = open_orders
            exchange.get_positions.return_value = open_positions
            exchange.cancel_order = AsyncMock(return_value={'status': 'CANCELLED'})
            exchange.close_position = AsyncMock(return_value={'status': 'FILLED'})
            
            # Execute emergency procedures
            recovery_result = await disaster_recovery.execute_emergency_procedures()
            
            assert recovery_result['orders_cancelled'] == 2
            assert recovery_result['positions_closed'] == 2
            assert recovery_result['snapshot_created'] is True
            assert recovery_result['notifications_sent'] is True
    
    @pytest.mark.asyncio
    async def test_account_switching_audit_trail(self, setup_infrastructure, master_account, sub_accounts):
        """Test that account switching is properly audited."""
        infra = await setup_infrastructure
        account_manager = infra['account_manager']
        repo = infra['repo']
        event_bus = infra['event_bus']
        
        # Track events
        events_published = []
        event_bus.publish = AsyncMock(side_effect=lambda e: events_published.append(e))
        
        # Switch between accounts
        await account_manager.switch_account(
            from_account='MASTER_001',
            to_account='SUB_001',
            user_id='USER_001'
        )
        
        await account_manager.switch_account(
            from_account='SUB_001',
            to_account='SUB_002',
            user_id='USER_001'
        )
        
        # Verify audit events
        assert len(events_published) == 2
        assert events_published[0].event_type == 'ACCOUNT_SWITCHED'
        assert events_published[0].data['from_account'] == 'MASTER_001'
        assert events_published[0].data['to_account'] == 'SUB_001'
        assert events_published[1].data['from_account'] == 'SUB_001'
        assert events_published[1].data['to_account'] == 'SUB_002'
    
    @pytest.mark.asyncio
    async def test_institutional_feature_access_control(self, setup_infrastructure, master_account):
        """Test tier-based access control for institutional features."""
        infra = await setup_infrastructure
        
        # Test with Strategist tier (should have access)
        master_account.tier = Tier.STRATEGIST
        
        # All institutional features should be accessible
        assert await infra['account_manager'].can_use_multi_account(master_account)
        assert await infra['compliance_reporter'].can_generate_reports(master_account)
        assert await infra['risk_dashboard'].can_view_advanced_metrics(master_account)
        assert await infra['tax_optimizer'].can_optimize_taxes(master_account)
        
        # Test with Hunter tier (should not have access)
        hunter_account = Account(
            account_id='HUNTER_001',
            tier=Tier.HUNTER,
            balance_usdt=Decimal('5000.00')
        )
        
        assert not await infra['account_manager'].can_use_multi_account(hunter_account)
        assert not await infra['compliance_reporter'].can_generate_reports(hunter_account)
        assert not await infra['risk_dashboard'].can_view_advanced_metrics(hunter_account)
        assert not await infra['tax_optimizer'].can_optimize_taxes(hunter_account)
    
    @pytest.mark.asyncio
    async def test_decimal_precision_across_workflow(self, setup_infrastructure):
        """Test that Decimal precision is maintained throughout all workflows."""
        infra = await setup_infrastructure
        
        # Test precise values through various components
        precise_value = Decimal('123456.789012345678901234567890')
        
        # Risk calculation
        risk_result = await infra['risk_dashboard'].calculate_position_risk(
            position_value=precise_value,
            volatility=Decimal('0.02')
        )
        assert isinstance(risk_result, Decimal)
        
        # Tax calculation
        tax_result = await infra['tax_optimizer'].calculate_tax_on_gain(
            gain_amount=precise_value,
            tax_rate=Decimal('0.23')
        )
        assert isinstance(tax_result, Decimal)
        
        # Reconciliation difference
        recon_diff = infra['reconciliation_engine'].calculate_discrepancy(
            db_value=precise_value,
            exchange_value=Decimal('123456.789012345678901234567891')
        )
        assert isinstance(recon_diff, Decimal)
        assert recon_diff == Decimal('0.000000000000000000000000000001')