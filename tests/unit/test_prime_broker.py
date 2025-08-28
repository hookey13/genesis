"""Unit tests for PrimeBrokerAdapter - Integration point validation for prime brokers."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from genesis.exchange.prime_broker import (
    PrimeBrokerAdapter,
    PrimeBrokerType,
    AllocationInstruction,
    ExecutionReport,
    PrimeBrokerConfig,
    MultiVenueOrder,
    VenueAllocation
)
from genesis.core.models import Order, OrderType, OrderSide, Position, Account


class TestPrimeBrokerAdapter:
    """Test suite for prime broker integration adapter."""
    
    @pytest.fixture
    def gs_config(self):
        """Goldman Sachs prime broker configuration."""
        return PrimeBrokerConfig(
            broker_type=PrimeBrokerType.GOLDMAN_SACHS,
            api_endpoint="https://api.gs.com/trading",
            api_key="test_api_key",
            api_secret="test_secret",
            account_id="GS_ACCOUNT_001",
            supported_venues=["BINANCE", "COINBASE", "KRAKEN"],
            allocation_strategy="SMART_ROUTING"
        )
    
    @pytest.fixture
    def ms_config(self):
        """Morgan Stanley prime broker configuration."""
        return PrimeBrokerConfig(
            broker_type=PrimeBrokerType.MORGAN_STANLEY,
            api_endpoint="https://api.morganstanley.com/trading",
            api_key="test_api_key",
            api_secret="test_secret",
            account_id="MS_ACCOUNT_001",
            supported_venues=["CME", "ICE", "BINANCE"],
            allocation_strategy="BEST_EXECUTION"
        )
    
    @pytest.fixture
    def gs_adapter(self, gs_config):
        """Create Goldman Sachs adapter instance."""
        return PrimeBrokerAdapter(config=gs_config)
    
    @pytest.fixture
    def ms_adapter(self, ms_config):
        """Create Morgan Stanley adapter instance."""
        return PrimeBrokerAdapter(config=ms_config)
    
    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id="ORD_123456",
            client_order_id="CLIENT_789",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("45000.00"),
            time_in_force="GTC",
            account_id="ACC_001"
        )
    
    @pytest.mark.asyncio
    async def test_goldman_sachs_authentication(self, gs_adapter, gs_config):
        """Test Goldman Sachs API authentication."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'access_token': 'test_token', 'expires_in': 3600}
            )
            
            # Execute
            token = await gs_adapter.authenticate()
            
            # Verify
            assert token == 'test_token'
            assert gs_adapter.is_authenticated()
            mock_post.assert_called_with(
                f"{gs_config.api_endpoint}/auth",
                json={'api_key': gs_config.api_key, 'api_secret': gs_config.api_secret}
            )
    
    @pytest.mark.asyncio
    async def test_morgan_stanley_authentication(self, ms_adapter, ms_config):
        """Test Morgan Stanley API authentication."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={'session_id': 'ms_session_123', 'ttl': 7200}
            )
            
            # Execute
            session = await ms_adapter.authenticate()
            
            # Verify
            assert session == 'ms_session_123'
            assert ms_adapter.is_authenticated()
    
    @pytest.mark.asyncio
    async def test_multi_venue_order_routing(self, gs_adapter, sample_order):
        """Test smart order routing across multiple venues."""
        # Setup
        venue_allocations = [
            VenueAllocation(venue="BINANCE", quantity=Decimal("5.0"), price=Decimal("44999.00")),
            VenueAllocation(venue="COINBASE", quantity=Decimal("3.0"), price=Decimal("45001.00")),
            VenueAllocation(venue="KRAKEN", quantity=Decimal("2.0"), price=Decimal("45000.50"))
        ]
        
        with patch.object(gs_adapter, 'calculate_smart_routing', return_value=venue_allocations):
            with patch.object(gs_adapter, 'send_order_to_venue') as mock_send:
                mock_send.return_value = AsyncMock(return_value={'status': 'FILLED'})
                
                # Execute
                execution = await gs_adapter.route_order(sample_order)
                
                # Verify
                assert execution.total_quantity == sample_order.quantity
                assert len(execution.venue_fills) == 3
                assert mock_send.call_count == 3
    
    @pytest.mark.asyncio
    async def test_allocation_instruction_processing(self, gs_adapter):
        """Test processing of post-trade allocation instructions."""
        # Setup
        allocation = AllocationInstruction(
            master_order_id="MASTER_001",
            allocations=[
                {"account_id": "SUB_001", "quantity": Decimal("3.0")},
                {"account_id": "SUB_002", "quantity": Decimal("5.0")},
                {"account_id": "SUB_003", "quantity": Decimal("2.0")}
            ],
            symbol="BTC/USD",
            executed_price=Decimal("45000.00")
        )
        
        with patch.object(gs_adapter, 'process_allocation') as mock_process:
            mock_process.return_value = AsyncMock(return_value={'status': 'ALLOCATED'})
            
            # Execute
            result = await gs_adapter.allocate_trade(allocation)
            
            # Verify
            assert result['status'] == 'ALLOCATED'
            assert sum(a["quantity"] for a in allocation.allocations) == Decimal("10.0")
    
    @pytest.mark.asyncio
    async def test_best_execution_analysis(self, ms_adapter):
        """Test best execution price discovery across venues."""
        # Setup
        market_data = {
            "CME": {"bid": Decimal("44998.00"), "ask": Decimal("45002.00"), "liquidity": Decimal("100.0")},
            "ICE": {"bid": Decimal("44999.00"), "ask": Decimal("45001.00"), "liquidity": Decimal("50.0")},
            "BINANCE": {"bid": Decimal("45000.00"), "ask": Decimal("45000.50"), "liquidity": Decimal("200.0")}
        }
        
        with patch.object(ms_adapter, 'fetch_market_data', return_value=market_data):
            # Execute
            best_venue = await ms_adapter.find_best_execution(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=Decimal("10.0")
            )
            
            # Verify
            assert best_venue.venue == "BINANCE"  # Best ask price
            assert best_venue.expected_price == Decimal("45000.50")
    
    @pytest.mark.asyncio
    async def test_position_aggregation_across_venues(self, gs_adapter):
        """Test aggregation of positions across multiple trading venues."""
        # Setup
        venue_positions = {
            "BINANCE": [
                {"symbol": "BTC/USD", "quantity": Decimal("5.0"), "avg_price": Decimal("45000.00")}
            ],
            "COINBASE": [
                {"symbol": "BTC/USD", "quantity": Decimal("3.0"), "avg_price": Decimal("44950.00")}
            ],
            "KRAKEN": [
                {"symbol": "BTC/USD", "quantity": Decimal("2.0"), "avg_price": Decimal("45100.00")}
            ]
        }
        
        with patch.object(gs_adapter, 'fetch_venue_positions', side_effect=lambda v: venue_positions[v]):
            # Execute
            aggregated = await gs_adapter.get_aggregated_positions()
            
            # Verify
            btc_position = aggregated["BTC/USD"]
            assert btc_position["total_quantity"] == Decimal("10.0")
            # Weighted average price
            expected_avg = (Decimal("5.0") * Decimal("45000.00") + 
                          Decimal("3.0") * Decimal("44950.00") + 
                          Decimal("2.0") * Decimal("45100.00")) / Decimal("10.0")
            assert abs(btc_position["avg_price"] - expected_avg) < Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_risk_limit_checking(self, gs_adapter):
        """Test pre-trade risk limit validation."""
        # Setup
        risk_limits = {
            "max_position_size": Decimal("100.0"),
            "max_order_size": Decimal("20.0"),
            "max_daily_volume": Decimal("1000.0"),
            "current_position": Decimal("85.0"),
            "daily_volume": Decimal("500.0")
        }
        
        with patch.object(gs_adapter, 'get_risk_limits', return_value=risk_limits):
            # Test order that passes
            order1 = Order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=Decimal("10.0")
            )
            assert await gs_adapter.check_risk_limits(order1) is True
            
            # Test order that exceeds position limit
            order2 = Order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=Decimal("20.0")
            )
            assert await gs_adapter.check_risk_limits(order2) is False
    
    @pytest.mark.asyncio
    async def test_execution_report_processing(self, ms_adapter):
        """Test processing of execution reports from prime broker."""
        # Setup
        raw_report = {
            "order_id": "PB_ORDER_123",
            "client_order_id": "CLIENT_456",
            "symbol": "BTC/USD",
            "side": "BUY",
            "executed_quantity": 10.0,
            "average_price": 45000.50,
            "fills": [
                {"venue": "CME", "quantity": 5.0, "price": 45000.00, "timestamp": "2024-01-01T10:00:00Z"},
                {"venue": "BINANCE", "quantity": 5.0, "price": 45001.00, "timestamp": "2024-01-01T10:00:01Z"}
            ],
            "commission": 25.00,
            "status": "FILLED"
        }
        
        # Execute
        report = ms_adapter.parse_execution_report(raw_report)
        
        # Verify
        assert report.client_order_id == "CLIENT_456"
        assert report.total_quantity == Decimal("10.0")
        assert report.average_price == Decimal("45000.50")
        assert len(report.fills) == 2
        assert report.total_commission == Decimal("25.00")
        assert all(isinstance(fill["quantity"], Decimal) for fill in report.fills)
    
    @pytest.mark.asyncio
    async def test_margin_requirement_calculation(self, gs_adapter):
        """Test margin requirement calculations for leveraged trading."""
        # Setup
        position = Position(
            symbol="BTC/USD",
            size=Decimal("10.0"),
            entry_price=Decimal("45000.00"),
            leverage=Decimal("5.0")
        )
        
        # Execute
        margin = await gs_adapter.calculate_margin_requirement(position)
        
        # Verify
        expected_margin = (position.size * position.entry_price) / position.leverage
        assert margin.initial_margin == expected_margin
        assert margin.maintenance_margin == expected_margin * Decimal("0.5")  # Typical 50% maintenance
        assert isinstance(margin.initial_margin, Decimal)
        assert isinstance(margin.maintenance_margin, Decimal)
    
    @pytest.mark.asyncio
    async def test_commission_calculation(self, ms_adapter):
        """Test commission calculation based on prime broker fee schedule."""
        # Setup
        fee_schedule = {
            "maker_fee": Decimal("0.0002"),  # 2 bps
            "taker_fee": Decimal("0.0004"),  # 4 bps
            "min_fee": Decimal("1.00"),
            "volume_discount": [
                {"threshold": Decimal("1000000"), "discount": Decimal("0.1")},
                {"threshold": Decimal("10000000"), "discount": Decimal("0.2")}
            ]
        }
        
        with patch.object(ms_adapter, 'get_fee_schedule', return_value=fee_schedule):
            # Execute
            commission = await ms_adapter.calculate_commission(
                order_value=Decimal("450000.00"),  # $450k order
                order_type="TAKER",
                monthly_volume=Decimal("5000000.00")  # $5M monthly volume
            )
            
            # Verify
            base_fee = Decimal("450000.00") * Decimal("0.0004")
            discounted_fee = base_fee * Decimal("0.9")  # 10% discount
            assert commission == max(discounted_fee, fee_schedule["min_fee"])
    
    @pytest.mark.asyncio
    async def test_settlement_instruction_generation(self, gs_adapter):
        """Test generation of settlement instructions."""
        # Setup
        trades = [
            {"symbol": "BTC/USD", "quantity": Decimal("5.0"), "price": Decimal("45000.00"), "venue": "BINANCE"},
            {"symbol": "ETH/USD", "quantity": Decimal("50.0"), "price": Decimal("3000.00"), "venue": "COINBASE"}
        ]
        
        # Execute
        settlement = await gs_adapter.generate_settlement_instructions(
            trades=trades,
            settlement_date=datetime(2024, 1, 3, tzinfo=timezone.utc)
        )
        
        # Verify
        assert settlement.total_value == Decimal("375000.00")  # 5*45000 + 50*3000
        assert len(settlement.instructions) == 2
        assert settlement.settlement_date == datetime(2024, 1, 3, tzinfo=timezone.utc)
        assert all(isinstance(inst["value"], Decimal) for inst in settlement.instructions)
    
    @pytest.mark.asyncio
    async def test_connection_failover(self, gs_adapter):
        """Test failover to backup connection endpoints."""
        # Setup
        primary_endpoint = "https://api.gs.com/trading"
        backup_endpoints = [
            "https://api-backup1.gs.com/trading",
            "https://api-backup2.gs.com/trading"
        ]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Primary fails, first backup succeeds
            mock_post.side_effect = [
                Exception("Connection timeout"),
                MagicMock(__aenter__=AsyncMock(return_value=MagicMock(
                    json=AsyncMock(return_value={'status': 'connected'})
                )))
            ]
            
            # Execute
            connection = await gs_adapter.connect_with_failover(
                primary=primary_endpoint,
                backups=backup_endpoints
            )
            
            # Verify
            assert connection['status'] == 'connected'
            assert mock_post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_regulatory_reporting(self, ms_adapter):
        """Test regulatory reporting data generation."""
        # Setup
        trade = {
            "trade_id": "TRADE_001",
            "symbol": "BTC/USD",
            "quantity": Decimal("10.0"),
            "price": Decimal("45000.00"),
            "timestamp": datetime.now(timezone.utc),
            "counterparty": "BINANCE",
            "client_id": "CLIENT_001"
        }
        
        # Execute
        regulatory_report = await ms_adapter.generate_regulatory_report(
            trade=trade,
            report_type="MiFID_II"
        )
        
        # Verify
        assert regulatory_report.trade_id == "TRADE_001"
        assert regulatory_report.lei_code is not None  # Legal Entity Identifier
        assert regulatory_report.transaction_reference is not None
        assert regulatory_report.reporting_timestamp is not None
        assert isinstance(regulatory_report.notional_value, Decimal)
    
    def test_decimal_precision_preservation(self, gs_adapter):
        """Test that Decimal precision is preserved in all calculations."""
        # Setup
        order = Order(
            symbol="BTC/USD",
            quantity=Decimal("1.234567890123456789"),
            price=Decimal("45678.987654321098765")
        )
        
        # Execute
        order_value = gs_adapter.calculate_order_value(order)
        
        # Verify
        expected = order.quantity * order.price
        assert order_value == expected
        assert isinstance(order_value, Decimal)
        # Check precision is maintained
        assert str(order_value).split('.')[1][:10] == str(expected).split('.')[1][:10]