# Source Tree

The directory structure physically enforces tier boundaries and supports the evolutionary path from monolith to services. Tier-specific features are literally in separate directories that can be conditionally loaded.

```plaintext
genesis/
├── .env.example                    # Template for environment variables
├── .gitignore                      # Never commit secrets or data
├── Makefile                        # Simple commands: make run, make test, make deploy
├── README.md                       # Setup and architecture overview
├── requirements/
│   ├── base.txt                   # Core dependencies (all tiers)
│   ├── sniper.txt                 # -r base.txt only
│   ├── hunter.txt                 # -r base.txt + slicing libs
│   ├── strategist.txt             # -r hunter.txt + advanced analytics
│   └── dev.txt                    # Testing and development tools
│
├── alembic/                        # Database migrations
│   ├── versions/
│   │   ├── 001_initial_schema.py
│   │   ├── 002_add_correlation_view.py
│   │   └── 003_sqlite_to_postgres.py
│   └── alembic.ini
│
├── config/
│   ├── __init__.py
│   ├── settings.py                # Pydantic settings with validation
│   ├── tier_gates.yaml            # Tier progression requirements
│   └── trading_rules.yaml         # Risk limits by tier
│
├── genesis/                        # Main application package
│   ├── __init__.py
│   ├── __main__.py               # Entry point: python -m genesis
│   │
│   ├── core/                      # Domain core (all tiers)
│   │   ├── __init__.py
│   │   ├── models.py              # Domain models (Position, Order, etc.)
│   │   ├── events.py              # Event definitions
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── constants.py           # Enums and constants
│   │
│   ├── engine/                    # Trading engine
│   │   ├── __init__.py
│   │   ├── event_bus.py          # Priority lanes for events
│   │   ├── state_machine.py      # Tier state machine with decorators
│   │   ├── risk_engine.py        # Position sizing and limits
│   │   └── executor/
│   │       ├── __init__.py
│   │       ├── base.py           # Abstract executor
│   │       ├── market.py         # Simple market orders (Sniper)
│   │       ├── iceberg.py        # Order slicing (Hunter+)
│   │       └── vwap.py           # VWAP execution (Strategist+)
│   │
│   ├── tilt/                      # Psychological monitoring
│   │   ├── __init__.py
│   │   ├── detector.py           # Behavioral analysis
│   │   ├── baseline.py           # Baseline calculation
│   │   ├── interventions.py      # Intervention strategies
│   │   └── indicators/
│   │       ├── __init__.py
│   │       ├── click_speed.py
│   │       ├── cancel_rate.py
│   │       ├── revenge_trading.py
│   │       └── position_sizing.py
│   │
│   ├── strategies/                # Trading strategies by tier
│   │   ├── __init__.py
│   │   ├── loader.py             # Dynamic strategy loading by tier
│   │   ├── base.py               # Abstract strategy
│   │   ├── sniper/               # $500-$2k strategies
│   │   │   ├── __init__.py
│   │   │   ├── simple_arb.py    # Basic arbitrage
│   │   │   └── spread_capture.py
│   │   ├── hunter/               # $2k-$10k strategies  
│   │   │   ├── __init__.py
│   │   │   ├── multi_pair.py    # Concurrent pair trading
│   │   │   └── mean_reversion.py
│   │   └── strategist/           # $10k+ strategies
│   │       ├── __init__.py      # LOCKED until tier reached
│   │       ├── statistical_arb.py
│   │       └── market_making.py
│   │
│   ├── exchange/                  # Exchange interaction
│   │   ├── __init__.py
│   │   ├── gateway.py            # Binance API wrapper
│   │   ├── websocket_manager.py  # WS connection resilience
│   │   ├── circuit_breaker.py   # Failure protection
│   │   └── time_sync.py          # NTP synchronization
│   │
│   ├── data/                      # Data layer
│   │   ├── __init__.py
│   │   ├── repository.py         # Abstract repository
│   │   ├── sqlite_repo.py        # SQLite implementation
│   │   ├── postgres_repo.py      # PostgreSQL implementation
│   │   ├── migrations.py         # Migration coordinator
│   │   └── models_db.py          # SQLAlchemy models
│   │
│   ├── analytics/                 # Performance analysis
│   │   ├── __init__.py
│   │   ├── metrics.py            # Performance calculations
│   │   ├── forensics.py          # Event replay and analysis
│   │   ├── reports.py            # Report generation
│   │   └── correlation.py        # Correlation monitoring
│   │
│   ├── ui/                        # Terminal interface
│   │   ├── __init__.py
│   │   ├── app.py                # Textual application
│   │   ├── dashboard.py          # Main trading view
│   │   ├── commands.py           # Command parser
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── positions.py      # Position display
│   │   │   ├── pnl.py           # P&L widget
│   │   │   ├── tilt_indicator.py # Tilt warning display
│   │   │   └── tier_progress.py  # Gate completion
│   │   └── themes/
│   │       ├── __init__.py
│   │       └── zen_garden.py     # Calm color scheme
│   │
│   ├── api/                       # Internal REST API
│   │   ├── __init__.py
│   │   ├── server.py             # FastAPI app (added at $2k)
│   │   ├── routes.py             # API endpoints
│   │   └── auth.py               # API key validation
│   │
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── decorators.py         # @requires_tier, @with_timeout
│       ├── logger.py             # Structlog configuration
│       ├── math.py               # Decimal operations
│       └── validators.py         # Input validation
│
├── scripts/                        # Operational scripts
│   ├── backup.sh                 # Backup to DigitalOcean Spaces
│   ├── deploy.sh                 # Deployment script
│   ├── emergency_close.py        # Close all positions
│   ├── migrate_db.py             # Database migration
│   └── verify_checksums.py       # Code integrity check
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── unit/
│   │   ├── test_risk_engine.py
│   │   ├── test_state_machine.py
│   │   ├── test_tilt_detector.py
│   │   └── test_executor.py
│   ├── integration/
│   │   ├── test_order_flow.py
│   │   ├── test_tier_transition.py
│   │   └── test_emergency_demotion.py
│   └── fixtures/
│       ├── market_data.json
│       └── test_events.json
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml        # Local development
│   └── docker-compose.prod.yml   # Production deployment
│
├── terraform/                      # Infrastructure as Code (added at $2k)
│   ├── main.tf
│   ├── digitalocean.tf
│   └── monitoring.tf
│
├── docs/
│   ├── architecture.md           # This document
│   ├── runbook.md                # Emergency procedures
│   └── post_mortem_template.md   # For learning from failures
│
└── .genesis/                      # Runtime data (git-ignored)
    ├── data/
    │   ├── genesis.db            # SQLite database
    │   └── backups/              # Local backups
    ├── logs/
    │   ├── trading.log           # Main log
    │   ├── audit.log             # All events
    │   └── tilt.log              # Behavioral monitoring
    └── state/
        ├── tier_state.json       # Current tier and gates
        └── checksum.sha256       # Code integrity
```
