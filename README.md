# Project GENESIS 🚀

An evolutionary cryptocurrency trading system with tier-based progression and tilt detection, designed to grow from $500 to $100k through disciplined, systematic trading.

## 🎯 Project Overview

Project GENESIS is a sophisticated cryptocurrency trading platform that implements:
- **Tier-based progression system** (Sniper → Hunter → Strategist)
- **Psychological tilt detection** to prevent emotional trading
- **Risk management** with strict position sizing and loss limits
- **Beautiful terminal UI** with a "Digital Zen Garden" aesthetic
- **Automated backup and recovery** systems

The system is designed to start small and evolve as capital grows, unlocking more sophisticated strategies and capabilities at each tier.

## 🏗️ Architecture

```
genesis/
├── core/          # Domain models and business logic
├── engine/        # Trading engine with state machine
├── tilt/          # Psychological monitoring system
├── strategies/    # Tier-specific trading strategies
├── exchange/      # Exchange API integration
├── data/          # Data persistence layer
├── analytics/     # Performance analysis
├── ui/            # Terminal user interface
└── utils/         # Shared utilities
```

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.11.8 (exact version required)
- Git
- Docker (optional, for containerized deployment)
- Binance API credentials

### Python Version Management

We recommend using pyenv to manage Python versions:

**Linux/macOS:**
```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Install Python 3.11.8
pyenv install 3.11.8
pyenv local 3.11.8
```

**Windows:**
```powershell
# Install pyenv-win (if not already installed)
git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv

# Install Python 3.11.8
pyenv install 3.11.8
pyenv local 3.11.8
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hookey13/genesis.git
   cd genesis
   ```

2. **Set up Python environment (Automated):**
   
   **Linux/macOS:**
   ```bash
   # Set tier (optional, defaults to sniper)
   export TIER=sniper  # or hunter, strategist
   
   # Activate environment and install dependencies
   source scripts/activate_env.sh
   ```
   
   **Windows PowerShell:**
   ```powershell
   # Set tier (optional, defaults to sniper)
   $env:TIER = "sniper"  # or hunter, strategist
   
   # Activate environment and install dependencies
   .\scripts\activate_env.ps1
   ```

   **Manual Setup (Alternative):**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   make setup
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials and settings
   ```

4. **Run the application:**
   ```bash
   make run
   ```

### Docker Deployment

For production deployment using Docker:

```bash
# Build and run with Docker
make build-docker
make run-docker

# View logs
make logs

# Stop containers
make stop-docker
```

## 📋 Available Commands

Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make setup` | Complete initial setup |
| `make install` | Install production dependencies |
| `make install-dev` | Install development dependencies |
| `make run` | Run the application |
| `make test` | Run all tests |
| `make format` | Format code with black |
| `make lint` | Run ruff linter |
| `make deploy` | Deploy to production |
| `make backup` | Run backup script |

## 🧪 Testing

The project uses pytest for testing with comprehensive coverage requirements:

```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run specific test suites
make test-unit        # Unit tests only
make test-integration # Integration tests only
```

Coverage requirements:
- 100% for money-handling paths
- 90% for risk management and tilt detection
- 70% for UI and analytics

## 🛠️ Development

### Code Standards

- **Python 3.11.8** exclusively (no 3.12 features)
- **Black** formatter with 88-character line length
- **Ruff** linter for fast code checking
- **Type hints** mandatory for all functions
- **Structured logging** with structlog (no print statements)

### Pre-commit Hooks

Pre-commit hooks are configured to ensure code quality:

```bash
# Install pre-commit hooks
make pre-commit

# Run manually
pre-commit run --all-files
```

### Contributing Guidelines

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass and coverage requirements are met
4. Run formatters and linters
5. Submit a pull request with clear description

## 📊 Tier System

The system implements three trading tiers with progressive unlocking:

| Tier | Capital Range | Features |
|------|--------------|----------|
| **Sniper** | $500-$2k | Single pair, market orders, basic arbitrage |
| **Hunter** | $2k-$10k | Multi-pair, iceberg orders, mean reversion |
| **Strategist** | $10k+ | Statistical arbitrage, VWAP execution, market making |

## 🔒 Security

- API credentials stored in environment variables
- Never commit `.env` files or secrets
- All data encrypted at rest
- Automated backups to DigitalOcean Spaces
- Circuit breakers for API failures

## 📝 Configuration

Key configuration options in `.env`:

```bash
# Exchange
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true

# Trading
TRADING_TIER=sniper
MAX_POSITION_SIZE_USDT=100.0
MAX_DAILY_LOSS_USDT=50.0

# Tilt Detection
TILT_CLICK_SPEED_THRESHOLD=5.0
TILT_CANCEL_RATE_THRESHOLD=0.5
```

See `.env.example` for all available options.

## 🚨 Emergency Procedures

In case of issues:

1. **Emergency position closure:**
   ```bash
   python scripts/emergency_close.py
   ```

2. **Database recovery:**
   ```bash
   python scripts/migrate_db.py
   ```

3. **Backup restoration:**
   ```bash
   bash scripts/restore_backup.sh
   ```

See [docs/runbook.md](docs/runbook.md) for detailed procedures.

## 📈 Performance Monitoring

The system includes comprehensive monitoring:

- Real-time P&L tracking
- Tilt detection indicators
- Correlation monitoring
- Performance analytics
- Automated alerts on critical events

## 🗺️ Roadmap

- [x] Project infrastructure setup
- [ ] Core trading engine implementation
- [ ] Binance API integration
- [ ] Tilt detection system
- [ ] Terminal UI development
- [ ] Sniper tier strategies
- [ ] Hunter tier features
- [ ] Strategist tier capabilities
- [ ] Production deployment

## 📄 License

This project is proprietary software. All rights reserved.

## 🤝 Support

For issues or questions:
- Check [docs/](docs/) for documentation
- Review [docs/runbook.md](docs/runbook.md) for troubleshooting
- Submit issues via GitHub

---

**Remember:** This system is designed for disciplined, systematic trading. The tilt detection system is there to protect you from yourself. Trust the process, respect the limits, and let compounding work its magic.

*"Slow is smooth, smooth is fast."*