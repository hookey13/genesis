#!/bin/bash
# Genesis Trading System - Complete Development Environment Setup
# This script sets up a complete development environment for Genesis

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Genesis Trading System - Dev Environment Setup       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install pyenv
install_pyenv() {
    echo -e "${YELLOW}Installing pyenv...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew install pyenv
        else
            curl https://pyenv.run | bash
        fi
    else
        # Linux
        curl https://pyenv.run | bash
    fi
    
    # Add to shell profile
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    
    # Source for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
}

# Step 1: Check Python version
echo -e "${YELLOW}[1/10] Checking Python version...${NC}"
if command_exists pyenv; then
    echo -e "${GREEN}✓ pyenv is installed${NC}"
    
    # Install Python 3.11.8 if not available
    if ! pyenv versions | grep -q "3.11.8"; then
        echo -e "${YELLOW}Installing Python 3.11.8...${NC}"
        pyenv install 3.11.8
    fi
    
    # Set local Python version
    cd "$PROJECT_ROOT"
    pyenv local 3.11.8
    echo -e "${GREEN}✓ Python 3.11.8 is set as local version${NC}"
else
    echo -e "${YELLOW}pyenv not found. Would you like to install it? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        install_pyenv
        pyenv install 3.11.8
        pyenv local 3.11.8
    else
        echo -e "${RED}⚠ pyenv is recommended for Python version management${NC}"
    fi
fi

# Step 2: Create virtual environment
echo -e "${YELLOW}[2/10] Setting up virtual environment...${NC}"
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    python3.11 -m venv "$PROJECT_ROOT/.venv"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Step 3: Upgrade pip and install pip-tools
echo -e "${YELLOW}[3/10] Upgrading pip and installing pip-tools...${NC}"
pip install --quiet --upgrade pip pip-tools wheel setuptools
echo -e "${GREEN}✓ pip and pip-tools installed${NC}"

# Step 4: Install Poetry
echo -e "${YELLOW}[4/10] Installing Poetry...${NC}"
if ! command_exists poetry; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}✓ Poetry installed${NC}"
else
    echo -e "${GREEN}✓ Poetry already installed${NC}"
fi

# Step 5: Install dependencies
echo -e "${YELLOW}[5/10] Installing project dependencies...${NC}"

# Detect tier from environment or use default
TIER=${TIER:-sniper}
echo -e "${BLUE}Installing for tier: $TIER${NC}"

# Install base requirements
if [ -f "$PROJECT_ROOT/requirements/base.txt" ]; then
    pip install -q -r "$PROJECT_ROOT/requirements/base.txt"
fi

# Install tier-specific requirements
if [ -f "$PROJECT_ROOT/requirements/${TIER}.txt" ]; then
    pip install -q -r "$PROJECT_ROOT/requirements/${TIER}.txt"
fi

# Install dev requirements
if [ -f "$PROJECT_ROOT/requirements/dev.txt" ]; then
    pip install -q -r "$PROJECT_ROOT/requirements/dev.txt"
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Step 6: Setup pre-commit hooks
echo -e "${YELLOW}[6/10] Setting up pre-commit hooks...${NC}"
if [ -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠ .pre-commit-config.yaml not found${NC}"
fi

# Step 7: Setup Git configuration
echo -e "${YELLOW}[7/10] Configuring Git...${NC}"
git config --local core.autocrlf input
git config --local core.eol lf
echo -e "${GREEN}✓ Git configured for consistent line endings${NC}"

# Step 8: Create necessary directories
echo -e "${YELLOW}[8/10] Creating project directories...${NC}"
mkdir -p "$PROJECT_ROOT/.genesis/data"
mkdir -p "$PROJECT_ROOT/.genesis/logs"
mkdir -p "$PROJECT_ROOT/.genesis/state"
mkdir -p "$PROJECT_ROOT/docs/setup"
mkdir -p "$PROJECT_ROOT/docs/deployment"
mkdir -p "$PROJECT_ROOT/docs/security"
echo -e "${GREEN}✓ Project directories created${NC}"

# Step 9: Setup environment variables
echo -e "${YELLOW}[9/10] Setting up environment variables...${NC}"
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "${GREEN}✓ .env file created from template${NC}"
        echo -e "${YELLOW}⚠ Please edit .env with your API credentials${NC}"
    else
        # Create basic .env file
        cat > "$PROJECT_ROOT/.env" << EOF
# Genesis Trading System Configuration
# IMPORTANT: Never commit this file to version control

# Environment
GENESIS_ENV=development
TIER=sniper

# Binance API (Get from https://www.binance.com/en/my/settings/api-management)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true

# Database
DATABASE_URL=sqlite:///.genesis/data/genesis.db

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=.genesis/logs/genesis.log

# Trading Parameters
MAX_POSITION_SIZE_USDT=100.0
MAX_DAILY_LOSS_USDT=50.0

# Tilt Detection
TILT_ENABLED=true
TILT_CLICK_SPEED_THRESHOLD=5.0
TILT_CANCEL_RATE_THRESHOLD=0.5

# Monitoring
PROMETHEUS_PORT=9090
EOF
        echo -e "${GREEN}✓ .env file created with defaults${NC}"
        echo -e "${YELLOW}⚠ Please edit .env with your API credentials${NC}"
    fi
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Step 10: Run initial tests
echo -e "${YELLOW}[10/10] Running initial tests...${NC}"
echo -e "${BLUE}Running linting checks...${NC}"
if command_exists ruff; then
    ruff check genesis/ --fix || true
fi

if command_exists black; then
    black genesis/ --check || true
fi

echo -e "${BLUE}Running type checks...${NC}"
if command_exists mypy; then
    mypy genesis/ --ignore-missing-imports || true
fi

echo -e "${BLUE}Running unit tests...${NC}"
if command_exists pytest; then
    pytest tests/unit/ -v --tb=short -x || true
fi

# Final summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Development Environment Ready! 🎉              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Environment Summary:${NC}"
echo -e "  Python Version: $(python --version)"
echo -e "  Virtual Env:    $PROJECT_ROOT/.venv"
echo -e "  Trading Tier:   $TIER"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Edit ${YELLOW}.env${NC} with your Binance API credentials"
echo -e "  2. Run ${YELLOW}make test${NC} to verify everything works"
echo -e "  3. Run ${YELLOW}python -m genesis${NC} to start the trading system"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo -e "  ${YELLOW}make run${NC}        - Start the trading system"
echo -e "  ${YELLOW}make test${NC}       - Run all tests"
echo -e "  ${YELLOW}make format${NC}     - Format code with black"
echo -e "  ${YELLOW}make lint${NC}       - Run linting checks"
echo -e "  ${YELLOW}make clean${NC}      - Clean build artifacts"
echo ""
echo -e "${GREEN}Happy Trading! May the profits be with you 📈${NC}"