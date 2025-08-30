#!/bin/bash
# Genesis Trading Bot - Virtual Environment Activation Script
# For Linux/macOS systems

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}🚀 Genesis Trading Bot - Environment Setup${NC}"
echo "========================================"

# Check Python version
check_python_version() {
    local python_cmd=$1
    local required_version="3.11.8"
    
    if command -v $python_cmd &> /dev/null; then
        local version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
        if [[ $version == $required_version* ]]; then
            echo -e "${GREEN}✓ Found Python $version${NC}"
            return 0
        fi
    fi
    return 1
}

# Find appropriate Python command
PYTHON_CMD=""
if check_python_version "python3.11"; then
    PYTHON_CMD="python3.11"
elif check_python_version "python"; then
    PYTHON_CMD="python"
elif check_python_version "python3"; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}❌ Error: Python 3.11.8 is required but not found${NC}"
    echo "Please install Python 3.11.8 using pyenv or your system package manager"
    echo ""
    echo "Using pyenv:"
    echo "  pyenv install 3.11.8"
    echo "  pyenv local 3.11.8"
    exit 1
fi

# Virtual environment directory
VENV_DIR="$PROJECT_ROOT/.venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    echo -e "  Path: $VIRTUAL_ENV"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Upgrade pip to latest version
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --quiet --upgrade pip

# Install pip-tools if not present
if ! pip show pip-tools &> /dev/null; then
    echo -e "${YELLOW}Installing pip-tools...${NC}"
    pip install --quiet pip-tools
fi

# Determine which requirements to install based on TIER environment variable
TIER=${TIER:-sniper}
echo -e "${YELLOW}Installing requirements for tier: $TIER${NC}"

# Install appropriate requirements
case $TIER in
    sniper)
        REQUIREMENTS_FILE="$PROJECT_ROOT/requirements/sniper.txt"
        ;;
    hunter)
        REQUIREMENTS_FILE="$PROJECT_ROOT/requirements/hunter.txt"
        ;;
    strategist)
        REQUIREMENTS_FILE="$PROJECT_ROOT/requirements/strategist.txt"
        ;;
    *)
        echo -e "${RED}❌ Unknown tier: $TIER${NC}"
        exit 1
        ;;
esac

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Installing from $REQUIREMENTS_FILE...${NC}"
    pip install -q -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Requirements file not found: $REQUIREMENTS_FILE${NC}"
fi

# Install development dependencies if in dev mode
if [ "${DEV_MODE}" = "true" ] || [ "${ENV}" = "development" ]; then
    if [ -f "$PROJECT_ROOT/requirements/dev.txt" ]; then
        echo -e "${YELLOW}Installing development dependencies...${NC}"
        pip install -q -r "$PROJECT_ROOT/requirements/dev.txt"
        echo -e "${GREEN}✓ Development dependencies installed${NC}"
    fi
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export GENESIS_ROOT="$PROJECT_ROOT"

# Display final status
echo ""
echo -e "${GREEN}✅ Environment Ready!${NC}"
echo "========================================"
echo -e "Python:      $($PYTHON_CMD --version)"
echo -e "Pip:         $(pip --version | cut -d' ' -f2)"
echo -e "Tier:        $TIER"
echo -e "Project:     $PROJECT_ROOT"
echo -e "Virtual Env: $VIRTUAL_ENV"
echo ""
echo "To deactivate the environment, run: deactivate"