#!/bin/bash
# Script to install pyenv and Python 3.12 if not already installed

set -e  # Exit immediately if a command exits with a non-zero status

PYTHON_VERSION="3.12"
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Checking for pyenv installation...${NC}"

# Check if pyenv is installed
if command -v pyenv >/dev/null 2>&1; then
    echo -e "${GREEN}pyenv is already installed.${NC}"
else
    echo -e "${YELLOW}pyenv is not installed. Installing pyenv...${NC}"
    
    # Check the OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew >/dev/null 2>&1; then
            echo "Installing pyenv via Homebrew..."
            brew update
            brew install pyenv
        else
            echo -e "${YELLOW}Homebrew not found. Installing pyenv via curl...${NC}"
            curl https://pyenv.run | bash
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing pyenv dependencies and pyenv..."
        if command -v apt-get >/dev/null 2>&1; then
            # Debian/Ubuntu
            sudo apt-get update
            sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
                liblzma-dev python-openssl git
        elif command -v yum >/dev/null 2>&1; then
            # CentOS/RHEL/Fedora
            sudo yum install -y gcc make patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel tk-devel \
                libffi-devel xz-devel
        fi
        curl https://pyenv.run | bash
    else
        echo -e "${RED}Unsupported operating system. Please install pyenv manually.${NC}"
        exit 1
    fi
    
    # Add pyenv to PATH and initialize
    echo -e "${YELLOW}Adding pyenv to your shell configuration...${NC}"
    
    # Determine shell configuration file
    SHELL_CONFIG=""
    if [[ -f "$HOME/.bashrc" ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        echo -e "${RED}Could not determine shell configuration file. Please add pyenv to your PATH manually.${NC}"
        exit 1
    fi
    
    # Add pyenv to shell configuration if not already present
    if ! grep -q "pyenv init" "$SHELL_CONFIG"; then
        echo -e "\n# pyenv configuration" >> "$SHELL_CONFIG"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_CONFIG"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init --path)"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_CONFIG"
        
        echo -e "${GREEN}pyenv has been added to $SHELL_CONFIG.${NC}"
        echo -e "${YELLOW}Please restart your shell or run 'source $SHELL_CONFIG' to use pyenv.${NC}"
        
        # Source the configuration for the current session
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
fi

# Check if Python 3.12 is installed via pyenv
echo -e "${GREEN}Checking for Python $PYTHON_VERSION installation...${NC}"
if pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo -e "${GREEN}Python $PYTHON_VERSION is already installed via pyenv.${NC}"
else
    echo -e "${YELLOW}Python $PYTHON_VERSION is not installed. Installing...${NC}"
    pyenv install "$PYTHON_VERSION"
    echo -e "${GREEN}Python $PYTHON_VERSION has been installed.${NC}"
fi

# Set Python 3.12 as the local version for the project
echo -e "${GREEN}Setting Python $PYTHON_VERSION as the local version for this project...${NC}"
pyenv local "$PYTHON_VERSION"
echo -e "${GREEN}Python $PYTHON_VERSION is now set as the local version.${NC}"

# Create a virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo -e "${GREEN}Creating a virtual environment...${NC}"
    python -m venv .venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

echo -e "${GREEN}Installing project dependencies...${NC}"
pip install -e ".[dev]"

echo -e "${GREEN}Setup complete! Python $PYTHON_VERSION is installed and configured.${NC}"
echo -e "${YELLOW}To activate the virtual environment, run: source .venv/bin/activate${NC}"
