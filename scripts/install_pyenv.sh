#!/bin/bash
#
# install_pyenv.sh - Python Environment Setup Script
#
# This script automates the setup of a Python 3.12 development environment by:
# 1. Installing pyenv if not already installed
# 2. Installing Python 3.12 via pyenv if not already installed
# 3. Making Python 3.12 available as 'python3.12' in the PATH
# 4. Setting Python 3.12 as the local version for the project
#
# The script handles different operating systems (macOS, Linux) and
# installation methods (Homebrew, apt, yum) automatically.
#
# Usage: ./scripts/install_pyenv.sh
#

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
PYTHON_VERSION="3.12"

# Terminal colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}$1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}$1${NC}"
}

print_status "Checking for pyenv installation..."

# Step 1: Install pyenv if not already installed
# ---------------------------------------------
if command -v pyenv >/dev/null 2>&1; then
    print_status "pyenv is already installed."
else
    print_warning "pyenv is not installed. Installing pyenv..."
    
    # OS-specific installation methods
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
        if command -v brew >/dev/null 2>&1; then
            print_status "Installing pyenv via Homebrew..."
            brew update
            brew install pyenv
        else
            print_warning "Homebrew not found. Installing pyenv via curl..."
            curl https://pyenv.run | bash
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
        print_status "Installing pyenv dependencies and pyenv..."
        if command -v apt-get >/dev/null 2>&1; then
            # Debian/Ubuntu dependencies
            print_status "Installing dependencies for Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
                liblzma-dev python-openssl git
        elif command -v yum >/dev/null 2>&1; then
            # CentOS/RHEL/Fedora dependencies
            print_status "Installing dependencies for CentOS/RHEL/Fedora..."
            sudo yum install -y gcc make patch zlib-devel bzip2 bzip2-devel \
                readline-devel sqlite sqlite-devel openssl-devel tk-devel \
                libffi-devel xz-devel
        fi
        print_status "Installing pyenv..."
        curl https://pyenv.run | bash
    else
        print_error "Unsupported operating system. Please install pyenv manually."
        exit 1
    fi
    
    # Configure shell for pyenv
    print_warning "Adding pyenv to your shell configuration..."
    
    # Determine which shell configuration file to use
    SHELL_CONFIG=""
    if [[ -f "$HOME/.bashrc" ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [[ -f "$HOME/.zshrc" ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        print_error "Could not determine shell configuration file. Please add pyenv to your PATH manually."
        exit 1
    fi
    
    # Add pyenv initialization to shell config if not already present
    if ! grep -q "pyenv init" "$SHELL_CONFIG"; then
        print_status "Adding pyenv initialization to $SHELL_CONFIG..."
        echo -e "\n# pyenv configuration" >> "$SHELL_CONFIG"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_CONFIG"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init --path)"' >> "$SHELL_CONFIG"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_CONFIG"
        
        print_status "pyenv has been added to $SHELL_CONFIG."
        print_warning "Please restart your shell or run 'source $SHELL_CONFIG' to use pyenv."
        
        # Source the configuration for the current session
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
fi

# Step 2: Install Python 3.12 via pyenv if not already installed
# -------------------------------------------------------------
print_status "Checking for Python $PYTHON_VERSION installation..."
if pyenv versions | grep -q "$PYTHON_VERSION"; then
    print_status "Python $PYTHON_VERSION is already installed via pyenv."
else
    print_warning "Python $PYTHON_VERSION is not installed. Installing..."
    pyenv install "$PYTHON_VERSION"
    print_status "Python $PYTHON_VERSION has been installed."
fi

# Step 3: Set Python 3.12 as the local version for the project
# -----------------------------------------------------------
print_status "Setting Python $PYTHON_VERSION as the local version for this project..."
pyenv local "$PYTHON_VERSION"
print_status "Python $PYTHON_VERSION is now set as the local version."

# Step 4: Make Python 3.12 available in PATH via symlink
# -----------------------------------------------------
PYENV_PYTHON_PATH="$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python"
if [[ -f "$PYENV_PYTHON_PATH" ]]; then
    print_status "Creating symlink for python3.12..."
    
    # Try to create a system-wide symlink if we have sudo access
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        if [[ -d "/usr/local/bin" ]]; then
            if [[ ! -f "/usr/local/bin/python3.12" ]]; then
                print_status "Creating system-wide symlink in /usr/local/bin..."
                sudo ln -sf "$PYENV_PYTHON_PATH" /usr/local/bin/python3.12
                print_status "Created symlink in /usr/local/bin/python3.12"
            else
                print_warning "Symlink already exists at /usr/local/bin/python3.12"
            fi
        fi
    else
        # Create a user-level symlink if we don't have sudo access
        print_status "Creating user-level symlink in ~/.local/bin..."
        mkdir -p "$HOME/.local/bin"
        if [[ ! -f "$HOME/.local/bin/python3.12" ]]; then
            ln -sf "$PYENV_PYTHON_PATH" "$HOME/.local/bin/python3.12"
            print_status "Created symlink in $HOME/.local/bin/python3.12"
            
            # Add ~/.local/bin to PATH if not already there
            if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                export PATH="$HOME/.local/bin:$PATH"
                print_warning "Added $HOME/.local/bin to PATH in $SHELL_CONFIG"
            fi
        else
            print_warning "Symlink already exists at $HOME/.local/bin/python3.12"
        fi
    fi
else
    print_error "Could not find Python $PYTHON_VERSION installation at $PYENV_PYTHON_PATH"
    exit 1
fi

# Step 5: Verify the installation
# ------------------------------
if command -v python3.12 >/dev/null 2>&1; then
    print_status "python3.12 is now available in PATH"
    python3.12 --version
else
    print_warning "python3.12 is not directly available in PATH. You can use it through pyenv:"
    print_warning "pyenv shell $PYTHON_VERSION"
    print_warning "python --version"
fi

print_status "Setup complete! Python $PYTHON_VERSION is installed and configured."
