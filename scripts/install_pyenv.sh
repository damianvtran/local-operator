#!/bin/bash
#
# install_pyenv.sh - Python Environment Setup Script
#
# This script automates the setup of a Python 3.12 development environment by:
# 1. Installing pyenv if not already installed
# 2. Installing Python 3.12 via pyenv if not already installed
# 3. Making Python 3.12 available as 'python3.12' in the PATH
# 4. Setting Python 3.12 as the local version for the project
# 5. Configuring shell environment for pyenv based on detected shell type
#
# The script handles different operating systems (macOS, Linux),
# installation methods (Homebrew, apt, yum), and shell types (Bash, Zsh, Fish)
# automatically.
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
    
    # Detect the current shell
    configure_shell_for_pyenv
    
    # Source the configuration for the current session
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    if [[ "$SHELL" == *"fish"* ]]; then
        # For fish shell in the current session
        # We can't easily source fish config in a bash script, so just set the path
        export PATH="$PYENV_ROOT/bin:$PATH"
    else
        # For bash/zsh in the current session
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

# Function to configure shell environment for pyenv
# This function detects the user's shell and adds the appropriate
# pyenv configuration to the shell's configuration files
configure_shell_for_pyenv() {
    # Detect the current shell
    local current_shell
    if [[ -n "$SHELL" ]]; then
        current_shell=$(basename "$SHELL")
    else
        # Try to detect from /etc/passwd if $SHELL is not set
        current_shell=$(basename "$(grep "^$USER:" /etc/passwd | cut -d: -f7)")
    fi
    
    print_status "Detected shell: $current_shell"
    
    case "$current_shell" in
        bash)
            configure_bash
            ;;
        zsh)
            configure_zsh
            ;;
        fish)
            configure_fish
            ;;
        *)
            print_warning "Unsupported shell: $current_shell. Defaulting to bash configuration."
            configure_bash
            ;;
    esac
}

# Function to configure Bash shell for pyenv
# Adds pyenv configuration to .bashrc and the appropriate profile file
configure_bash() {
    print_status "Configuring Bash shell for pyenv..."
    
    # Always add to .bashrc for interactive shells
    if [[ -f "$HOME/.bashrc" ]]; then
        add_pyenv_to_bash_config "$HOME/.bashrc"
    else
        print_warning "No .bashrc found. Creating one..."
        touch "$HOME/.bashrc"
        add_pyenv_to_bash_config "$HOME/.bashrc"
    fi
    
    # Add to profile file for login shells
    # Check for profile files in order of preference
    if [[ -f "$HOME/.bash_profile" ]]; then
        add_pyenv_to_bash_config "$HOME/.bash_profile"
    elif [[ -f "$HOME/.bash_login" ]]; then
        add_pyenv_to_bash_config "$HOME/.bash_login"
    elif [[ -f "$HOME/.profile" ]]; then
        add_pyenv_to_bash_config "$HOME/.profile"
    else
        print_warning "No profile file found. Creating .profile..."
        touch "$HOME/.profile"
        add_pyenv_to_bash_config "$HOME/.profile"
    fi
    
    print_status "Bash configuration complete."
}

# Helper function to add pyenv configuration to a Bash config file
add_pyenv_to_bash_config() {
    local config_file="$1"
    
    if ! grep -q "PYENV_ROOT" "$config_file"; then
        print_status "Adding pyenv configuration to $config_file..."
        echo -e "\n# pyenv configuration" >> "$config_file"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$config_file"
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> "$config_file"
        echo 'eval "$(pyenv init - bash)"' >> "$config_file"
        print_status "pyenv configuration added to $config_file."
    else
        print_warning "pyenv configuration already exists in $config_file."
    fi
}

# Function to configure Zsh shell for pyenv
# Adds pyenv configuration to .zshrc and optionally to .zprofile or .zlogin
configure_zsh() {
    print_status "Configuring Zsh shell for pyenv..."
    
    # Add to .zshrc for interactive shells
    if [[ -f "$HOME/.zshrc" ]]; then
        add_pyenv_to_zsh_config "$HOME/.zshrc"
    else
        print_warning "No .zshrc found. Creating one..."
        touch "$HOME/.zshrc"
        add_pyenv_to_zsh_config "$HOME/.zshrc"
    fi
    
    # Optionally add to .zprofile or .zlogin for login shells
    if [[ -f "$HOME/.zprofile" ]]; then
        add_pyenv_to_zsh_config "$HOME/.zprofile"
    elif [[ -f "$HOME/.zlogin" ]]; then
        add_pyenv_to_zsh_config "$HOME/.zlogin"
    fi
    
    print_status "Zsh configuration complete."
}

# Helper function to add pyenv configuration to a Zsh config file
add_pyenv_to_zsh_config() {
    local config_file="$1"
    
    if ! grep -q "PYENV_ROOT" "$config_file"; then
        print_status "Adding pyenv configuration to $config_file..."
        echo -e "\n# pyenv configuration" >> "$config_file"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$config_file"
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> "$config_file"
        echo 'eval "$(pyenv init - zsh)"' >> "$config_file"
        print_status "pyenv configuration added to $config_file."
    else
        print_warning "pyenv configuration already exists in $config_file."
    fi
}

# Function to configure Fish shell for pyenv
# Adds pyenv configuration to config.fish and sets universal variables
configure_fish() {
    print_status "Configuring Fish shell for pyenv..."
    
    # Create Fish config directory if it doesn't exist
    mkdir -p "$HOME/.config/fish"
    
    # Add to config.fish
    local config_file="$HOME/.config/fish/config.fish"
    if [[ ! -f "$config_file" ]] || ! grep -q "pyenv init" "$config_file"; then
        print_status "Adding pyenv configuration to $config_file..."
        echo -e "\n# pyenv configuration" >> "$config_file"
        echo 'pyenv init - fish | source' >> "$config_file"
        print_status "pyenv configuration added to $config_file."
    else
        print_warning "pyenv configuration already exists in $config_file."
    fi
    
    # Set universal variables
    # Note: We can't directly run fish commands from bash,
    # so we'll create a temporary fish script and execute it
    local fish_script="/tmp/pyenv_fish_setup.fish"
    echo '#!/usr/bin/env fish' > "$fish_script"
    echo 'set -Ux PYENV_ROOT $HOME/.pyenv' >> "$fish_script"
    
    # Check fish version for the appropriate path command
    if fish -c "type -q fish_add_path" 2>/dev/null; then
        # Fish 3.2.0 or newer
        echo 'fish_add_path $PYENV_ROOT/bin' >> "$fish_script"
    else
        # Older Fish versions
        echo 'set -U fish_user_paths $PYENV_ROOT/bin $fish_user_paths' >> "$fish_script"
    fi
    
    chmod +x "$fish_script"
    
    if command -v fish >/dev/null 2>&1; then
        print_status "Setting Fish universal variables..."
        fish "$fish_script"
        rm "$fish_script"
        print_status "Fish universal variables set."
    else
        print_warning "Fish shell not found in PATH. Manual configuration required."
        print_warning "Please run the following commands in Fish shell:"
        print_warning "set -Ux PYENV_ROOT \$HOME/.pyenv"
        print_warning "fish_add_path \$PYENV_ROOT/bin  # For Fish 3.2.0+"
        print_warning "# OR"
        print_warning "set -U fish_user_paths \$PYENV_ROOT/bin \$fish_user_paths  # For older Fish versions"
    fi
    
    print_status "Fish configuration complete."
}

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
                # Detect shell and add to appropriate config
                if [[ "$SHELL" == *"fish"* ]]; then
                    # For Fish shell
                    mkdir -p "$HOME/.config/fish"
                    echo 'fish_add_path $HOME/.local/bin' >> "$HOME/.config/fish/config.fish"
                    print_warning "Added $HOME/.local/bin to PATH in Fish config"
                elif [[ "$SHELL" == *"zsh"* ]]; then
                    # For Zsh
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
                    print_warning "Added $HOME/.local/bin to PATH in .zshrc"
                else
                    # Default to Bash
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
                    print_warning "Added $HOME/.local/bin to PATH in .bashrc"
                fi
                export PATH="$HOME/.local/bin:$PATH"
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
