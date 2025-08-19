#!/bin/bash

# install_python.sh - Cross-platform Python 3.11 installation via pyenv
# Supports macOS, Linux, and Windows (via Git Bash/WSL)

set -e  # Exit on any error

PYTHON_VERSION="3.11.10"
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"

echo "🐍 Installing Python ${PYTHON_VERSION} via pyenv..."

# Function to detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

# Function to install pyenv if not present
install_pyenv() {
    local os=$1

    if command -v pyenv >/dev/null 2>&1; then
        echo "✅ pyenv is already installed"
        return 0
    fi

    echo "📦 Installing pyenv..."

    case $os in
        "macos")
            if command -v brew >/dev/null 2>&1; then
                echo "Installing pyenv via Homebrew..."
                brew install pyenv
            else
                echo "Installing pyenv via curl (Homebrew not found)..."
                curl https://pyenv.run | bash
            fi
            ;;
        "linux")
            echo "Installing pyenv via curl..."
            curl https://pyenv.run | bash
            ;;
        "windows")
            echo "Installing pyenv-win via git..."
            if [ ! -d "$PYENV_ROOT" ]; then
                git clone https://github.com/pyenv-win/pyenv-win.git "$PYENV_ROOT"
            fi
            ;;
        *)
            echo "❌ Unsupported operating system: $os"
            exit 1
            ;;
    esac
}

# Function to setup pyenv environment
setup_pyenv_env() {
    # Add pyenv to PATH if not already there
    if [[ ":$PATH:" != *":$PYENV_ROOT/bin:"* ]]; then
        export PATH="$PYENV_ROOT/bin:$PATH"
    fi

    # Initialize pyenv if available
    if command -v pyenv >/dev/null 2>&1; then
        eval "$(pyenv init -)"
    fi
}

# Function to install system dependencies
install_dependencies() {
    local os=$1

    echo "📋 Installing system dependencies for Python compilation..."

    case $os in
        "macos")
            if command -v brew >/dev/null 2>&1; then
                echo "Installing build dependencies via Homebrew..."
                brew install openssl readline sqlite3 xz zlib tcl-tk || true
            else
                echo "⚠️  Homebrew not found. Please install build dependencies manually."
                echo "   You may need: openssl, readline, sqlite3, xz, zlib, tcl-tk"
            fi
            ;;
        "linux")
            if command -v apt-get >/dev/null 2>&1; then
                echo "Installing build dependencies via apt..."
                sudo apt-get update
                sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
                    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
                    libffi-dev liblzma-dev || true
            elif command -v yum >/dev/null 2>&1; then
                echo "Installing build dependencies via yum..."
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel \
                    zlib-devel readline-devel sqlite-devel xz-devel tk-devel || true
            elif command -v dnf >/dev/null 2>&1; then
                echo "Installing build dependencies via dnf..."
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y gcc openssl-devel bzip2-devel libffi-devel \
                    zlib-devel readline-devel sqlite-devel xz-devel tk-devel || true
            else
                echo "⚠️  Package manager not found. Please install build dependencies manually."
            fi
            ;;
        "windows")
            echo "ℹ️  On Windows, pyenv-win handles dependencies automatically."
            ;;
    esac
}

# Function to install Python version (updated approach)
install_python_version() {
    echo "🔧 Installing Python ${PYTHON_VERSION}..."

    # Check if version is already installed
    if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        echo "✅ Python ${PYTHON_VERSION} is already installed"
    else
        echo "📥 Installing Python ${PYTHON_VERSION} (using pre-built binary if available)..."

        # Install with pyenv (will use pre-built if available)
        pyenv install "${PYTHON_VERSION}"
        echo "✅ Python ${PYTHON_VERSION} installed successfully"
    fi

    # Set as local version (better than global)
    echo "📍 Setting Python ${PYTHON_VERSION} as local version..."
    pyenv local "${PYTHON_VERSION}"
}

# Main execution
main() {
    echo "🚀 Starting Python ${PYTHON_VERSION} installation process..."

    # Detect operating system
    OS=$(detect_os)
    echo "🖥️  Detected operating system: $OS"

    # Install pyenv if needed
    install_pyenv "$OS"

    # Setup pyenv environment
    setup_pyenv_env

    # Install system dependencies
    install_dependencies "$OS"

    # Install Python version
    install_python_version


    echo ""
    echo "🎉 Python ${PYTHON_VERSION} installation completed!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "   2. Verify with: python --version"
    echo "   3. Install project dependencies: make setup"
    echo ""
}

# Run main function
main "$@"
