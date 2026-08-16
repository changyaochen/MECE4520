#!/bin/bash

# install_uv.sh - Cross-platform uv installation script
# Installs the latest stable version of uv (Python package manager)
# Supports macOS, Linux, and Windows (via Git Bash/WSL)

set -e  # Exit on any error

# Pin uv version - update this when you want to upgrade
UV_VERSION="0.8.12"  # Latest stable version as of August 2025
UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"

echo "📦 Installing uv v${UV_VERSION} (Python package manager)..."

# Function to detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

# Function to detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)   echo "x86_64" ;;
        arm64|aarch64)  echo "aarch64" ;;
        armv7l)         echo "armv7" ;;
        *)              echo "unknown" ;;
    esac
}

# Function to install uv via official installer
install_uv_official() {
    local os=$1

    echo "🌐 Installing uv via official installer..."

    case $os in
        "macos"|"linux")
            # Use the official installer script
            if command -v curl >/dev/null 2>&1; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
            elif command -v wget >/dev/null 2>&1; then
                wget -qO- https://astral.sh/uv/install.sh | sh
            else
                echo "❌ Neither curl nor wget found. Please install one of them."
                exit 1
            fi
            ;;
        "windows")
            # Use PowerShell installer for Windows
            if command -v powershell >/dev/null 2>&1; then
                powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
            else
                echo "⚠️  PowerShell not found. Trying curl fallback..."
                curl -LsSf https://astral.sh/uv/install.sh | sh
            fi
            ;;
        *)
            echo "❌ Unsupported operating system: $os"
            exit 1
            ;;
    esac
}

# Function to install uv via direct binary download (as fallback)
install_uv_binary() {
    local os=$1
    local arch=$2

    echo "🔧 Installing uv via direct binary download..."

    # Construct download URL based on OS and architecture
    local platform=""
    case $os in
        "macos")
            case $arch in
                "x86_64")   platform="apple-darwin" ;;
                "aarch64")  platform="apple-darwin" ;;
                *)          echo "❌ Unsupported macOS architecture: $arch"; exit 1 ;;
            esac
            ;;
        "linux")
            case $arch in
                "x86_64")   platform="unknown-linux-gnu" ;;
                "aarch64")  platform="unknown-linux-gnu" ;;
                *)          echo "❌ Unsupported Linux architecture: $arch"; exit 1 ;;
            esac
            ;;
        "windows")
            platform="pc-windows-msvc"
            ;;
        *)
            echo "❌ Unsupported operating system: $os"
            exit 1
            ;;
    esac

    # Download and install binary
    local download_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${arch}-${platform}.tar.gz"
    local temp_dir=$(mktemp -d)
    local binary_name="uv"

    if [[ "$os" == "windows" ]]; then
        download_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${arch}-${platform}.zip"
        binary_name="uv.exe"
    fi

    echo "📥 Downloading from: $download_url"

    cd "$temp_dir"
    if command -v curl >/dev/null 2>&1; then
        curl -LO "$download_url"
    elif command -v wget >/dev/null 2>&1; then
        wget "$download_url"
    else
        echo "❌ Neither curl nor wget found."
        exit 1
    fi

    # Extract and install
    local archive_name=$(basename "$download_url")
    if [[ "$archive_name" == *.tar.gz ]]; then
        tar -xzf "$archive_name"
    elif [[ "$archive_name" == *.zip ]]; then
        unzip "$archive_name"
    fi

    # Create install directory if it doesn't exist
    mkdir -p "$UV_INSTALL_DIR"

    # Find and copy the binary
    find . -name "$binary_name" -type f -exec cp {} "$UV_INSTALL_DIR/" \;
    chmod +x "$UV_INSTALL_DIR/$binary_name"

    # Clean up
    cd - >/dev/null
    rm -rf "$temp_dir"

    echo "✅ uv binary installed to: $UV_INSTALL_DIR/$binary_name"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying uv installation..."

    # Add common uv installation paths to PATH for verification
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$UV_INSTALL_DIR:$PATH"

    if command -v uv >/dev/null 2>&1; then
        local installed_version=$(uv --version | head -n1)
        echo "✅ uv installed successfully!"
        echo "📍 Location: $(which uv)"
        echo "🏷️  Version: $installed_version"

        return 0
    else
        echo "❌ uv installation verification failed"
        return 1
    fi
}

# Function to check if desired version is already installed
check_existing_installation() {
    echo "🔍 Checking if uv v${UV_VERSION} is already installed..."

    # Add common uv installation paths to PATH for checking
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$UV_INSTALL_DIR:$PATH"

    if command -v uv >/dev/null 2>&1; then
        local installed_version=$(uv --version 2>&1 | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0")
        echo "📦 Found existing uv installation: v$installed_version"

        if [ "$installed_version" = "$UV_VERSION" ]; then
            echo "✅ uv v${UV_VERSION} is already installed!"
            echo "📍 Location: $(which uv)"
            echo "🎉 Skipping installation process."
            return 0
        else
            echo "⚠️  Different version found (v$installed_version vs v${UV_VERSION} required)"
            echo "🔄 Proceeding with installation of v${UV_VERSION}..."
            return 1
        fi
    else
        echo "❌ uv not found in PATH"
        echo "📥 Proceeding with fresh installation..."
        return 1
    fi
}

# Function to setup PATH
setup_path() {
    echo "🛣️  Setting up PATH configuration..."

    # Common uv installation paths
    local uv_paths=(
        "$HOME/.cargo/bin"
        "$HOME/.local/bin"
        "$UV_INSTALL_DIR"
    )

    # Detect shell and config file
    local shell_name=$(basename "$SHELL")
    local rc_file=""

    case $shell_name in
        "bash")
            if [[ "$(uname -s)" == "Darwin" ]]; then
                rc_file="$HOME/.bash_profile"
            else
                rc_file="$HOME/.bashrc"
            fi
            ;;
        "zsh")
            rc_file="$HOME/.zshrc"
            ;;
        "fish")
            rc_file="$HOME/.config/fish/config.fish"
            ;;
        *)
            echo "⚠️  Unknown shell: $shell_name"
            echo "   Please add the following to your shell configuration:"
            for path in "${uv_paths[@]}"; do
                echo "   export PATH=\"$path:\$PATH\""
            done
            return 0
            ;;
    esac

    if [[ -n "$rc_file" ]]; then
        local path_added=false

        for uv_path in "${uv_paths[@]}"; do
            if [[ ! -f "$rc_file" ]] || ! grep -q "$uv_path" "$rc_file"; then
                echo "export PATH=\"$uv_path:\$PATH\"" >> "$rc_file"
                path_added=true
            fi
        done

        if [[ "$path_added" == true ]]; then
            echo "✅ Added uv to PATH in $rc_file"
            echo "ℹ️  Please restart your shell or run: source $rc_file"
        else
            echo "✅ uv is already in PATH configuration"
        fi
    fi
}

# Function to show usage examples
show_usage_examples() {
    echo ""
    echo "🎯 uv Usage Examples:"
    echo "   uv --version                    # Check version"
    echo "   uv pip install pandas          # Install packages"
    echo "   uv pip install -r requirements.txt  # Install from requirements"
    echo "   uv venv                         # Create virtual environment"
    echo "   uv pip sync pyproject.toml      # Sync dependencies from pyproject.toml"
    echo ""
    echo "📚 Learn more: https://github.com/astral-sh/uv"
}

# Main execution
main() {
    echo "🚀 Starting uv v${UV_VERSION} installation process..."

    # Check if desired version is already installed
    if check_existing_installation; then
        return 0
    fi

    # Detect system
    local os=$(detect_os)
    local arch=$(detect_arch)

    echo "🖥️  Detected system: $os ($arch)"

    # Try official installer first, fallback to binary download
    if install_uv_official "$os"; then
        echo "✅ Official installer completed"
    else
        echo "⚠️  Official installer failed, trying binary download..."
        install_uv_binary "$os" "$arch"
    fi

    # Verify installation
    if verify_installation; then
        setup_path
        show_usage_examples

        echo ""
        echo "🎉 uv v${UV_VERSION} installation completed successfully!"
        echo ""
        echo "📝 Next steps:"
        echo "   1. Restart your terminal or source your shell config"
        echo "   2. Verify with: uv --version"
        echo "   3. Use uv to manage Python dependencies"
        echo ""
    else
        echo "❌ Installation failed. Please check the error messages above."
        exit 1
    fi
}

# Run main function
main "$@"
