#!/bin/bash
# bin/check_env.sh - Simple environment check for required Python and uv versions

# Required versions
REQUIRED_PYTHON="3.11"
REQUIRED_UV="0.8.12"

# Function to extract semantic version (handles x.y.z, x.y, etc.)
extract_version() {
    local version_string="$1"
    # Extract semantic version: major.minor.patch (or just major.minor)
    echo "$version_string" | grep -oE '[0-9]+(\.[0-9]+)*' | head -n1
}

# Check Python
if ! command -v python >/dev/null 2>&1; then
    echo "❌ Python not found. Run: make install-python"
    exit 1
fi

PYTHON_VER=$(extract_version "$(python --version 2>&1)")
# For Python, we only compare major.minor
PYTHON_VER_SHORT=$(echo "$PYTHON_VER" | cut -d'.' -f1,2)
if [ "$(printf '%s\n' "$REQUIRED_PYTHON" "$PYTHON_VER_SHORT" | sort -V | head -n1)" != "$REQUIRED_PYTHON" ]; then
    echo "❌ Python $PYTHON_VER < $REQUIRED_PYTHON required. Run: make install-python"
    exit 1
fi

# Check uv
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv not found. Run: make install-uv"
    exit 1
fi

UV_VER=$(extract_version "$(uv --version 2>&1)")
if [ "$(printf '%s\n' "$REQUIRED_UV" "$UV_VER" | sort -V | head -n1)" != "$REQUIRED_UV" ]; then
    echo "❌ uv $UV_VER < $REQUIRED_UV required. Run: make install-uv"
    exit 1
fi

echo "✅ Environment ready: Python $PYTHON_VER, uv $UV_VER"
