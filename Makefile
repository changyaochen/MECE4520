.PHONY: build serve install-python install-uv

install-python:
	@echo "🐍 Installing Python 3.10 via pyenv..."
	@bin/install_python.sh

install-uv:
	@echo "📦 Installing uv package manager..."
	@bin/install_uv.sh

setup: install-python install-uv
	@echo "🚀 Setting up the project..."
	@$$HOME/.local/bin/uv sync || uv sync

jupyter:
	@bin/check_env.sh
	@echo "🚀 Starting Jupyter server..."
	@source .venv/bin/activate && jupyter lab




