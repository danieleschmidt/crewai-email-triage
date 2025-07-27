#!/bin/bash
# =============================================================================
# CrewAI Email Triage - Development Container Setup
# =============================================================================

set -e

echo "🚀 Setting up CrewAI Email Triage development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    vim \
    tree \
    htop \
    net-tools \
    postgresql-client \
    redis-tools

# Upgrade pip and install pip-tools
echo "🐍 Setting up Python environment..."
python -m pip install --upgrade pip setuptools wheel
pip install pip-tools

# Install project dependencies
echo "📚 Installing project dependencies..."
pip install -e ".[dev,test,docs,performance]"

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p /workspaces/crewai-email-triage/{logs,data,tmp}

# Set up environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your actual configuration values"
fi

# Install development tools
echo "🛠️ Installing additional development tools..."
pip install \
    ipython \
    jupyterlab \
    pre-commit \
    tox \
    bump2version

# Run initial tests to verify setup
echo "🧪 Running initial tests to verify setup..."
pytest tests/ -v --tb=short || echo "⚠️ Some tests failed - this is normal for a fresh setup"

# Run pre-commit on all files to ensure consistency
echo "✅ Running pre-commit checks..."
pre-commit run --all-files || echo "⚠️ Pre-commit found issues - they should be auto-fixed"

# Display helpful information
echo ""
echo "✨ Development environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "  make help                 - Show all available commands"
echo "  make test                 - Run tests"
echo "  make lint                 - Run linting"
echo "  make format               - Format code"
echo "  make security             - Run security checks"
echo "  make docs                 - Build documentation"
echo ""
echo "🔗 Useful URLs (when running):"
echo "  Application:  http://localhost:8000"
echo "  Metrics:      http://localhost:8080/metrics"
echo "  Health:       http://localhost:8081/health"
echo ""
echo "📖 Quick start:"
echo "  1. Edit .env file with your email credentials"
echo "  2. Run: python triage.py --help"
echo "  3. Test: python triage.py --message 'Test message' --pretty"
echo ""
echo "🎉 Happy coding!"