.PHONY: help install install-dev install-all test test-cov test-parallel test-integration
.PHONY: lint format typecheck security audit clean build dev docs serve-docs
.PHONY: pre-commit profile benchmark docker docker-build docker-run docker-clean
.PHONY: release release-test release-prod check health

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
PYTEST := pytest
COVERAGE := coverage
RUFF := ruff
BLACK := black
MYPY := mypy
BANDIT := bandit
SAFETY := safety
PRE_COMMIT := pre-commit

# Colors for output
BLUE := \033[36m
YELLOW := \033[33m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)CrewAI Email Triage - Development Commands$(RESET)"
	@echo ""
	@echo "$(YELLOW)Installation:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(install|setup)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(dev|format|lint|typecheck)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(test|coverage|benchmark)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Security & Quality:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(security|audit|pre-commit)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Build & Deploy:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(build|clean|docker|release)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Documentation:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(docs)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install package in development mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev,test]"

install-all: ## Install package with all optional dependencies
	$(PIP) install -e ".[dev,test,docs,performance]"

setup: install-dev pre-commit-install ## Full development setup

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(RESET)"
	@echo "$(YELLOW)Available commands:$(RESET)"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linting"
	@echo "  make format   - Format code"
	@echo "  make typecheck - Run type checking"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(BLACK) src tests
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

lint: ## Run linting with ruff
	@echo "$(BLUE)Running linting...$(RESET)"
	$(RUFF) check src tests

typecheck: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	$(MYPY) src tests

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTEST) tests/

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) -m "unit" tests/

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) -m "integration" tests/

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) --cov=src/crewai_email_triage --cov-report=term-missing --cov-report=html tests/

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTEST) -n auto tests/

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	$(PYTEST) -f tests/

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTEST) -m "performance" --benchmark-only tests/

# =============================================================================
# SECURITY & QUALITY
# =============================================================================

security: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(BANDIT) -r src/

audit: ## Run dependency vulnerability checks
	@echo "$(BLUE)Running dependency audit...$(RESET)"
	$(SAFETY) check --json

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	$(PRE_COMMIT) install

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(PRE_COMMIT) run --all-files

check: lint typecheck security test ## Run all quality checks

# =============================================================================
# BUILD & CLEAN
# =============================================================================

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package for distribution
	@echo "$(BLUE)Building package...$(RESET)"
	$(PYTHON) -m build

build-check: build ## Build and check package
	@echo "$(BLUE)Checking built package...$(RESET)"
	$(PYTHON) -m twine check dist/*

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t crewai-email-triage:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	docker run --rm -it crewai-email-triage:latest

docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker artifacts...$(RESET)"
	docker system prune -f

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	mkdocs build

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	mkdocs serve

# =============================================================================
# PERFORMANCE & PROFILING
# =============================================================================

profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats run_benchmarks.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile: ## Run memory profiling
	@echo "$(BLUE)Running memory profiling...$(RESET)"
	mprof run run_benchmarks.py
	mprof plot

# =============================================================================
# RELEASE MANAGEMENT
# =============================================================================

release-test: ## Test release to TestPyPI
	@echo "$(BLUE)Testing release to TestPyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

release-prod: ## Release to PyPI
	@echo "$(BLUE)Releasing to PyPI...$(RESET)"
	$(PYTHON) -m twine upload dist/*

release: clean build build-check ## Build and prepare for release
	@echo "$(GREEN)Package ready for release!$(RESET)"
	@echo "Run 'make release-test' to upload to TestPyPI"
	@echo "Run 'make release-prod' to upload to PyPI"

# =============================================================================
# HEALTH & MONITORING
# =============================================================================

health: ## Check application health
	@echo "$(BLUE)Checking application health...$(RESET)"
	$(PYTHON) -c "from src.crewai_email_triage.core import health_check; print('Health:', health_check())"

metrics: ## Show application metrics
	@echo "$(BLUE)Showing application metrics...$(RESET)"
	$(PYTHON) -c "from src.crewai_email_triage.metrics_export import show_metrics; show_metrics()"

# =============================================================================
# UTILITY TARGETS
# =============================================================================

env-check: ## Check environment configuration
	@echo "$(BLUE)Checking environment configuration...$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $${VIRTUAL_ENV:-Not activated}"
	@$(PYTHON) -c "import sys; print(f'Python path: {sys.executable}')"

requirements: ## Generate requirements.txt from pyproject.toml
	@echo "$(BLUE)Generating requirements.txt...$(RESET)"
	$(PIP) freeze > requirements.txt

upgrade-deps: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,docs,performance]"

# =============================================================================
# CI/CD SIMULATION
# =============================================================================

ci: clean install-dev check test-cov build ## Simulate CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully!$(RESET)"

ci-fast: install-dev lint test ## Fast CI checks
	@echo "$(GREEN)Fast CI checks completed!$(RESET)"