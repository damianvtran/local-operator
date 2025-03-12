# ============================================================================
# Local Operator Makefile
# ============================================================================
#
# This Makefile provides commands for development, testing, and running the
# Local Operator project. It includes targets for setting up the Python
# environment, running the server, executing tests, and maintaining code quality.
#
# Main targets:
# - install: Complete setup (Python + dependencies)
# - setup-python: Install pyenv and Python 3.12
# - server/dev-server: Run the server
# - test/coverage: Run tests
# - format/lint/type-check: Code quality tools
#

# Declare all targets as phony (not representing files)
.PHONY: server dev-server cli openapi test coverage format lint type-check security clean help setup-python install

# Default target when running 'make' without arguments
.DEFAULT_GOAL := help

# ============================================================================
# Setup Commands
# ============================================================================

# Install pyenv and Python 3.12 if not already installed
# This target ensures the correct Python version is available
setup-python: ## Install pyenv and Python 3.12 if not already installed
	@echo "Setting up Python environment..."
	@./scripts/install_pyenv.sh

# Complete installation target that depends on setup-python
# This will set up Python and install all project dependencies
install: setup-python ## Complete setup: install Python 3.12 and all dependencies
	@echo "Creating virtual environment in .venv..."
	@if command -v python3.12 >/dev/null 2>&1; then \
		python3.12 -m venv .venv; \
	else \
		echo "python3.12 not found in PATH, using pyenv to create virtual environment..."; \
		eval "$$(pyenv init -)" && pyenv shell 3.12 && python -m venv .venv; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	. .venv/bin/activate && pip install -e ".[dev]"

# ============================================================================
# Variables
# ============================================================================
# Python executable to use for commands
PYTHON := python3.12
# Test runner
PYTEST := pytest
# Directory for test coverage reports
COVERAGE_DIR := htmlcov
# Output file for OpenAPI specification
OPENAPI_OUTPUT := docs/openapi.json

# ============================================================================
# Server Commands
# ============================================================================
# Start the server without hot reload (for production-like environments)
server: ## Start the server
	local-operator serve

# Start the server with hot reload (for development)
dev-server: ## Start the server with hot reload
	local-operator serve --reload

# ============================================================================
# CLI Commands
# ============================================================================
# Start the CLI interface
cli: ## Start the CLI
	local-operator

# ============================================================================
# Documentation Commands
# ============================================================================
# Generate OpenAPI specification for API documentation
openapi: ## Generate OpenAPI specification
	$(PYTHON) -m local_operator.server.generate_openapi -o $(OPENAPI_OUTPUT)

# ============================================================================
# Testing Commands
# ============================================================================
# Run all tests
test: ## Run tests
	$(PYTEST)

# Generate test coverage report
coverage: ## Generate test coverage report
	$(PYTEST) --cov=local_operator --cov-report=html
	@echo "Coverage report generated in $(COVERAGE_DIR)/"

# ============================================================================
# Code Quality Commands
# ============================================================================
# Format code with black and isort
format: ## Format code with black and isort
	black .
	isort .

# Run linting with flake8
lint: ## Run linting with flake8
	flake8

# Run type checking with pyright
type-check: ## Run type checking with pyright
	pyright

# Run security audit with pip-audit
security: ## Run security audit with pip-audit
	pip-audit

# ============================================================================
# Cleanup Commands
# ============================================================================
# Clean up generated files and directories
clean: ## Clean up generated files
	rm -rf $(COVERAGE_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ============================================================================
# Help Command
# ============================================================================
# Display help information about available commands
help: ## Show this help message
	@echo "Local Operator Makefile Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
