# Local Operator Makefile

.PHONY: server dev-server cli openapi test coverage format lint type-check security clean help setup-python

# Default target
.DEFAULT_GOAL := help

# Setup commands
setup-python: ## Install pyenv and Python 3.12 if not already installed
	@echo "Setting up Python environment..."
	@./scripts/install_pyenv.sh

# Variables
PYTHON := python
PYTEST := pytest
COVERAGE_DIR := htmlcov
OPENAPI_OUTPUT := docs/openapi.json

# Server commands
server: ## Start the server
	local-operator serve

dev-server: ## Start the server with hot reload
	local-operator serve --reload

# CLI command
cli: ## Start the CLI
	local-operator

# OpenAPI generation
openapi: ## Generate OpenAPI specification
	$(PYTHON) -m local_operator.server.generate_openapi -o $(OPENAPI_OUTPUT)

# Test commands
test: ## Run tests
	$(PYTEST)

coverage: ## Generate test coverage report
	$(PYTEST) --cov=local_operator --cov-report=html
	@echo "Coverage report generated in $(COVERAGE_DIR)/"

# Code quality commands
format: ## Format code with black and isort
	black .
	isort .

lint: ## Run linting with flake8
	flake8

type-check: ## Run type checking with pyright
	pyright

security: ## Run security audit with pip-audit
	pip-audit

# Cleanup
clean: ## Clean up generated files
	rm -rf $(COVERAGE_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help command
help: ## Show this help message
	@echo "Local Operator Makefile Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
