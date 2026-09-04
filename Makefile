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
.PHONY: server dev-server cli openapi test coverage format lint type-check adapter-osworld security clean help setup-python install prototype-ramp

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
# Invoke formatters/linters the same way AGENTS.md documents the gates.
# Bare `.venv/bin/black` (and flake8/isort/pyright) console scripts carry a
# shebang baked in at install time; after a worktree that owned the venv
# is deleted they fail with `bad interpreter` and exit 126. `rc=126`
# disappears inside a pipeline (`cmd | tail` reports tail's 0), so a gate
# that never ran looks green. `python -m` uses the interpreter's module
# path; `uvx` fetches a pinned tool into an isolated cache. Neither reads
# those shebangs. black/isort are deliberately NOT in the dev extra, so
# format goes through uvx at the versions CI pins.
format: ## Format code with black and isort
	uvx --from black==26.1.0 black .
	uvx isort==5.13.2 .

# Run linting with flake8
# THROWAWAY. Delete this target with the module it runs, once the "lop
# underuses each theme's palette" question is settled (see that module's
# docstring for the measurements).
prototype-ramp: ## PROTOTYPE: build + open the whole-TUI colour variants
	env -u NO_COLOR .venv/bin/python -m local_operator.tui.prototype_markdown_ramp

lint: ## Run linting with flake8
	.venv/bin/python -m flake8 .

# Run type checking with pyright
type-check: ## Run type checking with pyright
	.venv/bin/python -m pyright --pythonpath .venv/bin/python .

# Build the OSWorld V2 evaluation adapter: lock, wheel, and the workspace
# materialisation command. Deliberately NOT wired into CI's default job — the
# workspace step needs a human-accepted gated HF dataset and an HF_TOKEN, so
# this is a developer/operator command, not a gate.
adapter-osworld: ## Build the OSWorld V2 adapter wheel and print workspace steps
	cd benchmarks/osworld_v2_adapter && uv lock && uv build --wheel --out-dir dist/
	@echo "Wheel built under benchmarks/osworld_v2_adapter/dist/."
	@echo "Materialise the workspace (needs HF_TOKEN) with:"
	@echo "  python scripts/build_osworld_adapter.py --benchmark-release osworld-v2-2026.08.08 --out <workspace>"

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
