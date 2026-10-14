# Contributing to Local Operator

Thank you for your interest in contributing to Local Operator! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions. By participating in this project, you agree to abide by its [MIT License](LICENSE).

## Project Structure

```shell
.
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Project license
├── Makefile                     # Makefile for common development tasks
├── docs                         # Documentation files and resources
├── local_operator
│   ├── admin.py                 # Tools for managing agents and conversations
│   ├── agents.py                # Agent data structures and registry management
│   ├── cli.py                   # Command line interface implementation
│   ├── clients                  # API clients for external services
│   │   ├── openrouter.py        # OpenRouter API client
│   │   └── serpapi.py           # SerpAPI client
│   ├── config.py                # Configuration management and settings
│   ├── console.py               # Terminal output formatting and display
│   ├── credentials.py           # Secure storage of API keys and credentials
│   ├── executor.py              # Code execution with safety checks
│   ├── mocks.py                 # Mock objects for testing
│   ├── model                    # Language model configurations
│   │   ├── configure.py         # Model configuration settings
│   │   └── registry.py          # Model pricing, context limits, and configuration
│   ├── operator.py              # Core environment manager for model interactions
│   ├── prompts.py               # System prompts and message templates
│   ├── server.py                # HTTP API server implementation
│   ├── tools.py                 # Utility functions and tool registry
│   └── types.py                 # Type definitions and enums
├── scripts                      # Utility scripts for development
│   └── install_pyenv.sh         # Script to install pyenv and Python 3.12
├── pyproject.toml               # Package metadata, dependencies and extras
├── static                       # Static assets and resources
└── tests                        # Test suite directory
    ├── conftest.py              # Pytest configuration and fixtures
    └── unit                     # Unit test implementations
```

## Dependency Graph

Refer to the [dependency graph](docs/dependencies.md) for a visual representation of the project structure.

## Getting Started

### Prerequisites

- Python 3.12+ (will be installed via pyenv if not available)
- Dependencies are declared in `pyproject.toml`. The default install is
  deliberately lean; optional features live behind extras (`server`, `mcp`,
  `images`, `tokenizer`, and `all`). `make install` installs `.[dev]`, which
  pulls every extra plus the test and lint tooling.

### Development Setup

1. If you don't have contributor access, **Fork** the repository on GitHub
2. **Clone** your forked repository:

   ```bash
   git clone https://github.com/your-username/local-operator.git
   ```

3. You can set up your Python environment in one of two ways:

   **Option 1:** Use the provided install command (recommended):

   ```bash
   make install
   ```

   This will:

   - Install pyenv if not already installed
   - Install Python 3.12 via pyenv if not already installed
   - Make Python 3.12 available as `python3.12` in your PATH
   - Create a virtual environment using Python 3.12
   - Install all project dependencies

   **Option 2:** Step-by-step setup:

   ```bash
   # First, set up Python environment (installs pyenv and Python 3.12)
   make setup-python

   # Then, manually create a virtual environment and install dependencies
   # (python3.12 should now be available in your PATH)
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .[dev]
   ```

## Development Workflow with Makefile

We provide a Makefile to simplify common development tasks. You can see all available commands by running:

```bash
make help
```

### Common Commands

- **Install Project** (sets up Python environment and installs dependencies):

  ```bash
  make install
  ```

- **Setup Python Environment** (installs pyenv and Python 3.12 if needed):

  ```bash
  make setup-python
  ```

- **Start the Server**:

  ```bash
  make server
  ```

- **Start the Server with Hot Reload** (for development):

  ```bash
  make dev-server
  ```

- **Start the CLI**:

  ```bash
  make cli
  ```

- **Generate OpenAPI Specification**:

  ```bash
  make openapi
  ```

- **Run Tests**:

  ```bash
  make test
  ```

- **Generate Test Coverage Report**:

  ```bash
  make coverage
  ```

### Code Quality Commands

- **Format Code** (black and isort):

  ```bash
  make format
  ```

- **Run Linting** (flake8):

  ```bash
  make lint
  ```

- **Run Type Checking** (pyright):

  ```bash
  make type-check
  ```

- **Run Security Audit** (pip-audit):

  ```bash
  make security
  ```

Always run these quality checks before submitting a pull request. They will also be run in the CI pipeline on any branches with the `dev-` prefix and on PRs merged to `main`.

## Testing Guidelines

We use pytest for testing with async support:

- Keep tests in the tests/ directory
- Aim for 80%+ test coverage for new features
- Mark async tests with `@pytest.mark.asyncio`

### Flake rules

- Wait for the fact an assertion reads, not a fixed sleep or a count of unrelated
  loop turns. TUI tests that require an adopted session can use
  `await_session_boot(app)` from `tests.unit.tui.conftest`; tests deliberately
  holding boot open must not use that helper. `pilot.pause()` drains UI messages,
  not the factory's thread. Network copy storage and the owner's acknowledgement
  are separate writes: wait on the acknowledgement before asserting its ledger.
- Use per-test temporary directories and OS-assigned ports. Isolate HOME/config,
  strip inherited session/terminal variables, and restore locale, TZ, environment,
  cwd and module-global state after mutation. See `AGENTS.md` for the required
  isolation prefix and bounded process-group cleanup.
- Inject a clock for expiry tests; use exact arithmetic for logical clocks and
  avoid real-wall-clock performance assertions when a structural invariant exists.
  A timeout is a deadlock backstop, never evidence that work has completed.
- Reproduce before changing a wait. For a deterministic boot scheduling probe,
  run **one targeted test**, serially, with `LOP_TEST_SLOW_BOOT_WARM_S=0.4` in its
  isolated environment. This delays only the existing warm-up thread; it does not
  disable the production boot path. Do not apply it to the whole suite by default.
- Repeat touched node IDs serially in fresh isolated environments, within
  `scripts/run_bounded.py` limits. No repeat/random-order plugin is required by
  this repository. A rerun that passes is evidence to investigate, not permission
  to hide the failure with retries, skips or xfails. Record the failing run URL,
  signature, actual before/after rates and any cases still unresolved.

## Contribution Workflow

Please follow the following steps to contribute:

1. For accounts without contributor access, create your fork of the repository
2. Create a new branch for your changes with the `dev-` prefix

   ```bash
   git checkout -b dev-my-feature
   # or
   git checkout -b dev-issue-number-description
   ```

3. Commit changes with descriptive messages

   - Use the present tense
   - Keep commits small and atomic

   ```bash
   git add .
   git commit -m "Add feature for specific component"
   ```

4. Update documentation if adding new features or changing behavior

5. Push your changes to your fork

   ```bash
   git push origin dev-my-feature
   ```

6. Create a Pull Request against the main branch of the upstream repository

## Release Process

Once approved and merged, maintainers will package changes into releases with the following procedure:

1. Update the version in `pyproject.toml`
2. Create a git tag with prefix `v` (for example `v0.2.0`)
3. Push the tag to the upstream repository on the commit to release on `main`
4. Create a release from the tag on GitHub with a concise release name and a description of the changes
5. CD will trigger on release creation and package and upload the version in `pyproject.toml` to PyPI

**For pre-release versions**:

- Use the `ax` or `bx` suffix (for example `v0.2.0a1`)
- Update the version in `pyproject.toml` on the `dev-` branch
- Releases can be created by maintainers from the `dev-` branch if they are not ready to be merged to main

## Pull Request Checklist

- Tests added/updated
- Documentation updated (README, docstrings)
- Code formatted with black and isort
- Linting passes (flake8)
- Type checking passes (pyright)
- All CI checks pass
- Security considerations addressed (for code execution features)

## Reporting Issues

When filing an issue, please include:

1. Local Operator version
2. Python version
3. Hosting platform (DeepSeek/Ollama/OpenAI)
4. Steps to reproduce
5. Expected vs actual behavior
6. Error logs (if applicable)

## Security Considerations

For features involving code execution or system operations:

- Clearly document security implications
- Maintain safety checks and user confirmations
- Add tests for potentially dangerous operations
- Follow principle of least privilege in file/system access

## Feature Requests

We welcome innovative ideas! When proposing new features:

- Explain the use case and target audience
- Outline potential implementation strategy
- Discuss security implications
- Suggest documentation needs

## Documentation

Help us improve documentation by:

- Fixing typos/outdated information
- Adding usage examples
- Updating the dependency graph in `docs/dependencies.md`
- Improving section organization
- Translating documentation (if applicable)

## Need Help?

Join our Discussions to:

- Ask questions about implementation
- Discuss architectural decisions
- Propose major changes before coding
- Share your use cases

Thank you for helping make Local Operator better! 🚀
