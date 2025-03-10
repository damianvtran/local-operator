# Contributing to Local Operator

Thank you for your interest in contributing to Local Operator! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions. By participating in this project, you agree to abide by its [GPL 3.0 License](LICENSE).

## Project Structure

```shell
.
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Project license
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
├── setup.py                     # Package installation configuration
├── static                       # Static assets and resources
└── tests                        # Test suite directory
    ├── conftest.py              # Pytest configuration and fixtures
    └── unit                     # Unit test implementations
```

## Dependency Graph

Refer to the [dependency graph](docs/dependencies.md) for a visual representation of the project structure.

## Getting Started

### Prerequisites

- Python 3.12+
- Dependencies in requirements.txt

### Development Setup

1. If you don't have contributor access, **Fork** the repository on GitHub
2. **Clone** your forked repository:

   ```bash
   git clone https://github.com/your-username/local-operator.git
   ```

3. Set up a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -e .[dev]
   ```

## Code Style & Quality

We enforce consistent code style and quality checks:

- **Formatting**: Uses black and isort

  ```bash
  black .
  isort .
  ```

- **Linting**: Uses flake8

  ```bash
  flake8
  ```

- **Type Checking**: Uses pyright

  ```bash
  pyright
  ```

- **Dependency Scanning**: Uses pip-audit

  ```bash
  pip-audit
  ```

Always run these tools before submitting a pull request.  They will also be run in the CI pipeline on any branches with the `dev-` prefix and on PRs merged to `main`.

## Testing

We use pytest for testing with async support:

```bash
pytest
```

- Keep tests in the tests/ directory
- Aim for 80%+ test coverage for new features
- Mark async tests with `@pytest.mark.asyncio`

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
