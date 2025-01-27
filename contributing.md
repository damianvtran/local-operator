# Contributing to Local Operator

Thank you for your interest in contributing to Local Operator! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions. By participating in this project, you agree to abide by its [MIT License](LICENSE).

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) (recommended for dependency management)
- Basic understanding of command-line interfaces

### Development Setup

1. **Fork** the repository on GitHub
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

Always run these tools before submitting a pull request.  They will also be run in the CI pipeline on any branches with the `dev-` prefix.

## Testing

We use pytest for testing with async support:

```bash
pytest
```

- Keep tests in the tests/ directory
- Aim for 60%+ test coverage for new features
- Mark async tests with `@pytest.mark.asyncio`

## Contribution Workflow

Please follow the following steps to contribute:

1. Create your fork of the repository
2. Create a new branch for your changes

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
- Improving section organization
- Translating documentation (if applicable)

## Need Help?

Join our Discussions to:

- Ask questions about implementation
- Discuss architectural decisions
- Propose major changes before coding
- Share your use cases

Thank you for helping make Local Operator better! 🚀
