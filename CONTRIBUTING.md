# Contributing to CrewAI Email Triage

Thank you for taking the time to contribute!

## Getting Started
1. Fork the repository and create your branch from `main`.
2. Install dependencies:
   ```bash
   pip install -e .[test]
   ```
3. Ensure the test suite passes:
   ```bash
   pytest -q
   ```
4. Run static analysis:
   ```bash
   ruff check .
   bandit -r src -q
   ```

## Commit Messages
- Use present-tense imperative style: `Add logging`, `Fix bug in triage`.
- Keep commits focused and small.

## Pull Requests
- Include a clear description of the change and link to open issues.
- Update documentation and tests as needed.
- Ensure CI checks pass before requesting review.

## Security
Report any security vulnerabilities privately by opening a security issue on GitHub.
