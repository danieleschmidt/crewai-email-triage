# Manual Setup Requirements

## Repository Configuration
### Branch Protection Rules (Admin Required)
```bash
# Configure via GitHub Settings > Branches
• Protect 'main' branch
• Require pull request reviews (1 minimum)
• Require status checks to pass
• Restrict pushes to 'main'
```

### GitHub Actions Workflows (Admin Required)
Create these workflow files in `.github/workflows/`:

• **`test.yml`**: Test on Python 3.8-3.12, run pytest with coverage
• **`lint.yml`**: Run ruff, bandit, and pre-commit hooks
• **`release.yml`**: Automated semantic versioning and PyPI publishing
• **`docs.yml`**: Build and deploy documentation to GitHub Pages

### Repository Settings
• Enable Issues, Discussions, and Security Advisories
• Configure Topics: `email`, `triage`, `automation`, `python`, `ai`
• Set Repository Description and Homepage URL
• Enable vulnerability alerts and automated security updates

### External Integrations
• **Dependabot**: Configure for Python dependencies (weekly)
• **CodeQL**: Enable GitHub Advanced Security scanning
• **Monitoring**: Configure observability and alerting tools

## References
• [GitHub Repository Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)
• [GitHub Actions Workflows](https://docs.github.com/en/actions/using-workflows)
• [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot)