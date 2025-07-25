# Dependency Management Strategy

## Overview

This document outlines the dependency management strategy for the CrewAI Email Triage project, including version pinning rationale and security considerations.

## Production Dependencies

### Core Dependencies
- **cryptography** (>=43.0.0,<46.0.0)
  - Used for secure credential storage and encryption
  - Version range allows patch updates while preventing major breaking changes
  - Minimum version 43.0.0 ensures security fixes and stable API
  - Maximum version <46.0.0 prevents untested major version updates

## Development Dependencies

### Test Dependencies (Optional Extra: `test`)
All test dependencies use compatible release clauses (`~=`) to allow patch updates while preventing minor/major version changes that could break CI:

- **pytest** (~=8.4.0) - Test framework
- **pytest-cov** (~=6.2.0) - Coverage reporting
- **ruff** (~=0.12.0) - Fast Python linter
- **bandit** (~=1.8.0) - Security vulnerability scanner
- **pre-commit** (~=4.2.0) - Git pre-commit hooks
- **pytest-xdist** (~=3.8.0) - Parallel test execution

## Dependency Pinning Rationale

### Security
- All dependencies are regularly scanned for vulnerabilities using `pip-audit`
- Version constraints prevent automatic updates to potentially vulnerable versions
- Compatible release clauses allow security patches while maintaining stability

### Reproducibility
- Pinned test dependencies ensure consistent CI/CD behavior
- Development teams can reproduce identical test environments
- Build artifacts are deterministic across environments

### Stability
- Version ranges for production dependencies balance security with stability
- Test dependencies use stricter pinning to prevent CI flakiness
- Manual dependency updates allow for proper testing before deployment

## Updating Dependencies

### Security Updates
1. Run `pip-audit` to check for vulnerabilities
2. Update affected packages within existing constraints when possible
3. If major version update required, test thoroughly and update constraints

### Regular Updates
1. Review dependency updates quarterly
2. Test updates in isolated environment
3. Update version constraints if tests pass
4. Document any breaking changes or migration requirements

### Process
```bash
# Check for vulnerabilities
pip-audit

# Update dependencies within constraints
pip install -e ".[test]" --upgrade

# Test with new versions
python -m pytest

# Update constraints if needed
# Edit pyproject.toml and test again
```

## Installation

### Production
```bash
pip install crewai-email-triage
```

### Development
```bash
pip install -e ".[test]"
```

## Security Scanning

Dependencies are automatically scanned for known vulnerabilities. Current status:
- ✅ No known vulnerabilities found (as of 2025-07-25)
- All dependencies use secure, maintained versions
- Regular security updates applied within version constraints

## Notes

- Python requirement: >=3.8 (maintains compatibility with modern Python versions)
- Only essential dependencies are included in production
- Test dependencies are optional to minimize production footprint
- All version constraints are documented and justified above