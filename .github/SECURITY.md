# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of CrewAI Email Triage seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities through one of these channels:

1. **GitHub Security Advisories** (Preferred)
   - Go to the [Security tab](https://github.com/crewai/email-triage/security/advisories)
   - Click "Report a vulnerability"
   - Fill out the form with detailed information

2. **Email**
   - Send an email to: security@crewai-email-triage.com
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### 📋 What to Include

When reporting a vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and severity assessment
- **Reproduction**: Step-by-step reproduction instructions
- **Environment**: Affected versions, operating systems, etc.
- **Proof of Concept**: Code snippets or screenshots (if applicable)
- **Suggested Fix**: Any ideas for fixing the issue (optional)

### 🚀 Response Timeline

- **Initial Response**: Within 24 hours
- **Vulnerability Assessment**: Within 72 hours
- **Status Updates**: Every 7 days until resolved
- **Fix Timeline**: Critical issues within 7 days, others within 30 days

### 🛡️ Security Measures

We implement several security measures:

#### Code Security
- **Static Analysis**: Bandit security linting
- **Dependency Scanning**: Safety and pip-audit checks
- **Secret Detection**: Pre-commit hooks and CI scanning
- **Container Scanning**: Trivy vulnerability scanning
- **Code Review**: All changes require review

#### Runtime Security
- **Input Validation**: All user inputs are sanitized
- **Secure Credentials**: Encrypted credential storage
- **Rate Limiting**: Protection against abuse
- **Access Controls**: Principle of least privilege
- **Audit Logging**: Security events are logged

#### Infrastructure Security
- **HTTPS**: All communications encrypted in transit
- **Secrets Management**: Environment-based secret handling
- **Container Security**: Minimal attack surface
- **Network Security**: Firewall and network segmentation

### 🔄 Security Updates

#### Automated Security
- **Dependabot**: Automated dependency updates
- **Security Advisories**: GitHub security advisory monitoring
- **CVE Monitoring**: Continuous vulnerability assessment
- **Pre-commit Hooks**: Security checks before commits

#### Manual Reviews
- **Quarterly Security Reviews**: Comprehensive security assessment
- **Penetration Testing**: Annual third-party security testing
- **Code Audits**: Regular security-focused code reviews
- **Threat Modeling**: Regular threat assessment updates

### 📚 Security Best Practices

#### For Users
- **Environment Variables**: Never commit secrets to version control
- **Access Controls**: Use strong, unique passwords
- **Updates**: Keep the application updated to latest versions
- **Network Security**: Use secure networks for email access
- **Monitoring**: Monitor application logs for suspicious activity

#### For Contributors
- **Secure Development**: Follow secure coding practices
- **Testing**: Include security tests in contributions
- **Dependencies**: Use only trusted, maintained dependencies
- **Review**: Participate in security-focused code reviews
- **Training**: Stay updated on security best practices

### 🏆 Security Hall of Fame

We recognize security researchers who help improve our security:

*(No reports yet - be the first!)*

### 📞 Contact

For security-related questions or concerns:

- **Security Team**: security@crewai-email-triage.com
- **General Questions**: Use GitHub Discussions
- **Bug Reports**: Use GitHub Issues (for non-security bugs only)

### 🔗 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [GitHub Security Features](https://docs.github.com/en/github/managing-security-vulnerabilities)

---

Thank you for helping keep CrewAI Email Triage secure! 🙏