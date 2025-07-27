# ADR-0005: Security credential handling

**Date**: 2025-07-27
**Status**: Accepted
**Deciders**: Development Team

## Context

The email triage system needs to handle sensitive credentials for accessing email providers (Gmail, IMAP servers). We need a secure approach that prevents credential leakage while maintaining usability.

## Decision Drivers

* Security requirement to never store credentials in code
* Need for environment-specific credential management
* Prevention of credential leakage in logs or version control
* Usability for local development and production deployment
* Compliance with security best practices

## Considered Options

* Environment variables for all credentials
* External secret management systems (HashiCorp Vault, AWS Secrets Manager)
* Encrypted configuration files
* OAuth 2.0 flows with token refresh
* System keychain integration

## Decision Outcome

Chosen option: "Environment variables for all credentials", because they provide a secure, widely-supported approach that works well in both development and production environments without requiring additional infrastructure.

### Positive Consequences

* No credentials stored in code or configuration files
* Works well with container deployments
* Supported by all deployment platforms
* Simple to implement and understand
* Good separation between code and secrets

### Negative Consequences

* Credentials visible in process environment
* Manual management of credential rotation
* No built-in encryption at rest

## Pros and Cons of the Options

### Environment variables for all credentials

* Good, because they keep secrets out of code
* Good, because they work in all deployment environments
* Good, because they're simple to implement
* Bad, because they're visible in process environment
* Bad, because they don't provide encryption at rest

### External secret management systems

* Good, because they provide centralized secret management
* Good, because they support automatic rotation
* Good, because they provide audit trails
* Bad, because they add infrastructure complexity
* Bad, because they require additional operational overhead

### OAuth 2.0 flows with token refresh

* Good, because they eliminate long-lived credentials
* Good, because they provide better security model
* Good, because they support automatic token refresh
* Bad, because they add implementation complexity
* Bad, because they require web browser interaction for setup

### Encrypted configuration files

* Good, because they provide encryption at rest
* Good, because they can be version controlled (encrypted)
* Bad, because they require key management
* Bad, because they add complexity to deployment

## Links

* [Secure credentials implementation](../../src/crewai_email_triage/secure_credentials.py)
* [Environment configuration](../../src/crewai_email_triage/env_config.py)