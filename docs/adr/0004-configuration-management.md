# ADR-0004: Configuration management approach

**Date**: 2025-07-27
**Status**: Accepted
**Deciders**: Development Team

## Context

The email triage system needs configurable classification keywords, priority scoring, and various operational parameters. We need a flexible configuration system that supports both defaults and customization.

## Decision Drivers

* Need for easy customization without code changes
* Support for different environments (dev, test, prod)
* User-friendly configuration format
* Validation and error handling
* Environment variable integration

## Considered Options

* JSON configuration files with environment variable overrides
* YAML configuration files
* Environment variables only
* Python configuration modules
* TOML configuration files

## Decision Outcome

Chosen option: "JSON configuration files with environment variable overrides", because JSON provides a simple, widely-supported format that's easy to validate and parse, while environment variables allow for deployment-specific overrides.

### Positive Consequences

* Simple and widely supported format
* Easy to validate with JSON Schema
* Good tooling support in editors
* Clear separation between defaults and overrides
* Version controllable configuration

### Negative Consequences

* No support for comments in JSON
* Less human-readable than YAML
* Limited data types compared to Python modules

## Pros and Cons of the Options

### JSON configuration files with environment variables

* Good, because JSON is simple and widely supported
* Good, because environment variables provide deployment flexibility
* Good, because it's easy to validate
* Bad, because JSON doesn't support comments
* Bad, because it's less readable than YAML

### YAML configuration files

* Good, because it's human-readable and supports comments
* Good, because it supports complex data structures
* Bad, because YAML parsing can be complex and error-prone
* Bad, because it's less standardized than JSON

### Environment variables only

* Good, because it's simple and deployment-friendly
* Good, because it integrates well with container environments
* Bad, because it's difficult to manage complex configurations
* Bad, because it lacks structure for nested data

### Python configuration modules

* Good, because it provides maximum flexibility
* Good, because it allows for dynamic configuration
* Bad, because it's a security risk (code execution)
* Bad, because it's harder to validate and manage

## Links

* [Default configuration](../../src/crewai_email_triage/default_config.json)
* [Configuration loading](../../src/crewai_email_triage/config.py)