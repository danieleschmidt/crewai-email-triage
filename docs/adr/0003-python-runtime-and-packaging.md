# ADR-0003: Python runtime and packaging

**Date**: 2025-07-27
**Status**: Accepted
**Deciders**: Development Team

## Context

We need to choose a programming language and packaging system for the email triage application. The system needs to handle text processing, integrate with email providers, and provide a CLI interface.

## Decision Drivers

* Text processing capabilities
* Rich ecosystem for email and NLP libraries
* CLI development ease
* Team expertise
* Deployment simplicity
* Performance requirements

## Considered Options

* Python with setuptools
* Node.js with npm
* Go with modules
* Rust with Cargo

## Decision Outcome

Chosen option: "Python with setuptools", because Python provides excellent text processing libraries, email handling capabilities, and the team has strong Python expertise.

### Positive Consequences

* Rich ecosystem for text processing and email
* Excellent testing frameworks (pytest)
* Strong team expertise
* Rapid development capabilities
* Good CLI libraries available

### Negative Consequences

* Slower execution compared to compiled languages
* Dependency management complexity
* GIL limitations for CPU-bound tasks

## Pros and Cons of the Options

### Python with setuptools

* Good, because of rich NLP and email libraries
* Good, because of team expertise
* Good, because of rapid development
* Bad, because of performance limitations
* Bad, because of dependency complexity

### Node.js with npm

* Good, because of fast development cycle
* Good, because of large package ecosystem
* Bad, because less suitable for text processing
* Bad, because of limited team expertise

### Go with modules

* Good, because of excellent performance
* Good, because of simple deployment
* Bad, because of limited text processing libraries
* Bad, because of learning curve for team

### Rust with Cargo

* Good, because of excellent performance and safety
* Good, because of modern tooling
* Bad, because of steep learning curve
* Bad, because of limited ecosystem for this use case

## Links

* [pyproject.toml configuration](../../pyproject.toml)