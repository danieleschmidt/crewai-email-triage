# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records for the CrewAI Email Triage project.

## ADR Format

We use the format proposed by Michael Nygard in his article "Documenting Architecture Decisions":

- **Title**: A short phrase describing the architectural decision
- **Status**: Proposed, Accepted, Deprecated, or Superseded
- **Context**: The forces at play, including technological, political, social, and project local
- **Decision**: The change that we're proposing or have agreed to implement
- **Consequences**: What becomes easier or more difficult to do because of this change

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [ADR-0002](0002-multi-agent-pipeline-architecture.md) | Multi-agent pipeline architecture | Accepted |
| [ADR-0003](0003-python-runtime-and-packaging.md) | Python runtime and packaging | Accepted |
| [ADR-0004](0004-configuration-management.md) | Configuration management approach | Accepted |
| [ADR-0005](0005-security-credential-handling.md) | Security credential handling | Accepted |

## Creating New ADRs

1. Copy the template from `template.md`
2. Number sequentially (e.g., `0006-description.md`)
3. Fill in all sections
4. Update this README index
5. Create pull request for review