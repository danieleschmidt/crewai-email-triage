# 🧭 Project Vision

Smart assistant that classifies, prioritizes, summarizes and drafts replies for email messages. Targets busy professionals needing quick triage of high volumes of mail. Provides simple CLI and Gmail integration.

# 📅 12-Week Roadmap

## I1 - Foundations & Security
- **Themes**: Security, Developer UX
- **Goals / Epics**
  - Harden credential handling and remove hardcoded values
  - Stabilize CI with reproducible setup and lint checks
- **Definition of Done**
  - Secrets loaded from environment or vault
  - CI runs lint, security scan and tests with no failures

## I2 - Performance & Scaling
- **Themes**: Performance, Observability
- **Goals / Epics**
  - Optimize batch processing and measure latency
  - Add structured logging and basic metrics
- **Definition of Done**
  - Batch mode processes at least 100 emails/min locally
  - Logs include request id and timing info

## I3 - Advanced Features
- **Themes**: User Experience, Integrations
- **Goals / Epics**
  - Extend provider support beyond Gmail
  - Add learning mode using user feedback
- **Definition of Done**
  - IMAP/Outlook providers implemented
  - Feedback loop updates priority model

# ✅ Epic & Task Checklist

### 🔒 Increment 1: Security & Refactoring
- [x] [EPIC] Eliminate hardcoded secrets
  - [x] Load credentials from environment securely
  - [x] Add pre-commit hook scanning for secrets
- [x] [EPIC] Improve CI stability
  - [x] Replace flaky integration tests
  - [x] Enable parallel test execution

### ⚡️ Increment 2: Performance & Observability
- [ ] [EPIC] Optimize batch triage
  - [ ] Profile pipeline with >100 emails
  - [x] Introduce async or multiprocessing
- [ ] [EPIC] Structured logging
  - [ ] Add request id to all log lines
  - [ ] Export metrics to Prometheus format

### 💻 Increment 3: Advanced Features
- [ ] [EPIC] Multi-provider support
  - [ ] Implement Outlook client
  - [ ] Configurable IMAP provider
- [ ] [EPIC] Learning mode
  - [ ] Collect user feedback on responses
  - [ ] Adjust priority scores based on history

# ⚠️ Risks & Mitigation
- Misused credentials → Use OAuth or secure secrets store
- High latency with large batches → Introduce async processing and measure
- Spam or malicious emails → Add input sanitization and scanning
- CI failures slow team → Isolate flaky tests and cache dependencies
- Feature creep → Review roadmap each sprint and keep scope focused

# 📊 KPIs & Metrics
- [ ] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

# 👥 Ownership & Roles (Optional)
- DevOps: maintain CI/CD and secret management
- Backend: implement agents and providers
- QA: expand automated test suites
