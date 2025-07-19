# Technical Backlog - Impact Ranked (WSJF)

## Critical Issues (High Impact, Low Effort)

### 1. Missing Package Dependencies & Import Structure [WSJF: 90]
- **Impact**: 30 (Critical - project won't run)
- **Effort**: 3 (Low - quick fix)
- **Issue**: Missing `__init__.py` imports, package structure broken
- **Evidence**: `ModuleNotFoundError: No module named 'crewai_email_triage'` in triage.py:10
- **Risk**: Complete system failure

### 2. Hardcoded Gmail Credentials Vulnerability [WSJF: 80] 
- **Impact**: 25 (High - security risk)
- **Effort**: 5 (Medium - requires OAuth integration)
- **Issue**: Password authentication instead of OAuth2
- **Evidence**: provider.py:26 stores plaintext passwords
- **Risk**: Credential exposure, deprecated Gmail auth

### 3. Missing Error Handling in Email Parsing [WSJF: 75]
- **Impact**: 20 (Medium - runtime failures)
- **Effort**: 4 (Low-medium - add try/catch blocks)
- **Issue**: No error handling for malformed emails
- **Evidence**: provider.py:43-46 raw decode without validation
- **Risk**: Runtime crashes on malformed emails

## Performance Issues (Medium Impact, Low-Medium Effort)

### 4. Inefficient Agent Instantiation in Batch Mode [WSJF: 60]
- **Impact**: 15 (Medium - performance degradation)
- **Effort**: 4 (Medium - refactor to reuse instances)
- **Issue**: Creates new agent instances per message in parallel mode
- **Evidence**: pipeline.py:94-103 creates agents per thread
- **Risk**: Memory and CPU overhead at scale

### 5. Missing Request ID and Structured Logging [WSJF: 55]
- **Impact**: 10 (Low-medium - observability)
- **Effort**: 6 (Medium - implement logging framework)
- **Issue**: Debug logs lack correlation IDs
- **Evidence**: pipeline.py:30-40 basic logger.debug calls
- **Risk**: Difficult debugging in production

## Feature Gaps (High Impact, High Effort)

### 6. No Input Sanitization/Validation [WSJF: 50]
- **Impact**: 25 (High - security)
- **Effort**: 10 (High - comprehensive validation)
- **Issue**: Raw email content processed without sanitization
- **Evidence**: core.py:19 processes content directly
- **Risk**: XSS, injection attacks

### 7. Missing Test Coverage for Integration Paths [WSJF: 45]
- **Impact**: 15 (Medium - quality assurance)
- **Effort**: 8 (High - comprehensive test suite)
- **Issue**: No end-to-end pipeline tests
- **Evidence**: Tests focus on individual components
- **Risk**: Integration bugs in production

### 8. No Metrics Export (Prometheus/OpenTelemetry) [WSJF: 40]
- **Impact**: 10 (Low-medium - observability)
- **Effort**: 8 (High - metrics infrastructure)
- **Issue**: METRICS dict not exported
- **Evidence**: pipeline.py:18 local metrics only
- **Risk**: Limited production monitoring

## Technical Debt (Medium Impact, Medium Effort)

### 9. Inconsistent Return Type Parsing [WSJF: 35]
- **Impact**: 10 (Low-medium - maintainability)
- **Effort**: 5 (Medium - standardize agent responses)
- **Issue**: String replacement for agent outputs
- **Evidence**: pipeline.py:29,31,37,39 manual string parsing
- **Risk**: Fragile parsing logic

### 10. Missing Configuration Validation [WSJF: 30]
- **Impact**: 10 (Low-medium - reliability)
- **Effort**: 6 (Medium - schema validation)
- **Issue**: No validation of config.json structure
- **Evidence**: config.py:17 direct JSON load
- **Risk**: Runtime errors from malformed config

## WSJF Calculation: (Business Value + Time Criticality + Risk Reduction) / Job Size
- Business Value: 1-10 scale
- Time Criticality: 1-10 scale  
- Risk Reduction: 1-10 scale
- Job Size (Effort): 1-10 scale

## Next Actions
1. Fix package structure and imports (#1)
2. Implement OAuth2 for Gmail (#2) 
3. Add comprehensive error handling (#3)
4. Optimize agent reuse in batch processing (#4)