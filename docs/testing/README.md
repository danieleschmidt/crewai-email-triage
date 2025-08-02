# Testing Documentation

## Overview

CrewAI Email Triage uses a comprehensive testing strategy that includes unit tests, integration tests, performance tests, and end-to-end tests. This document provides guidelines for writing, running, and maintaining tests.

## Test Structure

```
tests/
├── conftest.py                 # Global pytest configuration and fixtures
├── fixtures/                   # Test data and configuration fixtures
│   ├── sample_emails.json     # Email test data
│   └── test_configs.json      # Configuration test data
├── unit/                       # Unit tests (implied in root)
├── integration/                # Integration tests
├── performance/                # Performance and benchmark tests
├── e2e/                       # End-to-end tests
└── test_performance_config.py  # Performance testing utilities
```

## Test Categories

### Unit Tests
- **Location**: `tests/test_*.py`
- **Purpose**: Test individual components in isolation
- **Marker**: `@pytest.mark.unit`
- **Characteristics**:
  - Fast execution (< 1 second per test)
  - No external dependencies
  - Mocked external services
  - High coverage of edge cases

### Integration Tests  
- **Location**: `tests/integration/`
- **Purpose**: Test component interactions
- **Marker**: `@pytest.mark.integration`
- **Characteristics**:
  - Test multiple components together
  - May use test databases or services
  - Validate data flow between components

### Performance Tests
- **Location**: `tests/performance/`
- **Purpose**: Validate performance requirements
- **Marker**: `@pytest.mark.performance`
- **Characteristics**:
  - Measure response times and throughput
  - Memory usage validation
  - Load testing scenarios
  - Benchmark tracking

### End-to-End Tests
- **Location**: `tests/e2e/`
- **Purpose**: Test complete user workflows
- **Marker**: `@pytest.mark.e2e`
- **Characteristics**:
  - Full system testing
  - Real or realistic test environments
  - User scenario validation

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/crewai_email_triage --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m performance            # Performance tests only
pytest -m "not slow"             # Skip slow tests

# Run tests in parallel
pytest -n auto                   # Auto-detect CPU cores
pytest -n 4                      # Use 4 workers

# Run with detailed output
pytest -v --tb=short             # Verbose with short tracebacks
pytest -v --tb=long              # Verbose with full tracebacks
```

### Performance Testing

```bash
# Run performance benchmarks
pytest -m performance --benchmark-only

# Run memory profiling tests  
pytest -m performance_memory --profile

# Run load testing
pytest -m performance_load -v

# Generate performance report
pytest -m performance --benchmark-json=benchmark_results.json
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Unit Test Coverage**: 90%+ for core modules
- **Integration Coverage**: 70%+ for critical paths
- **Exclusions**: Test files, migrations, generated code

```bash
# Generate coverage report
pytest --cov=src/crewai_email_triage \
       --cov-report=html \
       --cov-report=term-missing \
       --cov-fail-under=80
```

## Writing Tests

### Test Conventions

1. **File Naming**: `test_<module_name>.py`
2. **Function Naming**: `test_<functionality>_<expected_result>`
3. **Class Naming**: `Test<ClassName>`
4. **Fixture Naming**: `<purpose>_<type>` (e.g., `sample_email`, `mock_provider`)

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from crewai_email_triage.classifier import ClassifierAgent

class TestClassifierAgent:
    """Test suite for ClassifierAgent."""
    
    def test_classify_urgent_email_returns_urgent_category(self, test_config):
        """Test that urgent emails are classified correctly."""
        # Arrange
        agent = ClassifierAgent(test_config)
        urgent_email = {
            "subject": "URGENT: Server down",
            "body": "Critical system failure"
        }
        
        # Act
        result = agent.classify(urgent_email)
        
        # Assert
        assert result["category"] == "urgent"
        assert result["confidence"] > 0.8
        
    @pytest.mark.parametrize("email_data,expected_category", [
        ({"subject": "Meeting tomorrow", "body": "Project discussion"}, "work"),
        ({"subject": "Birthday party", "body": "Come celebrate!"}, "personal"),
        ({"subject": "Unsubscribe now", "body": "Click to unsubscribe"}, "spam"),
    ])
    def test_classify_email_categories(self, test_config, email_data, expected_category):
        """Test classification of various email categories."""
        agent = ClassifierAgent(test_config)
        result = agent.classify(email_data)
        assert result["category"] == expected_category
```

### Performance Test Example

```python
@pytest.mark.performance
def test_single_email_processing_performance(
    sample_emails, 
    performance_monitor, 
    performance_thresholds
):
    """Test single email processing meets performance requirements."""
    email = sample_emails[0]
    
    # Measure processing time
    performance_monitor.start_timer("single_email")
    pipeline = EmailTriagePipeline()
    result = pipeline.process(email)
    duration = performance_monitor.end_timer("single_email")
    
    # Assert performance requirements
    assert duration < performance_thresholds.SINGLE_EMAIL_COMPLETE_PIPELINE
    assert result is not None
```

## Test Fixtures and Data

### Using Sample Data

```python
def test_email_classification(sample_emails):
    """Test using fixture data."""
    urgent_emails = [e for e in sample_emails if e["expected_classification"]["category"] == "urgent"]
    assert len(urgent_emails) > 0
    
    for email in urgent_emails:
        # Test with predefined expected results
        result = classifier.classify(email)
        expected = email["expected_classification"]
        assert result["category"] == expected["category"]
```

### Custom Fixtures

```python
@pytest.fixture
def custom_email_batch():
    """Create a custom batch of emails for specific test needs."""
    return [
        {"subject": f"Test email {i}", "body": f"Content {i}"}
        for i in range(10)
    ]
```

## Mocking Guidelines

### External Service Mocking

```python
@pytest.fixture
def mock_gmail_provider():
    """Mock Gmail provider for testing."""
    with patch('crewai_email_triage.provider.GmailProvider') as mock:
        provider = Mock()
        provider.fetch_messages.return_value = []
        provider.send_response.return_value = True
        mock.return_value = provider
        yield provider

def test_email_fetching(mock_gmail_provider):
    """Test email fetching with mocked provider."""
    emails = mock_gmail_provider.fetch_messages()
    assert isinstance(emails, list)
    mock_gmail_provider.fetch_messages.assert_called_once()
```

### Environment Variable Mocking

```python
@patch.dict(os.environ, {
    "GMAIL_USER": "test@example.com",
    "GMAIL_PASSWORD": "test_password"
})
def test_configuration_loading():
    """Test configuration loading with mocked environment."""
    config = load_configuration()
    assert config.gmail_user == "test@example.com"
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.performance
def test_batch_processing_throughput(performance_test_data, performance_monitor):
    """Benchmark batch processing throughput."""
    emails = performance_test_data["medium_batch"]  # 100 emails
    
    performance_monitor.start_timer("batch_processing")
    pipeline = EmailTriagePipeline()
    results = pipeline.process_batch(emails)
    duration = performance_monitor.end_timer("batch_processing")
    
    throughput = len(emails) / duration
    assert throughput >= 10  # At least 10 emails per second
    assert len(results) == len(emails)
```

### Memory Testing

```python
@pytest.mark.performance_memory
def test_memory_usage_within_limits(memory_profiler, performance_thresholds):
    """Test that memory usage stays within acceptable limits."""
    memory_profiler.reset_baseline()
    
    # Process large batch
    pipeline = EmailTriagePipeline()
    large_batch = generate_test_emails(1000)
    results = pipeline.process_batch(large_batch)
    
    memory_delta = memory_profiler.memory_delta()
    assert memory_delta < performance_thresholds.BATCH_PROCESSING_MAX_MEMORY
```

## Continuous Integration

### GitHub Actions Integration

The test suite runs automatically on:
- Pull requests
- Pushes to main branch
- Nightly scheduled runs

### Test Stages

1. **Unit Tests**: Fast feedback (< 2 minutes)
2. **Integration Tests**: Component validation (< 5 minutes)  
3. **Performance Tests**: Benchmark validation (< 10 minutes)
4. **E2E Tests**: Full system validation (< 15 minutes)

### Quality Gates

- All tests must pass
- Coverage threshold: 80%
- Performance regression: < 10% slowdown
- No security vulnerabilities
- Code quality checks pass

## Debugging Tests

### Common Issues

1. **Flaky Tests**: Use proper mocking and fixtures
2. **Slow Tests**: Profile and optimize, add `@pytest.mark.slow`
3. **Environment Issues**: Use mock environment variables
4. **Resource Leaks**: Proper cleanup in fixtures

### Debugging Commands

```bash
# Debug single test
pytest tests/test_classifier.py::test_classify_urgent -v -s

# Run with debugger
pytest --pdb tests/test_classifier.py::test_classify_urgent

# Show local variables on failure
pytest --tb=long --showlocals

# Capture output
pytest -s tests/test_classifier.py  # Show print statements
```

## Best Practices

### Test Design

1. **AAA Pattern**: Arrange, Act, Assert
2. **Single Responsibility**: One concept per test
3. **Descriptive Names**: Clear test intentions
4. **Independent Tests**: No test dependencies
5. **Fast Feedback**: Quick test execution

### Data Management

1. **Use Fixtures**: Consistent test data
2. **Parametrize Tests**: Multiple scenarios
3. **Mock External Dependencies**: Isolation
4. **Clean Up**: Proper resource cleanup

### Performance Considerations

1. **Baseline Measurements**: Track performance trends
2. **Reasonable Thresholds**: Achievable but meaningful
3. **Environment Consistency**: Reproducible results
4. **Resource Monitoring**: Memory and CPU usage

### Maintenance

1. **Regular Updates**: Keep tests current
2. **Refactor with Code**: Maintain test quality
3. **Remove Obsolete Tests**: Clean up unused tests
4. **Documentation**: Keep this guide updated

## Tools and Extensions

### Recommended pytest Plugins

```bash
pip install pytest-cov          # Coverage reporting
pip install pytest-xdist        # Parallel execution
pip install pytest-mock         # Enhanced mocking
pip install pytest-benchmark    # Performance benchmarking
pip install pytest-html         # HTML reports
pip install pytest-asyncio      # Async test support
```

### IDE Integration

- **VS Code**: Python Test Discovery
- **PyCharm**: Integrated test runner
- **Vim/Neovim**: vim-test plugin

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

For questions or suggestions about testing, please open an issue or contact the development team.