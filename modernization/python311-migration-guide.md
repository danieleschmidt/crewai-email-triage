# Python 3.11+ Migration Guide for Email Triage Service

## Overview
Comprehensive migration guide to leverage Python 3.11+ features for enhanced performance, type safety, and developer experience in email processing workloads.

## Performance Improvements

### 1. Exception Performance (3.11+)
Python 3.11 exceptions are 10-30% faster. Update error handling patterns:

```python
# Before (Python 3.8-3.10)
def process_email_with_fallback(email_data):
    try:
        return classify_email(email_data)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return fallback_classification(email_data)

# After (Python 3.11+) - Leverages faster exception handling
async def process_email_with_fallback(email_data):
    try:
        return await classify_email_async(email_data)
    except ClassificationError as e:
        # Specific exception handling is now more performant
        logger.error(f"Classification failed: {e}")
        return await fallback_classification_async(email_data)
```

### 2. Faster Startup (3.11+)
Utilize cached bytecode for faster module loading:

```python
# Precompile critical modules for production deployment
import py_compile
import compileall

# Add to deployment script
compileall.compile_dir('src/crewai_email_triage', force=True, quiet=1)
```

## Type Safety Enhancements

### 1. Self Type (3.11+)
```python
from typing import Self
from dataclasses import dataclass

@dataclass
class EmailClassification:
    category: str
    confidence: float
    
    def with_enhanced_confidence(self, boost: float) -> Self:
        return EmailClassification(
            category=self.category,
            confidence=min(1.0, self.confidence + boost)
        )
```

### 2. Literal String Types (3.11+)
```python
from typing import LiteralString

def secure_email_query(query: LiteralString) -> list:
    # Ensures query is a literal string to prevent injection
    return execute_database_query(query)

# Usage - prevents dynamic query construction
results = secure_email_query("SELECT * FROM emails WHERE category = 'urgent'")
```

### 3. Generic Self for Builders (3.11+)
```python
from typing import Generic, TypeVar, Self

T = TypeVar('T')

class EmailProcessor(Generic[T]):
    def __init__(self, data: T) -> None:
        self.data = data
        
    def with_preprocessing(self, preprocessor) -> Self:
        self.data = preprocessor(self.data)
        return self
        
    def with_validation(self, validator) -> Self:
        if not validator(self.data):
            raise ValueError("Email validation failed")
        return self
```

## Pattern Matching Adoption

### Email Classification with Pattern Matching (3.10+)
```python
def classify_email_advanced(email_data: dict) -> str:
    match email_data:
        case {"subject": subject, "priority": "urgent"} if "URGENT" in subject.upper():
            return "high_priority"
        case {"sender": sender} if sender.endswith("@company.com"):
            return "internal"
        case {"attachments": attachments} if len(attachments) > 5:
            return "bulk_attachment"
        case {"body_length": length} if length > 10000:
            return "long_form"
        case _:
            return "standard"
```

### Error Handling with Pattern Matching
```python
def handle_email_processing_error(error: Exception) -> str:
    match error:
        case ConnectionError(msg) if "timeout" in str(msg).lower():
            return "retry_with_backoff"
        case ValueError(msg) if "invalid email format" in str(msg):
            return "skip_with_logging"
        case MemoryError():
            return "reduce_batch_size"
        case Exception() as e if hasattr(e, 'code') and e.code == 429:
            return "rate_limit_backoff"
        case _:
            return "escalate_to_human"
```

## Async Enhancements

### Task Groups (3.11+)
```python
import asyncio
from asyncio import TaskGroup

async def process_email_batch_concurrent(emails: list) -> list:
    """Process multiple emails concurrently with proper error handling."""
    results = []
    
    async with TaskGroup() as tg:
        tasks = [
            tg.create_task(process_single_email(email))
            for email in emails
        ]
    
    # All tasks completed successfully if we reach here
    return [task.result() for task in tasks]
```

### Exception Groups (3.11+)
```python
import asyncio
from asyncio import TaskGroup

async def robust_email_processing(emails: list) -> tuple[list, list]:
    """Process emails with comprehensive error collection."""
    successful_results = []
    errors = []
    
    try:
        async with TaskGroup() as tg:
            tasks = [
                tg.create_task(process_email_with_retry(email))
                for email in emails
            ]
            
    except* ConnectionError as eg:
        # Handle all connection errors together
        for error in eg.exceptions:
            errors.append(f"Connection failed: {error}")
            
    except* ValueError as eg:
        # Handle all validation errors together
        for error in eg.exceptions:
            errors.append(f"Validation failed: {error}")
    
    return successful_results, errors
```

## Performance Profiling Integration

### Built-in Profiling (3.11+)
```python
import sys
import cProfile
from contextlib import contextmanager

@contextmanager
def profile_email_processing():
    """Context manager for profiling email processing performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        # Save profile for analysis
        profiler.dump_stats('email_processing_profile.prof')
```

## Migration Checklist

### Phase 1: Compatibility Assessment
- [ ] Update pyproject.toml to require Python 3.11+
- [ ] Run existing tests with Python 3.11+
- [ ] Verify all dependencies support Python 3.11+
- [ ] Update CI/CD pipeline Python version

### Phase 2: Type Safety Migration
- [ ] Replace `typing.Self` imports with `typing_extensions.Self`
- [ ] Add `LiteralString` types for security-critical functions
- [ ] Update generic classes to use `Self` return types
- [ ] Run mypy with strictest settings

### Phase 3: Pattern Matching Adoption
- [ ] Identify complex if/elif chains for pattern matching conversion
- [ ] Update error handling to use pattern matching
- [ ] Refactor email classification logic
- [ ] Add pattern matching to configuration parsing

### Phase 4: Async Modernization
- [ ] Replace manual exception handling with ExceptionGroup
- [ ] Adopt TaskGroup for concurrent operations
- [ ] Update timeout handling patterns
- [ ] Optimize async context managers

### Phase 5: Performance Validation
- [ ] Benchmark exception handling performance
- [ ] Measure startup time improvements
- [ ] Profile memory usage changes
- [ ] Validate overall throughput improvements