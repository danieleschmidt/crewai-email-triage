# 🔌 Plugin Development Guide - CrewAI Email Triage

## Overview
Comprehensive guide for developing plugins for the CrewAI Email Triage system. The plugin architecture provides a secure, scalable framework for extending email processing capabilities.

## Plugin Architecture

### Core Concepts
- **Base Plugin Classes**: Abstract interfaces for different plugin types
- **Plugin Registry**: Centralized management and discovery system  
- **Security Sandbox**: Safe execution environment with restrictions
- **Performance Monitoring**: Built-in metrics and optimization
- **Configuration Management**: JSON-based settings and priorities

### Plugin Types

#### 1. Email Processor Plugins
Process email content and extract insights:
```python
from crewai_email_triage.plugin_architecture import EmailProcessorPlugin, PluginMetadata

class SentimentAnalysisPlugin(EmailProcessorPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="sentiment_analysis",
            version="1.0.0",
            description="Analyzes email sentiment and emotional tone",
            author="Your Name"
        )
    
    def initialize(self) -> bool:
        # Initialize resources (models, connections, etc.)
        return True
    
    def process_email(self, email_content: str, metadata: dict) -> dict:
        # Your processing logic here
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "emotional_tone": "professional"
        }
    
    def cleanup(self) -> None:
        # Clean up resources
        pass
```

#### 2. CLI Command Plugins  
Add custom CLI commands:
```python
from crewai_email_triage.plugin_architecture import CLICommandPlugin

class CustomAnalysisCommand(CLICommandPlugin):
    def get_command_name(self) -> str:
        return "custom-analysis"
    
    def get_command_help(self) -> str:
        return "Run custom analysis on emails"
    
    def add_arguments(self, parser) -> None:
        parser.add_argument("--depth", type=int, default=1, help="Analysis depth")
    
    def execute_command(self, args) -> dict:
        # Command implementation
        return {"status": "completed", "results": {...}}
```

## Development Workflow

### 1. Setup Development Environment
```bash
# Clone repository
git clone <repo-url>
cd crewai-email-triage

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[test,dev]"

# Create plugins directory (if not exists)
mkdir -p plugins
```

### 2. Create Plugin Structure
```bash
# Create plugin file
touch plugins/my_plugin.py

# Update plugin configuration
vim plugin_config.json
```

### 3. Basic Plugin Template
```python
"""
My Custom Plugin
Description of what this plugin does.
"""

from typing import Any, Dict
from crewai_email_triage.plugin_architecture import EmailProcessorPlugin, PluginMetadata, PluginConfig


class MyCustomPlugin(EmailProcessorPlugin):
    """Custom plugin for specific email processing."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_custom_plugin",
            version="1.0.0",
            description="Description of plugin functionality",
            author="Your Name",
            dependencies=[],  # External dependencies if any
            min_api_version="1.0.0",
            max_api_version="2.0.0"
        )
    
    def initialize(self) -> bool:
        """Initialize plugin resources."""
        try:
            # Initialize any resources, models, connections
            self.logger.info("Initializing custom plugin")
            
            # Access plugin configuration
            settings = self.config.settings
            self.threshold = settings.get('threshold', 0.5)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {e}")
            return False
    
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process email and return results."""
        try:
            # Your processing logic here
            result = self._analyze_email(email_content, metadata)
            
            return {
                'analysis_type': 'custom_analysis',
                'result': result,
                'confidence': 0.95,
                'processing_time_ms': self._get_processing_time(),
                'metadata': {
                    'plugin_version': self.get_metadata().version,
                    'threshold_used': self.threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return {
                'error': str(e),
                'analysis_type': 'custom_analysis',
                'result': None
            }
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.logger.info("Cleaning up custom plugin")
        # Release resources, close connections, etc.
    
    def get_processing_priority(self) -> int:
        """Return processing priority (lower = higher priority)."""
        return self.config.priority
    
    def _analyze_email(self, content: str, metadata: dict) -> dict:
        """Private method for email analysis logic."""
        # Implement your analysis logic
        return {
            'word_count': len(content.split()),
            'contains_keywords': self._check_keywords(content),
            'sentiment_detected': 'neutral'
        }
    
    def _check_keywords(self, content: str) -> list:
        """Check for specific keywords."""
        keywords = ['urgent', 'important', 'deadline', 'asap']
        found = [kw for kw in keywords if kw.lower() in content.lower()]
        return found
    
    def _get_processing_time(self) -> float:
        """Get processing time in milliseconds."""
        # Implementation depends on how you track timing
        return 10.5
```

### 4. Configuration
Add plugin to `plugin_config.json`:
```json
{
  "my_custom_plugin": {
    "enabled": true,
    "settings": {
      "threshold": 0.7,
      "enable_advanced_features": true,
      "max_processing_time": 5000
    },
    "priority": 150
  }
}
```

## Best Practices

### Security Guidelines
1. **Input Validation**: Always validate input parameters
2. **Resource Limits**: Implement timeouts and resource constraints
3. **Error Handling**: Graceful error handling without exposing internals
4. **Logging**: Use structured logging for debugging

```python
def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    # Input validation
    if not email_content or not isinstance(email_content, str):
        return {'error': 'Invalid email content'}
    
    if len(email_content) > 50000:  # 50KB limit
        return {'error': 'Email content too large'}
    
    # Resource constraints
    start_time = time.time()
    try:
        result = self._process_with_timeout(email_content, timeout=5.0)
        processing_time = (time.time() - start_time) * 1000
        
        # Structured logging
        self.logger.info("Email processed", extra={
            'processing_time_ms': processing_time,
            'content_length': len(email_content),
            'result_keys': list(result.keys()) if result else None
        })
        
        return result
        
    except Exception as e:
        self.logger.error(f"Processing failed: {e}")
        return {'error': 'Processing failed', 'details': str(e)}
```

### Performance Optimization
1. **Efficient Algorithms**: Use optimal data structures and algorithms
2. **Caching**: Cache expensive computations when appropriate
3. **Resource Management**: Properly manage memory and connections
4. **Async Processing**: Consider async operations for I/O bound tasks

```python
class OptimizedPlugin(EmailProcessorPlugin):
    def initialize(self) -> bool:
        # Initialize cache for expensive operations
        self.cache = {}
        self.cache_max_size = 1000
        return True
    
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Check cache first
        cache_key = hashlib.md5(email_content.encode()).hexdigest()
        if cache_key in self.cache:
            self.logger.debug("Cache hit for email processing")
            return self.cache[cache_key]
        
        # Process email
        result = self._expensive_analysis(email_content)
        
        # Cache result (with size limit)
        if len(self.cache) < self.cache_max_size:
            self.cache[cache_key] = result
        
        return result
```

### Error Handling
```python
def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Pre-processing validation
        if not self._validate_input(email_content, metadata):
            return self._error_result("Invalid input")
        
        # Main processing
        result = self._core_processing(email_content, metadata)
        
        # Post-processing validation
        if not self._validate_output(result):
            return self._error_result("Invalid output generated")
        
        return result
        
    except ValueError as e:
        self.logger.warning(f"Validation error: {e}")
        return self._error_result(f"Validation failed: {e}")
        
    except ConnectionError as e:
        self.logger.error(f"Connection error: {e}")
        return self._error_result("Service temporarily unavailable")
        
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}", exc_info=True)
        return self._error_result("Processing failed")
    
def _error_result(self, message: str) -> Dict[str, Any]:
    """Generate standardized error result."""
    return {
        'error': True,
        'message': message,
        'plugin_name': self.get_metadata().name,
        'timestamp': time.time()
    }
```

## Testing Your Plugin

### Unit Tests
Create `tests/test_my_plugin.py`:
```python
import unittest
from unittest.mock import Mock
from plugins.my_plugin import MyCustomPlugin
from crewai_email_triage.plugin_architecture import PluginConfig


class TestMyCustomPlugin(unittest.TestCase):
    def setUp(self):
        config = PluginConfig(enabled=True, settings={'threshold': 0.5})
        self.plugin = MyCustomPlugin(config)
        self.plugin.initialize()
    
    def test_email_processing(self):
        email = "This is a test email with urgent content"
        metadata = {'from': 'test@example.com'}
        
        result = self.plugin.process_email(email, metadata)
        
        self.assertIsInstance(result, dict)
        self.assertIn('analysis_type', result)
        self.assertEqual(result['analysis_type'], 'custom_analysis')
    
    def test_keyword_detection(self):
        plugin = self.plugin
        keywords = plugin._check_keywords("This is urgent and important")
        
        self.assertIn('urgent', keywords)
        self.assertIn('important', keywords)
    
    def test_error_handling(self):
        result = self.plugin.process_email(None, {})
        self.assertIn('error', result)
    
    def tearDown(self):
        self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()
```

### Integration Testing
```bash
# Test plugin loading
python test_plugin_system.py

# Test with real email processing
python triage.py --analyze-plugins --message "Test email content"

# Performance testing
python triage.py --benchmark-plugins --iterations 100
```

## Advanced Features

### Plugin Dependencies
Handle external dependencies gracefully:
```python
def initialize(self) -> bool:
    try:
        import numpy as np
        import pandas as pd
        self.np = np
        self.pd = pd
        return True
    except ImportError as e:
        self.logger.warning(f"Optional dependency not available: {e}")
        self.fallback_mode = True
        return True  # Still initialize, but with reduced functionality
```

### Async Processing
For I/O bound operations:
```python
import asyncio
from typing import Dict, Any

class AsyncPlugin(EmailProcessorPlugin):
    async def async_process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Async processing logic
        async with aiohttp.ClientSession() as session:
            result = await self._fetch_external_analysis(session, email_content)
        return result
    
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Run async code in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create task
            task = asyncio.create_task(self.async_process_email(email_content, metadata))
            return asyncio.ensure_future(task).result()
        else:
            return loop.run_until_complete(self.async_process_email(email_content, metadata))
```

### Plugin Communication
Plugins can share data through metadata:
```python
def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    # Check if another plugin has already processed this
    if 'sentiment_analysis' in metadata.get('plugin_results', {}):
        sentiment = metadata['plugin_results']['sentiment_analysis']['sentiment']
        self.logger.info(f"Using sentiment from previous plugin: {sentiment}")
    
    result = self._my_analysis(email_content)
    
    # Share results with other plugins
    if 'plugin_results' not in metadata:
        metadata['plugin_results'] = {}
    metadata['plugin_results'][self.get_metadata().name] = result
    
    return result
```

## Deployment & Distribution

### Plugin Packaging
Create a distributable plugin package:
```bash
plugins/
├── my_plugin/
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   ├── config.json
│   └── README.md
└── my_plugin.py  # Entry point
```

### Plugin Installation
```bash
# Install plugin
python -m pip install my-crewai-plugin

# Or copy to plugins directory
cp my_plugin.py plugins/
```

### Configuration Management
```json
{
  "my_plugin": {
    "enabled": true,
    "settings": {
      "api_key": "${MY_PLUGIN_API_KEY}",
      "endpoint": "https://api.example.com",
      "timeout": 30,
      "retry_attempts": 3
    },
    "priority": 100
  }
}
```

## Troubleshooting

### Common Issues
1. **Plugin not loading**: Check plugin structure and configuration
2. **Import errors**: Verify dependencies are installed
3. **Performance issues**: Check for resource leaks or inefficient algorithms
4. **Security violations**: Review plugin security scan results

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Test plugin in isolation
python -c "
from plugins.my_plugin import MyCustomPlugin
from crewai_email_triage.plugin_architecture import PluginConfig

config = PluginConfig(enabled=True, settings={})
plugin = MyCustomPlugin(config)
print(plugin.initialize())
result = plugin.process_email('Test email', {})
print(result)
plugin.cleanup()
"
```

### Performance Profiling
```python
# Add timing to your plugin
import time
from functools import wraps

def profile_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        self.logger.info(f"{func.__name__} took {(end_time - start_time) * 1000:.2f}ms")
        return result
    return wrapper

class MyPlugin(EmailProcessorPlugin):
    @profile_method
    def process_email(self, email_content: str, metadata: dict) -> dict:
        # Your processing logic
        return result
```

## Contributing

### Plugin Submission
1. Follow coding standards and security guidelines
2. Include comprehensive tests
3. Provide documentation and examples
4. Submit for security review

### Community Plugins
- Share plugins with the community
- Follow naming conventions: `crewai-plugin-<name>`
- Include proper licensing and attribution
- Maintain backward compatibility

---

## Resources

- [Plugin Architecture Documentation](src/crewai_email_triage/plugin_architecture.py)
- [Security Framework](src/crewai_email_triage/plugin_security.py)
- [Performance Framework](src/crewai_email_triage/plugin_scaling.py)
- [Example Plugins](plugins/)
- [Testing Framework](test_plugin_system.py)

For questions or support, please refer to the main project documentation or create an issue in the repository.