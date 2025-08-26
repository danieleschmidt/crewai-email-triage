"""
Plugin Security and Validation Framework
Provides comprehensive security measures for plugin system.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .plugin_architecture import BasePlugin, PluginMetadata


@dataclass
class SecurityViolation:
    """Represents a security violation detected in a plugin."""
    severity: str  # 'low', 'medium', 'high', 'critical'
    violation_type: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of plugin validation."""
    is_valid: bool
    violations: List[SecurityViolation]
    warnings: List[str]
    score: float  # 0.0 to 1.0, where 1.0 is completely secure


class PluginSandbox:
    """Sandbox environment for safe plugin execution."""
    
    def __init__(self):
        self.restricted_modules = {
            'os', 'sys', 'subprocess', 'multiprocessing', 'threading',
            'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
            'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
            '__import__', 'eval', 'exec', 'compile', 'open'
        }
        self.restricted_attributes = {
            '__subclasses__', '__bases__', '__mro__', '__globals__',
            '__code__', '__func__', '__self__', '__dict__'
        }
        self.allowed_builtins = {
            'abs', 'all', 'any', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'int', 'isinstance', 'len', 'list',
            'map', 'max', 'min', 'range', 'round', 'str', 'sum', 'tuple',
            'type', 'zip', 'hasattr', 'getattr'
        }
    
    def is_safe_import(self, module_name: str) -> bool:
        """Check if module import is safe."""
        # Allow standard library modules that are safe
        safe_modules = {
            'json', 'datetime', 'time', 'math', 'random', 're',
            'collections', 'itertools', 'functools', 'operator',
            'logging', 'typing', 'dataclasses', 'pathlib'
        }
        
        # Allow crewai_email_triage modules
        if module_name.startswith('crewai_email_triage'):
            return True
            
        # Check against restricted modules
        if module_name in self.restricted_modules:
            return False
            
        # Allow safe standard library modules
        return module_name in safe_modules
    
    def validate_plugin_code(self, code: str, file_path: str) -> ValidationResult:
        """Validate plugin code for security issues."""
        violations = []
        warnings = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for dangerous imports
            if re.search(r'import\s+(?:os|sys|subprocess|socket)', line):
                violations.append(SecurityViolation(
                    severity='high',
                    violation_type='dangerous_import',
                    description=f"Dangerous import detected: {line.strip()}",
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()
                ))
            
            # Check for eval/exec usage
            if re.search(r'\b(eval|exec|compile)\s*\(', line):
                violations.append(SecurityViolation(
                    severity='critical',
                    violation_type='code_execution',
                    description=f"Code execution function detected: {line.strip()}",
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()
                ))
            
            # Check for file operations
            if re.search(r'\bopen\s*\(.*[\'"]w|a[\'"]', line):
                violations.append(SecurityViolation(
                    severity='medium',
                    violation_type='file_write',
                    description=f"File write operation detected: {line.strip()}",
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()
                ))
            
            # Check for network operations
            if re.search(r'(urllib|requests|http|socket)', line):
                violations.append(SecurityViolation(
                    severity='medium',
                    violation_type='network_access',
                    description=f"Network access detected: {line.strip()}",
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()
                ))
            
            # Check for __getattribute__ or other dangerous methods
            if re.search(r'__(?:getattribute|setattr|delattr|import)__', line):
                violations.append(SecurityViolation(
                    severity='high',
                    violation_type='reflection_abuse',
                    description=f"Dangerous reflection detected: {line.strip()}",
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip()
                ))
        
        # Calculate security score
        critical_count = len([v for v in violations if v.severity == 'critical'])
        high_count = len([v for v in violations if v.severity == 'high'])
        medium_count = len([v for v in violations if v.severity == 'medium'])
        
        # Score calculation: start with 1.0, subtract based on violations
        score = 1.0
        score -= critical_count * 0.4
        score -= high_count * 0.2  
        score -= medium_count * 0.1
        score = max(0.0, score)
        
        is_valid = critical_count == 0 and high_count < 3
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            score=score
        )


class PluginValidator:
    """Validates plugin safety and compliance."""
    
    def __init__(self):
        self.sandbox = PluginSandbox()
        self.logger = logging.getLogger("plugin_validator")
    
    def validate_plugin_file(self, plugin_path: str) -> ValidationResult:
        """Validate a plugin file for security compliance."""
        try:
            with open(plugin_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.sandbox.validate_plugin_code(code, plugin_path)
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                violations=[SecurityViolation(
                    severity='critical',
                    violation_type='validation_error',
                    description=f"Failed to validate plugin: {e}",
                    file_path=plugin_path
                )],
                warnings=[],
                score=0.0
            )
    
    def validate_plugin_metadata(self, metadata: PluginMetadata) -> List[str]:
        """Validate plugin metadata."""
        warnings = []
        
        # Check for suspicious names
        if re.search(r'(hack|crack|exploit|backdoor|trojan)', metadata.name.lower()):
            warnings.append(f"Suspicious plugin name: {metadata.name}")
        
        # Check version format
        if not re.match(r'^\d+\.\d+\.\d+$', metadata.version):
            warnings.append(f"Invalid version format: {metadata.version}")
        
        # Check author information
        if not metadata.author or len(metadata.author) < 3:
            warnings.append("Missing or incomplete author information")
        
        return warnings
    
    def validate_plugin_instance(self, plugin: BasePlugin) -> Tuple[bool, List[str]]:
        """Validate a plugin instance at runtime."""
        issues = []
        
        try:
            # Check if plugin follows expected interface
            if not hasattr(plugin, 'get_metadata'):
                issues.append("Plugin missing get_metadata method")
            
            if not hasattr(plugin, 'initialize'):
                issues.append("Plugin missing initialize method")
            
            if not hasattr(plugin, 'cleanup'):
                issues.append("Plugin missing cleanup method")
            
            # Validate metadata
            try:
                metadata = plugin.get_metadata()
                metadata_warnings = self.validate_plugin_metadata(metadata)
                issues.extend(metadata_warnings)
            except Exception as e:
                issues.append(f"Failed to get plugin metadata: {e}")
            
            # Check for dangerous attributes
            for attr_name in dir(plugin):
                if attr_name.startswith('_') and attr_name not in ['_logger', '_config']:
                    issues.append(f"Suspicious private attribute: {attr_name}")
            
        except Exception as e:
            issues.append(f"Plugin validation failed: {e}")
        
        return len(issues) == 0, issues


class SecurePluginRegistry:
    """Enhanced plugin registry with security features."""
    
    def __init__(self):
        from .plugin_architecture import PluginRegistry
        self.base_registry = PluginRegistry()
        self.validator = PluginValidator()
        self.security_log = []
        self.logger = logging.getLogger("secure_plugin_registry")
        self.quarantined_plugins: Set[str] = set()
    
    def register_plugin_securely(self, plugin: BasePlugin, plugin_path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Register a plugin with security validation."""
        issues = []
        
        try:
            metadata = plugin.get_metadata()
            
            # Check if plugin is quarantined
            if metadata.name in self.quarantined_plugins:
                issues.append(f"Plugin {metadata.name} is quarantined")
                return False, issues
            
            # Validate plugin file if path provided
            if plugin_path:
                validation_result = self.validator.validate_plugin_file(plugin_path)
                
                if not validation_result.is_valid:
                    critical_violations = [v for v in validation_result.violations if v.severity == 'critical']
                    if critical_violations:
                        self.quarantine_plugin(metadata.name, f"Critical security violations: {len(critical_violations)}")
                        return False, [f"Critical security violations detected in {plugin_path}"]
                
                # Log security warnings
                for violation in validation_result.violations:
                    self.logger.warning(f"Security violation in {metadata.name}: {violation.description}")
            
            # Validate plugin instance
            is_valid, instance_issues = self.validator.validate_plugin_instance(plugin)
            issues.extend(instance_issues)
            
            if not is_valid:
                self.logger.error(f"Plugin {metadata.name} failed instance validation")
                return False, issues
            
            # Register with base registry
            success = self.base_registry.register_plugin(plugin)
            
            if success:
                self.security_log.append({
                    'timestamp': self._get_timestamp(),
                    'action': 'register',
                    'plugin_name': metadata.name,
                    'success': True,
                    'issues': issues
                })
                self.logger.info(f"Securely registered plugin: {metadata.name}")
            
            return success, issues
            
        except Exception as e:
            error_msg = f"Failed to securely register plugin: {e}"
            self.logger.error(error_msg)
            return False, [error_msg]
    
    def quarantine_plugin(self, plugin_name: str, reason: str) -> None:
        """Quarantine a plugin for security reasons."""
        self.quarantined_plugins.add(plugin_name)
        self.security_log.append({
            'timestamp': self._get_timestamp(),
            'action': 'quarantine',
            'plugin_name': plugin_name,
            'reason': reason
        })
        self.logger.warning(f"Quarantined plugin {plugin_name}: {reason}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        return {
            'total_plugins': len(self.base_registry._plugins),
            'quarantined_plugins': len(self.quarantined_plugins),
            'quarantined_list': list(self.quarantined_plugins),
            'security_events': len(self.security_log),
            'recent_events': self.security_log[-10:] if self.security_log else []
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def __getattr__(self, name):
        """Delegate to base registry."""
        return getattr(self.base_registry, name)


class PluginErrorHandler:
    """Comprehensive error handling for plugins."""
    
    def __init__(self):
        self.error_log = []
        self.recovery_strategies = {}
        self.logger = logging.getLogger("plugin_error_handler")
    
    def handle_plugin_error(self, plugin_name: str, error: Exception, context: str) -> bool:
        """Handle plugin error with recovery strategies."""
        error_info = {
            'plugin_name': plugin_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': self._get_timestamp()
        }
        
        self.error_log.append(error_info)
        self.logger.error(f"Plugin {plugin_name} error in {context}: {error}")
        
        # Attempt recovery
        if plugin_name in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[plugin_name]
                return strategy(error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed for {plugin_name}: {recovery_error}")
        
        return False
    
    def register_recovery_strategy(self, plugin_name: str, strategy_func) -> None:
        """Register a recovery strategy for a plugin."""
        self.recovery_strategies[plugin_name] = strategy_func
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_log:
            return {'total_errors': 0}
        
        error_types = {}
        plugin_errors = {}
        
        for error in self.error_log:
            error_type = error['error_type']
            plugin_name = error['plugin_name']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            plugin_errors[plugin_name] = plugin_errors.get(plugin_name, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'plugin_errors': plugin_errors,
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()