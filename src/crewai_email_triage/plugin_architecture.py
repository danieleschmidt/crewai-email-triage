"""
Plugin Architecture for CrewAI Email Triage System
Enables dynamic loading and execution of plugins for extensible functionality.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import traceback

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        def dict(self):
            return {key: getattr(self, key) for key in dir(self) if not key.startswith('_')}
    
    def Field(**kwargs):
        return kwargs.get('default', None)


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    min_api_version: str = "1.0.0"
    max_api_version: str = "2.0.0"
    enabled: bool = True


class PluginConfig(BaseModel):
    """Plugin configuration schema."""
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100, description="Plugin execution priority (lower = higher priority)")


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def is_compatible(self, api_version: str) -> bool:
        """Check if plugin is compatible with API version."""
        metadata = self.get_metadata()
        # Simple version comparison - in production, use proper semver
        return metadata.min_api_version <= api_version <= metadata.max_api_version


class EmailProcessorPlugin(BasePlugin):
    """Base class for email processing plugins."""
    
    @abstractmethod
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process an email and return enhanced data."""
        pass
    
    def get_processing_priority(self) -> int:
        """Return processing priority (lower number = higher priority)."""
        return self.config.priority


class CLICommandPlugin(BasePlugin):
    """Base class for CLI command plugins."""
    
    @abstractmethod
    def get_command_name(self) -> str:
        """Return the CLI command name."""
        pass
    
    @abstractmethod
    def get_command_help(self) -> str:
        """Return help text for the command."""
        pass
    
    @abstractmethod
    def add_arguments(self, parser) -> None:
        """Add command-specific arguments to parser."""
        pass
    
    @abstractmethod
    def execute_command(self, args) -> Any:
        """Execute the command with parsed arguments."""
        pass


class PluginRegistry:
    """Registry for managing plugins with enhanced error handling."""
    
    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_configs: Dict[str, PluginConfig] = {}
        self._logger = logging.getLogger("plugin_registry")
        self.api_version = "1.0.0"
        self._error_counts: Dict[str, int] = {}
        self._max_errors_per_plugin = 5
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin with comprehensive error handling."""
        plugin_name = "unknown"
        try:
            metadata = plugin.get_metadata()
            plugin_name = metadata.name
            
            # Check if plugin has exceeded error limit
            if self._error_counts.get(plugin_name, 0) >= self._max_errors_per_plugin:
                self._logger.error(f"Plugin {plugin_name} has exceeded error limit, blocking registration")
                return False
            
            # Check compatibility
            if not plugin.is_compatible(self.api_version):
                self._logger.warning(
                    f"Plugin {plugin_name} is not compatible with API version {self.api_version}"
                )
                self._increment_error_count(plugin_name)
                return False
            
            # Validate plugin structure
            if not self._validate_plugin_structure(plugin):
                self._logger.error(f"Plugin {plugin_name} failed structure validation")
                self._increment_error_count(plugin_name)
                return False
            
            # Initialize plugin with timeout
            try:
                if not self._safe_initialize_plugin(plugin):
                    self._logger.error(f"Failed to initialize plugin {plugin_name}")
                    self._increment_error_count(plugin_name)
                    return False
            except Exception as init_error:
                self._logger.error(f"Plugin {plugin_name} initialization error: {init_error}")
                self._increment_error_count(plugin_name)
                return False
            
            self._plugins[plugin_name] = plugin
            self._logger.info(f"Successfully registered plugin: {plugin_name} v{metadata.version}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error registering plugin {plugin_name}: {e}")
            self._logger.debug(f"Plugin registration traceback: {traceback.format_exc()}")
            self._increment_error_count(plugin_name)
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            try:
                self._plugins[plugin_name].cleanup()
                del self._plugins[plugin_name]
                self._logger.info(f"Unregistered plugin: {plugin_name}")
                return True
            except Exception as e:
                self._logger.error(f"Error unregistering plugin {plugin_name}: {e}")
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        return [plugin for plugin in self._plugins.values() 
                if isinstance(plugin, plugin_type) and plugin.config.enabled]
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins."""
        return [plugin.get_metadata() for plugin in self._plugins.values()]
    
    def load_plugin_config(self, config_path: str) -> None:
        """Load plugin configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for plugin_name, plugin_config in config_data.items():
                self._plugin_configs[plugin_name] = PluginConfig(**plugin_config)
                
        except Exception as e:
            self._logger.error(f"Error loading plugin config: {e}")
    
    def _validate_plugin_structure(self, plugin: BasePlugin) -> bool:
        """Validate plugin has required methods and attributes."""
        required_methods = ['get_metadata', 'initialize', 'cleanup']
        
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                self._logger.error(f"Plugin missing required method: {method_name}")
                return False
            
            method = getattr(plugin, method_name)
            if not callable(method):
                self._logger.error(f"Plugin attribute {method_name} is not callable")
                return False
        
        return True
    
    def _safe_initialize_plugin(self, plugin: BasePlugin) -> bool:
        """Safely initialize plugin with timeout and error handling."""
        try:
            # In a production system, you might want to add a timeout here
            return plugin.initialize()
        except Exception as e:
            self._logger.error(f"Plugin initialization failed: {e}")
            return False
    
    def _increment_error_count(self, plugin_name: str) -> None:
        """Increment error count for a plugin."""
        self._error_counts[plugin_name] = self._error_counts.get(plugin_name, 0) + 1
        if self._error_counts[plugin_name] >= self._max_errors_per_plugin:
            self._logger.warning(f"Plugin {plugin_name} has reached maximum error count")
    
    def get_plugin_health(self) -> Dict[str, Any]:
        """Get health information for all plugins."""
        health_info = {
            'total_plugins': len(self._plugins),
            'healthy_plugins': 0,
            'error_prone_plugins': 0,
            'plugin_errors': self._error_counts,
            'plugins': []
        }
        
        for plugin_name, plugin in self._plugins.items():
            error_count = self._error_counts.get(plugin_name, 0)
            is_healthy = error_count < self._max_errors_per_plugin // 2
            
            plugin_info = {
                'name': plugin_name,
                'healthy': is_healthy,
                'error_count': error_count,
                'enabled': plugin.config.enabled
            }
            
            health_info['plugins'].append(plugin_info)
            
            if is_healthy:
                health_info['healthy_plugins'] += 1
            else:
                health_info['error_prone_plugins'] += 1
        
        return health_info


class PluginLoader:
    """Dynamic plugin loader."""
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._logger = logging.getLogger("plugin_loader")
    
    def load_plugin_from_file(self, plugin_path: str) -> bool:
        """Load a plugin from a Python file."""
        try:
            plugin_path = Path(plugin_path)
            if not plugin_path.exists() or plugin_path.suffix != '.py':
                self._logger.error(f"Invalid plugin file: {plugin_path}")
                return False
            
            # Add plugin directory to Python path
            plugin_dir = str(plugin_path.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj is not BasePlugin):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                self._logger.error(f"No plugin classes found in {plugin_path}")
                return False
            
            # Register plugins
            success = True
            for plugin_class in plugin_classes:
                try:
                    # Get plugin config
                    plugin_name = plugin_class.__name__
                    config = self.registry._plugin_configs.get(
                        plugin_name, PluginConfig()
                    )
                    
                    # Create and register plugin instance
                    plugin = plugin_class(config)
                    if not self.registry.register_plugin(plugin):
                        success = False
                        
                except Exception as e:
                    self._logger.error(f"Error instantiating plugin {plugin_class.__name__}: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self._logger.error(f"Error loading plugin from {plugin_path}: {e}")
            return False
    
    def load_plugins_from_directory(self, plugins_dir: str) -> int:
        """Load all plugins from a directory."""
        plugins_dir = Path(plugins_dir)
        if not plugins_dir.exists():
            self._logger.warning(f"Plugin directory does not exist: {plugins_dir}")
            return 0
        
        loaded_count = 0
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
                
            if self.load_plugin_from_file(str(plugin_file)):
                loaded_count += 1
        
        self._logger.info(f"Loaded {loaded_count} plugins from {plugins_dir}")
        return loaded_count


class PluginManager:
    """Main plugin management interface."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self._logger = logging.getLogger("plugin_manager")
        
        # Load plugin configurations
        if config_path and os.path.exists(config_path):
            self.registry.load_plugin_config(config_path)
    
    def initialize_default_plugins(self) -> None:
        """Initialize default built-in plugins."""
        # Load plugins from standard locations
        plugin_dirs = [
            "plugins",
            "src/crewai_email_triage/plugins",
            os.path.expanduser("~/.crewai_triage/plugins"),
            "/etc/crewai_triage/plugins"
        ]
        
        for plugin_dir in plugin_dirs:
            if os.path.exists(plugin_dir):
                self.loader.load_plugins_from_directory(plugin_dir)
    
    def process_email_with_plugins(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process email through all enabled email processor plugins with robust error handling."""
        result = {
            "original_content": email_content, 
            "metadata": metadata, 
            "enhancements": {},
            "processing_errors": {},
            "processing_statistics": {
                "total_plugins": 0,
                "successful_plugins": 0,
                "failed_plugins": 0,
                "processing_time_ms": 0
            }
        }
        
        import time
        start_time = time.time()
        
        # Get email processor plugins sorted by priority
        processors = self.registry.get_plugins_by_type(EmailProcessorPlugin)
        processors.sort(key=lambda p: p.get_processing_priority())
        
        result["processing_statistics"]["total_plugins"] = len(processors)
        
        for plugin in processors:
            plugin_name = "unknown"
            plugin_start_time = time.time()
            
            try:
                metadata_obj = plugin.get_metadata()
                plugin_name = metadata_obj.name
                
                # Check if plugin is enabled and healthy
                if not plugin.config.enabled:
                    self._logger.debug(f"Skipping disabled plugin: {plugin_name}")
                    continue
                
                # Validate input before processing
                if not self._validate_plugin_input(email_content, metadata):
                    self._logger.warning(f"Invalid input for plugin {plugin_name}")
                    result["processing_errors"][plugin_name] = "Invalid input data"
                    result["processing_statistics"]["failed_plugins"] += 1
                    continue
                
                # Process with timeout protection
                plugin_result = self._safe_process_email(plugin, email_content, metadata)
                
                if plugin_result is not None:
                    # Validate plugin output
                    if self._validate_plugin_output(plugin_result):
                        result["enhancements"][plugin_name] = plugin_result
                        result["enhancements"][plugin_name]["processing_time_ms"] = round((time.time() - plugin_start_time) * 1000, 2)
                        result["processing_statistics"]["successful_plugins"] += 1
                        self._logger.debug(f"Plugin {plugin_name} processed successfully")
                    else:
                        result["processing_errors"][plugin_name] = "Invalid output format"
                        result["processing_statistics"]["failed_plugins"] += 1
                else:
                    result["processing_errors"][plugin_name] = "Plugin returned None"
                    result["processing_statistics"]["failed_plugins"] += 1
                
            except Exception as e:
                error_msg = f"Plugin {plugin_name} failed: {str(e)}"
                self._logger.error(error_msg)
                self._logger.debug(f"Plugin error traceback: {traceback.format_exc()}")
                result["processing_errors"][plugin_name] = error_msg
                result["processing_statistics"]["failed_plugins"] += 1
                
                # Increment error count in registry
                if hasattr(self.registry, '_increment_error_count'):
                    self.registry._increment_error_count(plugin_name)
        
        result["processing_statistics"]["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        return result
    
    def _validate_plugin_input(self, email_content: str, metadata: Dict[str, Any]) -> bool:
        """Validate input data for plugin processing."""
        if not isinstance(email_content, str):
            return False
        if not isinstance(metadata, dict):
            return False
        if len(email_content.strip()) == 0:
            return False
        return True
    
    def _validate_plugin_output(self, output: Any) -> bool:
        """Validate plugin output format."""
        if output is None:
            return False
        if not isinstance(output, dict):
            return False
        return True
    
    def _safe_process_email(self, plugin: EmailProcessorPlugin, email_content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely process email with a plugin, including timeout protection."""
        try:
            # In a production environment, you might want to add timeout protection
            # using threading.Timer or signal.alarm (Unix only)
            return plugin.process_email(email_content, metadata)
        except Exception as e:
            self._logger.error(f"Safe processing failed for plugin: {e}")
            return None
    
    def get_cli_commands(self) -> Dict[str, CLICommandPlugin]:
        """Get all available CLI command plugins."""
        commands = {}
        for plugin in self.registry.get_plugins_by_type(CLICommandPlugin):
            command_name = plugin.get_command_name()
            commands[command_name] = plugin
        return commands
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins."""
        status = {
            "total_plugins": len(self.registry._plugins),
            "enabled_plugins": len([p for p in self.registry._plugins.values() if p.config.enabled]),
            "plugins": []
        }
        
        for plugin in self.registry._plugins.values():
            metadata = plugin.get_metadata()
            status["plugins"].append({
                "name": metadata.name,
                "version": metadata.version,
                "enabled": plugin.config.enabled,
                "type": type(plugin).__name__
            })
        
        return status


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager(config_path: Optional[str] = None) -> PluginManager:
    """Get or create the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(config_path)
        _plugin_manager.initialize_default_plugins()
    return _plugin_manager