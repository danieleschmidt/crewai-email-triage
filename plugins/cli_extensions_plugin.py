"""
CLI Extensions Plugin
Adds additional CLI commands for plugin management and analysis.
"""

import json
from typing import Any, Dict

from crewai_email_triage.plugin_architecture import CLICommandPlugin, PluginMetadata, PluginConfig, get_plugin_manager


class PluginStatusCommand(CLICommandPlugin):
    """CLI command to show plugin status."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="plugin_status_command",
            version="1.0.0",
            description="CLI command to show plugin status and information",
            author="CrewAI Team"
        )
    
    def initialize(self) -> bool:
        """Initialize the command."""
        self.logger.info("Initializing plugin status command")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def get_command_name(self) -> str:
        """Return command name."""
        return "plugin-status"
    
    def get_command_help(self) -> str:
        """Return command help text."""
        return "Show status and information about loaded plugins"
    
    def add_arguments(self, parser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--detailed", 
            action="store_true", 
            help="Show detailed plugin information"
        )
        parser.add_argument(
            "--plugin-name",
            help="Show information for specific plugin"
        )
    
    def execute_command(self, args) -> Any:
        """Execute the command."""
        plugin_manager = get_plugin_manager()
        status = plugin_manager.get_plugin_status()
        
        if args.plugin_name:
            # Show specific plugin info
            plugin = plugin_manager.registry.get_plugin(args.plugin_name)
            if not plugin:
                return {"error": f"Plugin '{args.plugin_name}' not found"}
            
            metadata = plugin.get_metadata()
            return {
                "plugin": {
                    "name": metadata.name,
                    "version": metadata.version,
                    "description": metadata.description,
                    "author": metadata.author,
                    "dependencies": metadata.dependencies,
                    "enabled": plugin.config.enabled,
                    "type": type(plugin).__name__,
                    "config": plugin.config.dict() if hasattr(plugin.config, 'dict') else str(plugin.config)
                }
            }
        
        if args.detailed:
            # Show detailed status
            detailed_status = status.copy()
            detailed_plugins = []
            
            for plugin_info in status["plugins"]:
                plugin = plugin_manager.registry.get_plugin(plugin_info["name"])
                if plugin:
                    metadata = plugin.get_metadata()
                    detailed_info = plugin_info.copy()
                    detailed_info.update({
                        "description": metadata.description,
                        "author": metadata.author,
                        "dependencies": metadata.dependencies,
                        "api_compatibility": f"{metadata.min_api_version} - {metadata.max_api_version}"
                    })
                    detailed_plugins.append(detailed_info)
            
            detailed_status["plugins"] = detailed_plugins
            return detailed_status
        
        return status


class AnalyzeWithPluginsCommand(CLICommandPlugin):
    """CLI command to analyze email with all available plugins."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="analyze_with_plugins_command",
            version="1.0.0",
            description="Analyze email content using all available plugins",
            author="CrewAI Team"
        )
    
    def initialize(self) -> bool:
        """Initialize the command."""
        self.logger.info("Initializing analyze with plugins command")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def get_command_name(self) -> str:
        """Return command name."""
        return "analyze-plugins"
    
    def get_command_help(self) -> str:
        """Return command help text."""
        return "Analyze email content using all available plugins for enhanced insights"
    
    def add_arguments(self, parser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--message",
            help="Email message to analyze"
        )
        parser.add_argument(
            "--file",
            help="File containing email message to analyze"
        )
        parser.add_argument(
            "--metadata",
            help="Additional metadata as JSON string"
        )
    
    def execute_command(self, args) -> Any:
        """Execute the command."""
        # Get email content
        if args.message:
            email_content = args.message
        elif args.file:
            try:
                with open(args.file, 'r') as f:
                    email_content = f.read()
            except Exception as e:
                return {"error": f"Failed to read file: {e}"}
        else:
            return {"error": "Either --message or --file must be provided"}
        
        # Parse metadata
        metadata = {}
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except Exception as e:
                return {"error": f"Invalid metadata JSON: {e}"}
        
        # Process with plugins
        plugin_manager = get_plugin_manager()
        try:
            result = plugin_manager.process_email_with_plugins(email_content, metadata)
            
            # Add summary
            result["analysis_summary"] = {
                "total_plugins": len(result["enhancements"]),
                "successful_analyses": len([e for e in result["enhancements"].values() if "error" not in e]),
                "failed_analyses": len([e for e in result["enhancements"].values() if "error" in e])
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}