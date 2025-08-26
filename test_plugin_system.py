#!/usr/bin/env python3
"""
Simple test script for plugin system functionality.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crewai_email_triage.plugin_architecture import get_plugin_manager


def test_plugin_loading():
    """Test plugin loading and basic functionality."""
    print("🔌 Testing Plugin System...")
    
    # Initialize plugin manager with config
    plugin_manager = get_plugin_manager("plugin_config.json")
    
    # Get plugin status
    status = plugin_manager.get_plugin_status()
    print(f"✅ Loaded {status['total_plugins']} plugins")
    print(f"📊 Enabled: {status['enabled_plugins']}")
    
    # List plugins
    print("\n📋 Plugin Details:")
    for plugin_info in status['plugins']:
        print(f"  - {plugin_info['name']} v{plugin_info['version']} ({plugin_info['type']})")
    
    # Test email processing with plugins
    test_email = "I'm really excited about this urgent project deadline tomorrow!"
    print(f"\n📧 Testing email analysis...")
    print(f"Email: '{test_email}'")
    
    result = plugin_manager.process_email_with_plugins(test_email, {"from": "test@example.com"})
    
    print("\n🧠 Analysis Results:")
    for plugin_name, analysis in result["enhancements"].items():
        print(f"\n{plugin_name}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    
    # Test CLI commands
    cli_commands = plugin_manager.get_cli_commands()
    print(f"\n🖥️  Available CLI Commands: {len(cli_commands)}")
    for command_name, plugin in cli_commands.items():
        print(f"  --{command_name}: {plugin.get_command_help()}")
    
    print("\n✅ Plugin system test completed successfully!")


if __name__ == "__main__":
    test_plugin_loading()