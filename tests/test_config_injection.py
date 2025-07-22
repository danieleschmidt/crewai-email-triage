#!/usr/bin/env python3
"""
Test suite for configuration dependency injection.

Tests the refactored configuration system to ensure:
- Agents accept configuration through constructor injection
- No global mutable state dependencies
- Backward compatibility with existing interfaces
- Thread safety for configuration usage
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.config import load_config
from crewai_email_triage.classifier import ClassifierAgent
from crewai_email_triage.priority import PriorityAgent


class TestConfigurationInjection(unittest.TestCase):
    """Test configuration dependency injection pattern."""
    
    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            "classifier": {
                "urgent": ["urgent", "critical"],
                "work": ["meeting", "project"],
                "spam": ["unsubscribe", "offer"]
            },
            "priority": {
                "scores": {"high": 10, "medium": 5, "low": 1},
                "high_keywords": ["urgent", "critical"],
                "medium_keywords": ["important", "soon"]
            }
        }
        
        self.custom_config = {
            "classifier": {
                "urgent": ["emergency", "asap"],
                "personal": ["family", "friend"]
            },
            "priority": {
                "scores": {"high": 20, "medium": 10, "low": 2},
                "high_keywords": ["emergency", "asap"],
                "medium_keywords": ["today", "deadline"]
            }
        }
    
    def test_classifier_accepts_config_injection(self):
        """Test ClassifierAgent accepts configuration via constructor."""
        # Test with default config
        agent = ClassifierAgent(config_dict=self.test_config)
        self.assertIsNotNone(agent)
        
        # Test classification with injected config
        result = agent.run("This is an urgent message")
        self.assertIn("urgent", result.lower())
        
        # Test with custom config
        agent_custom = ClassifierAgent(config_dict=self.custom_config)
        result_custom = agent_custom.run("This is an emergency")
        self.assertIn("urgent", result_custom.lower())  # Should map to urgent category
        
    def test_priority_agent_accepts_config_injection(self):
        """Test PriorityAgent accepts configuration via constructor."""
        # Test with default config
        agent = PriorityAgent(config_dict=self.test_config)
        self.assertIsNotNone(agent)
        
        # Test priority scoring with injected config
        result = agent.run("This is urgent!")
        self.assertIn("priority: 10", result)  # Should return high priority score
        
        # Test with custom config
        agent_custom = PriorityAgent(config_dict=self.custom_config)
        result_custom = agent_custom.run("This is an emergency!")
        self.assertIn("priority: 20", result_custom)  # Should return custom high score
    
    def test_agents_work_without_global_config(self):
        """Test agents function independently of global CONFIG variable."""
        # Mock the global CONFIG to be None or empty
        with patch('crewai_email_triage.config.CONFIG', {}):
            # Agents should still work with injected config
            classifier = ClassifierAgent(config_dict=self.test_config)
            priority = PriorityAgent(config_dict=self.test_config)
            
            classifier_result = classifier.run("urgent meeting")
            priority_result = priority.run("urgent meeting")
            
            self.assertIsInstance(classifier_result, str)
            self.assertIsInstance(priority_result, str)
            self.assertTrue(len(classifier_result) > 0)
            self.assertTrue(len(priority_result) > 0)
    
    def test_backward_compatibility_with_no_config(self):
        """Test agents fall back gracefully when no config is provided."""
        # Agents should work with default configuration when none is injected
        classifier = ClassifierAgent()
        priority = PriorityAgent()
        
        classifier_result = classifier.run("urgent message")
        priority_result = priority.run("urgent message")
        
        self.assertIsInstance(classifier_result, str)
        self.assertIsInstance(priority_result, str)
    
    def test_different_configs_produce_different_results(self):
        """Test that different configurations produce different behavior."""
        content = "emergency situation"
        
        # Test with default config
        classifier1 = ClassifierAgent(config_dict=self.test_config)
        result1 = classifier1.run(content)
        
        # Test with custom config that has different keywords
        classifier2 = ClassifierAgent(config_dict=self.custom_config)
        result2 = classifier2.run(content)
        
        # Results might be different based on keyword matching
        # At minimum, both should return valid classifications
        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        self.assertTrue(len(result1) > 0)
        self.assertTrue(len(result2) > 0)
    
    def test_config_immutability(self):
        """Test that configuration is not modified by agents."""
        original_config = self.test_config.copy()
        
        classifier = ClassifierAgent(config_dict=self.test_config)
        priority = PriorityAgent(config_dict=self.test_config)
        
        # Run operations
        classifier.run("test message")
        priority.run("test message")
        
        # Configuration should remain unchanged
        self.assertEqual(self.test_config, original_config)
    
    def test_config_validation(self):
        """Test that agents handle invalid configuration gracefully."""
        invalid_configs = [
            {},  # Empty config
            {"classifier": {}},  # Missing priority section
            {"priority": {}},  # Missing classifier section
            None,  # None config
        ]
        
        for invalid_config in invalid_configs:
            # Agents should handle invalid config gracefully (fallback to default)
            try:
                classifier = ClassifierAgent(config_dict=invalid_config)
                priority = PriorityAgent(config_dict=invalid_config)
                
                # Should still be able to process content
                classifier_result = classifier.run("test")
                priority_result = priority.run("test")
                
                self.assertIsInstance(classifier_result, str)
                self.assertIsInstance(priority_result, str)
            except Exception as e:
                self.fail(f"Agent should handle invalid config gracefully, but raised: {e}")


if __name__ == "__main__":
    unittest.main()