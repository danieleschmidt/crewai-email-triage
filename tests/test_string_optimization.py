"""Tests for string operation optimization in agents."""

import unittest
from crewai_email_triage import ClassifierAgent, PriorityAgent


class TestClassifierStringOptimization(unittest.TestCase):
    """Test classifier agent string operation efficiency."""

    def test_classifier_basic_functionality_preserved(self):
        """Ensure optimization preserves basic classification."""
        agent = ClassifierAgent()
        assert agent.run("This is urgent") == "category: urgent"
        assert agent.run("Schedule a meeting") == "category: work"
        assert agent.run("Random content") == "category: general"
        assert agent.run(None) == "category: unknown"
        assert agent.run("") == "category: unknown"

    def test_classifier_case_insensitive_preserved(self):
        """Ensure case-insensitive matching still works."""
        agent = ClassifierAgent()
        assert agent.run("URGENT REQUEST") == "category: urgent"
        assert agent.run("Meeting Tomorrow") == "category: work"
        assert agent.run("uRgEnT cAsE") == "category: urgent"

    def test_classifier_multiple_keywords_preserved(self):
        """Ensure multiple keyword matching works correctly."""
        agent = ClassifierAgent()
        # Should match first category found in config order
        content_with_multiple_matches = "urgent meeting request"
        result = agent.run(content_with_multiple_matches)
        # Should match whichever category appears first in config iteration (urgent comes first)
        assert result == "category: urgent"


class TestPriorityStringOptimization(unittest.TestCase):
    """Test priority agent string operation efficiency."""

    def test_priority_basic_functionality_preserved(self):
        """Ensure optimization preserves basic priority scoring."""
        agent = PriorityAgent()
        assert agent.run("This is urgent") == "priority: 10"
        assert agent.run("Project deadline tomorrow") == "priority: 8"
        assert agent.run("Regular email") == "priority: 5"
        assert agent.run(None) == "priority: 0"
        assert agent.run("") == "priority: 0"

    def test_priority_uppercase_detection_preserved(self):
        """Ensure uppercase detection still works."""
        agent = PriorityAgent()
        assert agent.run("PLEASE RESPOND") == "priority: 10"
        assert agent.run("HELP ME") == "priority: 10"

    def test_priority_exclamation_detection_preserved(self):
        """Ensure exclamation mark detection still works."""
        agent = PriorityAgent()
        assert agent.run("Need this reviewed!") == "priority: 8"
        assert agent.run("Thanks!") == "priority: 8"

    def test_priority_mixed_case_keywords_preserved(self):
        """Ensure mixed case keyword matching works."""
        agent = PriorityAgent()
        assert agent.run("URGENT deadline tomorrow") == "priority: 10"
        assert agent.run("Project DEADLINE") == "priority: 8"

    def test_priority_multiple_indicators_preserved(self):
        """Ensure highest priority wins with multiple indicators."""
        agent = PriorityAgent()
        # High urgency keyword should trump medium + exclamation
        assert agent.run("urgent deadline!") == "priority: 10"
        # Uppercase should trump medium keyword
        assert agent.run("DEADLINE TOMORROW") == "priority: 10"


class TestStringOperationEfficiency(unittest.TestCase):
    """Test that string operations are performed efficiently."""

    def test_classifier_single_normalization(self):
        """Test that content is normalized once for efficiency."""
        agent = ClassifierAgent()
        # Mock test - in real implementation we'd verify normalization happens once
        # This test ensures the behavior is preserved after optimization
        content = "MIXED CaSe CoNtEnT"
        result = agent.run(content)
        assert result.startswith("category:")

    def test_priority_efficient_checks(self):
        """Test that priority checks are performed efficiently."""
        agent = PriorityAgent()
        # Test with content that triggers multiple checks
        content = "URGENT deadline tomorrow!"
        result = agent.run(content)
        assert result == "priority: 10"

    def test_large_content_performance(self):
        """Test performance with larger content strings."""
        agent_classifier = ClassifierAgent()
        agent_priority = PriorityAgent()
        
        # Create a large string with keywords at the end to test efficiency
        large_content = "Lorem ipsum " * 1000 + " urgent meeting"
        
        # Should still work correctly
        assert agent_classifier.run(large_content) == "category: urgent"
        assert agent_priority.run(large_content) == "priority: 10"

    def test_repeated_processing_efficiency(self):
        """Test efficiency with repeated processing of similar content."""
        agent_classifier = ClassifierAgent()
        agent_priority = PriorityAgent()
        
        test_cases = [
            "urgent matter",
            "URGENT MATTER", 
            "Urgent Matter",
            "uRgEnT mAtTeR"
        ]
        
        for content in test_cases:
            assert agent_classifier.run(content) == "category: urgent"
            assert agent_priority.run(content) == "priority: 10"