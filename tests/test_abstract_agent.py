"""Tests for the abstract Agent base class implementation."""

import pytest
from crewai_email_triage import Agent, LegacyAgent, ClassifierAgent


def test_agent_is_abstract():
    """Test that the Agent class cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class Agent"):
        Agent()


def test_concrete_agents_inherit_from_agent():
    """Test that all concrete agents properly inherit from the abstract Agent."""
    classifier = ClassifierAgent()
    assert isinstance(classifier, Agent)
    assert hasattr(classifier, 'run')
    

def test_legacy_agent_provides_backward_compatibility():
    """Test that LegacyAgent maintains backward compatibility."""
    legacy = LegacyAgent()
    assert isinstance(legacy, Agent)
    
    # Test that it behaves like the old Agent class
    result = legacy.run("Hello")
    assert result == "Processed: Hello"
    
    result = legacy.run(None)
    assert result == ""


def test_concrete_agent_must_implement_run():
    """Test that concrete agents must implement the run method."""
    
    class IncompleteAgent(Agent):
        pass  # Missing run method implementation
    
    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteAgent"):
        IncompleteAgent()


def test_concrete_agent_with_run_method_works():
    """Test that concrete agents with run method can be instantiated."""
    
    class ValidAgent(Agent):
        def run(self, content: str | None) -> str:
            return f"processed: {content or 'empty'}"
    
    agent = ValidAgent()
    assert agent.run("test") == "processed: test"
    assert agent.run(None) == "processed: empty"


def test_agent_interface_contract():
    """Test that the Agent interface contract is properly enforced."""
    # All existing agents should follow the contract
    agents = [ClassifierAgent()]
    
    for agent in agents:
        assert hasattr(agent, 'run')
        assert callable(getattr(agent, 'run'))
        
        # Test with valid content
        result = agent.run("test email content")
        assert isinstance(result, str)
        
        # Test with None
        result = agent.run(None)
        assert isinstance(result, str)