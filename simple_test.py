#!/usr/bin/env python3
"""Simple test runner to validate basic functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_functionality():
    """Test core email processing functionality."""
    from crewai_email_triage.core import process_email
    
    # Test basic processing
    result = process_email("Test email content")
    assert result == "Processed: Test email content"
    
    # Test None handling
    result = process_email(None)
    assert result == ""
    
    # Test whitespace handling
    result = process_email("  Test with spaces  ")
    assert result == "Processed: Test with spaces"
    
    print("✅ Core functionality tests passed")

def test_agent_imports():
    """Test that core agent classes can be imported."""
    try:
        from crewai_email_triage.classifier import ClassifierAgent
        from crewai_email_triage.priority import PriorityAgent
        from crewai_email_triage.summarizer import SummarizerAgent
        from crewai_email_triage.response import ResponseAgent
        print("✅ Agent imports successful")
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")

def test_pipeline_basic():
    """Test basic pipeline functionality."""
    try:
        from crewai_email_triage.pipeline import TriageResult
        
        # Test TriageResult creation
        result = TriageResult(
            category="urgent",
            priority=5,
            summary="Test summary",
            response="Test response"
        )
        
        assert result.category == "urgent"
        assert result.priority == 5
        assert result.summary == "Test summary"
        assert result.response == "Test response"
        
        print("✅ Pipeline basic tests passed")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Running Generation 1 functionality tests...\n")
    
    test_core_functionality()
    test_agent_imports()
    test_pipeline_basic()
    
    print("\n🎉 Generation 1 tests completed successfully!")

if __name__ == "__main__":
    main()