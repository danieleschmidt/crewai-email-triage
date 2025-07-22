"""Test enhanced SummarizerAgent implementation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from crewai_email_triage.summarizer import SummarizerAgent


class TestEnhancedSummarizer:
    """Test enhanced summarization functionality."""
    
    def test_basic_functionality_preserved(self):
        """Test that basic functionality still works."""
        agent = SummarizerAgent()
        
        # Should handle None gracefully
        result = agent.run(None)
        assert result == "summary:"
        
        # Should handle empty string
        result = agent.run("")
        assert result == "summary:"
        
        # Should handle basic content
        result = agent.run("Hello world.")
        assert result.startswith("summary:")
        assert "Hello world" in result
    
    def test_single_sentence_summarization(self):
        """Test summarization of single sentence content."""
        agent = SummarizerAgent()
        
        content = "This is a simple sentence about a meeting."
        result = agent.run(content)
        
        assert result.startswith("summary:")
        # Should return the sentence since it's already short
        assert "meeting" in result
        assert len(result.replace("summary:", "").strip()) > 0
    
    def test_multi_sentence_summarization(self):
        """Test summarization of multiple sentences."""
        agent = SummarizerAgent()
        
        content = """
        The quarterly meeting will be held next Tuesday at 2 PM in the main conference room.
        We will be discussing the new project timeline and budget allocations.
        Please bring your laptops and the latest progress reports.
        Lunch will be provided for all attendees.
        """
        
        result = agent.run(content)
        
        assert result.startswith("summary:")
        summary_text = result.replace("summary:", "").strip()
        
        # Should be shorter than original
        assert len(summary_text) < len(content)
        
        # Should contain key information
        assert any(word in summary_text.lower() for word in ['meeting', 'tuesday', 'project'])
    
    def test_max_length_configuration(self):
        """Test configurable maximum summary length."""
        # Test with custom max_length
        config = {"summarizer": {"max_length": 50}}
        agent = SummarizerAgent(config_dict=config)
        
        long_content = "This is a very long piece of content that should be truncated because it exceeds the maximum length limit that we have configured for the summarizer component."
        
        result = agent.run(long_content)
        
        assert result.startswith("summary:")
        summary_text = result.replace("summary:", "").strip()
        
        # Should respect max_length (50 chars)
        assert len(summary_text) <= 50
        
        # Should include ellipsis if truncated
        if len(summary_text) == 50:
            assert summary_text.endswith("...")
    
    def test_key_sentence_extraction(self):
        """Test extraction of key sentences from longer content."""
        agent = SummarizerAgent()
        
        content = """
        Hello, I hope this email finds you well.
        I am writing to inform you about the urgent deadline for the Q4 project submission.
        The deadline has been moved up to next Friday due to client requirements.
        Please ensure all deliverables are completed by then.
        Let me know if you have any questions or concerns.
        Best regards, John.
        """
        
        result = agent.run(content)
        
        assert result.startswith("summary:")
        summary_text = result.replace("summary:", "").strip()
        
        # Should contain the most important information
        assert any(word in summary_text.lower() for word in ['urgent', 'deadline', 'friday'])
        
        # Should be significantly shorter than original
        assert len(summary_text) < len(content) * 0.6  # At least 40% reduction
    
    def test_email_structure_awareness(self):
        """Test summarization that's aware of email structure."""
        agent = SummarizerAgent()
        
        content = """
        Subject: Urgent: System Maintenance Window
        
        Hi Team,
        
        This is to notify you that we have scheduled an urgent system maintenance window for tonight from 11 PM to 3 AM EST.
        During this time, the production systems will be unavailable.
        
        Please plan accordingly and ensure all critical processes are completed before 11 PM.
        
        If you have any concerns, please reach out immediately.
        
        Thanks,
        IT Operations
        """
        
        result = agent.run(content)
        
        assert result.startswith("summary:")
        summary_text = result.replace("summary:", "").strip()
        
        # Should extract key information about maintenance
        assert any(word in summary_text.lower() for word in ['maintenance', 'tonight', '11', 'pm', 'systems'])
        
        # Should not include greeting/closing fluff
        assert "hi team" not in summary_text.lower()
        assert "thanks" not in summary_text.lower() or "it operations" not in summary_text.lower()
    
    def test_summarization_strategy_config(self):
        """Test different summarization strategies via configuration."""
        # Test extractive strategy (extract key sentences)
        extractive_config = {
            "summarizer": {
                "strategy": "extractive",
                "max_length": 100
            }
        }
        agent_extractive = SummarizerAgent(config_dict=extractive_config)
        
        # Test truncation strategy (first N chars/sentences)
        truncation_config = {
            "summarizer": {
                "strategy": "truncation",
                "max_length": 80
            }
        }
        agent_truncation = SummarizerAgent(config_dict=truncation_config)
        
        content = """
        This is important information about a critical deadline.
        We need to complete the project by Friday.
        There are many details about formatting and structure.
        The budget has been approved for additional resources.
        Please contact me if you have any questions about the implementation.
        """
        
        result_extractive = agent_extractive.run(content)
        result_truncation = agent_truncation.run(content)
        
        # Both should start with summary:
        assert result_extractive.startswith("summary:")
        assert result_truncation.startswith("summary:")
        
        # Results may differ based on strategy
        extractive_text = result_extractive.replace("summary:", "").strip()
        truncation_text = result_truncation.replace("summary:", "").strip()
        
        # Both should respect max_length
        assert len(extractive_text) <= 100
        assert len(truncation_text) <= 80
    
    def test_error_handling_and_edge_cases(self):
        """Test robust error handling for edge cases."""
        agent = SummarizerAgent()
        
        # Test with very short content
        result = agent.run("Hi.")
        assert result == "summary: Hi."
        
        # Test with whitespace-only content
        result = agent.run("   \n\t  ")
        assert result == "summary:"
        
        # Test with special characters and encoding
        result = agent.run("Café résumé naïve coöperate 日本語 🎉")
        assert result.startswith("summary:")
        
        # Test with HTML-like content (should handle gracefully)
        result = agent.run("<p>This looks like HTML content</p>")
        assert result.startswith("summary:")
        assert "HTML" in result or "content" in result
    
    def test_backward_compatibility(self):
        """Test that enhanced agent maintains backward compatibility."""
        # Default config should work like before
        agent = SummarizerAgent()
        
        # Simple content should work as expected
        result = agent.run("Simple message.")
        assert result == "summary: Simple message."
        
        # Should handle existing test cases
        result = agent.run("urgent meeting tomorrow")
        assert result.startswith("summary:")
        assert "urgent" in result or "meeting" in result
    
    def test_performance_requirements(self):
        """Test that enhanced summarizer meets performance requirements."""
        import time
        
        agent = SummarizerAgent()
        
        # Test with moderately long content
        content = " ".join([f"This is sentence number {i} in a longer document." for i in range(50)])
        
        start_time = time.time()
        result = agent.run(content)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (< 100ms for 50 sentences)
        assert elapsed < 0.1
        
        # Should still produce valid output
        assert result.startswith("summary:")
        assert len(result.replace("summary:", "").strip()) > 0


def run_enhanced_summarizer_tests():
    """Run all enhanced summarizer tests."""
    print("🧪 Testing Enhanced SummarizerAgent Implementation")
    print("=" * 60)
    
    test_instance = TestEnhancedSummarizer()
    
    tests = [
        ("Basic Functionality", test_instance.test_basic_functionality_preserved),
        ("Single Sentence", test_instance.test_single_sentence_summarization),
        ("Multi Sentence", test_instance.test_multi_sentence_summarization),
        ("Max Length Config", test_instance.test_max_length_configuration),
        ("Key Sentence Extraction", test_instance.test_key_sentence_extraction),
        ("Email Structure", test_instance.test_email_structure_awareness),
        ("Strategy Configuration", test_instance.test_summarization_strategy_config),
        ("Error Handling", test_instance.test_error_handling_and_edge_cases),
        ("Backward Compatibility", test_instance.test_backward_compatibility),
        ("Performance", test_instance.test_performance_requirements),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 {test_name}...")
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All enhanced summarizer tests passed!")
        return True
    else:
        print("🎯 Implementation needed to pass all tests")
        return False


if __name__ == "__main__":
    success = run_enhanced_summarizer_tests()
    exit(0 if success else 1)