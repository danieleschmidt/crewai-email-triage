"""Test enhanced ResponseAgent implementation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from crewai_email_triage.response import ResponseAgent


class TestEnhancedResponseAgent:
    """Test enhanced response generation functionality."""
    
    def test_basic_functionality_preserved(self):
        """Test that basic functionality still works."""
        agent = ResponseAgent()
        
        # Should handle None gracefully
        result = agent.run(None)
        assert result == "response:"
        
        # Should handle empty string
        result = agent.run("")
        assert result == "response:"
        
        # Should handle basic content
        result = agent.run("Hello")
        assert result.startswith("response:")
        assert len(result.replace("response:", "").strip()) > 0
    
    def test_template_based_responses(self):
        """Test customizable template-based responses."""
        # Test with custom template
        config = {
            "response": {
                "template": "Thank you for reaching out.",
                "signature": "Best regards,\nSupport Team"
            }
        }
        agent = ResponseAgent(config_dict=config)
        
        result = agent.run("Need help with account")
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip()
        
        assert "Thank you for reaching out" in response_text
        assert "Support Team" in response_text
    
    def test_context_aware_responses(self):
        """Test responses that adapt to email content."""
        agent = ResponseAgent()
        
        # Test urgent email response
        urgent_content = "URGENT: System is down, need immediate help!"
        result = agent.run(urgent_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should indicate urgency awareness
        assert any(word in response_text for word in [
            'urgent', 'immediately', 'priority', 'quickly', 'asap'
        ])
    
    def test_meeting_request_response(self):
        """Test specialized responses for meeting requests."""
        agent = ResponseAgent()
        
        meeting_content = "Can we schedule a meeting next Tuesday at 2 PM to discuss the project?"
        result = agent.run(meeting_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should acknowledge the meeting request
        assert any(word in response_text for word in [
            'meeting', 'schedule', 'calendar', 'available', 'time'
        ])
    
    def test_question_response_patterns(self):
        """Test responses that address questions appropriately."""
        agent = ResponseAgent()
        
        question_content = "How do I reset my password? I've forgotten it."
        result = agent.run(question_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should provide helpful response to questions
        assert any(word in response_text for word in [
            'help', 'assist', 'guide', 'steps', 'instructions', 'support'
        ])
    
    def test_complaint_response_handling(self):
        """Test empathetic responses to complaints or issues."""
        agent = ResponseAgent()
        
        complaint_content = "I'm very frustrated with the service. This is the third time this issue has occurred."
        result = agent.run(complaint_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should show empathy and concern
        assert any(word in response_text for word in [
            'sorry', 'apologize', 'understand', 'concern', 'resolve'
        ])
    
    def test_thank_you_acknowledgment(self):
        """Test responses to thank you messages."""
        agent = ResponseAgent()
        
        thanks_content = "Thank you so much for your help with the project. It was exactly what we needed."
        result = agent.run(thanks_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should acknowledge the thanks
        assert any(phrase in response_text for phrase in [
            'welcome', 'glad', 'happy', 'pleased', 'anytime'
        ])
    
    def test_response_tone_configuration(self):
        """Test configurable response tone."""
        # Test formal tone
        formal_config = {
            "response": {
                "tone": "formal",
                "signature": "Regards,\nCustomer Service"
            }
        }
        agent_formal = ResponseAgent(config_dict=formal_config)
        
        # Test casual tone
        casual_config = {
            "response": {
                "tone": "casual",
                "signature": "Thanks,\nThe Team"
            }
        }
        agent_casual = ResponseAgent(config_dict=casual_config)
        
        content = "I need help with my account"
        
        formal_result = agent_formal.run(content)
        casual_result = agent_casual.run(content)
        
        # Both should respond appropriately
        assert formal_result.startswith("response:")
        assert casual_result.startswith("response:")
        
        formal_text = formal_result.replace("response:", "").strip()
        casual_text = casual_result.replace("response:", "").strip()
        
        # Formal should be more structured
        assert "Regards" in formal_text
        # Casual should be more relaxed
        assert "Thanks" in casual_text
    
    def test_response_length_control(self):
        """Test configurable response length."""
        # Test brief responses
        brief_config = {"response": {"style": "brief"}}
        agent_brief = ResponseAgent(config_dict=brief_config)
        
        # Test detailed responses
        detailed_config = {"response": {"style": "detailed"}}
        agent_detailed = ResponseAgent(config_dict=detailed_config)
        
        content = "I have a question about the new features"
        
        brief_result = agent_brief.run(content)
        detailed_result = agent_detailed.run(content)
        
        assert brief_result.startswith("response:")
        assert detailed_result.startswith("response:")
        
        brief_text = brief_result.replace("response:", "").strip()
        detailed_text = detailed_result.replace("response:", "").strip()
        
        # Detailed should be longer than brief
        assert len(detailed_text) >= len(brief_text)
        
        # Brief should be concise
        assert len(brief_text.split()) <= 20  # Roughly 20 words max
    
    def test_sentiment_based_responses(self):
        """Test responses that adapt to email sentiment."""
        agent = ResponseAgent()
        
        # Positive sentiment
        positive_content = "Great job on the presentation! It was fantastic and very helpful."
        pos_result = agent.run(positive_content)
        pos_text = pos_result.replace("response:", "").strip().lower()
        
        # Should reflect positive sentiment
        assert any(word in pos_text for word in [
            'glad', 'happy', 'pleased', 'great', 'wonderful'
        ])
        
        # Negative sentiment
        negative_content = "This is terrible. Nothing works as expected and I'm very disappointed."
        neg_result = agent.run(negative_content)
        neg_text = neg_result.replace("response:", "").strip().lower()
        
        # Should address negative sentiment appropriately
        assert any(word in neg_text for word in [
            'sorry', 'apologize', 'understand', 'improve', 'resolve'
        ])
    
    def test_auto_reply_indicators(self):
        """Test handling of auto-reply and out-of-office messages."""
        agent = ResponseAgent()
        
        auto_reply_content = "This is an automated reply. I am currently out of office until Monday."
        result = agent.run(auto_reply_content)
        
        assert result.startswith("response:")
        response_text = result.replace("response:", "").strip().lower()
        
        # Should provide appropriate response to auto-replies
        assert any(phrase in response_text for phrase in [
            'noted', 'acknowledged', 'received', 'thank you', 'understood'
        ])
    
    def test_error_handling_and_edge_cases(self):
        """Test robust error handling for edge cases."""
        agent = ResponseAgent()
        
        # Test with very long content
        long_content = "This is a very long email. " * 100
        result = agent.run(long_content)
        assert result.startswith("response:")
        
        # Test with special characters
        special_content = "Hello! @#$%^&*() 日本語 🎉 How are you?"
        result = agent.run(special_content)
        assert result.startswith("response:")
        
        # Test with HTML-like content
        html_content = "<p>This <b>looks</b> like HTML content</p>"
        result = agent.run(html_content)
        assert result.startswith("response:")
        
        # Test with only whitespace
        result = agent.run("   \n\t  ")
        assert result == "response:"
    
    def test_backward_compatibility(self):
        """Test that enhanced agent maintains backward compatibility."""
        # Default config should work as expected
        agent = ResponseAgent()
        
        result = agent.run("Test message")
        assert result.startswith("response:")
        
        # Should handle existing template configuration
        config = {
            "response": {
                "template": "Thanks for your email",
                "signature": "Best regards"
            }
        }
        agent_with_config = ResponseAgent(config_dict=config)
        result = agent_with_config.run("Hello")
        
        response_text = result.replace("response:", "").strip()
        assert "Thanks for your email" in response_text
        assert "Best regards" in response_text
    
    def test_performance_requirements(self):
        """Test that enhanced response agent meets performance requirements."""
        import time
        
        agent = ResponseAgent()
        
        # Test with moderate content
        content = "I need assistance with my account settings and password recovery process."
        
        start_time = time.time()
        result = agent.run(content)
        elapsed = time.time() - start_time
        
        # Should complete quickly (< 50ms for response generation)
        assert elapsed < 0.05
        
        # Should still produce valid output
        assert result.startswith("response:")
        assert len(result.replace("response:", "").strip()) > 0


def run_enhanced_response_agent_tests():
    """Run all enhanced response agent tests."""
    print("🧪 Testing Enhanced ResponseAgent Implementation")
    print("=" * 60)
    
    test_instance = TestEnhancedResponseAgent()
    
    tests = [
        ("Basic Functionality", test_instance.test_basic_functionality_preserved),
        ("Template Responses", test_instance.test_template_based_responses),
        ("Context Awareness", test_instance.test_context_aware_responses),
        ("Meeting Requests", test_instance.test_meeting_request_response),
        ("Question Patterns", test_instance.test_question_response_patterns),
        ("Complaint Handling", test_instance.test_complaint_response_handling),
        ("Thank You Acknowledgment", test_instance.test_thank_you_acknowledgment),
        ("Tone Configuration", test_instance.test_response_tone_configuration),
        ("Length Control", test_instance.test_response_length_control),
        ("Sentiment Adaptation", test_instance.test_sentiment_based_responses),
        ("Auto-Reply Handling", test_instance.test_auto_reply_indicators),
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
        print("🎉 All enhanced response agent tests passed!")
        return True
    else:
        print("🎯 Implementation needed to pass all tests")
        return False


if __name__ == "__main__":
    success = run_enhanced_response_agent_tests()
    exit(0 if success else 1)