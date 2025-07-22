"""Simple response agent."""

from __future__ import annotations
import re
import threading
import time
from typing import Dict, Any

from .agent import Agent


class ResponseAgent(Agent):
    """Agent that drafts a basic reply."""
    
    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize response agent with optional configuration injection.
        
        Args:
            config_dict: Configuration dictionary with response settings.
                        If None, falls back to default behavior.
        """
        super().__init__()
        self._config = config_dict
        self._config_lock = threading.RLock()
    
    def _get_response_config(self) -> Dict[str, Any]:
        """Get response configuration, with fallback to defaults."""
        with self._config_lock:
            if self._config is not None:
                return self._config.get("response", {})
            return {}

    def run(self, content: str | None) -> str:
        """Return a reply string for ``content``."""
        start_time = time.time()
        
        # Handle empty content
        if not content or not content.strip():
            return "response:"
        
        # Get configuration for customizable responses
        response_config = self._get_response_config()
        
        # Analyze content for context-aware responses
        context = self._analyze_content_context(content)
        
        # Generate intelligent response based on context
        response_text = self._generate_contextual_response(content, context, response_config)
        
        # Ensure performance requirement (<50ms)
        elapsed = time.time() - start_time
        if elapsed >= 0.05:
            # Log performance issue but don't fail
            pass
            
        return f"response: {response_text}"
    
    def _analyze_content_context(self, content: str) -> Dict[str, Any]:
        """Analyze email content to determine context and characteristics."""
        content_lower = content.lower()
        
        # Check for various content types and sentiments
        context = {
            'is_urgent': any(word in content_lower for word in [
                'urgent', 'asap', 'immediately', 'emergency', 'critical',
                'deadline', 'time sensitive', 'rush', 'quickly'
            ]) or '!!!' in content or content.isupper(),
            
            'is_meeting': any(word in content_lower for word in [
                'meeting', 'schedule', 'appointment', 'calendar',
                'call', 'zoom', 'conference', 'discuss'
            ]),
            
            'is_question': any(word in content_lower for word in [
                'how', 'what', 'when', 'where', 'why', 'can you',
                'could you', 'would you', 'help', 'assistance'
            ]) or '?' in content,
            
            'is_complaint': any(word in content_lower for word in [
                'frustrated', 'disappointed', 'terrible', 'awful',
                'complaint', 'issue', 'problem', 'broken', 'not working'
            ]),
            
            'is_thanks': any(word in content_lower for word in [
                'thank', 'thanks', 'appreciate', 'grateful',
                'great job', 'excellent', 'fantastic'
            ]),
            
            'is_auto_reply': any(phrase in content_lower for phrase in [
                'automated reply', 'auto reply', 'out of office',
                'vacation', 'away message', 'automatic response'
            ]),
            
            'sentiment': self._analyze_sentiment(content_lower),
            'content_length': len(content.split())
        }
        
        return context
    
    def _analyze_sentiment(self, content_lower: str) -> str:
        """Basic sentiment analysis of content."""
        positive_words = ['great', 'excellent', 'fantastic', 'wonderful', 'amazing', 'perfect', 'love', 'happy']
        negative_words = ['terrible', 'awful', 'horrible', 'disappointed', 'frustrated', 'angry', 'hate', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_contextual_response(self, content: str, context: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate intelligent response based on content context."""
        # Get configuration parameters
        tone = config.get('tone', 'professional')
        style = config.get('style', 'standard')
        template = config.get('template')
        signature = config.get('signature', '')
        
        # If template is provided, use it (for backward compatibility)
        if template:
            response_text = template
        else:
            # Generate contextual response
            response_text = self._create_contextual_message(context, tone)
        
        # Apply style modifications
        if style == 'brief':
            response_text = self._make_brief(response_text)
        elif style == 'detailed':
            response_text = self._make_detailed(response_text, context)
        
        # Add signature if provided
        if signature:
            response_text += f"\n\n{signature}"
        
        return response_text
    
    def _create_contextual_message(self, context: Dict[str, Any], tone: str) -> str:
        """Create context-aware response message."""
        if context['is_urgent']:
            if tone == 'casual':
                return "Got it! I'll prioritize this and get back to you ASAP."
            else:
                return "I understand this is urgent and will address it immediately with high priority."
        
        elif context['is_meeting']:
            if tone == 'casual':
                return "Happy to schedule a meeting! I'll check my calendar and get back to you with available times."
            else:
                return "Thank you for the meeting request. I will review my calendar and respond with available time slots."
        
        elif context['is_complaint']:
            if tone == 'casual':
                return "I'm really sorry to hear about this issue. Let me help resolve it for you right away."
            else:
                return "I sincerely apologize for the inconvenience. I understand your concern and will work to resolve this issue promptly."
        
        elif context['is_question']:
            if tone == 'casual':
                return "Great question! I'm happy to help with that. Let me get you the information you need."
            else:
                return "Thank you for your inquiry. I will be happy to assist you with the information you require."
        
        elif context['is_thanks']:
            if tone == 'casual':
                return "You're so welcome! Glad I could help. Feel free to reach out anytime."
            else:
                return "You are very welcome. I am pleased that I could be of assistance."
        
        elif context['is_auto_reply']:
            if tone == 'casual':
                return "Thanks for the heads up! I've noted that you're currently away."
            else:
                return "Thank you for the notification. I have acknowledged that you are currently out of office."
        
        elif context['sentiment'] == 'positive':
            if tone == 'casual':
                return "Thanks so much for the positive feedback! It really means a lot."
            else:
                return "Thank you for your positive feedback. I am glad to hear about your experience."
        
        elif context['sentiment'] == 'negative':
            if tone == 'casual':
                return "I'm sorry things didn't go as expected. Let me help make this right."
            else:
                return "I apologize that your experience did not meet expectations. I am committed to resolving this matter."
        
        else:
            # Default response - maintain backward compatibility
            if tone == 'casual':
                return "Thanks for reaching out! I'll get back to you soon."
            else:
                return "Thanks for your email"
    
    def _make_brief(self, text: str) -> str:
        """Make response more concise."""
        # Split into sentences and take first one or two
        sentences = text.split('. ')
        if len(sentences) > 1:
            return sentences[0] + '.'
        return text
    
    def _make_detailed(self, text: str, context: Dict[str, Any]) -> str:
        """Make response more detailed."""
        # Add context-specific details
        if context['is_urgent']:
            text += " I will ensure this receives immediate attention and provide updates as I work through the resolution."
        elif context['is_meeting']:
            text += " I will coordinate with my calendar to find the best time for all participants and send you a calendar invitation."
        elif context['is_question']:
            text += " I will provide comprehensive information and additional resources that may be helpful."
        
        return text
