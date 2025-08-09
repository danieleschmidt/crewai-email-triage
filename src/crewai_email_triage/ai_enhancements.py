"""AI-powered enhancements for advanced email triage capabilities."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .logging_utils import get_logger
from .metrics_export import get_metrics_collector
from .agent_responses import AgentResponse

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


@dataclass
class EmailContext:
    """Enhanced email context with AI-extracted metadata."""
    
    sender_domain: Optional[str] = None
    recipient_count: int = 0
    thread_id: Optional[str] = None
    reply_chain_length: int = 0
    
    # AI-extracted features
    sentiment_score: float = 0.0
    urgency_indicators: List[str] = field(default_factory=list)
    topic_keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    
    # Business context
    customer_tier: Optional[str] = None
    business_hours: bool = True
    time_zone: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sender_domain": self.sender_domain,
            "recipient_count": self.recipient_count,
            "thread_id": self.thread_id,
            "reply_chain_length": self.reply_chain_length,
            "sentiment_score": self.sentiment_score,
            "urgency_indicators": self.urgency_indicators,
            "topic_keywords": self.topic_keywords,
            "entities": self.entities,
            "customer_tier": self.customer_tier,
            "business_hours": self.business_hours,
            "time_zone": self.time_zone,
        }


@dataclass
class IntelligentTriageResult:
    """Enhanced triage result with AI insights."""
    
    # Standard triage fields
    category: str
    priority: int
    summary: str
    response: str
    
    # AI enhancements
    confidence_score: float = 0.0
    context: EmailContext = field(default_factory=EmailContext)
    
    # Advanced insights
    escalation_recommendation: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    related_tickets: List[str] = field(default_factory=list)
    
    # Timing and metadata
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category,
            "priority": self.priority,
            "summary": self.summary,
            "response": self.response,
            "confidence_score": self.confidence_score,
            "context": self.context.to_dict(),
            "escalation_recommendation": self.escalation_recommendation,
            "suggested_actions": self.suggested_actions,
            "related_tickets": self.related_tickets,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
        }


class AdvancedEmailAnalyzer:
    """Advanced AI-powered email analysis with context extraction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the analyzer with optional configuration."""
        self.config = config or {}
        self.sentiment_keywords = self._load_sentiment_keywords()
        self.urgency_patterns = self._load_urgency_patterns()
        
        logger.info("AdvancedEmailAnalyzer initialized")
    
    def analyze_email(self, content: str, headers: Optional[Dict] = None) -> EmailContext:
        """Perform comprehensive email analysis and extract context."""
        start_time = time.perf_counter()
        
        try:
            context = EmailContext()
            
            # Extract basic metadata
            if headers:
                context.sender_domain = self._extract_sender_domain(headers.get('from', ''))
                context.recipient_count = self._count_recipients(headers)
                context.thread_id = headers.get('message-id')
                context.reply_chain_length = self._calculate_reply_chain(headers)
            
            # AI-powered content analysis
            context.sentiment_score = self._analyze_sentiment(content)
            context.urgency_indicators = self._detect_urgency(content)
            context.topic_keywords = self._extract_keywords(content)
            context.entities = self._extract_entities(content)
            
            # Business context inference
            context.business_hours = self._is_business_hours()
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            _metrics_collector.increment_counter("email_analysis_operations")
            _metrics_collector.record_histogram("email_analysis_time_ms", processing_time)
            
            logger.debug("Email analysis completed",
                        extra={'processing_time_ms': processing_time,
                              'sentiment_score': context.sentiment_score,
                              'urgency_count': len(context.urgency_indicators),
                              'keyword_count': len(context.topic_keywords)})
            
            return context
            
        except Exception as e:
            _metrics_collector.increment_counter("email_analysis_errors")
            logger.error("Email analysis failed", extra={'error': str(e)})
            return EmailContext()
    
    def _load_sentiment_keywords(self) -> Dict[str, float]:
        """Load sentiment analysis keywords and weights."""
        return {
            # Positive sentiment
            'thank': 1.0, 'appreciate': 1.0, 'excellent': 1.0, 'great': 0.8,
            'good': 0.6, 'pleased': 0.8, 'satisfied': 0.8, 'happy': 1.0,
            
            # Negative sentiment
            'urgent': -0.8, 'problem': -0.8, 'issue': -0.6, 'error': -0.8,
            'broken': -1.0, 'frustrated': -1.0, 'angry': -1.2, 'disappointed': -0.8,
            'complaint': -1.0, 'terrible': -1.2, 'awful': -1.2, 'horrible': -1.2,
            
            # Neutral/informational
            'information': 0.0, 'question': 0.0, 'inquiry': 0.0, 'request': 0.0,
        }
    
    def _load_urgency_patterns(self) -> List[re.Pattern]:
        """Load regex patterns for urgency detection."""
        patterns = [
            r'\b(?:urgent|asap|immediately|emergency|critical)\b',
            r'\b(?:deadline|due\s+(?:today|tomorrow))\b',
            r'\b(?:high\s+priority|time[-\s]sensitive)\b',
            r'\b(?:please\s+(?:rush|expedite|hurry))\b',
            r'!{2,}',  # Multiple exclamation marks
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _extract_sender_domain(self, from_header: str) -> Optional[str]:
        """Extract domain from sender email address."""
        if not from_header:
            return None
        
        # Extract email from "Name <email@domain.com>" format
        email_match = re.search(r'<([^>]+)>', from_header)
        if email_match:
            email = email_match.group(1)
        else:
            email = from_header.strip()
        
        # Extract domain
        domain_match = re.search(r'@([a-zA-Z0-9.-]+)', email)
        return domain_match.group(1) if domain_match else None
    
    def _count_recipients(self, headers: Dict) -> int:
        """Count number of recipients from email headers."""
        recipients = 0
        
        # Count To recipients
        to_header = headers.get('to', '')
        if to_header:
            recipients += len(re.findall(r'[^,;]+', to_header))
        
        # Count CC recipients
        cc_header = headers.get('cc', '')
        if cc_header:
            recipients += len(re.findall(r'[^,;]+', cc_header))
        
        return recipients
    
    def _calculate_reply_chain(self, headers: Dict) -> int:
        """Calculate length of reply chain from headers."""
        in_reply_to = headers.get('in-reply-to')
        references = headers.get('references', '')
        
        if not in_reply_to and not references:
            return 0
        
        # Count references to estimate chain length
        if references:
            return len(references.split())
        
        return 1 if in_reply_to else 0
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of email content."""
        if not content:
            return 0.0
        
        words = re.findall(r'\b\w+\b', content.lower())
        sentiment_sum = 0.0
        word_count = 0
        
        for word in words:
            if word in self.sentiment_keywords:
                sentiment_sum += self.sentiment_keywords[word]
                word_count += 1
        
        # Normalize by word count, cap at [-1.0, 1.0]
        if word_count == 0:
            return 0.0
        
        sentiment = sentiment_sum / word_count
        return max(-1.0, min(1.0, sentiment))
    
    def _detect_urgency(self, content: str) -> List[str]:
        """Detect urgency indicators in email content."""
        indicators = []
        
        for pattern in self.urgency_patterns:
            matches = pattern.findall(content)
            indicators.extend(matches)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(indicators))
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract topic keywords from email content."""
        # Simple keyword extraction based on word frequency and length
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'way', 'come', 'could', 'each', 'like', 'make',
            'many', 'over', 'said', 'them', 'very', 'what', 'with', 'have', 'from',
            'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'will',
            'about', 'after', 'back', 'other', 'right', 'than', 'their', 'think',
            'where', 'being', 'every', 'great', 'might', 'still', 'take', 'work',
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        return sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:10]
    
    def _extract_entities(self, content: str) -> List[Dict[str, str]]:
        """Extract named entities from email content."""
        entities = []
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        for email in emails:
            entities.append({'type': 'email', 'value': email})
        
        # Phone numbers (simple pattern)
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content)
        for phone in phones:
            entities.append({'type': 'phone', 'value': phone})
        
        # URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        for url in urls:
            entities.append({'type': 'url', 'value': url})
        
        # Ticket IDs (common patterns)
        tickets = re.findall(r'\b(?:ticket|case|id|#)\s*[:\-#]?\s*([A-Z0-9]{3,})\b', content, re.IGNORECASE)
        for ticket in tickets:
            entities.append({'type': 'ticket', 'value': ticket})
        
        return entities
    
    def _is_business_hours(self) -> bool:
        """Determine if current time is within business hours."""
        import datetime
        
        now = datetime.datetime.now()
        # Simple business hours: Monday-Friday, 9 AM to 5 PM
        return (now.weekday() < 5 and 9 <= now.hour < 17)


class SmartResponseGenerator:
    """Advanced response generation with contextual awareness."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the smart response generator."""
        self.config = config or {}
        self.response_templates = self._load_response_templates()
        
        logger.info("SmartResponseGenerator initialized")
    
    def generate_contextual_response(
        self,
        content: str,
        category: str,
        priority: int,
        context: EmailContext,
        base_response: str
    ) -> Tuple[str, List[str]]:
        """Generate a contextual response with suggested actions."""
        
        start_time = time.perf_counter()
        
        try:
            # Select appropriate template based on context
            template = self._select_response_template(category, priority, context)
            
            # Generate personalized response
            personalized_response = self._personalize_response(
                template, content, context, base_response
            )
            
            # Generate action suggestions
            suggested_actions = self._generate_action_suggestions(category, priority, context)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            _metrics_collector.increment_counter("contextual_response_operations")
            _metrics_collector.record_histogram("contextual_response_time_ms", processing_time)
            
            logger.debug("Contextual response generated",
                        extra={'processing_time_ms': processing_time,
                              'template_used': template.get('name', 'default'),
                              'actions_count': len(suggested_actions)})
            
            return personalized_response, suggested_actions
            
        except Exception as e:
            _metrics_collector.increment_counter("contextual_response_errors")
            logger.error("Contextual response generation failed", extra={'error': str(e)})
            return base_response, []
    
    def _load_response_templates(self) -> Dict[str, Dict]:
        """Load response templates for different scenarios."""
        return {
            'urgent_customer': {
                'name': 'urgent_customer',
                'template': "Thank you for contacting us regarding your urgent matter. We understand the importance of resolving this quickly and have escalated your request to our priority support team.",
                'tone': 'professional',
                'escalation': True,
            },
            'complaint': {
                'name': 'complaint',
                'template': "We sincerely apologize for the inconvenience you've experienced. Your feedback is valuable to us, and we are committed to resolving this issue promptly.",
                'tone': 'apologetic',
                'escalation': True,
            },
            'inquiry': {
                'name': 'inquiry',
                'template': "Thank you for your inquiry. We're happy to provide you with the information you need.",
                'tone': 'helpful',
                'escalation': False,
            },
            'technical': {
                'name': 'technical',
                'template': "Thank you for reporting this technical issue. Our technical support team will investigate and provide a resolution.",
                'tone': 'professional',
                'escalation': False,
            },
            'billing': {
                'name': 'billing',
                'template': "Thank you for your billing inquiry. Our accounts team will review your request and respond within 24 hours.",
                'tone': 'reassuring',
                'escalation': False,
            },
        }
    
    def _select_response_template(
        self, 
        category: str, 
        priority: int, 
        context: EmailContext
    ) -> Dict:
        """Select appropriate response template based on context."""
        
        # High priority or negative sentiment gets urgent treatment
        if priority >= 8 or context.sentiment_score < -0.5:
            if 'complaint' in category.lower() or context.sentiment_score < -0.7:
                return self.response_templates.get('complaint', self.response_templates['inquiry'])
            else:
                return self.response_templates.get('urgent_customer', self.response_templates['inquiry'])
        
        # Category-based selection
        category_lower = category.lower()
        if 'technical' in category_lower or 'bug' in category_lower:
            return self.response_templates.get('technical', self.response_templates['inquiry'])
        elif 'billing' in category_lower or 'payment' in category_lower:
            return self.response_templates.get('billing', self.response_templates['inquiry'])
        elif 'complaint' in category_lower:
            return self.response_templates.get('complaint', self.response_templates['inquiry'])
        
        # Default to inquiry template
        return self.response_templates['inquiry']
    
    def _personalize_response(
        self,
        template: Dict,
        content: str,
        context: EmailContext,
        base_response: str
    ) -> str:
        """Personalize response based on context and content."""
        
        response = template['template']
        
        # Add customer tier acknowledgment
        if context.customer_tier:
            tier_acknowledgment = {
                'premium': " As a valued premium customer, your request will receive priority attention.",
                'enterprise': " As our enterprise partner, we'll ensure this receives immediate attention from our dedicated team.",
                'standard': "",
            }
            response += tier_acknowledgment.get(context.customer_tier, "")
        
        # Add urgency acknowledgment
        if context.urgency_indicators:
            response += " We recognize the urgent nature of your request."
        
        # Add business hours context
        if not context.business_hours:
            response += " While this message was received outside business hours, we'll respond as soon as possible during our next business day."
        
        # Incorporate base response if it adds value
        if base_response and base_response != "No response generated" and len(base_response) > 20:
            response += f"\n\n{base_response}"
        
        return response.strip()
    
    def _generate_action_suggestions(
        self,
        category: str,
        priority: int,
        context: EmailContext
    ) -> List[str]:
        """Generate suggested actions based on analysis."""
        
        actions = []
        
        # Priority-based actions
        if priority >= 8:
            actions.append("Escalate to senior support team")
            actions.append("Set up follow-up within 2 hours")
        elif priority >= 6:
            actions.append("Schedule follow-up within 24 hours")
        
        # Sentiment-based actions
        if context.sentiment_score < -0.5:
            actions.append("Review for customer retention risk")
            actions.append("Consider compensation or goodwill gesture")
        
        # Category-based actions
        category_lower = category.lower()
        if 'technical' in category_lower:
            actions.append("Assign to technical support team")
            actions.append("Collect system logs if needed")
        elif 'billing' in category_lower:
            actions.append("Forward to billing department")
            actions.append("Review account status")
        
        # Context-based actions
        if context.entities:
            actions.append("Extract and validate contact information")
        
        if context.related_tickets:
            actions.append("Check for related ticket history")
        
        return actions[:5]  # Limit to top 5 actions


async def intelligent_triage_email(
    content: str,
    headers: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> IntelligentTriageResult:
    """Perform intelligent email triage with AI enhancements."""
    
    start_time = time.perf_counter()
    
    try:
        # Initialize AI components
        analyzer = AdvancedEmailAnalyzer(config)
        response_generator = SmartResponseGenerator(config)
        
        # Perform standard triage first (import here to avoid circular imports)
        from .pipeline import triage_email_enhanced
        
        standard_result = triage_email_enhanced(content, config)
        
        # Perform AI analysis
        context = analyzer.analyze_email(content, headers)
        
        # Generate intelligent response
        enhanced_response, suggested_actions = response_generator.generate_contextual_response(
            content,
            standard_result.category,
            standard_result.priority,
            context,
            standard_result.response
        )
        
        # Calculate confidence score based on various factors
        confidence_score = _calculate_confidence_score(standard_result, context)
        
        # Generate escalation recommendation
        escalation_recommendation = _determine_escalation(
            standard_result.category,
            standard_result.priority,
            context
        )
        
        # Create enhanced result
        result = IntelligentTriageResult(
            category=standard_result.category,
            priority=standard_result.priority,
            summary=standard_result.summary,
            response=enhanced_response,
            confidence_score=confidence_score,
            context=context,
            escalation_recommendation=escalation_recommendation,
            suggested_actions=suggested_actions,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        
        _metrics_collector.increment_counter("intelligent_triage_operations")
        _metrics_collector.record_histogram("intelligent_triage_time_ms", result.processing_time_ms)
        
        logger.info("Intelligent triage completed",
                   extra={'category': result.category,
                         'priority': result.priority,
                         'confidence': result.confidence_score,
                         'processing_time_ms': result.processing_time_ms})
        
        return result
        
    except Exception as e:
        _metrics_collector.increment_counter("intelligent_triage_errors")
        logger.error("Intelligent triage failed", extra={'error': str(e)})
        
        # Fallback to standard result
        from .pipeline import triage_email_enhanced
        standard_result = triage_email_enhanced(content, config)
        
        return IntelligentTriageResult(
            category=standard_result.category,
            priority=standard_result.priority,
            summary=standard_result.summary,
            response=standard_result.response,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )


def _calculate_confidence_score(standard_result, context: EmailContext) -> float:
    """Calculate confidence score for the triage result."""
    
    confidence = 0.5  # Base confidence
    
    # Increase confidence based on clear indicators
    if context.urgency_indicators:
        confidence += 0.1 * min(len(context.urgency_indicators), 3)
    
    if context.topic_keywords:
        confidence += 0.1 * min(len(context.topic_keywords), 2)
    
    if abs(context.sentiment_score) > 0.3:
        confidence += 0.1
    
    if context.entities:
        confidence += 0.05 * min(len(context.entities), 4)
    
    # Decrease confidence for uncertain scenarios
    if standard_result.errors:
        confidence -= 0.1 * len(standard_result.errors)
    
    if standard_result.warnings:
        confidence -= 0.05 * len(standard_result.warnings)
    
    return max(0.0, min(1.0, confidence))


def _determine_escalation(category: str, priority: int, context: EmailContext) -> Optional[str]:
    """Determine if escalation is needed and provide recommendation."""
    
    escalation_reasons = []
    
    # High priority requires escalation
    if priority >= 9:
        escalation_reasons.append("Critical priority level")
    
    # Very negative sentiment suggests escalation
    if context.sentiment_score < -0.8:
        escalation_reasons.append("Highly negative customer sentiment")
    
    # Premium customers get escalated faster
    if context.customer_tier in ['premium', 'enterprise'] and priority >= 6:
        escalation_reasons.append(f"High-value {context.customer_tier} customer")
    
    # Multiple urgency indicators
    if len(context.urgency_indicators) >= 3:
        escalation_reasons.append("Multiple urgency indicators detected")
    
    # Outside business hours for urgent matters
    if not context.business_hours and priority >= 7:
        escalation_reasons.append("Urgent matter received outside business hours")
    
    if escalation_reasons:
        return f"Recommended escalation: {'; '.join(escalation_reasons)}"
    
    return None