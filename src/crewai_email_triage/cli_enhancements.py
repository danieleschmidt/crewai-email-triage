"""Enhanced CLI functionality with advanced AI features."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Dict, List, Optional, Any

from .ai_enhancements import intelligent_triage_email, IntelligentTriageResult
from .logging_utils import get_logger
from .metrics_export import get_metrics_collector

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


class AdvancedCLIProcessor:
    """Advanced CLI processor with AI-enhanced capabilities."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced CLI processor."""
        self.config = config or {}
        logger.info("AdvancedCLIProcessor initialized")
    
    async def process_intelligent_triage(
        self,
        content: str,
        headers: Optional[Dict] = None,
        output_format: str = 'json',
        show_insights: bool = True
    ) -> str:
        """Process email with intelligent AI triage."""
        
        start_time = time.perf_counter()
        
        try:
            # Perform intelligent triage
            result = await intelligent_triage_email(content, headers, self.config)
            
            # Format output based on requested format
            if output_format == 'json':
                return self._format_json_output(result, show_insights)
            elif output_format == 'detailed':
                return self._format_detailed_output(result)
            elif output_format == 'executive':
                return self._format_executive_summary(result)
            elif output_format == 'actions':
                return self._format_action_list(result)
            else:
                return self._format_standard_output(result)
        
        except Exception as e:
            logger.error("Intelligent triage processing failed", extra={'error': str(e)})
            return json.dumps({
                "error": "Processing failed",
                "message": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }, indent=2)
    
    def _format_json_output(self, result: IntelligentTriageResult, show_insights: bool) -> str:
        """Format result as JSON with optional insights."""
        
        output_data = {
            "triage_result": {
                "category": result.category,
                "priority": result.priority,
                "summary": result.summary,
                "response": result.response,
                "confidence_score": result.confidence_score,
            },
            "metadata": {
                "processing_time_ms": result.processing_time_ms,
                "timestamp": result.timestamp,
            }
        }
        
        if show_insights:
            output_data["ai_insights"] = {
                "context": result.context.to_dict(),
                "escalation_recommendation": result.escalation_recommendation,
                "suggested_actions": result.suggested_actions,
                "related_tickets": result.related_tickets,
            }
        
        return json.dumps(output_data, indent=2)
    
    def _format_detailed_output(self, result: IntelligentTriageResult) -> str:
        """Format result as detailed human-readable report."""
        
        lines = [
            "=" * 80,
            "📧 INTELLIGENT EMAIL TRIAGE REPORT",
            "=" * 80,
            "",
            f"🏷️  CLASSIFICATION: {result.category.upper()}",
            f"⚡ PRIORITY: {result.priority}/10",
            f"🎯 CONFIDENCE: {result.confidence_score:.1%}",
            f"⏱️  PROCESSING TIME: {result.processing_time_ms:.2f}ms",
            "",
            "📝 SUMMARY:",
            f"   {result.summary}",
            "",
            "💬 SUGGESTED RESPONSE:",
            f"   {result.response}",
            "",
        ]
        
        # AI Context Section
        if result.context:
            lines.extend([
                "🧠 AI ANALYSIS:",
                f"   • Sentiment Score: {result.context.sentiment_score:.2f}",
                f"   • Business Hours: {'Yes' if result.context.business_hours else 'No'}",
            ])
            
            if result.context.sender_domain:
                lines.append(f"   • Sender Domain: {result.context.sender_domain}")
            
            if result.context.urgency_indicators:
                lines.append(f"   • Urgency Indicators: {', '.join(result.context.urgency_indicators)}")
            
            if result.context.topic_keywords:
                keywords = ', '.join(result.context.topic_keywords[:5])
                lines.append(f"   • Key Topics: {keywords}")
            
            if result.context.entities:
                entity_summary = {}
                for entity in result.context.entities:
                    entity_type = entity['type']
                    entity_summary[entity_type] = entity_summary.get(entity_type, 0) + 1
                
                entity_counts = [f"{count} {etype}(s)" for etype, count in entity_summary.items()]
                lines.append(f"   • Entities Found: {', '.join(entity_counts)}")
            
            lines.append("")
        
        # Escalation Recommendation
        if result.escalation_recommendation:
            lines.extend([
                "🚨 ESCALATION RECOMMENDATION:",
                f"   {result.escalation_recommendation}",
                "",
            ])
        
        # Suggested Actions
        if result.suggested_actions:
            lines.extend([
                "✅ SUGGESTED ACTIONS:",
            ])
            for i, action in enumerate(result.suggested_actions, 1):
                lines.append(f"   {i}. {action}")
            lines.append("")
        
        # Related Tickets
        if result.related_tickets:
            lines.extend([
                "🔗 RELATED TICKETS:",
                f"   {', '.join(result.related_tickets)}",
                "",
            ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_executive_summary(self, result: IntelligentTriageResult) -> str:
        """Format result as executive summary for leadership."""
        
        # Determine status indicator
        status_indicator = "🟢"  # Green
        if result.priority >= 8 or (result.context and result.context.sentiment_score < -0.5):
            status_indicator = "🔴"  # Red
        elif result.priority >= 6:
            status_indicator = "🟡"  # Yellow
        
        lines = [
            f"{status_indicator} EMAIL TRIAGE EXECUTIVE SUMMARY",
            "",
            f"Category: {result.category.title()}",
            f"Priority: {result.priority}/10",
            f"Confidence: {result.confidence_score:.0%}",
            "",
            f"Key Issue: {result.summary}",
        ]
        
        # Add critical insights
        if result.context:
            if result.context.customer_tier:
                lines.append(f"Customer Tier: {result.context.customer_tier.title()}")
            
            if result.context.sentiment_score < -0.3:
                lines.append(f"⚠️  Negative sentiment detected ({result.context.sentiment_score:.2f})")
        
        if result.escalation_recommendation:
            lines.extend([
                "",
                "🚨 ESCALATION NEEDED:",
                f"   {result.escalation_recommendation}",
            ])
        
        # Top action
        if result.suggested_actions:
            lines.extend([
                "",
                f"Primary Action: {result.suggested_actions[0]}",
            ])
        
        return "\n".join(lines)
    
    def _format_action_list(self, result: IntelligentTriageResult) -> str:
        """Format result as actionable task list."""
        
        lines = [
            "📋 ACTION ITEMS FOR EMAIL TRIAGE",
            "=" * 40,
        ]
        
        # Priority-based immediate actions
        if result.priority >= 8:
            lines.extend([
                "🚨 IMMEDIATE ACTIONS (High Priority):",
                f"   • Respond within 2 hours",
                f"   • Escalate to senior team",
                "",
            ])
        elif result.priority >= 6:
            lines.extend([
                "⚡ URGENT ACTIONS (Medium Priority):",
                f"   • Respond within 24 hours",
                "",
            ])
        
        # AI-suggested actions
        if result.suggested_actions:
            lines.extend([
                "🤖 AI-SUGGESTED ACTIONS:",
            ])
            for i, action in enumerate(result.suggested_actions, 1):
                lines.append(f"   {i}. {action}")
            lines.append("")
        
        # Context-based actions
        if result.context:
            context_actions = []
            
            if result.context.entities:
                context_actions.append("Verify and update contact information")
            
            if result.context.sentiment_score < -0.5:
                context_actions.append("Schedule customer retention call")
            
            if not result.context.business_hours:
                context_actions.append("Acknowledge out-of-hours receipt")
            
            if context_actions:
                lines.extend([
                    "📊 CONTEXT-BASED ACTIONS:",
                ])
                for i, action in enumerate(context_actions, 1):
                    lines.append(f"   {i}. {action}")
                lines.append("")
        
        # Follow-up actions
        lines.extend([
            "🔄 FOLLOW-UP ACTIONS:",
            "   • Set reminder for response tracking",
            "   • Update customer record with interaction",
            "   • Log resolution outcome",
        ])
        
        return "\n".join(lines)
    
    def _format_standard_output(self, result: IntelligentTriageResult) -> str:
        """Format result in standard triage format."""
        
        return json.dumps({
            "category": result.category,
            "priority": result.priority,
            "summary": result.summary,
            "response": result.response,
            "confidence": result.confidence_score,
            "processing_time_ms": result.processing_time_ms,
        }, indent=2)
    
    async def process_batch_intelligent(
        self,
        messages: List[str],
        headers_list: Optional[List[Dict]] = None,
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[IntelligentTriageResult]:
        """Process multiple messages with intelligent triage."""
        
        start_time = time.perf_counter()
        
        if not headers_list:
            headers_list = [None] * len(messages)
        
        try:
            if parallel:
                # Process in parallel using asyncio
                tasks = [
                    intelligent_triage_email(msg, headers, self.config)
                    for msg, headers in zip(messages, headers_list)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing failed for message {i}", 
                                   extra={'error': str(result)})
                        # Create fallback result
                        processed_results.append(IntelligentTriageResult(
                            category="error",
                            priority=0,
                            summary="Processing failed",
                            response="Unable to process message",
                        ))
                    else:
                        processed_results.append(result)
                
                results = processed_results
            else:
                # Process sequentially
                results = []
                for msg, headers in zip(messages, headers_list):
                    try:
                        result = await intelligent_triage_email(msg, headers, self.config)
                        results.append(result)
                    except Exception as e:
                        logger.error("Sequential processing failed", extra={'error': str(e)})
                        results.append(IntelligentTriageResult(
                            category="error",
                            priority=0,
                            summary="Processing failed",
                            response="Unable to process message",
                        ))
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            _metrics_collector.increment_counter("intelligent_batch_operations")
            _metrics_collector.record_histogram("intelligent_batch_time_ms", processing_time)
            _metrics_collector.set_gauge("intelligent_batch_size", len(messages))
            
            logger.info(f"Intelligent batch processing completed",
                       extra={'message_count': len(messages),
                             'processing_time_ms': processing_time,
                             'parallel': parallel})
            
            return results
        
        except Exception as e:
            logger.error("Batch intelligent processing failed", extra={'error': str(e)})
            _metrics_collector.increment_counter("intelligent_batch_errors")
            return []
    
    def format_batch_report(
        self,
        results: List[IntelligentTriageResult],
        format_type: str = 'summary'
    ) -> str:
        """Format batch processing results into a comprehensive report."""
        
        if not results:
            return "No results to report."
        
        if format_type == 'summary':
            return self._format_batch_summary(results)
        elif format_type == 'detailed':
            return self._format_batch_detailed(results)
        elif format_type == 'analytics':
            return self._format_batch_analytics(results)
        else:
            return json.dumps([result.to_dict() for result in results], indent=2)
    
    def _format_batch_summary(self, results: List[IntelligentTriageResult]) -> str:
        """Format batch results as executive summary."""
        
        total_messages = len(results)
        high_priority = len([r for r in results if r.priority >= 8])
        medium_priority = len([r for r in results if 5 <= r.priority < 8])
        low_priority = len([r for r in results if r.priority < 5])
        
        avg_confidence = sum(r.confidence_score for r in results) / total_messages if total_messages > 0 else 0
        
        escalation_needed = len([r for r in results if r.escalation_recommendation])
        
        # Category distribution
        categories = {}
        for result in results:
            categories[result.category] = categories.get(result.category, 0) + 1
        
        lines = [
            "📊 BATCH PROCESSING SUMMARY",
            "=" * 50,
            "",
            f"📧 Total Messages: {total_messages}",
            f"🔴 High Priority (8-10): {high_priority}",
            f"🟡 Medium Priority (5-7): {medium_priority}",
            f"🟢 Low Priority (0-4): {low_priority}",
            "",
            f"🎯 Average Confidence: {avg_confidence:.1%}",
            f"🚨 Escalations Needed: {escalation_needed}",
            "",
            "📂 Category Breakdown:",
        ]
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_messages) * 100
            lines.append(f"   • {category.title()}: {count} ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def _format_batch_detailed(self, results: List[IntelligentTriageResult]) -> str:
        """Format batch results with detailed information."""
        
        lines = [
            "📧 DETAILED BATCH PROCESSING REPORT",
            "=" * 60,
            "",
        ]
        
        # High priority items first
        high_priority_results = [r for r in results if r.priority >= 8]
        if high_priority_results:
            lines.extend([
                "🚨 HIGH PRIORITY ITEMS:",
                "-" * 30,
            ])
            
            for i, result in enumerate(high_priority_results, 1):
                lines.extend([
                    f"{i}. {result.category.upper()} (Priority {result.priority})",
                    f"   Summary: {result.summary[:100]}...",
                    f"   Confidence: {result.confidence_score:.1%}",
                ])
                
                if result.escalation_recommendation:
                    lines.append(f"   🚨 Escalation: {result.escalation_recommendation}")
                
                lines.append("")
        
        # Medium priority summary
        medium_priority_results = [r for r in results if 5 <= r.priority < 8]
        if medium_priority_results:
            lines.extend([
                f"⚡ MEDIUM PRIORITY ITEMS: {len(medium_priority_results)} items",
                "-" * 30,
            ])
            
            for result in medium_priority_results[:5]:  # Show top 5
                lines.append(f"• {result.category.title()}: {result.summary[:80]}...")
            
            if len(medium_priority_results) > 5:
                lines.append(f"• ... and {len(medium_priority_results) - 5} more")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_batch_analytics(self, results: List[IntelligentTriageResult]) -> str:
        """Format batch results with analytics and insights."""
        
        if not results:
            return "No data for analytics."
        
        # Calculate metrics
        total_messages = len(results)
        avg_processing_time = sum(r.processing_time_ms for r in results) / total_messages
        avg_confidence = sum(r.confidence_score for r in results) / total_messages
        
        # Sentiment analysis
        sentiment_scores = [r.context.sentiment_score for r in results if r.context]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Business hours analysis
        business_hours_count = len([r for r in results if r.context and r.context.business_hours])
        business_hours_pct = (business_hours_count / total_messages) * 100 if total_messages > 0 else 0
        
        # Priority distribution
        priority_dist = {}
        for result in results:
            priority_range = "High (8-10)" if result.priority >= 8 else "Medium (5-7)" if result.priority >= 5 else "Low (0-4)"
            priority_dist[priority_range] = priority_dist.get(priority_range, 0) + 1
        
        lines = [
            "📈 BATCH ANALYTICS REPORT",
            "=" * 50,
            "",
            "⏱️  PERFORMANCE METRICS:",
            f"   • Average Processing Time: {avg_processing_time:.2f}ms",
            f"   • Average Confidence Score: {avg_confidence:.1%}",
            "",
            "📊 CONTENT ANALYSIS:",
            f"   • Average Sentiment: {avg_sentiment:.2f}",
            f"   • Business Hours Messages: {business_hours_pct:.1f}%",
            "",
            "🎯 PRIORITY DISTRIBUTION:",
        ]
        
        for priority_range, count in priority_dist.items():
            percentage = (count / total_messages) * 100
            lines.append(f"   • {priority_range}: {count} ({percentage:.1f}%)")
        
        # Recommendations
        lines.extend([
            "",
            "💡 RECOMMENDATIONS:",
        ])
        
        if priority_dist.get("High (8-10)", 0) > total_messages * 0.2:
            lines.append("   • High volume of urgent messages - consider staffing adjustment")
        
        if avg_sentiment < -0.3:
            lines.append("   • Negative sentiment trend - review customer satisfaction")
        
        if business_hours_pct < 60:
            lines.append("   • Consider extended support hours for better coverage")
        
        if avg_confidence < 0.7:
            lines.append("   • Low confidence scores - review AI model tuning")
        
        return "\n".join(lines)


def run_async_cli_function(func, *args, **kwargs):
    """Helper function to run async CLI functions in sync context."""
    
    try:
        # Get or create event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, func(*args, **kwargs))
                return future.result()
        else:
            # If no loop is running, use asyncio.run
            return loop.run_until_complete(func(*args, **kwargs))
    except RuntimeError:
        # Fallback to asyncio.run for new event loop
        return asyncio.run(func(*args, **kwargs))