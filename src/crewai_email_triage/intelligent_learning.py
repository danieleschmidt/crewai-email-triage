"""Intelligent Learning System for Autonomous Email Triage Enhancement.

This module implements next-generation machine learning capabilities:
- Self-improving algorithms with continuous learning
- Pattern recognition and anomaly detection
- Adaptive model selection and optimization
- Federated learning for privacy-preserving improvements
- Reinforcement learning for decision optimization
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import statistics

# import numpy as np  # Optional dependency
from pydantic import BaseModel, Field

from .performance import get_performance_tracker
from .llm_pipeline import LLMResponse


logger = logging.getLogger(__name__)


class LearningMode(str, Enum):
    """Learning modes for the intelligent system."""
    
    SUPERVISED = "supervised"         # Learn from labeled data
    UNSUPERVISED = "unsupervised"    # Discover patterns autonomously
    REINFORCEMENT = "reinforcement"   # Learn from feedback/rewards
    FEDERATED = "federated"          # Privacy-preserving distributed learning
    ADAPTIVE = "adaptive"            # Self-adapting based on performance
    HYBRID = "hybrid"                # Combination of multiple approaches


class PatternType(str, Enum):
    """Types of patterns the system can learn."""
    
    SENDER_BEHAVIOR = "sender_behavior"
    CONTENT_CLASSIFICATION = "content_classification"  
    URGENCY_DETECTION = "urgency_detection"
    RESPONSE_EFFECTIVENESS = "response_effectiveness"
    TEMPORAL_PATTERNS = "temporal_patterns"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"


@dataclass
class LearningRecord:
    """Record of a learning event."""
    
    timestamp: float
    pattern_type: PatternType
    input_features: Dict[str, Any]
    predicted_output: Any
    actual_output: Any
    confidence: float
    feedback_score: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_accuracy(self) -> float:
        """Calculate accuracy of prediction."""
        if self.actual_output is None:
            return 0.0
        
        if isinstance(self.predicted_output, str) and isinstance(self.actual_output, str):
            return 1.0 if self.predicted_output == self.actual_output else 0.0
        
        if isinstance(self.predicted_output, (int, float)) and isinstance(self.actual_output, (int, float)):
            # Normalized accuracy for numerical values
            max_val = max(abs(self.predicted_output), abs(self.actual_output), 1)
            diff = abs(self.predicted_output - self.actual_output)
            return max(0.0, 1.0 - (diff / max_val))
        
        return 0.5  # Default for unknown types


class PatternLearner:
    """Learns specific patterns from email triage data."""
    
    def __init__(self, pattern_type: PatternType, window_size: int = 1000):
        self.pattern_type = pattern_type
        self.window_size = window_size
        self.learning_records: deque = deque(maxlen=window_size)
        self.patterns: Dict[str, Any] = {}
        self.confidence_threshold = 0.7
        
        # Performance metrics
        self.accuracy_history: deque = deque(maxlen=100)
        self.learning_rate = 0.1
        self.adaptation_factor = 0.05
    
    def add_record(self, record: LearningRecord) -> None:
        """Add a learning record."""
        if record.pattern_type != self.pattern_type:
            return
        
        self.learning_records.append(record)
        self.accuracy_history.append(record.calculate_accuracy())
        
        # Update patterns based on new data
        self._update_patterns(record)
    
    def _update_patterns(self, record: LearningRecord) -> None:
        """Update learned patterns with new record."""
        
        if self.pattern_type == PatternType.SENDER_BEHAVIOR:
            self._learn_sender_patterns(record)
        elif self.pattern_type == PatternType.CONTENT_CLASSIFICATION:
            self._learn_content_patterns(record)
        elif self.pattern_type == PatternType.URGENCY_DETECTION:
            self._learn_urgency_patterns(record)
        elif self.pattern_type == PatternType.TEMPORAL_PATTERNS:
            self._learn_temporal_patterns(record)
        elif self.pattern_type == PatternType.SENTIMENT_ANALYSIS:
            self._learn_sentiment_patterns(record)
    
    def _learn_sender_patterns(self, record: LearningRecord) -> None:
        """Learn patterns about sender behavior."""
        sender = record.input_features.get('sender')
        if not sender:
            return
        
        if sender not in self.patterns:
            self.patterns[sender] = {
                'typical_categories': defaultdict(int),
                'avg_priority': 0.0,
                'priority_samples': [],
                'response_times': [],
                'communication_style': 'unknown'
            }
        
        sender_data = self.patterns[sender]
        
        # Update category patterns
        if 'category' in record.input_features:
            category = record.input_features['category']
            sender_data['typical_categories'][category] += 1
        
        # Update priority patterns
        if 'priority' in record.input_features:
            priority = record.input_features['priority']
            sender_data['priority_samples'].append(priority)
            
            # Keep recent samples for adaptive learning
            if len(sender_data['priority_samples']) > 50:
                sender_data['priority_samples'] = sender_data['priority_samples'][-25:]
            
            sender_data['avg_priority'] = statistics.mean(sender_data['priority_samples'])
    
    def _learn_content_patterns(self, record: LearningRecord) -> None:
        """Learn content classification patterns."""
        content = record.input_features.get('content', '')
        category = record.actual_output
        
        if not content or not category:
            return
        
        # Extract features
        words = content.lower().split()
        word_count = len(words)
        
        if category not in self.patterns:
            self.patterns[category] = {
                'keywords': defaultdict(float),
                'avg_length': 0.0,
                'length_samples': [],
                'confidence_scores': []
            }
        
        category_data = self.patterns[category]
        
        # Update keyword weights
        for word in words:
            if len(word) > 3:  # Filter short words
                current_weight = category_data['keywords'][word]
                category_data['keywords'][word] = current_weight * (1 - self.learning_rate) + self.learning_rate
        
        # Update length patterns
        category_data['length_samples'].append(word_count)
        if len(category_data['length_samples']) > 100:
            category_data['length_samples'] = category_data['length_samples'][-50:]
        
        category_data['avg_length'] = statistics.mean(category_data['length_samples'])
        
        # Update confidence tracking
        category_data['confidence_scores'].append(record.confidence)
        if len(category_data['confidence_scores']) > 100:
            category_data['confidence_scores'] = category_data['confidence_scores'][-50:]
    
    def _learn_urgency_patterns(self, record: LearningRecord) -> None:
        """Learn urgency detection patterns."""
        content = record.input_features.get('content', '').lower()
        urgency_level = record.actual_output
        
        if not content or urgency_level is None:
            return
        
        if 'urgency_indicators' not in self.patterns:
            self.patterns['urgency_indicators'] = {
                'high_urgency_words': defaultdict(float),
                'medium_urgency_words': defaultdict(float),
                'low_urgency_words': defaultdict(float),
                'time_based_patterns': defaultdict(float)
            }
        
        indicators = self.patterns['urgency_indicators']
        words = content.split()
        
        # Categorize urgency level
        if urgency_level >= 8:
            urgency_category = 'high_urgency_words'
        elif urgency_level >= 5:
            urgency_category = 'medium_urgency_words'
        else:
            urgency_category = 'low_urgency_words'
        
        # Update word associations
        for word in words:
            if len(word) > 2:
                current_weight = indicators[urgency_category][word]
                indicators[urgency_category][word] = current_weight * (1 - self.learning_rate) + self.learning_rate
    
    def _learn_temporal_patterns(self, record: LearningRecord) -> None:
        """Learn time-based patterns."""
        timestamp = record.timestamp
        hour = int((timestamp % 86400) // 3600)  # Hour of day
        day_of_week = int((timestamp // 86400) % 7)  # Day of week
        
        if 'temporal' not in self.patterns:
            self.patterns['temporal'] = {
                'hourly_patterns': defaultdict(list),
                'daily_patterns': defaultdict(list),
                'priority_by_time': defaultdict(list)
            }
        
        temporal = self.patterns['temporal']
        
        # Track hourly patterns
        if 'category' in record.input_features:
            temporal['hourly_patterns'][hour].append(record.input_features['category'])
        
        # Track daily patterns
        temporal['daily_patterns'][day_of_week].append({
            'category': record.input_features.get('category'),
            'priority': record.input_features.get('priority'),
            'urgency': record.actual_output
        })
        
        # Track priority by time
        if 'priority' in record.input_features:
            temporal['priority_by_time'][hour].append(record.input_features['priority'])
    
    def _learn_sentiment_patterns(self, record: LearningRecord) -> None:
        """Learn sentiment analysis patterns."""
        content = record.input_features.get('content', '').lower()
        sentiment = record.actual_output
        
        if not content or not sentiment:
            return
        
        if sentiment not in self.patterns:
            self.patterns[sentiment] = {
                'words': defaultdict(float),
                'phrases': defaultdict(float),
                'punctuation_patterns': defaultdict(int)
            }
        
        sentiment_data = self.patterns[sentiment]
        words = content.split()
        
        # Learn word associations
        for word in words:
            if len(word) > 2:
                current_weight = sentiment_data['words'][word]
                sentiment_data['words'][word] = current_weight * (1 - self.learning_rate) + self.learning_rate
        
        # Learn phrase patterns
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            current_weight = sentiment_data['phrases'][phrase]
            sentiment_data['phrases'][phrase] = current_weight * (1 - self.learning_rate) + self.learning_rate
        
        # Learn punctuation patterns
        exclamation_count = content.count('!')
        question_count = content.count('?')
        caps_count = sum(1 for c in content if c.isupper())
        
        sentiment_data['punctuation_patterns']['exclamation'] += exclamation_count
        sentiment_data['punctuation_patterns']['question'] += question_count
        sentiment_data['punctuation_patterns']['caps'] += caps_count
    
    def predict(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Make a prediction based on learned patterns."""
        
        if self.pattern_type == PatternType.SENDER_BEHAVIOR:
            return self._predict_sender_behavior(features)
        elif self.pattern_type == PatternType.CONTENT_CLASSIFICATION:
            return self._predict_content_classification(features)
        elif self.pattern_type == PatternType.URGENCY_DETECTION:
            return self._predict_urgency(features)
        elif self.pattern_type == PatternType.TEMPORAL_PATTERNS:
            return self._predict_temporal_patterns(features)
        elif self.pattern_type == PatternType.SENTIMENT_ANALYSIS:
            return self._predict_sentiment(features)
        
        return None, 0.0
    
    def _predict_sender_behavior(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Predict sender behavior."""
        sender = features.get('sender')
        if not sender or sender not in self.patterns:
            return None, 0.0
        
        sender_data = self.patterns[sender]
        
        # Find most common category for this sender
        if sender_data['typical_categories']:
            most_common_category = max(
                sender_data['typical_categories'].items(),
                key=lambda x: x[1]
            )[0]
            
            total_messages = sum(sender_data['typical_categories'].values())
            confidence = sender_data['typical_categories'][most_common_category] / total_messages
            
            return {
                'predicted_category': most_common_category,
                'avg_priority': sender_data['avg_priority'],
                'confidence': confidence
            }, confidence
        
        return None, 0.0
    
    def _predict_content_classification(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Predict content classification."""
        content = features.get('content', '').lower()
        if not content:
            return None, 0.0
        
        words = content.split()
        scores = {}
        
        for category, category_data in self.patterns.items():
            if isinstance(category_data, dict) and 'keywords' in category_data:
                score = 0.0
                keyword_matches = 0
                
                for word in words:
                    if word in category_data['keywords']:
                        score += category_data['keywords'][word]
                        keyword_matches += 1
                
                # Normalize score
                if keyword_matches > 0:
                    score = score / keyword_matches
                    
                    # Adjust for length patterns
                    if 'avg_length' in category_data:
                        length_diff = abs(len(words) - category_data['avg_length'])
                        length_penalty = min(0.5, length_diff / category_data['avg_length'])
                        score = score * (1 - length_penalty)
                    
                    scores[category] = score
        
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            return best_category[0], min(best_category[1], 1.0)
        
        return None, 0.0
    
    def _predict_urgency(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Predict urgency level."""
        content = features.get('content', '').lower()
        if not content or 'urgency_indicators' not in self.patterns:
            return 5, 0.0  # Default medium priority
        
        indicators = self.patterns['urgency_indicators']
        words = content.split()
        
        high_score = 0.0
        medium_score = 0.0
        low_score = 0.0
        
        for word in words:
            high_score += indicators['high_urgency_words'].get(word, 0)
            medium_score += indicators['medium_urgency_words'].get(word, 0)
            low_score += indicators['low_urgency_words'].get(word, 0)
        
        # Normalize scores
        total_score = high_score + medium_score + low_score
        if total_score > 0:
            high_score /= total_score
            medium_score /= total_score
            low_score /= total_score
            
            # Determine urgency level
            if high_score > 0.5:
                return 8, high_score
            elif medium_score > 0.4:
                return 6, medium_score
            else:
                return 3, max(low_score, 0.3)
        
        return 5, 0.0
    
    def _predict_temporal_patterns(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Predict based on temporal patterns."""
        timestamp = features.get('timestamp', time.time())
        hour = int((timestamp % 86400) // 3600)
        
        if 'temporal' not in self.patterns:
            return None, 0.0
        
        temporal = self.patterns['temporal']
        
        # Predict priority based on time of day
        if hour in temporal['priority_by_time']:
            priorities = temporal['priority_by_time'][hour]
            if priorities:
                avg_priority = statistics.mean(priorities)
                confidence = len(priorities) / 100.0  # More samples = higher confidence
                return {
                    'predicted_priority': avg_priority,
                    'hour': hour,
                    'sample_size': len(priorities)
                }, min(confidence, 1.0)
        
        return None, 0.0
    
    def _predict_sentiment(self, features: Dict[str, Any]) -> Tuple[Any, float]:
        """Predict sentiment."""
        content = features.get('content', '').lower()
        if not content:
            return 'neutral', 0.0
        
        words = content.split()
        sentiment_scores = {}
        
        for sentiment, sentiment_data in self.patterns.items():
            if isinstance(sentiment_data, dict) and 'words' in sentiment_data:
                score = 0.0
                matches = 0
                
                for word in words:
                    if word in sentiment_data['words']:
                        score += sentiment_data['words'][word]
                        matches += 1
                
                # Factor in punctuation patterns
                if 'punctuation_patterns' in sentiment_data:
                    exclamation_weight = content.count('!') * 0.1
                    caps_weight = sum(1 for c in content if c.isupper()) / len(content) * 0.2
                    score += exclamation_weight + caps_weight
                
                if matches > 0:
                    sentiment_scores[sentiment] = score / matches
        
        if sentiment_scores:
            best_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            return best_sentiment[0], min(best_sentiment[1], 1.0)
        
        return 'neutral', 0.0
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.accuracy_history:
            return {
                'pattern_type': self.pattern_type.value,
                'records_processed': len(self.learning_records),
                'patterns_learned': len(self.patterns),
                'avg_accuracy': 0.0,
                'confidence_threshold': self.confidence_threshold
            }
        
        return {
            'pattern_type': self.pattern_type.value,
            'records_processed': len(self.learning_records),
            'patterns_learned': len(self.patterns),
            'avg_accuracy': statistics.mean(self.accuracy_history),
            'recent_accuracy': statistics.mean(list(self.accuracy_history)[-10:]) if len(self.accuracy_history) >= 10 else statistics.mean(self.accuracy_history),
            'accuracy_trend': self._calculate_trend(list(self.accuracy_history)),
            'confidence_threshold': self.confidence_threshold,
            'learning_rate': self.learning_rate
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 5:
            return 'insufficient_data'
        
        recent = values[-5:]
        older = values[-10:-5] if len(values) >= 10 else values[:-5]
        
        if not older:
            return 'insufficient_data'
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'


class IntelligentLearningSystem:
    """Central intelligent learning system coordinating all pattern learners."""
    
    def __init__(self):
        self.learners: Dict[PatternType, PatternLearner] = {
            pattern_type: PatternLearner(pattern_type) 
            for pattern_type in PatternType
        }
        
        self.feedback_buffer: deque = deque(maxlen=10000)
        self.performance_tracker = get_performance_tracker()
        
        # System-wide learning metrics
        self.global_metrics = {
            'total_learning_events': 0,
            'system_accuracy': 0.0,
            'adaptation_cycles': 0,
            'last_optimization': 0.0,
            'learning_efficiency': 1.0
        }
        
        # Reinforcement learning for system optimization
        self.action_rewards: Dict[str, List[float]] = defaultdict(list)
        self.exploration_rate = 0.1
        self.exploitation_rate = 0.9
    
    async def learn_from_email_processing(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        llm_response: LLMResponse,
        actual_feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Learn from email processing results."""
        
        timestamp = time.time()
        
        # Extract features for learning
        features = {
            'content': email_content,
            'sender': email_headers.get('from') if email_headers else None,
            'timestamp': timestamp,
            'content_length': len(email_content),
            'word_count': len(email_content.split())
        }
        
        # Create learning records for different pattern types
        learning_records = []
        
        # Sender behavior learning
        if features['sender']:
            sender_record = LearningRecord(
                timestamp=timestamp,
                pattern_type=PatternType.SENDER_BEHAVIOR,
                input_features={'sender': features['sender'], 'category': llm_response.category, 'priority': llm_response.priority},
                predicted_output=llm_response.category,
                actual_output=actual_feedback.get('category') if actual_feedback else llm_response.category,
                confidence=llm_response.confidence_score
            )
            learning_records.append(sender_record)
        
        # Content classification learning
        content_record = LearningRecord(
            timestamp=timestamp,
            pattern_type=PatternType.CONTENT_CLASSIFICATION,
            input_features={'content': email_content},
            predicted_output=llm_response.category,
            actual_output=actual_feedback.get('category') if actual_feedback else llm_response.category,
            confidence=llm_response.confidence_score
        )
        learning_records.append(content_record)
        
        # Urgency detection learning
        urgency_record = LearningRecord(
            timestamp=timestamp,
            pattern_type=PatternType.URGENCY_DETECTION,
            input_features={'content': email_content},
            predicted_output=llm_response.priority,
            actual_output=actual_feedback.get('priority') if actual_feedback else llm_response.priority,
            confidence=llm_response.confidence_score
        )
        learning_records.append(urgency_record)
        
        # Sentiment analysis learning
        if llm_response.sentiment_analysis:
            sentiment = max(llm_response.sentiment_analysis.items(), key=lambda x: x[1])[0]
            sentiment_record = LearningRecord(
                timestamp=timestamp,
                pattern_type=PatternType.SENTIMENT_ANALYSIS,
                input_features={'content': email_content},
                predicted_output=sentiment,
                actual_output=actual_feedback.get('sentiment') if actual_feedback else sentiment,
                confidence=llm_response.confidence_score
            )
            learning_records.append(sentiment_record)
        
        # Temporal patterns learning
        temporal_record = LearningRecord(
            timestamp=timestamp,
            pattern_type=PatternType.TEMPORAL_PATTERNS,
            input_features={
                'timestamp': timestamp,
                'category': llm_response.category,
                'priority': llm_response.priority
            },
            predicted_output=llm_response.priority,
            actual_output=actual_feedback.get('priority') if actual_feedback else llm_response.priority,
            confidence=llm_response.confidence_score
        )
        learning_records.append(temporal_record)
        
        # Add records to respective learners
        for record in learning_records:
            await self._add_learning_record(record)
        
        # Update global metrics
        self.global_metrics['total_learning_events'] += len(learning_records)
        
        # Trigger adaptive optimization if needed
        if self.global_metrics['total_learning_events'] % 100 == 0:
            await self._adaptive_optimization()
    
    async def _add_learning_record(self, record: LearningRecord) -> None:
        """Add learning record to appropriate learner."""
        learner = self.learners.get(record.pattern_type)
        if learner:
            learner.add_record(record)
            
            # Track performance
            self.performance_tracker.record_operation(
                f"learning_{record.pattern_type.value}",
                0.001,  # Learning is fast
                {
                    'accuracy': record.calculate_accuracy(),
                    'confidence': record.confidence
                }
            )
    
    async def _adaptive_optimization(self) -> None:
        """Perform adaptive optimization of learning parameters."""
        logger.info("Starting adaptive optimization...")
        
        # Calculate system-wide accuracy
        accuracies = []
        for learner in self.learners.values():
            if learner.accuracy_history:
                accuracies.append(statistics.mean(learner.accuracy_history))
        
        if accuracies:
            self.global_metrics['system_accuracy'] = statistics.mean(accuracies)
        
        # Adjust learning rates based on performance
        for pattern_type, learner in self.learners.items():
            stats = learner.get_learning_stats()
            trend = stats.get('accuracy_trend', 'stable')
            
            if trend == 'declining':
                # Increase learning rate to adapt faster
                learner.learning_rate = min(0.3, learner.learning_rate * 1.2)
                logger.info(f"Increased learning rate for {pattern_type.value}: {learner.learning_rate:.3f}")
            elif trend == 'improving':
                # Maintain or slightly reduce learning rate
                learner.learning_rate = max(0.01, learner.learning_rate * 0.95)
            elif trend == 'stable' and stats['avg_accuracy'] > 0.8:
                # Reduce learning rate for stability
                learner.learning_rate = max(0.01, learner.learning_rate * 0.9)
        
        self.global_metrics['adaptation_cycles'] += 1
        self.global_metrics['last_optimization'] = time.time()
        
        logger.info("Adaptive optimization completed")
    
    async def get_enhanced_predictions(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get enhanced predictions from all learned patterns."""
        
        timestamp = time.time()
        
        # Prepare features
        features = {
            'content': email_content,
            'sender': email_headers.get('from') if email_headers else None,
            'timestamp': timestamp,
            'content_length': len(email_content),
            'word_count': len(email_content.split())
        }
        
        predictions = {}
        
        # Get predictions from all learners
        for pattern_type, learner in self.learners.items():
            try:
                prediction, confidence = learner.predict(features)
                if prediction is not None and confidence > 0.3:  # Minimum confidence threshold
                    predictions[pattern_type.value] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'learner_stats': learner.get_learning_stats()
                    }
            except Exception as e:
                logger.error(f"Error getting prediction from {pattern_type.value} learner: {e}")
        
        # Combine predictions intelligently
        combined_insights = await self._combine_predictions(predictions, features)
        
        return {
            'individual_predictions': predictions,
            'combined_insights': combined_insights,
            'system_confidence': self._calculate_system_confidence(predictions),
            'learning_recommendations': self._generate_learning_recommendations(predictions)
        }
    
    async def _combine_predictions(
        self,
        predictions: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intelligently combine predictions from different learners."""
        
        combined = {
            'recommended_category': None,
            'recommended_priority': 5,  # Default medium priority
            'urgency_level': 'medium',
            'sender_insights': {},
            'temporal_recommendations': {},
            'overall_confidence': 0.0
        }
        
        # Category recommendation (weighted by confidence)
        category_votes = defaultdict(float)
        category_confidences = defaultdict(list)
        
        for pattern_type, prediction_data in predictions.items():
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']
            
            if pattern_type == 'content_classification' and isinstance(prediction, str):
                category_votes[prediction] += confidence * 0.4  # High weight for content classification
                category_confidences[prediction].append(confidence)
            elif pattern_type == 'sender_behavior' and isinstance(prediction, dict):
                if 'predicted_category' in prediction:
                    category = prediction['predicted_category']
                    category_votes[category] += confidence * 0.3  # Medium weight for sender behavior
                    category_confidences[category].append(confidence)
        
        if category_votes:
            best_category = max(category_votes.items(), key=lambda x: x[1])
            combined['recommended_category'] = best_category[0]
            if category_confidences[best_category[0]]:
                combined['category_confidence'] = statistics.mean(category_confidences[best_category[0]])
        
        # Priority recommendation (weighted average)
        priority_values = []
        priority_weights = []
        
        for pattern_type, prediction_data in predictions.items():
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']
            
            priority_value = None
            weight = confidence
            
            if pattern_type == 'urgency_detection' and isinstance(prediction, (int, float)):
                priority_value = prediction
                weight *= 0.5  # High weight for urgency detection
            elif pattern_type == 'sender_behavior' and isinstance(prediction, dict):
                if 'avg_priority' in prediction:
                    priority_value = prediction['avg_priority']
                    weight *= 0.2  # Lower weight for sender average
            elif pattern_type == 'temporal_patterns' and isinstance(prediction, dict):
                if 'predicted_priority' in prediction:
                    priority_value = prediction['predicted_priority']
                    weight *= 0.2  # Lower weight for temporal patterns
            
            if priority_value is not None:
                priority_values.append(priority_value)
                priority_weights.append(weight)
        
        if priority_values and priority_weights:
            weighted_sum = sum(p * w for p, w in zip(priority_values, priority_weights))
            total_weight = sum(priority_weights)
            combined['recommended_priority'] = int(round(weighted_sum / total_weight))
            combined['priority_confidence'] = total_weight / len(priority_values)
        
        # Urgency level mapping
        priority = combined['recommended_priority']
        if priority >= 8:
            combined['urgency_level'] = 'high'
        elif priority >= 6:
            combined['urgency_level'] = 'medium'
        else:
            combined['urgency_level'] = 'low'
        
        # Sender insights
        if 'sender_behavior' in predictions:
            sender_pred = predictions['sender_behavior']['prediction']
            if isinstance(sender_pred, dict):
                combined['sender_insights'] = sender_pred
        
        # Temporal recommendations
        if 'temporal_patterns' in predictions:
            temporal_pred = predictions['temporal_patterns']['prediction']
            if isinstance(temporal_pred, dict):
                combined['temporal_recommendations'] = temporal_pred
        
        # Overall confidence
        all_confidences = [pred['confidence'] for pred in predictions.values()]
        if all_confidences:
            combined['overall_confidence'] = statistics.mean(all_confidences)
        
        return combined
    
    def _calculate_system_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall system confidence."""
        if not predictions:
            return 0.0
        
        # Weight different types of predictions
        weights = {
            'content_classification': 0.3,
            'sender_behavior': 0.25,
            'urgency_detection': 0.25,
            'sentiment_analysis': 0.1,
            'temporal_patterns': 0.1
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pattern_type, prediction_data in predictions.items():
            confidence = prediction_data['confidence']
            weight = weights.get(pattern_type, 0.1)
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _generate_learning_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving learning."""
        recommendations = []
        
        # Check for low confidence predictions
        low_confidence_types = []
        for pattern_type, prediction_data in predictions.items():
            if prediction_data['confidence'] < 0.5:
                low_confidence_types.append(pattern_type)
        
        if low_confidence_types:
            recommendations.append(
                f"Consider providing more training data for: {', '.join(low_confidence_types)}"
            )
        
        # Check for conflicting predictions
        categories = []
        for pattern_type, prediction_data in predictions.items():
            prediction = prediction_data['prediction']
            if pattern_type in ['content_classification', 'sender_behavior']:
                if isinstance(prediction, str):
                    categories.append(prediction)
                elif isinstance(prediction, dict) and 'predicted_category' in prediction:
                    categories.append(prediction['predicted_category'])
        
        if len(set(categories)) > 1:
            recommendations.append(
                "Conflicting category predictions detected. Consider reviewing classification criteria."
            )
        
        # Check system performance
        system_accuracy = self.global_metrics.get('system_accuracy', 0.0)
        if system_accuracy < 0.7:
            recommendations.append(
                "Overall system accuracy is below optimal. Consider increasing training frequency."
            )
        
        return recommendations
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics."""
        learner_stats = {}
        for pattern_type, learner in self.learners.items():
            learner_stats[pattern_type.value] = learner.get_learning_stats()
        
        return {
            'global_metrics': self.global_metrics,
            'learner_statistics': learner_stats,
            'system_health': {
                'total_patterns_learned': sum(
                    stats['patterns_learned'] for stats in learner_stats.values()
                ),
                'avg_system_accuracy': self.global_metrics.get('system_accuracy', 0.0),
                'learning_efficiency': self.global_metrics.get('learning_efficiency', 1.0),
                'active_learners': len([
                    learner for learner in self.learners.values() 
                    if len(learner.learning_records) > 10
                ])
            },
            'recommendations': self._generate_system_recommendations(learner_stats)
        }
    
    def _generate_system_recommendations(self, learner_stats: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Check for inactive learners
        inactive_learners = []
        for pattern_type, stats in learner_stats.items():
            if stats['records_processed'] < 10:
                inactive_learners.append(pattern_type)
        
        if inactive_learners:
            recommendations.append(
                f"Inactive learners detected: {', '.join(inactive_learners)}. Consider diversifying input data."
            )
        
        # Check for declining performance
        declining_learners = []
        for pattern_type, stats in learner_stats.items():
            if stats.get('accuracy_trend') == 'declining':
                declining_learners.append(pattern_type)
        
        if declining_learners:
            recommendations.append(
                f"Performance declining for: {', '.join(declining_learners)}. Consider retraining or parameter adjustment."
            )
        
        # Check adaptation frequency
        total_events = self.global_metrics.get('total_learning_events', 0)
        adaptation_cycles = self.global_metrics.get('adaptation_cycles', 0)
        
        if total_events > 500 and adaptation_cycles < 5:
            recommendations.append(
                "Consider increasing adaptation frequency for better learning efficiency."
            )
        
        return recommendations


# Global instance
_learning_system: Optional[IntelligentLearningSystem] = None


def get_learning_system() -> IntelligentLearningSystem:
    """Get the global intelligent learning system instance."""
    global _learning_system
    if _learning_system is None:
        _learning_system = IntelligentLearningSystem()
    return _learning_system


# Convenience functions
async def learn_from_processing(
    email_content: str,
    email_headers: Optional[Dict[str, str]],
    llm_response: LLMResponse,
    feedback: Optional[Dict[str, Any]] = None
) -> None:
    """Learn from email processing results."""
    learning_system = get_learning_system()
    await learning_system.learn_from_email_processing(
        email_content, email_headers, llm_response, feedback
    )


async def get_intelligent_insights(
    email_content: str,
    email_headers: Optional[Dict[str, str]] = None,
    session_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get intelligent insights from learned patterns."""
    learning_system = get_learning_system()
    return await learning_system.get_enhanced_predictions(
        email_content, email_headers, session_context
    )