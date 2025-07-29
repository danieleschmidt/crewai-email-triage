#!/usr/bin/env python3
"""
Advanced threat detection system for email triage service.
Implements ML-based anomaly detection, behavioral analysis, and automated response.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Protocol, Tuple
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib

class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatCategory(Enum):
    """Categories of threats."""
    AUTHENTICATION_ANOMALY = "authentication_anomaly"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTACK = "injection_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALWARE = "malware"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    API_ABUSE = "api_abuse"

@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    event_id: str
    source_ip: str
    user_id: Optional[str]
    event_type: str
    endpoint: str
    user_agent: str
    payload_size: int
    response_code: int
    processing_time_ms: float
    features: Dict[str, Any]

@dataclass
class ThreatDetection:
    """Detected threat information."""
    detection_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    confidence_score: float
    description: str
    affected_resources: List[str]
    indicators_of_compromise: List[str]
    recommended_actions: List[str]
    raw_events: List[SecurityEvent]

class ThreatDetector(ABC):
    """Abstract base class for threat detectors."""
    
    @abstractmethod
    async def analyze_events(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Analyze security events and return detected threats."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Return the name of this detector."""
        pass

class AnomalyDetector(ThreatDetector):
    """ML-based anomaly detection for security events."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'payload_size', 'processing_time_ms',
            'requests_per_minute', 'unique_endpoints', 'error_rate'
        ]
        
    def get_detector_name(self) -> str:
        return "ML_Anomaly_Detector"
    
    async def train_baseline(self, baseline_events: List[SecurityEvent]) -> None:
        """Train the anomaly detection model on baseline normal behavior."""
        if len(baseline_events) < 100:
            logging.warning("Insufficient baseline data for training anomaly detector")
            return
        
        # Extract features from baseline events
        features = self._extract_features(baseline_events)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train isolation forest
        self.model.fit(features_scaled)
        self.is_trained = True
        
        logging.info(f"Trained anomaly detector with {len(baseline_events)} baseline events")
    
    async def analyze_events(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Analyze events for anomalies."""
        if not self.is_trained or len(events) == 0:
            return []
        
        detections = []
        
        # Extract features
        features = self._extract_features(events)
        features_scaled = self.scaler.transform(features)
        
        # Predict anomalies
        anomaly_scores = self.model.decision_function(features_scaled)
        is_anomaly = self.model.predict(features_scaled) == -1
        
        # Process detected anomalies
        for i, (event, is_anom, score) in enumerate(zip(events, is_anomaly, anomaly_scores)):
            if is_anom:
                confidence = self._calculate_confidence(score)
                threat_level = self._determine_threat_level(confidence, event)
                
                detection = ThreatDetection(
                    detection_id=self._generate_detection_id(event),
                    timestamp=datetime.now(),
                    threat_level=threat_level,
                    threat_category=ThreatCategory.BEHAVIORAL_ANOMALY,
                    confidence_score=confidence,
                    description=f"Anomalous behavior detected from {event.source_ip}",
                    affected_resources=[event.endpoint],
                    indicators_of_compromise=[
                        f"source_ip:{event.source_ip}",
                        f"user_agent:{hashlib.md5(event.user_agent.encode()).hexdigest()[:8]}"
                    ],
                    recommended_actions=self._get_recommended_actions(threat_level),
                    raw_events=[event]
                )
                
                detections.append(detection)
        
        return detections
    
    def _extract_features(self, events: List[SecurityEvent]) -> np.ndarray:
        """Extract ML features from security events."""
        features = []
        
        # Group events by source IP for behavioral analysis
        ip_groups = {}
        for event in events:
            if event.source_ip not in ip_groups:
                ip_groups[event.source_ip] = []
            ip_groups[event.source_ip].append(event)
        
        for ip, ip_events in ip_groups.items():
            # Calculate behavioral features for this IP
            timestamps = [e.timestamp for e in ip_events]
            if len(timestamps) < 2:
                continue
                
            # Time-based features
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            requests_per_minute = len(ip_events) / max(time_span / 60, 1)
            
            # Activity patterns
            hours = [t.hour for t in timestamps]
            days = [t.weekday() for t in timestamps]
            avg_hour = np.mean(hours)
            avg_day = np.mean(days)
            
            # Request characteristics
            payload_sizes = [e.payload_size for e in ip_events]
            processing_times = [e.processing_time_ms for e in ip_events]
            response_codes = [e.response_code for e in ip_events]
            
            avg_payload_size = np.mean(payload_sizes)
            avg_processing_time = np.mean(processing_times)
            
            # Endpoint diversity
            unique_endpoints = len(set(e.endpoint for e in ip_events))
            
            # Error rate
            error_count = sum(1 for code in response_codes if code >= 400)
            error_rate = error_count / len(response_codes)
            
            feature_vector = [
                avg_hour,
                avg_day,
                avg_payload_size,
                avg_processing_time,
                requests_per_minute,
                unique_endpoints,
                error_rate
            ]
            
            features.append(feature_vector)
        
        return np.array(features) if features else np.array([]).reshape(0, len(self.feature_names))
    
    def _calculate_confidence(self, anomaly_score: float) -> float:
        """Calculate confidence score from anomaly score."""
        # Normalize anomaly score to confidence (0-1)
        # Lower scores indicate higher anomaly (higher confidence)
        confidence = max(0.0, min(1.0, (0.5 - anomaly_score) * 2))
        return confidence
    
    def _determine_threat_level(self, confidence: float, event: SecurityEvent) -> ThreatLevel:
        """Determine threat level based on confidence and event characteristics."""
        if confidence > 0.9:
            return ThreatLevel.CRITICAL
        elif confidence > 0.7:
            return ThreatLevel.HIGH
        elif confidence > 0.5:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_detection_id(self, event: SecurityEvent) -> str:
        """Generate unique detection ID."""
        content = f"{event.timestamp.isoformat()}{event.source_ip}{event.endpoint}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_recommended_actions(self, threat_level: ThreatLevel) -> List[str]:
        """Get recommended actions based on threat level."""
        base_actions = ["monitor_source_ip", "analyze_request_patterns"]
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            base_actions.extend([
                "block_source_ip",
                "notify_security_team",
                "preserve_evidence"
            ])
        
        if threat_level == ThreatLevel.CRITICAL:
            base_actions.extend([
                "activate_incident_response",
                "isolate_affected_systems"
            ])
        
        return base_actions

class InjectionAttackDetector(ThreatDetector):
    """Detector for injection attacks (SQL, NoSQL, Command injection)."""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"(\s*(union|select|insert|update|delete|drop|create|alter)\s+)",
            r"(\s*;\s*(select|insert|update|delete|drop|create|alter)\s+)",
            r"(\s*'\s*(or|and)\s+\w+\s*=\s*\w+)",
            r"(\s*'\s*(or|and)\s+\d+\s*=\s*\d+)",
            r"(\s*'\s*or\s+'1'\s*=\s*'1')",
        ]
        
        self.command_injection_patterns = [
            r"(\s*;\s*(ls|pwd|whoami|cat|grep|find|ps|netstat)\s*)",
            r"(\s*\|\s*(ls|pwd|whoami|cat|grep|find|ps|netstat)\s*)",
            r"(\s*&&\s*(ls|pwd|whoami|cat|grep|find|ps|netstat)\s*)",
            r"(\s*`[^`]*`)",
            r"(\$\([^)]*\))",
        ]
    
    def get_detector_name(self) -> str:
        return "Injection_Attack_Detector"
    
    async def analyze_events(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Analyze events for injection attacks."""
        detections = []
        
        for event in events:
            injection_indicators = self._detect_injection_patterns(event)
            
            if injection_indicators:
                confidence = self._calculate_injection_confidence(injection_indicators)
                threat_level = ThreatLevel.HIGH if confidence > 0.8 else ThreatLevel.MEDIUM
                
                detection = ThreatDetection(
                    detection_id=self._generate_detection_id(event),
                    timestamp=datetime.now(),
                    threat_level=threat_level,
                    threat_category=ThreatCategory.INJECTION_ATTACK,
                    confidence_score=confidence,
                    description=f"Potential injection attack detected from {event.source_ip}",
                    affected_resources=[event.endpoint],
                    indicators_of_compromise=[
                        f"source_ip:{event.source_ip}",
                        f"injection_patterns:{len(injection_indicators)}"
                    ] + injection_indicators,
                    recommended_actions=[
                        "block_source_ip",
                        "analyze_request_payload",
                        "check_application_logs",
                        "validate_input_sanitization"
                    ],
                    raw_events=[event]
                )
                
                detections.append(detection)
        
        return detections
    
    def _detect_injection_patterns(self, event: SecurityEvent) -> List[str]:
        """Detect injection patterns in the event."""
        import re
        
        indicators = []
        
        # Check user agent and other string fields for injection patterns
        text_fields = [
            event.user_agent,
            event.endpoint,
            str(event.features.get('query_params', '')),
            str(event.features.get('post_data', ''))
        ]
        
        for field in text_fields:
            field_lower = field.lower()
            
            # Check SQL injection patterns
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, field_lower, re.IGNORECASE):
                    indicators.append(f"sql_pattern:{pattern[:20]}...")
            
            # Check command injection patterns
            for pattern in self.command_injection_patterns:
                if re.search(pattern, field_lower, re.IGNORECASE):
                    indicators.append(f"cmd_pattern:{pattern[:20]}...")
        
        return indicators
    
    def _calculate_injection_confidence(self, indicators: List[str]) -> float:
        """Calculate confidence based on injection indicators."""
        base_confidence = min(0.9, len(indicators) * 0.3)
        return base_confidence
    
    def _generate_detection_id(self, event: SecurityEvent) -> str:
        """Generate unique detection ID."""
        content = f"injection_{event.timestamp.isoformat()}_{event.source_ip}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class RateLimitingDetector(ThreatDetector):
    """Detector for rate limiting violations and API abuse."""
    
    def __init__(self, rate_limit_threshold: int = 100):
        self.rate_limit_threshold = rate_limit_threshold  # requests per minute
        self.ip_request_history: Dict[str, List[datetime]] = {}
    
    def get_detector_name(self) -> str:
        return "Rate_Limiting_Detector"
    
    async def analyze_events(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Analyze events for rate limiting violations."""
        detections = []
        current_time = datetime.now()
        
        # Update request history
        for event in events:
            ip = event.source_ip
            if ip not in self.ip_request_history:
                self.ip_request_history[ip] = []
            
            self.ip_request_history[ip].append(event.timestamp)
        
        # Check for violations
        for ip, timestamps in self.ip_request_history.items():
            # Remove old timestamps (older than 1 minute)
            cutoff_time = current_time - timedelta(minutes=1)
            recent_timestamps = [ts for ts in timestamps if ts > cutoff_time]
            self.ip_request_history[ip] = recent_timestamps
            
            # Check if rate limit exceeded
            if len(recent_timestamps) > self.rate_limit_threshold:
                # Find events from this IP
                ip_events = [e for e in events if e.source_ip == ip]
                
                if ip_events:
                    violation_ratio = len(recent_timestamps) / self.rate_limit_threshold
                    threat_level = self._determine_threat_level_for_rate_limit(violation_ratio)
                    
                    detection = ThreatDetection(
                        detection_id=self._generate_detection_id(ip, current_time),
                        timestamp=current_time,
                        threat_level=threat_level,
                        threat_category=ThreatCategory.API_ABUSE,
                        confidence_score=min(0.95, violation_ratio / 2),
                        description=f"Rate limit violation detected from {ip}: "
                                   f"{len(recent_timestamps)} requests in 1 minute "
                                   f"(limit: {self.rate_limit_threshold})",
                        affected_resources=list(set(e.endpoint for e in ip_events)),
                        indicators_of_compromise=[
                            f"source_ip:{ip}",
                            f"request_count:{len(recent_timestamps)}",
                            f"violation_ratio:{violation_ratio:.2f}"
                        ],
                        recommended_actions=[
                            "apply_rate_limiting",
                            "block_source_ip_temporarily",
                            "analyze_request_patterns",
                            "check_for_bot_behavior"
                        ],
                        raw_events=ip_events[-10:]  # Include last 10 events
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def _determine_threat_level_for_rate_limit(self, violation_ratio: float) -> ThreatLevel:
        """Determine threat level based on violation ratio."""
        if violation_ratio > 5.0:
            return ThreatLevel.CRITICAL
        elif violation_ratio > 3.0:
            return ThreatLevel.HIGH
        elif violation_ratio > 2.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_detection_id(self, ip: str, timestamp: datetime) -> str:
        """Generate unique detection ID."""
        content = f"rate_limit_{ip}_{timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class ThreatDetectionEngine:
    """Main threat detection engine orchestrating multiple detectors."""
    
    def __init__(self):
        self.detectors: List[ThreatDetector] = []
        self.detection_history: List[ThreatDetection] = []
        
    def register_detector(self, detector: ThreatDetector) -> None:
        """Register a threat detector."""
        self.detectors.append(detector)
        logging.info(f"Registered detector: {detector.get_detector_name()}")
    
    async def analyze_events(self, events: List[SecurityEvent]) -> List[ThreatDetection]:
        """Analyze events with all registered detectors."""
        all_detections = []
        
        # Run all detectors concurrently
        detection_tasks = [
            detector.analyze_events(events) 
            for detector in self.detectors
        ]
        
        detector_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Collect results from all detectors
        for i, result in enumerate(detector_results):
            if isinstance(result, Exception):
                logging.error(f"Detector {self.detectors[i].get_detector_name()} failed: {result}")
                continue
            
            all_detections.extend(result)
        
        # Deduplicate and prioritize detections
        deduplicated_detections = self._deduplicate_detections(all_detections)
        prioritized_detections = self._prioritize_detections(deduplicated_detections)
        
        # Store in history
        self.detection_history.extend(prioritized_detections)
        
        # Keep only recent history (last 1000 detections)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        return prioritized_detections
    
    def _deduplicate_detections(self, detections: List[ThreatDetection]) -> List[ThreatDetection]:
        """Remove duplicate detections based on similar characteristics."""
        unique_detections = []
        seen_signatures = set()
        
        for detection in detections:
            # Create signature based on source IP, threat category, and time window
            time_window = detection.timestamp.replace(second=0, microsecond=0)
            source_ips = set()
            
            for event in detection.raw_events:
                source_ips.add(event.source_ip)
            
            signature = f"{sorted(source_ips)}_{detection.threat_category.value}_{time_window}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_detections.append(detection)
        
        return unique_detections
    
    def _prioritize_detections(self, detections: List[ThreatDetection]) -> List[ThreatDetection]:
        """Sort detections by priority (threat level and confidence)."""
        priority_order = {
            ThreatLevel.CRITICAL: 5,
            ThreatLevel.HIGH: 4,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.LOW: 2,
            ThreatLevel.INFO: 1
        }
        
        return sorted(
            detections,
            key=lambda d: (priority_order[d.threat_level], d.confidence_score),
            reverse=True
        )

# Example usage and testing
async def main_threat_detection_example():
    """Demonstrate threat detection system."""
    
    # Create threat detection engine
    engine = ThreatDetectionEngine()
    
    # Register detectors
    anomaly_detector = AnomalyDetector(contamination=0.1)
    injection_detector = InjectionAttackDetector()
    rate_limit_detector = RateLimitingDetector(rate_limit_threshold=50)
    
    engine.register_detector(anomaly_detector)
    engine.register_detector(injection_detector)
    engine.register_detector(rate_limit_detector)
    
    # Generate sample security events
    sample_events = []
    current_time = datetime.now()
    
    # Normal events
    for i in range(20):
        event = SecurityEvent(
            timestamp=current_time - timedelta(minutes=i),
            event_id=f"event_{i}",
            source_ip=f"192.168.1.{100 + i % 10}",
            user_id=f"user_{i % 5}",
            event_type="api_request",
            endpoint="/api/emails/classify",
            user_agent="Mozilla/5.0 (legitimate user agent)",
            payload_size=1024,
            response_code=200,
            processing_time_ms=150.0,
            features={"query_params": "category=inbox"}
        )
        sample_events.append(event)
    
    # Malicious events
    malicious_events = [
        SecurityEvent(
            timestamp=current_time,
            event_id="malicious_1",
            source_ip="10.0.0.1",
            user_id=None,
            event_type="api_request",
            endpoint="/api/emails/search",
            user_agent="sqlmap/1.0",
            payload_size=2048,
            response_code=200,
            processing_time_ms=500.0,
            features={"query_params": "q=' OR 1=1 --"}
        ),
        # Rate limiting violation events
        *[
            SecurityEvent(
                timestamp=current_time - timedelta(seconds=i),
                event_id=f"rate_limit_{i}",
                source_ip="192.168.1.200",
                user_id="attacker",
                event_type="api_request",
                endpoint="/api/emails/list",
                user_agent="bot/1.0",
                payload_size=512,
                response_code=200,
                processing_time_ms=50.0,
                features={}
            ) for i in range(60)  # 60 requests in 60 seconds = rate limit violation
        ]
    ]
    
    sample_events.extend(malicious_events)
    
    # Train anomaly detector with normal events
    normal_events = sample_events[:20]
    await anomaly_detector.train_baseline(normal_events)
    
    # Analyze all events
    print("Analyzing security events...")
    detections = await engine.analyze_events(sample_events)
    
    # Display results
    print(f"\nDetected {len(detections)} threats:")
    for detection in detections:
        print(f"\n- {detection.threat_category.value.upper()}")
        print(f"  Threat Level: {detection.threat_level.value}")
        print(f"  Confidence: {detection.confidence_score:.2%}")
        print(f"  Description: {detection.description}")
        print(f"  Affected Resources: {', '.join(detection.affected_resources)}")
        print(f"  Recommended Actions: {', '.join(detection.recommended_actions)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_threat_detection_example())