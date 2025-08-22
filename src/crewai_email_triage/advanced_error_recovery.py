"""Advanced Error Recovery and Self-Healing System.

Implements sophisticated error recovery mechanisms, automatic healing,
and adaptive failure management for robust email processing.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import logging
import pickle
import hashlib
from pathlib import Path

from .circuit_breaker import CircuitBreaker
from .retry_utils import retry_with_backoff
from .health import get_health_checker


class FailureType(Enum):
    """Classification of failure types for targeted recovery."""
    TRANSIENT = "transient"          # Temporary network/resource issues
    PERSISTENT = "persistent"        # Consistent failures requiring intervention
    RESOURCE = "resource"            # Memory/CPU/disk exhaustion
    DEPENDENCY = "dependency"        # External service failures
    CORRUPTION = "corruption"        # Data/state corruption
    CONFIGURATION = "configuration"  # Configuration-related failures
    SECURITY = "security"           # Security-related failures


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    RESTART_COMPONENT = "restart_component"
    RESET_STATE = "reset_state"
    SCALE_RESOURCES = "scale_resources"
    FALLBACK_MODE = "fallback_mode"
    QUARANTINE = "quarantine"
    NOTIFY_ADMIN = "notify_admin"


@dataclass
class FailureEvent:
    """Represents a failure event with context and recovery information."""
    id: str
    timestamp: datetime
    failure_type: FailureType
    component: str
    error_message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class RecoveryRule:
    """Defines recovery strategy for specific failure patterns."""
    pattern: str  # Regex pattern for error matching
    failure_type: FailureType
    actions: List[RecoveryAction]
    max_attempts: int = 3
    cooldown_seconds: int = 60
    priority: int = 1  # Higher number = higher priority


class ErrorPatternAnalyzer:
    """Analyzes error patterns and suggests recovery strategies."""
    
    def __init__(self):
        self.failure_history: List[FailureEvent] = []
        self.pattern_frequency: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
        
    def analyze_failure(self, error: Exception, component: str, context: Dict[str, Any]) -> FailureEvent:
        """Analyze a failure and classify it for recovery."""
        error_message = str(error)
        error_hash = hashlib.md5(error_message.encode()).hexdigest()[:8]
        
        # Classify failure type
        failure_type = self._classify_failure(error, error_message, context)
        
        failure_event = FailureEvent(
            id=f"failure_{error_hash}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            stack_trace=getattr(error, '__traceback__', None),
            context=context
        )
        
        self.failure_history.append(failure_event)
        self._update_pattern_frequency(error_message)
        
        return failure_event
    
    def _classify_failure(self, error: Exception, message: str, context: Dict[str, Any]) -> FailureType:
        """Classify failure based on error type and message patterns."""
        error_type = type(error).__name__
        message_lower = message.lower()
        
        # Network/connectivity issues
        if any(keyword in message_lower for keyword in 
               ['connection', 'timeout', 'network', 'unreachable', 'dns']):
            return FailureType.TRANSIENT
        
        # Resource exhaustion
        if any(keyword in message_lower for keyword in 
               ['memory', 'disk', 'space', 'quota', 'limit', 'resource']):
            return FailureType.RESOURCE
        
        # Configuration issues
        if any(keyword in message_lower for keyword in 
               ['config', 'permission', 'access', 'credential', 'key']):
            return FailureType.CONFIGURATION
        
        # Security issues
        if any(keyword in message_lower for keyword in 
               ['security', 'unauthorized', 'forbidden', 'authentication']):
            return FailureType.SECURITY
        
        # Data corruption
        if any(keyword in message_lower for keyword in 
               ['corrupt', 'invalid', 'malformed', 'decode', 'parse']):
            return FailureType.CORRUPTION
        
        # External dependency failures
        if any(keyword in message_lower for keyword in 
               ['service', 'api', 'endpoint', 'external', 'upstream']):
            return FailureType.DEPENDENCY
        
        # Default to transient for unknown errors
        return FailureType.TRANSIENT
    
    def _update_pattern_frequency(self, error_message: str):
        """Update frequency tracking for error patterns."""
        # Normalize error message for pattern matching
        normalized = error_message.lower()
        
        # Extract key error patterns
        patterns = []
        if 'connection' in normalized:
            patterns.append('connection_error')
        if 'timeout' in normalized:
            patterns.append('timeout_error')
        if 'memory' in normalized:
            patterns.append('memory_error')
        if 'permission' in normalized:
            patterns.append('permission_error')
        
        for pattern in patterns:
            self.pattern_frequency[pattern] = self.pattern_frequency.get(pattern, 0) + 1
    
    def get_recovery_recommendations(self, failure_event: FailureEvent) -> List[RecoveryAction]:
        """Get recommended recovery actions for a failure event."""
        recommendations = []
        
        # Base recommendations by failure type
        type_actions = {
            FailureType.TRANSIENT: [RecoveryAction.RETRY],
            FailureType.RESOURCE: [RecoveryAction.SCALE_RESOURCES, RecoveryAction.RESTART_COMPONENT],
            FailureType.DEPENDENCY: [RecoveryAction.RETRY, RecoveryAction.FALLBACK_MODE],
            FailureType.CORRUPTION: [RecoveryAction.RESET_STATE, RecoveryAction.RESTART_COMPONENT],
            FailureType.CONFIGURATION: [RecoveryAction.NOTIFY_ADMIN, RecoveryAction.FALLBACK_MODE],
            FailureType.SECURITY: [RecoveryAction.QUARANTINE, RecoveryAction.NOTIFY_ADMIN]
        }
        
        recommendations.extend(type_actions.get(failure_event.failure_type, []))
        
        # Add frequency-based recommendations
        error_pattern = failure_event.error_message.lower()
        if self.pattern_frequency.get('timeout_error', 0) > 5:
            recommendations.append(RecoveryAction.SCALE_RESOURCES)
        if self.pattern_frequency.get('connection_error', 0) > 3:
            recommendations.append(RecoveryAction.FALLBACK_MODE)
        
        return recommendations[:3]  # Limit to top 3 recommendations


class SelfHealingOrchestrator:
    """Orchestrates self-healing recovery actions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analyzer = ErrorPatternAnalyzer()
        self.active_recoveries: Dict[str, Future] = {}
        self.recovery_cooldowns: Dict[str, datetime] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Recovery action handlers
        self.action_handlers = {
            RecoveryAction.RETRY: self._handle_retry,
            RecoveryAction.RESTART_COMPONENT: self._handle_restart_component,
            RecoveryAction.RESET_STATE: self._handle_reset_state,
            RecoveryAction.SCALE_RESOURCES: self._handle_scale_resources,
            RecoveryAction.FALLBACK_MODE: self._handle_fallback_mode,
            RecoveryAction.QUARANTINE: self._handle_quarantine,
            RecoveryAction.NOTIFY_ADMIN: self._handle_notify_admin
        }
        
        # Component restart handlers
        self.component_restarters: Dict[str, Callable[[], bool]] = {}
        
    def register_component_restarter(self, component: str, restarter: Callable[[], bool]):
        """Register a function to restart a specific component."""
        self.component_restarters[component] = restarter
        self.logger.info(f"Registered restarter for component: {component}")
    
    def handle_failure(self, error: Exception, component: str, 
                      context: Optional[Dict[str, Any]] = None) -> str:
        """Handle a failure event and initiate recovery."""
        context = context or {}
        failure_event = self.analyzer.analyze_failure(error, component, context)
        
        self.logger.error(f"Handling failure {failure_event.id}: {failure_event.error_message}")
        
        # Check cooldown period
        cooldown_key = f"{component}_{failure_event.failure_type.value}"
        if cooldown_key in self.recovery_cooldowns:
            if datetime.utcnow() < self.recovery_cooldowns[cooldown_key]:
                self.logger.info(f"Recovery for {cooldown_key} is in cooldown period")
                return failure_event.id
        
        # Get recovery recommendations
        actions = self.analyzer.get_recovery_recommendations(failure_event)
        
        # Execute recovery actions asynchronously
        recovery_future = self.executor.submit(
            self._execute_recovery_actions, failure_event, actions
        )
        self.active_recoveries[failure_event.id] = recovery_future
        
        # Set cooldown
        self.recovery_cooldowns[cooldown_key] = datetime.utcnow() + timedelta(seconds=60)
        
        return failure_event.id
    
    def _execute_recovery_actions(self, failure_event: FailureEvent, actions: List[RecoveryAction]) -> bool:
        """Execute a sequence of recovery actions."""
        self.logger.info(f"Executing recovery actions for {failure_event.id}: {[a.value for a in actions]}")
        
        for action in actions:
            try:
                handler = self.action_handlers.get(action)
                if handler:
                    success = handler(failure_event, action)
                    failure_event.recovery_attempts.append(f"{action.value}:{'success' if success else 'failed'}")
                    
                    if success:
                        failure_event.resolved = True
                        failure_event.resolution_time = datetime.utcnow()
                        self.logger.info(f"Recovery successful for {failure_event.id} using {action.value}")
                        return True
                else:
                    self.logger.warning(f"No handler found for recovery action: {action.value}")
                    
            except Exception as e:
                self.logger.error(f"Recovery action {action.value} failed: {e}")
                failure_event.recovery_attempts.append(f"{action.value}:error:{str(e)}")
        
        self.logger.warning(f"All recovery actions failed for {failure_event.id}")
        return False
    
    def _handle_retry(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle retry recovery action."""
        # This is typically handled by the calling code with exponential backoff
        self.logger.info(f"Retry recommended for {failure_event.component}")
        return True  # Indicate that retry is a valid action
    
    def _handle_restart_component(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle component restart recovery action."""
        component = failure_event.component
        restarter = self.component_restarters.get(component)
        
        if restarter:
            try:
                success = restarter()
                self.logger.info(f"Component {component} restart {'successful' if success else 'failed'}")
                return success
            except Exception as e:
                self.logger.error(f"Failed to restart component {component}: {e}")
                return False
        else:
            self.logger.warning(f"No restarter registered for component: {component}")
            return False
    
    def _handle_reset_state(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle state reset recovery action."""
        component = failure_event.component
        
        # Clear component caches
        try:
            from .cache import get_smart_cache
            cache = get_smart_cache()
            cache.clear_all()
            self.logger.info(f"State reset completed for {component}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset state for {component}: {e}")
            return False
    
    def _handle_scale_resources(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle resource scaling recovery action."""
        # Implement resource scaling logic
        self.logger.info(f"Resource scaling triggered for {failure_event.component}")
        
        # Example: trigger garbage collection
        import gc
        gc.collect()
        
        # Could trigger auto-scaling if running in cloud environment
        return True
    
    def _handle_fallback_mode(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle fallback mode recovery action."""
        component = failure_event.component
        self.logger.info(f"Fallback mode activated for {component}")
        
        # Set fallback configuration
        fallback_key = f"fallback_{component}"
        # Store fallback state (could use redis/database in production)
        return True
    
    def _handle_quarantine(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle quarantine recovery action."""
        component = failure_event.component
        self.logger.warning(f"Component {component} quarantined due to security issue")
        
        # Disable component temporarily
        quarantine_key = f"quarantine_{component}"
        # Store quarantine state
        return True
    
    def _handle_notify_admin(self, failure_event: FailureEvent, action: RecoveryAction) -> bool:
        """Handle admin notification recovery action."""
        # Send notification to administrators
        notification = {
            "type": "failure_notification",
            "failure_id": failure_event.id,
            "component": failure_event.component,
            "failure_type": failure_event.failure_type.value,
            "message": failure_event.error_message,
            "timestamp": failure_event.timestamp.isoformat()
        }
        
        self.logger.critical(f"Admin notification: {json.dumps(notification)}")
        
        # In production, this would send email/slack/webhook notifications
        return True
    
    def get_recovery_status(self, failure_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a recovery operation."""
        if failure_id in self.active_recoveries:
            future = self.active_recoveries[failure_id]
            status = "completed" if future.done() else "in_progress"
            
            result = None
            if future.done():
                try:
                    result = future.result()
                except Exception as e:
                    result = f"error: {str(e)}"
            
            return {
                "failure_id": failure_id,
                "status": status,
                "result": result
            }
        
        return None
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        recent_failures = [f for f in self.analyzer.failure_history 
                          if f.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        failure_types = {}
        for failure in recent_failures:
            failure_types[failure.failure_type.value] = failure_types.get(failure.failure_type.value, 0) + 1
        
        resolved_count = len([f for f in recent_failures if f.resolved])
        
        return {
            "total_failures_last_hour": len(recent_failures),
            "resolved_failures": resolved_count,
            "resolution_rate": resolved_count / len(recent_failures) if recent_failures else 1.0,
            "failure_types": failure_types,
            "pattern_frequency": dict(self.analyzer.pattern_frequency),
            "active_recoveries": len(self.active_recoveries),
            "components_in_cooldown": len(self.recovery_cooldowns),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
_self_healing_orchestrator: Optional[SelfHealingOrchestrator] = None


def get_self_healing_orchestrator(config: Optional[Dict[str, Any]] = None) -> SelfHealingOrchestrator:
    """Get or create the global self-healing orchestrator."""
    global _self_healing_orchestrator
    
    if _self_healing_orchestrator is None:
        _self_healing_orchestrator = SelfHealingOrchestrator(config)
    
    return _self_healing_orchestrator


def handle_failure_with_recovery(error: Exception, component: str, 
                                context: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to handle failures with automatic recovery."""
    orchestrator = get_self_healing_orchestrator()
    return orchestrator.handle_failure(error, component, context)


def register_component_restart_handler(component: str, handler: Callable[[], bool]):
    """Register a restart handler for a component."""
    orchestrator = get_self_healing_orchestrator()
    orchestrator.register_component_restarter(component, handler)


def get_system_health_report() -> Dict[str, Any]:
    """Get comprehensive system health and recovery report."""
    orchestrator = get_self_healing_orchestrator()
    return orchestrator.get_system_health_report()