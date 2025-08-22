"""Real-time Email Flow Intelligence System.

Advanced real-time processing capabilities for email flow analysis,
predictive routing, and intelligent queue management.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable, Any, Union
import logging
import statistics

from .resilience import resilience
from .metrics_export import get_metrics_collector
from .health import get_health_checker
from .pipeline import triage_email
from .validation import validate_email_content


class FlowPriority(Enum):
    """Email flow priority levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BULK = 1


class ProcessingStatus(Enum):
    """Processing status for email flow tracking."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class EmailFlowEvent:
    """Represents an email processing event in the flow."""
    id: str
    content: str
    headers: Optional[Dict[str, str]] = None
    priority: FlowPriority = FlowPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: ProcessingStatus = ProcessingStatus.QUEUED
    retry_count: int = 0
    max_retries: int = 3
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None


@dataclass
class FlowMetrics:
    """Real-time flow processing metrics."""
    total_processed: int = 0
    success_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    queue_depth: int = 0
    throughput_per_minute: float = 0.0
    error_rate: float = 0.0
    priority_distribution: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class IntelligentRouter:
    """Intelligent routing system for email flow optimization."""
    
    def __init__(self):
        self.route_weights = {
            FlowPriority.CRITICAL: 1.0,
            FlowPriority.HIGH: 0.8,
            FlowPriority.MEDIUM: 0.5,
            FlowPriority.LOW: 0.3,
            FlowPriority.BULK: 0.1
        }
        self.load_balance_history = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_priority_score(self, event: EmailFlowEvent) -> float:
        """Calculate dynamic priority score based on content and context."""
        base_score = self.route_weights[event.priority]
        
        # Time-based urgency boost
        time_factor = 1.0
        if event.timestamp:
            age_minutes = (datetime.utcnow() - event.timestamp).total_seconds() / 60
            if age_minutes > 30:  # Boost priority for older emails
                time_factor = min(1.5, 1.0 + (age_minutes / 120))
        
        # Content-based priority detection
        content_boost = 1.0
        urgent_keywords = ['urgent', 'asap', 'emergency', 'critical', 'immediately']
        if any(keyword in event.content.lower() for keyword in urgent_keywords):
            content_boost = 1.3
        
        # Retry penalty
        retry_penalty = max(0.7, 1.0 - (event.retry_count * 0.1))
        
        return base_score * time_factor * content_boost * retry_penalty
    
    def route_event(self, event: EmailFlowEvent) -> str:
        """Determine optimal routing queue for the event."""
        score = self.calculate_priority_score(event)
        
        if score >= 1.0:
            return "high_priority_queue"
        elif score >= 0.5:
            return "standard_queue"
        else:
            return "bulk_queue"


class RealTimeProcessor:
    """Core real-time email processing engine."""
    
    def __init__(self, max_workers: int = 8, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.queues = {
            "high_priority_queue": Queue(),
            "standard_queue": Queue(),
            "bulk_queue": Queue()
        }
        self.processing_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.router = IntelligentRouter()
        self.metrics = FlowMetrics()
        self.active_processors = 0
        self.running = False
        self.logger = logging.getLogger(__name__)
        self.error_handlers: List[Callable[[EmailFlowEvent, Exception], None]] = []
        
        # Performance tracking
        self.processing_times = []
        self.last_metrics_update = time.time()
        
    def add_error_handler(self, handler: Callable[[EmailFlowEvent, Exception], None]):
        """Add custom error handler for processing failures."""
        self.error_handlers.append(handler)
    
    def enqueue_event(self, event: EmailFlowEvent) -> bool:
        """Add event to appropriate processing queue."""
        try:
            if not validate_email_content(event.content):
                self.logger.warning(f"Invalid email content for event {event.id}")
                return False
            
            queue_name = self.router.route_event(event)
            self.queues[queue_name].put(event)
            
            self.logger.info(f"Event {event.id} routed to {queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enqueue event {event.id}: {e}")
            return False
    
    def process_single_event(self, event: EmailFlowEvent) -> Optional[Dict[str, Any]]:
        """Process a single email event with full error handling."""
        start_time = time.time()
        
        try:
            event.status = ProcessingStatus.PROCESSING
            
            # Apply resilience patterns
            with resilience.bulkhead("email_processing"):
                result = resilience.retry(
                    lambda: triage_email(event.content),
                    max_attempts=event.max_retries,
                    backoff_factor=1.5
                )
            
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            event.status = ProcessingStatus.COMPLETED
            event.processing_metadata.update({
                'processing_time_ms': processing_time,
                'completed_at': datetime.utcnow().isoformat(),
                'result': result
            })
            
            self.logger.info(f"Successfully processed event {event.id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            event.retry_count += 1
            event.error_details = str(e)
            
            if event.retry_count >= event.max_retries:
                event.status = ProcessingStatus.FAILED
                self.logger.error(f"Event {event.id} failed after {event.max_retries} retries: {e}")
                
                # Trigger error handlers
                for handler in self.error_handlers:
                    try:
                        handler(event, e)
                    except Exception as handler_error:
                        self.logger.error(f"Error handler failed: {handler_error}")
            else:
                event.status = ProcessingStatus.RETRYING
                self.logger.warning(f"Event {event.id} failed, retrying ({event.retry_count}/{event.max_retries}): {e}")
                
                # Re-enqueue with delay
                time.sleep(min(2 ** event.retry_count, 30))  # Exponential backoff
                self.enqueue_event(event)
            
            return None
    
    def update_metrics(self):
        """Update real-time processing metrics."""
        current_time = time.time()
        time_window = current_time - self.last_metrics_update
        
        if time_window > 0:
            # Calculate throughput
            recent_processing_times = [t for t in self.processing_times if t is not None]
            if recent_processing_times:
                self.metrics.avg_processing_time_ms = statistics.mean(recent_processing_times)
                self.metrics.throughput_per_minute = len(recent_processing_times) / (time_window / 60)
            
            # Update queue depths
            total_queue_depth = sum(q.qsize() for q in self.queues.values())
            self.metrics.queue_depth = total_queue_depth
            
            # Clear old processing times (keep last 5 minutes)
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-500:]
            
            self.metrics.last_updated = datetime.utcnow()
            self.last_metrics_update = current_time
    
    def start_processing(self):
        """Start the real-time processing engine."""
        self.running = True
        self.logger.info("Real-time processor started")
        
        # Start processing threads for each queue
        for queue_name, queue in self.queues.items():
            priority_weight = {"high_priority_queue": 3, "standard_queue": 2, "bulk_queue": 1}[queue_name]
            
            for _ in range(min(self.max_workers // 3 * priority_weight, self.max_workers)):
                threading.Thread(
                    target=self._process_queue_worker,
                    args=(queue, queue_name),
                    daemon=True
                ).start()
        
        # Start metrics update thread
        threading.Thread(target=self._metrics_updater, daemon=True).start()
    
    def _process_queue_worker(self, queue: Queue, queue_name: str):
        """Worker thread for processing events from a specific queue."""
        while self.running:
            try:
                event = queue.get(timeout=1.0)
                self.active_processors += 1
                
                try:
                    self.process_single_event(event)
                    self.metrics.total_processed += 1
                finally:
                    self.active_processors -= 1
                    queue.task_done()
                    
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Queue worker error in {queue_name}: {e}")
    
    def _metrics_updater(self):
        """Background thread for updating metrics."""
        while self.running:
            try:
                self.update_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")
    
    def stop_processing(self):
        """Stop the real-time processing engine."""
        self.running = False
        
        # Wait for active processors to complete
        timeout = 30
        start_time = time.time()
        while self.active_processors > 0 and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        self.processing_pool.shutdown(wait=True)
        self.logger.info("Real-time processor stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status and metrics."""
        return {
            "running": self.running,
            "active_processors": self.active_processors,
            "max_workers": self.max_workers,
            "queue_depths": {name: queue.qsize() for name, queue in self.queues.items()},
            "metrics": {
                "total_processed": self.metrics.total_processed,
                "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
                "throughput_per_minute": self.metrics.throughput_per_minute,
                "queue_depth": self.metrics.queue_depth,
                "last_updated": self.metrics.last_updated.isoformat()
            }
        }


class RealTimeIntelligenceManager:
    """Main manager for real-time email intelligence system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.processor = RealTimeProcessor(
            max_workers=self.config.get('max_workers', 8),
            batch_size=self.config.get('batch_size', 10)
        )
        self.event_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # Setup default error handler
        self.processor.add_error_handler(self._default_error_handler)
    
    def _default_error_handler(self, event: EmailFlowEvent, error: Exception):
        """Default error handler for failed processing events."""
        self.logger.error(f"Processing failed for event {event.id}: {error}")
        
        # Report to metrics collector
        metrics_collector = get_metrics_collector()
        metrics_collector.increment_counter("email_processing_errors", {
            "event_id": event.id,
            "priority": event.priority.name,
            "retry_count": str(event.retry_count)
        })
    
    def submit_email(self, content: str, headers: Optional[Dict[str, str]] = None, 
                    priority: FlowPriority = FlowPriority.MEDIUM) -> str:
        """Submit email for real-time processing."""
        self.event_counter += 1
        event_id = f"email_{self.event_counter}_{int(time.time())}"
        
        event = EmailFlowEvent(
            id=event_id,
            content=content,
            headers=headers,
            priority=priority
        )
        
        if self.processor.enqueue_event(event):
            self.logger.info(f"Email submitted for processing: {event_id}")
            return event_id
        else:
            raise ValueError(f"Failed to submit email for processing: {event_id}")
    
    def start(self):
        """Start the real-time intelligence system."""
        self.processor.start_processing()
        self.logger.info("Real-time intelligence system started")
    
    def stop(self):
        """Stop the real-time intelligence system."""
        self.processor.stop_processing()
        self.logger.info("Real-time intelligence system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        processor_status = self.processor.get_status()
        
        # Add health check information
        health_checker = get_health_checker()
        health_result = health_checker.check_health()
        
        return {
            "realtime_processor": processor_status,
            "system_health": {
                "status": health_result.status.value,
                "response_time_ms": health_result.response_time_ms,
                "healthy_checks": len([c for c in health_result.checks if c.status.name == "HEALTHY"])
            },
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }


# Global instance for easy access
_intelligence_manager: Optional[RealTimeIntelligenceManager] = None


def get_intelligence_manager(config: Optional[Dict[str, Any]] = None) -> RealTimeIntelligenceManager:
    """Get or create the global real-time intelligence manager."""
    global _intelligence_manager
    
    if _intelligence_manager is None:
        _intelligence_manager = RealTimeIntelligenceManager(config)
    
    return _intelligence_manager


def submit_email_for_processing(content: str, headers: Optional[Dict[str, str]] = None,
                               priority: FlowPriority = FlowPriority.MEDIUM) -> str:
    """Convenience function to submit email for real-time processing."""
    manager = get_intelligence_manager()
    return manager.submit_email(content, headers, priority)


def start_realtime_system(config: Optional[Dict[str, Any]] = None):
    """Start the real-time email intelligence system."""
    manager = get_intelligence_manager(config)
    manager.start()


def stop_realtime_system():
    """Stop the real-time email intelligence system."""
    global _intelligence_manager
    if _intelligence_manager:
        _intelligence_manager.stop()


def get_realtime_status() -> Dict[str, Any]:
    """Get current real-time system status."""
    manager = get_intelligence_manager()
    return manager.get_system_status()