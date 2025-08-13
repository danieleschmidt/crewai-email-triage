"""Real-time Streaming Processing for Advanced Email Triage.

This module provides next-generation streaming capabilities including:
- WebSocket-based real-time communication
- Server-sent events for streaming updates
- Reactive processing with backpressure handling
- Real-time collaboration and multi-user support
- Live dashboard updates and notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Set

from .llm_pipeline import get_llm_pipeline

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""

    EMAIL_RECEIVED = "email_received"
    PROCESSING_STARTED = "processing_started"
    CLASSIFICATION_COMPLETE = "classification_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    RESPONSE_GENERATED = "response_generated"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR_OCCURRED = "error_occurred"

    # Real-time updates
    QUEUE_STATUS_UPDATE = "queue_status_update"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_HEALTH_UPDATE = "system_health_update"
    USER_ACTIVITY_UPDATE = "user_activity_update"


@dataclass
class StreamEvent:
    """Real-time streaming event."""

    event_type: StreamEventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps({
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data": self.data,
            "metadata": self.metadata
        })

    def to_sse_format(self) -> str:
        """Convert event to Server-Sent Events format."""
        return f"event: {self.event_type.value}\nid: {self.event_id}\ndata: {json.dumps(self.data)}\n\n"


class StreamSubscription:
    """Subscription to real-time events."""

    def __init__(
        self,
        subscription_id: str,
        event_types: Set[StreamEventType],
        callback: Callable[[StreamEvent], None],
        filters: Optional[Dict[str, Any]] = None
    ):
        self.subscription_id = subscription_id
        self.event_types = event_types
        self.callback = callback
        self.filters = filters or {}
        self.created_at = time.time()
        self.last_event_time = time.time()
        self.event_count = 0
        self.active = True

    def matches_event(self, event: StreamEvent) -> bool:
        """Check if event matches subscription criteria."""
        if not self.active:
            return False

        if event.event_type not in self.event_types:
            return False

        # Apply filters
        for filter_key, filter_value in self.filters.items():
            if filter_key == "session_id" and event.session_id != filter_value or filter_key == "user_id" and event.user_id != filter_value or filter_key in event.data and event.data[filter_key] != filter_value:
                return False

        return True

    async def notify(self, event: StreamEvent) -> None:
        """Notify subscriber of event."""
        if self.matches_event(event):
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(event)
                else:
                    self.callback(event)

                self.last_event_time = time.time()
                self.event_count += 1

            except Exception as e:
                logger.error(f"Error in event callback for subscription {self.subscription_id}: {e}")


class RealTimeEventBus:
    """Real-time event bus for streaming updates."""

    def __init__(self, max_event_history: int = 1000):
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.event_history: deque = deque(maxlen=max_event_history)
        self.stats = {
            "events_published": 0,
            "active_subscriptions": 0,
            "total_subscriptions": 0,
            "start_time": time.time()
        }

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_subscriptions())
        self._stats_task = asyncio.create_task(self._publish_stats_updates())

        logger.info("RealTimeEventBus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        logger.info("RealTimeEventBus stopped")

    def subscribe(
        self,
        event_types: Set[StreamEventType],
        callback: Callable[[StreamEvent], None],
        filters: Optional[Dict[str, Any]] = None,
        subscription_id: Optional[str] = None
    ) -> str:
        """Subscribe to real-time events."""
        if subscription_id is None:
            subscription_id = str(uuid.uuid4())

        subscription = StreamSubscription(
            subscription_id, event_types, callback, filters
        )

        self.subscriptions[subscription_id] = subscription
        self.stats["total_subscriptions"] += 1
        self.stats["active_subscriptions"] = len(
            [s for s in self.subscriptions.values() if s.active]
        )

        logger.info(f"New subscription created: {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].active = False
            del self.subscriptions[subscription_id]

            self.stats["active_subscriptions"] = len(
                [s for s in self.subscriptions.values() if s.active]
            )

            logger.info(f"Subscription removed: {subscription_id}")
            return True
        return False

    async def publish(self, event: StreamEvent) -> None:
        """Publish event to all matching subscribers."""
        self.event_history.append(event)
        self.stats["events_published"] += 1

        # Notify all matching subscriptions
        notification_tasks = []
        for subscription in self.subscriptions.values():
            if subscription.matches_event(event):
                notification_tasks.append(subscription.notify(event))

        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)

        logger.debug(f"Published event: {event.event_type.value} to {len(notification_tasks)} subscribers")

    async def _cleanup_inactive_subscriptions(self) -> None:
        """Clean up inactive subscriptions periodically."""
        while self._running:
            try:
                current_time = time.time()
                inactive_ids = []

                for sub_id, subscription in self.subscriptions.items():
                    # Remove subscriptions inactive for more than 1 hour
                    if not subscription.active or (current_time - subscription.last_event_time > 3600):
                        inactive_ids.append(sub_id)

                for sub_id in inactive_ids:
                    del self.subscriptions[sub_id]

                if inactive_ids:
                    logger.info(f"Cleaned up {len(inactive_ids)} inactive subscriptions")

                self.stats["active_subscriptions"] = len(
                    [s for s in self.subscriptions.values() if s.active]
                )

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in subscription cleanup: {e}")
                await asyncio.sleep(60)

    async def _publish_stats_updates(self) -> None:
        """Publish periodic statistics updates."""
        while self._running:
            try:
                stats_event = StreamEvent(
                    event_type=StreamEventType.PERFORMANCE_UPDATE,
                    data={
                        "event_bus_stats": self.stats,
                        "uptime_seconds": time.time() - self.stats["start_time"]
                    }
                )

                await self.publish(stats_event)
                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error publishing stats updates: {e}")
                await asyncio.sleep(60)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self.stats,
            "event_history_size": len(self.event_history),
            "active_subscriptions": self.stats["active_subscriptions"],
            "uptime_seconds": time.time() - self.stats["start_time"]
        }


class StreamingEmailProcessor:
    """Streaming email processor with real-time updates."""

    def __init__(self, event_bus: Optional[RealTimeEventBus] = None):
        self.event_bus = event_bus or RealTimeEventBus()
        self.llm_pipeline = get_llm_pipeline()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Processing statistics
        self.stats = {
            "emails_processed": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "queue_size": 0,
            "active_sessions_count": 0
        }

    async def start_processing(self) -> None:
        """Start the streaming processor."""
        await self.event_bus.start()

        # Start background processing
        asyncio.create_task(self._process_queue())
        asyncio.create_task(self._update_stats())

        logger.info("StreamingEmailProcessor started")

    async def process_email_stream(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process email with real-time streaming updates."""

        if session_id is None:
            session_id = str(uuid.uuid4())

        # Track active session
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "user_id": user_id,
            "status": "processing"
        }

        try:
            # Email received event
            email_received_event = StreamEvent(
                event_type=StreamEventType.EMAIL_RECEIVED,
                session_id=session_id,
                user_id=user_id,
                data={
                    "content_length": len(email_content),
                    "has_headers": email_headers is not None,
                    "session_id": session_id
                }
            )

            yield email_received_event
            await self.event_bus.publish(email_received_event)

            # Processing started event
            processing_started_event = StreamEvent(
                event_type=StreamEventType.PROCESSING_STARTED,
                session_id=session_id,
                user_id=user_id,
                data={"timestamp": time.time()}
            )

            yield processing_started_event
            await self.event_bus.publish(processing_started_event)

            # Stream LLM processing
            async for chunk in self.llm_pipeline.stream_process_email(
                email_content, email_headers, session_id
            ):
                # Determine event type based on chunk content
                if "Analyzing" in chunk:
                    event_type = StreamEventType.CLASSIFICATION_COMPLETE
                elif "Generating" in chunk or "response" in chunk.lower():
                    event_type = StreamEventType.RESPONSE_GENERATED
                else:
                    event_type = StreamEventType.ANALYSIS_COMPLETE

                chunk_event = StreamEvent(
                    event_type=event_type,
                    session_id=session_id,
                    user_id=user_id,
                    data={"chunk": chunk, "progress": "streaming"}
                )

                yield chunk_event
                await self.event_bus.publish(chunk_event)

            # Get final processing result
            final_result = await self.llm_pipeline.process_email(
                email_content, email_headers, session_id
            )

            # Processing complete event
            processing_complete_event = StreamEvent(
                event_type=StreamEventType.PROCESSING_COMPLETE,
                session_id=session_id,
                user_id=user_id,
                data={
                    "result": final_result.model_dump(),
                    "processing_time_ms": final_result.processing_time_ms,
                    "confidence_score": final_result.confidence_score
                }
            )

            yield processing_complete_event
            await self.event_bus.publish(processing_complete_event)

            # Update statistics
            self.stats["emails_processed"] += 1
            if final_result.processing_time_ms:
                self.stats["total_processing_time"] += final_result.processing_time_ms
                self.stats["avg_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["emails_processed"]
                )

        except Exception as e:
            # Error event
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR_OCCURRED,
                session_id=session_id,
                user_id=user_id,
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time()
                }
            )

            yield error_event
            await self.event_bus.publish(error_event)

            logger.error(f"Error processing email stream: {e}")

        finally:
            # Clean up session
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "completed"
                self.active_sessions[session_id]["end_time"] = time.time()

    async def _process_queue(self) -> None:
        """Process queued emails in background."""
        while True:
            try:
                email_data = await self.processing_queue.get()

                # Process email with streaming
                async for event in self.process_email_stream(
                    email_data["content"],
                    email_data.get("headers"),
                    email_data.get("session_id"),
                    email_data.get("user_id")
                ):
                    # Events are already published in process_email_stream
                    pass

                self.processing_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processing: {e}")

    async def _update_stats(self) -> None:
        """Update processing statistics periodically."""
        while True:
            try:
                self.stats["queue_size"] = self.processing_queue.qsize()
                self.stats["active_sessions_count"] = len(
                    [s for s in self.active_sessions.values() if s["status"] == "processing"]
                )

                # Publish queue status update
                queue_status_event = StreamEvent(
                    event_type=StreamEventType.QUEUE_STATUS_UPDATE,
                    data={
                        "queue_size": self.stats["queue_size"],
                        "active_sessions": self.stats["active_sessions_count"],
                        "processing_stats": self.stats
                    }
                )

                await self.event_bus.publish(queue_status_event)
                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                await asyncio.sleep(30)

    async def queue_email(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Queue email for background processing."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        email_data = {
            "content": email_content,
            "headers": email_headers,
            "session_id": session_id,
            "user_id": user_id
        }

        await self.processing_queue.put(email_data)
        logger.info(f"Email queued for processing: session {session_id}")

        return session_id

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a processing session."""
        return self.active_sessions.get(session_id)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            **self.stats,
            "event_bus_stats": self.event_bus.get_stats(),
            "total_active_sessions": len(self.active_sessions),
            "completed_sessions": len([
                s for s in self.active_sessions.values() if s["status"] == "completed"
            ])
        }


# WebSocket integration helpers
class WebSocketEventHandler:
    """WebSocket handler for real-time events."""

    def __init__(self, websocket, user_id: str):
        self.websocket = websocket
        self.user_id = user_id
        self.subscription_id = None

    async def handle_event(self, event: StreamEvent) -> None:
        """Handle incoming event and send to WebSocket."""
        try:
            message = {
                "type": "event",
                "event": event.to_json()
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket event: {e}")

    def subscribe_to_events(
        self,
        event_bus: RealTimeEventBus,
        event_types: Optional[Set[StreamEventType]] = None
    ) -> str:
        """Subscribe to events through WebSocket."""
        if event_types is None:
            event_types = set(StreamEventType)

        self.subscription_id = event_bus.subscribe(
            event_types=event_types,
            callback=self.handle_event,
            filters={"user_id": self.user_id}
        )

        return self.subscription_id

    def unsubscribe(self, event_bus: RealTimeEventBus) -> None:
        """Unsubscribe from events."""
        if self.subscription_id:
            event_bus.unsubscribe(self.subscription_id)


# Global instances
_event_bus: Optional[RealTimeEventBus] = None
_streaming_processor: Optional[StreamingEmailProcessor] = None


def get_event_bus() -> RealTimeEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = RealTimeEventBus()
    return _event_bus


def get_streaming_processor() -> StreamingEmailProcessor:
    """Get the global streaming processor instance."""
    global _streaming_processor
    if _streaming_processor is None:
        _streaming_processor = StreamingEmailProcessor(get_event_bus())
    return _streaming_processor


# Convenience functions
async def stream_email_processing(
    content: str,
    headers: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> AsyncGenerator[StreamEvent, None]:
    """Stream email processing with real-time updates."""
    processor = get_streaming_processor()
    async for event in processor.process_email_stream(content, headers, session_id, user_id):
        yield event


async def subscribe_to_processing_updates(
    callback: Callable[[StreamEvent], None],
    event_types: Optional[Set[StreamEventType]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """Subscribe to real-time processing updates."""
    event_bus = get_event_bus()
    if event_types is None:
        event_types = {
            StreamEventType.PROCESSING_STARTED,
            StreamEventType.CLASSIFICATION_COMPLETE,
            StreamEventType.ANALYSIS_COMPLETE,
            StreamEventType.PROCESSING_COMPLETE,
            StreamEventType.ERROR_OCCURRED
        }

    return event_bus.subscribe(event_types, callback, filters)
