"""Auto-scaling system with load balancing and resource management."""

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

class ScalingDecision(Enum):
    """Auto-scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ResourceMetrics:
    """System resource metrics for scaling decisions."""
    cpu_percent: float
    memory_percent: float
    queue_size: int
    active_workers: int
    throughput_per_second: float
    response_time_avg_ms: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system."""
    min_workers: int = 1
    max_workers: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_queue_size: int = 100
    target_response_time_ms: float = 500.0
    scale_up_threshold: float = 0.8  # Scale up if metrics exceed 80% of target
    scale_down_threshold: float = 0.4  # Scale down if metrics below 40% of target
    cooldown_period_seconds: int = 60  # Minimum time between scaling actions

class WorkerPool:
    """Dynamic worker pool with auto-scaling capabilities."""

    def __init__(self, config: ScalingConfig, worker_factory: Callable):
        self.config = config
        self.worker_factory = worker_factory
        self._workers = []
        self._task_queue = Queue()
        self._result_queue = Queue()
        self._active_tasks = 0
        self._lock = threading.RLock()
        self._last_scaling_action = 0
        self._metrics_history = []

        # Initialize with minimum workers
        self._initialize_workers()

        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _initialize_workers(self):
        """Initialize minimum number of workers."""
        for _ in range(self.config.min_workers):
            self._add_worker()

    def _add_worker(self):
        """Add a new worker to the pool."""
        worker_id = len(self._workers)
        worker = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            daemon=True
        )
        worker.start()
        self._workers.append(worker)
        logger.info(f"Added worker {worker_id}, total workers: {len(self._workers)}")

    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        while self._monitoring:
            try:
                task, args, kwargs, result_callback = self._task_queue.get(timeout=1.0)

                with self._lock:
                    self._active_tasks += 1

                try:
                    start_time = time.time()
                    result = task(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    if result_callback:
                        result_callback(result, duration_ms, None)

                except Exception as e:
                    logger.error(f"Worker {worker_id} task failed: {e}")
                    if result_callback:
                        result_callback(None, 0, e)

                finally:
                    with self._lock:
                        self._active_tasks -= 1
                        self._task_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def submit_task(self, task: Callable, *args, **kwargs) -> None:
        """Submit task to worker pool."""
        self._task_queue.put((task, args, kwargs, None))

    def submit_task_with_callback(self, task: Callable, callback: Callable, *args, **kwargs):
        """Submit task with result callback."""
        self._task_queue.put((task, args, kwargs, callback))

    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_percent = (process.memory_info().rss / psutil.virtual_memory().total) * 100
        except:
            cpu_percent = 0.0
            memory_percent = 0.0

        # Calculate recent throughput
        recent_metrics = self._metrics_history[-60:]  # Last minute
        throughput = len(recent_metrics) / 60.0 if recent_metrics else 0.0

        # Calculate average response time
        if recent_metrics:
            response_times = [m.response_time_avg_ms for m in recent_metrics if m.response_time_avg_ms > 0]
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
        else:
            avg_response_time = 0.0

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            queue_size=self._task_queue.qsize(),
            active_workers=len([w for w in self._workers if w.is_alive()]),
            throughput_per_second=throughput,
            response_time_avg_ms=avg_response_time
        )

    def _should_scale_up(self, metrics: ResourceMetrics) -> bool:
        """Determine if we should scale up."""
        if len(self._workers) >= self.config.max_workers:
            return False

        # Check if any metric exceeds scale-up threshold
        cpu_ratio = metrics.cpu_percent / self.config.target_cpu_percent
        memory_ratio = metrics.memory_percent / self.config.target_memory_percent
        queue_ratio = metrics.queue_size / max(self.config.target_queue_size, 1)
        response_ratio = metrics.response_time_avg_ms / max(self.config.target_response_time_ms, 1)

        return (cpu_ratio > self.config.scale_up_threshold or
                memory_ratio > self.config.scale_up_threshold or
                queue_ratio > self.config.scale_up_threshold or
                response_ratio > self.config.scale_up_threshold)

    def _should_scale_down(self, metrics: ResourceMetrics) -> bool:
        """Determine if we should scale down."""
        if len(self._workers) <= self.config.min_workers:
            return False

        # Only scale down if all metrics are below scale-down threshold
        cpu_ratio = metrics.cpu_percent / self.config.target_cpu_percent
        memory_ratio = metrics.memory_percent / self.config.target_memory_percent
        queue_ratio = metrics.queue_size / max(self.config.target_queue_size, 1)
        response_ratio = metrics.response_time_avg_ms / max(self.config.target_response_time_ms, 1)

        return (cpu_ratio < self.config.scale_down_threshold and
                memory_ratio < self.config.scale_down_threshold and
                queue_ratio < self.config.scale_down_threshold and
                response_ratio < self.config.scale_down_threshold)

    def _make_scaling_decision(self, metrics: ResourceMetrics) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        current_time = time.time()

        # Respect cooldown period
        if current_time - self._last_scaling_action < self.config.cooldown_period_seconds:
            return ScalingDecision.MAINTAIN

        if self._should_scale_up(metrics):
            return ScalingDecision.SCALE_UP
        elif self._should_scale_down(metrics):
            return ScalingDecision.SCALE_DOWN
        else:
            return ScalingDecision.MAINTAIN

    def _execute_scaling_decision(self, decision: ScalingDecision, metrics: ResourceMetrics):
        """Execute scaling decision."""
        if decision == ScalingDecision.SCALE_UP:
            self._add_worker()
            self._last_scaling_action = time.time()
            logger.info(f"Scaled up: {len(self._workers)} workers (CPU: {metrics.cpu_percent:.1f}%, "
                       f"Queue: {metrics.queue_size}, Response: {metrics.response_time_avg_ms:.1f}ms)")

        elif decision == ScalingDecision.SCALE_DOWN:
            # Remove a worker by reducing the worker count
            # Workers will naturally exit when they finish current tasks
            if len(self._workers) > self.config.min_workers:
                self._workers.pop()
                self._last_scaling_action = time.time()
                logger.info(f"Scaled down: {len(self._workers)} workers")

    def _monitoring_loop(self):
        """Main monitoring and auto-scaling loop."""
        while self._monitoring:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()

                # Store metrics history
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > 300:  # Keep 5 minutes of history
                    self._metrics_history.pop(0)

                # Make scaling decision
                decision = self._make_scaling_decision(metrics)

                # Execute scaling decision
                if decision != ScalingDecision.MAINTAIN:
                    self._execute_scaling_decision(decision, metrics)

                # Sleep before next check
                time.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get current scaling statistics."""
        current_metrics = self.get_current_metrics()

        return {
            "current_workers": len(self._workers),
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "active_tasks": self._active_tasks,
            "queue_size": self._task_queue.qsize(),
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "throughput_per_second": current_metrics.throughput_per_second,
                "response_time_avg_ms": current_metrics.response_time_avg_ms
            },
            "last_scaling_action": self._last_scaling_action,
            "metrics_history_count": len(self._metrics_history)
        }

    def shutdown(self):
        """Shutdown worker pool."""
        self._monitoring = False

        # Wait for tasks to complete
        self._task_queue.join()

        # Stop monitoring thread
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        logger.info("Worker pool shutdown complete")

class LoadBalancer:
    """Simple load balancer for distributing work across multiple processors."""

    def __init__(self, processors: List[Callable]):
        self.processors = processors
        self.current_index = 0
        self._lock = threading.Lock()
        self._processor_stats = {i: {"requests": 0, "avg_time": 0.0} for i in range(len(processors))}

    def get_next_processor(self) -> Callable:
        """Get next processor using round-robin."""
        with self._lock:
            processor = self.processors[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.processors)
            return processor

    def process_with_load_balancing(self, task_data: Any) -> Any:
        """Process task with load balancing."""
        processor = self.get_next_processor()

        start_time = time.time()
        try:
            result = processor(task_data)
            duration = time.time() - start_time

            # Update processor stats
            processor_idx = self.processors.index(processor)
            stats = self._processor_stats[processor_idx]
            stats["requests"] += 1
            stats["avg_time"] = (stats["avg_time"] * (stats["requests"] - 1) + duration) / stats["requests"]

            return result
        except Exception as e:
            logger.error(f"Load balanced processing failed: {e}")
            raise

    def get_load_stats(self) -> Dict[str, Any]:
        """Get current load balancing statistics."""
        return {
            "total_processors": len(self.processors),
            "processor_stats": self._processor_stats.copy(),
            "current_index": self.current_index
        }

# Factory function for creating auto-scaling worker pools
def create_auto_scaling_pool(worker_factory: Callable,
                           min_workers: int = 1,
                           max_workers: int = 10,
                           **config_kwargs) -> WorkerPool:
    """Create auto-scaling worker pool."""
    config = ScalingConfig(
        min_workers=min_workers,
        max_workers=max_workers,
        **config_kwargs
    )
    return WorkerPool(config, worker_factory)
