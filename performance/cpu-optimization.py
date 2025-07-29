#!/usr/bin/env python3
"""
Advanced CPU optimization utilities for email triage processing.
Leverages Python 3.11+ features and modern performance patterns.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

import psutil

F = TypeVar("F", bound=Callable[..., Any])

class CPUOptimizer:
    """Advanced CPU optimization for email processing workloads."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.process_pool = None
        self.thread_pool = None
        
    @staticmethod
    def cpu_bound_task_optimizer(func: F) -> F:
        """Decorator for CPU-intensive email processing tasks."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                return await loop.run_in_executor(executor, func, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def io_bound_task_optimizer(max_workers: int = 20) -> Callable:
        """Decorator for I/O-intensive email operations."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    return await loop.run_in_executor(executor, func, *args, **kwargs)
            return wrapper
        return decorator

    @lru_cache(maxsize=1024)
    def optimize_batch_size(self, workload_type: str) -> int:
        """Calculate optimal batch size based on CPU and memory characteristics."""
        base_batch_sizes = {
            "classification": 32,
            "summarization": 16,
            "response_generation": 8
        }
        
        cpu_multiplier = min(self.cpu_count / 4, 2.0)
        return int(base_batch_sizes.get(workload_type, 16) * cpu_multiplier)