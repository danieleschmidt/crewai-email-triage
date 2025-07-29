#!/usr/bin/env python3
"""
Modern async patterns for Python 3.11+ email processing.
Leverages TaskGroups, ExceptionGroups, and advanced async features.
"""

import asyncio
from asyncio import TaskGroup, timeout
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator, Self
from collections.abc import AsyncIterable
import logging

@dataclass
class EmailProcessingResult:
    """Modern dataclass with enhanced type hints."""
    email_id: str
    classification: str
    summary: str | None = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    
    def with_summary(self, summary: str) -> Self:
        """Method chaining with Self type."""
        return EmailProcessingResult(
            email_id=self.email_id,
            classification=self.classification,
            summary=summary,
            confidence=self.confidence,
            processing_time_ms=self.processing_time_ms
        )

class ModernEmailProcessor:
    """Advanced email processor using Python 3.11+ async patterns."""
    
    def __init__(self, max_concurrent: int = 10, timeout_seconds: int = 30):
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    @asynccontextmanager
    async def processing_context(self):
        """Async context manager for resource management."""
        async with self.semaphore:
            start_time = asyncio.get_event_loop().time()
            try:
                yield
            finally:
                end_time = asyncio.get_event_loop().time()
                logging.info(f"Processing completed in {end_time - start_time:.2f}s")
    
    async def process_emails_concurrent(
        self, 
        emails: list[dict]
    ) -> tuple[list[EmailProcessingResult], list[Exception]]:
        """Process emails using TaskGroup with comprehensive error handling."""
        successful_results = []
        processing_errors = []
        
        try:
            async with TaskGroup() as tg:
                # Create tasks for all emails
                tasks = [
                    tg.create_task(
                        self._process_single_email_with_timeout(email),
                        name=f"email_{email.get('id', 'unknown')}"
                    )
                    for email in emails
                ]
                
        except* asyncio.TimeoutError as timeout_exceptions:
            # Handle all timeout errors collectively
            for exc in timeout_exceptions.exceptions:
                processing_errors.append(exc)
                logging.error(f"Email processing timeout: {exc}")
                
        except* ValueError as validation_exceptions:
            # Handle all validation errors collectively
            for exc in validation_exceptions.exceptions:
                processing_errors.append(exc)
                logging.error(f"Email validation error: {exc}")
                
        except* Exception as other_exceptions:
            # Handle any other exceptions
            for exc in other_exceptions.exceptions:
                processing_errors.append(exc)
                logging.error(f"Unexpected processing error: {exc}")
        
        # Collect successful results
        for task in tasks:
            if not task.cancelled() and task.done() and not task.exception():
                successful_results.append(task.result())
        
        return successful_results, processing_errors
    
    async def _process_single_email_with_timeout(
        self, 
        email: dict
    ) -> EmailProcessingResult:
        """Process single email with timeout and resource management."""
        async with self.processing_context():
            async with timeout(self.timeout_seconds):
                return await self._process_single_email(email)
    
    async def _process_single_email(self, email: dict) -> EmailProcessingResult:
        """Core email processing logic."""
        # Simulate async processing
        await asyncio.sleep(0.1)  # Simulate I/O-bound operation
        
        return EmailProcessingResult(
            email_id=email.get('id', 'unknown'),
            classification=await self._classify_email(email),
            confidence=0.95,
            processing_time_ms=100
        )
    
    async def _classify_email(self, email: dict) -> str:
        """Mock email classification."""
        await asyncio.sleep(0.05)  # Simulate ML inference
        
        # Pattern matching for classification
        match email:
            case {"subject": subject} if "urgent" in subject.lower():
                return "urgent"
            case {"sender": sender} if sender.endswith("@company.com"):
                return "internal"
            case {"attachments": attachments} if len(attachments) > 0:
                return "with_attachments"
            case _:
                return "standard"
    
    async def stream_process_emails(
        self, 
        email_stream: AsyncIterable[dict]
    ) -> AsyncGenerator[EmailProcessingResult, None]:
        """Stream processing with async generators."""
        batch = []
        batch_size = 5
        
        async for email in email_stream:
            batch.append(email)
            
            if len(batch) >= batch_size:
                # Process batch and yield results
                results, errors = await self.process_emails_concurrent(batch)
                
                for result in results:
                    yield result
                
                # Log errors but continue processing
                for error in errors:
                    logging.error(f"Batch processing error: {error}")
                
                batch.clear()
        
        # Process remaining emails in batch
        if batch:
            results, errors = await self.process_emails_concurrent(batch)
            for result in results:
                yield result

class AsyncEmailQueue:
    """Modern async queue implementation with backpressure handling."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
        self.processing_complete = asyncio.Event()
        
    async def producer(self, emails: list[dict]) -> None:
        """Produce emails to the queue with backpressure handling."""
        try:
            for email in emails:
                await self.queue.put(email)
                logging.debug(f"Queued email {email.get('id')}")
                
        except asyncio.QueueFull:
            logging.warning("Queue is full, applying backpressure")
            await asyncio.sleep(1)  # Backpressure delay
            
        finally:
            await self.queue.put(None)  # Sentinel value to signal completion
    
    async def consumer(
        self, 
        processor: ModernEmailProcessor
    ) -> AsyncIterator[EmailProcessingResult]:
        """Consume emails from queue and process them."""
        while True:
            email = await self.queue.get()
            
            if email is None:  # Sentinel value indicates completion
                self.processing_complete.set()
                break
                
            try:
                result = await processor._process_single_email(email)
                yield result
                
            except Exception as e:
                logging.error(f"Failed to process email {email.get('id')}: {e}")
                
            finally:
                self.queue.task_done()

# Example usage demonstrating modern patterns
async def main_processing_example():
    """Demonstrate modern async email processing patterns."""
    processor = ModernEmailProcessor(max_concurrent=5)
    
    # Sample email data
    sample_emails = [
        {"id": f"email_{i}", "subject": f"Test Subject {i}", "sender": "test@example.com"}
        for i in range(20)
    ]
    
    # Method 1: Concurrent processing with TaskGroup
    print("Processing with TaskGroup...")
    results, errors = await processor.process_emails_concurrent(sample_emails)
    print(f"Processed {len(results)} emails, {len(errors)} errors")
    
    # Method 2: Stream processing
    print("Stream processing...")
    async def email_stream():
        for email in sample_emails:
            yield email
            await asyncio.sleep(0.01)  # Simulate streaming delay
    
    async for result in processor.stream_process_emails(email_stream()):
        print(f"Streamed result: {result.email_id} -> {result.classification}")
    
    # Method 3: Queue-based processing
    print("Queue-based processing...")
    queue = AsyncEmailQueue(maxsize=10)
    
    # Start producer and consumer concurrently
    async with TaskGroup() as tg:
        tg.create_task(queue.producer(sample_emails[:10]))
        
        # Consume and collect results
        consumer_task = tg.create_task(
            collect_results(queue.consumer(processor))
        )
    
    await queue.processing_complete.wait()

async def collect_results(consumer: AsyncIterator[EmailProcessingResult]) -> list:
    """Helper to collect all results from async iterator."""
    results = []
    async for result in consumer:
        results.append(result)
    return results

if __name__ == "__main__":
    asyncio.run(main_processing_example())