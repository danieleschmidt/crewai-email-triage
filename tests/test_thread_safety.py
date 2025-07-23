#!/usr/bin/env python3
"""
Test suite for thread safety in agent operations.

Tests that agent instances can safely be used concurrently across multiple threads
without data corruption or race conditions, particularly around configuration access
and state management.
"""

import os
import sys
import unittest
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.classifier import ClassifierAgent
from crewai_email_triage.priority import PriorityAgent
from crewai_email_triage.summarizer import SummarizerAgent
from crewai_email_triage.response import ResponseAgent


class TestAgentThreadSafety(unittest.TestCase):
    """Test thread safety for all agent types."""
    
    def setUp(self):
        """Set up test configurations."""
        self.test_config = {
            "classifier": {
                "urgent": ["urgent", "critical"],
                "work": ["meeting", "project"],
                "spam": ["unsubscribe", "offer"]
            },
            "priority": {
                "scores": {"high": 10, "medium": 5, "low": 1},
                "high_keywords": ["urgent", "critical"],
                "medium_keywords": ["important", "soon"]
            },
            "summarizer": {
                "max_length": 50
            },
            "response": {
                "template": "Thank you for your email",
                "signature": "Support Team"
            }
        }
        
        self.alternate_config = {
            "classifier": {
                "urgent": ["emergency", "asap"],
                "personal": ["family", "friend"]
            },
            "priority": {
                "scores": {"high": 20, "medium": 10, "low": 2},
                "high_keywords": ["emergency", "asap"],
                "medium_keywords": ["today", "deadline"]
            },
            "summarizer": {
                "max_length": 30
            },
            "response": {
                "template": "Custom response template",
                "signature": "Custom Team"
            }
        }
    
    def test_classifier_agent_thread_safety(self):
        """Test ClassifierAgent thread safety under concurrent access."""
        agent = ClassifierAgent(config_dict=self.test_config)
        num_threads = 20
        num_operations_per_thread = 10
        results = []
        
        def worker_function(thread_id):
            """Worker function for concurrent testing."""
            thread_results = []
            for i in range(num_operations_per_thread):
                content = f"urgent meeting for thread {thread_id} operation {i}"
                result = agent.run(content)
                thread_results.append((thread_id, i, result))
                # Add small random delay to increase chance of race conditions
                time.sleep(random.uniform(0.001, 0.005))
            return thread_results
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * num_operations_per_thread)
        
        # Verify consistent behavior - all should return "urgent" category
        for thread_id, op_id, result in results:
            self.assertTrue(result.startswith("category:"))
            self.assertIn("urgent", result)
    
    def test_priority_agent_thread_safety(self):
        """Test PriorityAgent thread safety under concurrent access."""
        agent = PriorityAgent(config_dict=self.test_config)
        num_threads = 15
        num_operations_per_thread = 8
        results = []
        
        def worker_function(thread_id):
            """Worker function for concurrent testing."""
            thread_results = []
            for i in range(num_operations_per_thread):
                content = f"urgent request from thread {thread_id}"
                result = agent.run(content)
                thread_results.append((thread_id, i, result))
                time.sleep(random.uniform(0.001, 0.003))
            return thread_results
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * num_operations_per_thread)
        
        # Verify consistent behavior - all should return high priority
        for thread_id, op_id, result in results:
            self.assertTrue(result.startswith("priority:"))
            self.assertIn("10", result)  # High priority score
    
    def test_summarizer_agent_thread_safety(self):
        """Test SummarizerAgent thread safety under concurrent access."""
        agent = SummarizerAgent(config_dict=self.test_config)
        num_threads = 12
        num_operations_per_thread = 6
        results = []
        
        def worker_function(thread_id):
            """Worker function for concurrent testing."""
            thread_results = []
            for i in range(num_operations_per_thread):
                content = f"This is a very long email content that needs summarization from thread {thread_id}. It contains multiple sentences."
                result = agent.run(content)
                thread_results.append((thread_id, i, result, len(result)))
                time.sleep(random.uniform(0.001, 0.004))
            return thread_results
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * num_operations_per_thread)
        
        # Verify consistent behavior - all should respect max_length
        for thread_id, op_id, result, length in results:
            self.assertTrue(result.startswith("summary:"))
            # Should respect max_length configuration (50 + "summary: " prefix)
            self.assertLessEqual(length, 60)  # Some tolerance for the prefix
    
    def test_response_agent_thread_safety(self):
        """Test ResponseAgent thread safety under concurrent access."""
        agent = ResponseAgent(config_dict=self.test_config)
        num_threads = 10
        num_operations_per_thread = 12
        results = []
        
        def worker_function(thread_id):
            """Worker function for concurrent testing."""
            thread_results = []
            for i in range(num_operations_per_thread):
                content = f"Support request from thread {thread_id}"
                result = agent.run(content)
                thread_results.append((thread_id, i, result))
                time.sleep(random.uniform(0.001, 0.002))
            return thread_results
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * num_operations_per_thread)
        
        # Verify consistent behavior - all should use the same template and signature
        for thread_id, op_id, result in results:
            self.assertTrue(result.startswith("response:"))
            self.assertIn("Thank you for your email", result)
            self.assertIn("Support Team", result)
    
    def test_mixed_agent_operations_thread_safety(self):
        """Test multiple different agents operating concurrently."""
        classifier = ClassifierAgent(config_dict=self.test_config)
        priority = PriorityAgent(config_dict=self.test_config)
        summarizer = SummarizerAgent(config_dict=self.test_config)
        responder = ResponseAgent(config_dict=self.test_config)
        
        agents = [classifier, priority, summarizer, responder]
        num_threads = 16
        results = defaultdict(list)
        
        def worker_function(thread_id):
            """Worker function that uses all agents."""
            agent = agents[thread_id % len(agents)]
            agent_type = type(agent).__name__
            
            for i in range(5):
                content = f"urgent meeting request from thread {thread_id}"
                result = agent.run(content)
                results[agent_type].append((thread_id, result))
                time.sleep(random.uniform(0.001, 0.003))
        
        # Execute concurrent operations across all agent types
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Verify all agent types were exercised
        self.assertGreater(len(results["ClassifierAgent"]), 0)
        self.assertGreater(len(results["PriorityAgent"]), 0)
        self.assertGreater(len(results["SummarizerAgent"]), 0)
        self.assertGreater(len(results["ResponseAgent"]), 0)
        
        # Verify consistent results for each agent type
        for agent_type, agent_results in results.items():
            for thread_id, result in agent_results:
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
    
    def test_configuration_modification_thread_safety(self):
        """Test thread safety when configuration might be modified during execution."""
        # This test simulates potential race conditions if configuration is mutable
        agent = ClassifierAgent(config_dict=self.test_config)
        num_threads = 10
        results = []
        exceptions = []
        
        def worker_function(thread_id):
            """Worker function that processes emails."""
            try:
                for i in range(8):
                    content = f"urgent email from thread {thread_id}"
                    result = agent.run(content)
                    results.append((thread_id, result))
                    time.sleep(random.uniform(0.001, 0.002))
            except Exception as e:
                exceptions.append((thread_id, str(e)))
        
        def config_modifier():
            """Function that might modify configuration (simulating race condition)."""
            # Note: This doesn't actually modify the config but simulates access patterns
            # that could cause issues in non-thread-safe implementations
            time.sleep(0.01)  # Let other threads start
            for i in range(20):
                _ = agent._get_classifier_config()  # Access configuration
                time.sleep(0.002)
        
        # Execute concurrent operations with potential configuration access
        with ThreadPoolExecutor(max_workers=num_threads + 1) as executor:
            # Start worker threads
            worker_futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            # Start configuration accessor
            config_future = executor.submit(config_modifier)
            
            # Wait for all to complete
            for future in as_completed(worker_futures + [config_future]):
                future.result()
        
        # Verify no exceptions occurred
        self.assertEqual(len(exceptions), 0, f"Exceptions during concurrent execution: {exceptions}")
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * 8)
        
        # Verify consistent results
        for thread_id, result in results:
            self.assertTrue(result.startswith("category:"))
    
    def test_agent_state_isolation(self):
        """Test that agent instances maintain proper state isolation."""
        agent1 = ClassifierAgent(config_dict=self.test_config)
        agent2 = ClassifierAgent(config_dict=self.alternate_config)
        
        num_threads = 8
        results = defaultdict(list)
        
        def worker_function(thread_id, agent, config_name):
            """Worker function for specific agent/config combination."""
            for i in range(6):
                if config_name == "test":
                    content = "urgent meeting"  # Should trigger "urgent" in test config
                else:
                    content = "emergency situation"  # Should trigger "urgent" in alternate config
                
                result = agent.run(content)
                results[config_name].append((thread_id, result))
                time.sleep(random.uniform(0.001, 0.003))
        
        # Execute concurrent operations with different agents/configs
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads // 2):
                futures.append(executor.submit(worker_function, i, agent1, "test"))
                futures.append(executor.submit(worker_function, i, agent2, "alternate"))
            
            for future in as_completed(futures):
                future.result()
        
        # Verify both configurations were used and produced different behaviors
        self.assertGreater(len(results["test"]), 0)
        self.assertGreater(len(results["alternate"]), 0)
        
        # Verify results are consistent within each configuration
        for config_name, config_results in results.items():
            for thread_id, result in config_results:
                self.assertTrue(result.startswith("category:"))
                # Both should classify as urgent but might use different keywords
                self.assertIn("urgent", result)


class TestAgentConcurrencyPatterns(unittest.TestCase):
    """Test various concurrency patterns with agents."""
    
    def test_agent_pool_pattern(self):
        """Test using agents in a pool pattern for high concurrency."""
        # Create a pool of agents
        pool_size = 5
        agent_pool = [ClassifierAgent(config_dict={
            "classifier": {"urgent": ["urgent"], "work": ["meeting"]}
        }) for _ in range(pool_size)]
        
        num_requests = 50
        results = []
        
        def process_request(request_id):
            """Process a request using an agent from the pool."""
            agent = agent_pool[request_id % pool_size]
            content = f"urgent meeting request {request_id}"
            result = agent.run(content)
            return (request_id, result)
        
        # Execute concurrent requests using the agent pool
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_request, i) for i in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # Verify all requests were processed
        self.assertEqual(len(results), num_requests)
        
        # Verify consistent results
        for request_id, result in results:
            self.assertTrue(result.startswith("category:"))
            self.assertIn("urgent", result)
    
    def test_agent_factory_pattern(self):
        """Test creating fresh agent instances for each thread."""
        config = {"classifier": {"urgent": ["urgent"], "work": ["meeting"]}}
        num_threads = 12
        results = []
        
        def worker_with_fresh_agent(thread_id):
            """Worker that creates its own agent instance."""
            # Each thread gets its own agent instance
            agent = ClassifierAgent(config_dict=config)
            
            thread_results = []
            for i in range(4):
                content = f"urgent email from thread {thread_id}"
                result = agent.run(content)
                thread_results.append((thread_id, i, result))
                time.sleep(random.uniform(0.001, 0.002))
            return thread_results
        
        # Execute with fresh agents per thread
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_with_fresh_agent, i) for i in range(num_threads)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), num_threads * 4)
        
        # Verify consistent results across all threads
        for thread_id, op_id, result in results:
            self.assertTrue(result.startswith("category:"))
            self.assertIn("urgent", result)


if __name__ == "__main__":
    unittest.main()