"""
Performance Testing Plugin
Provides CLI commands for testing plugin performance and scaling capabilities.
"""

import time
import random
from typing import Any, Dict

from crewai_email_triage.plugin_architecture import CLICommandPlugin, PluginMetadata, PluginConfig


class PerformanceBenchmarkCommand(CLICommandPlugin):
    """CLI command to benchmark plugin performance."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="performance_benchmark_command",
            version="1.0.0",
            description="Benchmark plugin performance with various workloads",
            author="CrewAI Team"
        )
    
    def initialize(self) -> bool:
        """Initialize the command."""
        self.logger.info("Initializing performance benchmark command")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def get_command_name(self) -> str:
        """Return command name."""
        return "benchmark-plugins"
    
    def get_command_help(self) -> str:
        """Return command help text."""
        return "Run performance benchmarks on plugin system with concurrent and caching tests"
    
    def add_arguments(self, parser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--iterations", 
            type=int, 
            default=100,
            help="Number of benchmark iterations (default: 100)"
        )
        parser.add_argument(
            "--concurrent", 
            action="store_true",
            help="Test concurrent processing performance"
        )
        parser.add_argument(
            "--cache-test",
            action="store_true", 
            help="Test caching effectiveness"
        )
        parser.add_argument(
            "--load-test",
            action="store_true",
            help="Run load testing with multiple workers"
        )
    
    def execute_command(self, args) -> Any:
        """Execute the benchmark command."""
        try:
            from crewai_email_triage.plugin_scaling import ScalablePluginManager
            
            # Initialize scalable plugin manager
            plugin_manager = ScalablePluginManager(max_workers=4)
            
            # Generate test data
            test_emails = self._generate_test_emails(args.iterations)
            
            results = {
                'benchmark_config': {
                    'iterations': args.iterations,
                    'concurrent_enabled': args.concurrent,
                    'cache_test_enabled': args.cache_test,
                    'load_test_enabled': args.load_test
                },
                'tests': {}
            }
            
            # Sequential processing benchmark
            self.logger.info("Running sequential processing benchmark...")
            sequential_results = self._benchmark_sequential(plugin_manager, test_emails)
            results['tests']['sequential'] = sequential_results
            
            # Concurrent processing benchmark
            if args.concurrent:
                self.logger.info("Running concurrent processing benchmark...")
                concurrent_results = self._benchmark_concurrent(plugin_manager, test_emails)
                results['tests']['concurrent'] = concurrent_results
            
            # Cache effectiveness test
            if args.cache_test:
                self.logger.info("Running cache effectiveness test...")
                cache_results = self._test_cache_effectiveness(plugin_manager, test_emails[:10])
                results['tests']['cache'] = cache_results
            
            # Load testing
            if args.load_test:
                self.logger.info("Running load test...")
                load_results = self._load_test(plugin_manager, test_emails)
                results['tests']['load'] = load_results
            
            # Generate performance summary
            results['performance_summary'] = plugin_manager.get_performance_summary()
            results['optimization_recommendations'] = plugin_manager.optimize_configuration()
            
            return results
            
        except Exception as e:
            return {'error': f'Benchmark failed: {e}'}
    
    def _generate_test_emails(self, count: int) -> list:
        """Generate test email data."""
        email_templates = [
            "Urgent meeting request for tomorrow at 2 PM. Please confirm your availability.",
            "Thank you for the excellent presentation. I have a few follow-up questions.",
            "The project deadline has been moved to next Friday. Please adjust your timeline.",
            "Congratulations on the successful product launch! Outstanding work from the team.",
            "Please review the attached document and provide your feedback by end of week.",
            "System maintenance scheduled for this weekend. Expect temporary downtime.",
            "Welcome to the company! We're excited to have you join our team.",
            "Budget approval needed for the Q4 marketing campaign. Details attached.",
            "Performance review meeting scheduled for next Tuesday at 10 AM.",
            "Client feedback on the latest proposal is very positive. Great job!"
        ]
        
        emotions = ["excited", "concerned", "urgent", "pleased", "disappointed", "grateful"]
        priorities = ["high", "medium", "low", "critical"]
        
        test_emails = []
        for i in range(count):
            base_email = random.choice(email_templates)
            emotion = random.choice(emotions)
            priority = random.choice(priorities)
            
            # Add some variation
            enhanced_email = f"I'm {emotion} to inform you that {base_email.lower()}"
            
            test_emails.append({
                'content': enhanced_email,
                'metadata': {
                    'priority': priority,
                    'from': f'user{i % 10}@company.com',
                    'subject': f'Test Email {i}',
                    'timestamp': time.time()
                }
            })
        
        return test_emails
    
    def _benchmark_sequential(self, plugin_manager, test_emails: list) -> Dict[str, Any]:
        """Benchmark sequential processing."""
        start_time = time.time()
        results = []
        
        for email_data in test_emails:
            result = plugin_manager.process_email_scaled(
                email_data['content'],
                email_data['metadata'],
                concurrent=False
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            'total_time_seconds': round(total_time, 3),
            'avg_time_per_email_ms': round((total_time / len(test_emails)) * 1000, 2),
            'throughput_emails_per_second': round(len(test_emails) / total_time, 2),
            'total_emails_processed': len(test_emails)
        }
    
    def _benchmark_concurrent(self, plugin_manager, test_emails: list) -> Dict[str, Any]:
        """Benchmark concurrent processing.""" 
        start_time = time.time()
        results = []
        
        for email_data in test_emails:
            result = plugin_manager.process_email_scaled(
                email_data['content'],
                email_data['metadata'], 
                concurrent=True
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            'total_time_seconds': round(total_time, 3),
            'avg_time_per_email_ms': round((total_time / len(test_emails)) * 1000, 2),
            'throughput_emails_per_second': round(len(test_emails) / total_time, 2),
            'total_emails_processed': len(test_emails),
            'concurrent_processing': True
        }
    
    def _test_cache_effectiveness(self, plugin_manager, test_emails: list) -> Dict[str, Any]:
        """Test cache effectiveness."""
        # Clear cache first
        plugin_manager.cache.clear_cache()
        
        # First pass - populate cache
        start_time = time.time()
        for email_data in test_emails:
            plugin_manager.process_email_scaled(
                email_data['content'],
                email_data['metadata'],
                concurrent=False
            )
        first_pass_time = time.time() - start_time
        
        # Second pass - should use cache
        start_time = time.time()
        for email_data in test_emails:
            plugin_manager.process_email_scaled(
                email_data['content'],
                email_data['metadata'],
                concurrent=False
            )
        second_pass_time = time.time() - start_time
        
        cache_stats = plugin_manager.cache.get_cache_stats()
        
        return {
            'first_pass_time_seconds': round(first_pass_time, 3),
            'second_pass_time_seconds': round(second_pass_time, 3),
            'speedup_factor': round(first_pass_time / second_pass_time, 2) if second_pass_time > 0 else 0,
            'cache_statistics': cache_stats
        }
    
    def _load_test(self, plugin_manager, test_emails: list) -> Dict[str, Any]:
        """Perform load testing."""
        import threading
        import concurrent.futures
        
        # Test with different numbers of concurrent threads
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for thread_count in thread_counts:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                
                # Submit all emails for processing
                for email_data in test_emails:
                    future = executor.submit(
                        plugin_manager.process_email_scaled,
                        email_data['content'],
                        email_data['metadata'],
                        True
                    )
                    futures.append(future)
                
                # Wait for all to complete
                completed_count = 0
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        completed_count += 1
                    except Exception as e:
                        self.logger.error(f"Load test task failed: {e}")
            
            total_time = time.time() - start_time
            
            results[f'{thread_count}_threads'] = {
                'total_time_seconds': round(total_time, 3),
                'completed_emails': completed_count,
                'throughput_emails_per_second': round(completed_count / total_time, 2) if total_time > 0 else 0,
                'thread_count': thread_count
            }
        
        return results


class CacheManagementCommand(CLICommandPlugin):
    """CLI command for cache management operations."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="cache_management_command",
            version="1.0.0", 
            description="Manage plugin result caches",
            author="CrewAI Team"
        )
    
    def initialize(self) -> bool:
        """Initialize the command."""
        self.logger.info("Initializing cache management command")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def get_command_name(self) -> str:
        """Return command name."""
        return "cache-management"
    
    def get_command_help(self) -> str:
        """Return command help text."""
        return "Manage plugin caches - view stats, clear cache, optimize settings"
    
    def add_arguments(self, parser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--stats", 
            action="store_true",
            help="Show cache statistics"
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear all caches"
        )
        parser.add_argument(
            "--optimize",
            action="store_true", 
            help="Optimize cache configuration"
        )
    
    def execute_command(self, args) -> Any:
        """Execute the cache management command."""
        try:
            from crewai_email_triage.plugin_scaling import ScalablePluginManager
            
            plugin_manager = ScalablePluginManager()
            
            result = {}
            
            if args.stats:
                cache_stats = plugin_manager.cache.get_cache_stats()
                result['cache_statistics'] = cache_stats
            
            if args.clear:
                plugin_manager.cache.clear_cache()
                result['cache_cleared'] = True
                
            if args.optimize:
                optimization = plugin_manager.optimize_configuration()
                result['optimization_report'] = optimization
            
            if not any([args.stats, args.clear, args.optimize]):
                result = plugin_manager.cache.get_cache_stats()
            
            return result
            
        except Exception as e:
            return {'error': f'Cache management failed: {e}'}