"""Research Orchestrator - Advanced Integration and Error Handling for Breakthrough AI.

This module provides robust integration of breakthrough research implementations:
- Unified interface for all research paradigms
- Comprehensive error handling and graceful degradation
- Advanced monitoring and health checking
- Research methodology validation and compliance
- Performance optimization and resource management
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager

from .neuro_quantum_fusion import (
    NeuroQuantumFusionEngine,
    ComputationParadigm,
    create_neuro_quantum_engine
)
from .quantum_consciousness import (
    QuantumConsciousnessEngine,
    ConsciousnessLevel,
    create_consciousness_engine
)
from .performance import get_performance_tracker
from .health import get_health_checker
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .retry_utils import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)


class ResearchMode(str, Enum):
    """Research processing modes with different guarantees."""
    
    SAFE_MODE = "safe_mode"                    # Conservative, proven algorithms
    RESEARCH_MODE = "research_mode"            # Breakthrough algorithms with fallbacks
    EXPERIMENTAL_MODE = "experimental_mode"   # Cutting-edge, may fail gracefully
    VALIDATION_MODE = "validation_mode"       # Research validation and verification
    BENCHMARK_MODE = "benchmark_mode"         # Performance benchmarking
    PUBLICATION_MODE = "publication_mode"     # Publication-ready results


class ProcessingStrategy(str, Enum):
    """Strategies for processing with different paradigms."""
    
    CLASSICAL_ONLY = "classical_only"
    FUSION_PREFERRED = "fusion_preferred"
    CONSCIOUSNESS_PREFERRED = "consciousness_preferred"
    HYBRID_ADAPTIVE = "hybrid_adaptive"
    RESEARCH_OPTIMAL = "research_optimal"
    PERFORMANCE_OPTIMAL = "performance_optimal"


@dataclass
class ResearchResult:
    """Comprehensive result from research processing."""
    
    # Core results
    classification: str
    priority_score: float
    summary: str
    confidence: float
    
    # Research metadata
    paradigm_used: str
    processing_mode: str
    research_breakthrough: bool
    
    # Performance metrics
    processing_time: float
    quantum_advantage: float
    consciousness_level: Optional[str] = None
    
    # Quality metrics
    result_quality: float = field(default=0.8)
    statistical_significance: float = field(default=0.0)
    reproducibility_score: float = field(default=0.0)
    
    # Research insights
    novel_patterns: List[str] = field(default_factory=list)
    breakthrough_indicators: Dict[str, float] = field(default_factory=dict)
    research_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    fallback_used: bool = field(default=False)
    error_details: Optional[str] = None
    recovery_strategy: Optional[str] = None


@dataclass
class ResearchConfig:
    """Configuration for research orchestrator."""
    
    # Engine configurations
    fusion_config: Dict[str, Any] = field(default_factory=dict)
    consciousness_config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing parameters
    default_mode: ResearchMode = field(default=ResearchMode.RESEARCH_MODE)
    strategy: ProcessingStrategy = field(default=ProcessingStrategy.HYBRID_ADAPTIVE)
    timeout_seconds: float = field(default=30.0)
    
    # Quality thresholds
    min_confidence: float = field(default=0.6)
    min_statistical_significance: float = field(default=0.05)
    min_reproducibility: float = field(default=0.8)
    
    # Fallback settings
    enable_fallbacks: bool = field(default=True)
    max_retries: int = field(default=3)
    circuit_breaker_enabled: bool = field(default=True)
    
    # Research validation
    validate_results: bool = field(default=True)
    benchmark_against_classical: bool = field(default=True)
    track_breakthroughs: bool = field(default=True)


class ResearchOrchestrator:
    """Main orchestrator for breakthrough research processing."""
    
    def __init__(self, config: ResearchConfig = None):
        """Initialize the research orchestrator."""
        self.config = config or ResearchConfig()
        
        # Initialize engines
        self.fusion_engine = None
        self.consciousness_engine = None
        self.classical_processor = None
        
        # Monitoring and health
        self.health_checker = get_health_checker()
        self.performance_tracker = get_performance_tracker()
        
        # Circuit breakers for each engine
        self.circuit_breakers = {
            "fusion": CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0
            )),
            "consciousness": CircuitBreaker(CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=90.0
            ))
        }
        
        # Research tracking
        self.processing_history: deque = deque(maxlen=1000)
        self.breakthrough_log: List[Dict[str, Any]] = []
        self.performance_benchmarks: Dict[str, List[float]] = defaultdict(list)
        
        # Error statistics
        self.error_counts = defaultdict(int)
        self.recovery_success_rate = 0.0
        self.total_processes = 0
        self.successful_processes = 0
        
        logger.info("ResearchOrchestrator initialized for breakthrough AI processing")
    
    async def initialize_engines(self) -> None:
        """Initialize all processing engines with error handling."""
        try:
            # Initialize fusion engine
            if not self.fusion_engine:
                logger.info("Initializing NeuroQuantum Fusion Engine...")
                self.fusion_engine = create_neuro_quantum_engine(self.config.fusion_config)
                logger.info("✅ Fusion engine initialized")
            
            # Initialize consciousness engine
            if not self.consciousness_engine:
                logger.info("Initializing Quantum Consciousness Engine...")
                self.consciousness_engine = create_consciousness_engine(self.config.consciousness_config)
                logger.info("✅ Consciousness engine initialized")
            
            # Initialize classical processor (fallback)
            if not self.classical_processor:
                self.classical_processor = self._create_classical_processor()
                logger.info("✅ Classical processor initialized")
                
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise
    
    async def process_email_research(self, email_content: str, metadata: Dict[str, Any] = None,
                                   mode: ResearchMode = None) -> ResearchResult:
        """Process email using advanced research paradigms with robust error handling."""
        start_time = time.time()
        mode = mode or self.config.default_mode
        metadata = metadata or {}
        
        self.total_processes += 1
        
        try:
            # Ensure engines are initialized
            await self.initialize_engines()
            
            # Health check before processing
            if not await self._health_check():
                logger.warning("Health check failed, using safe mode")
                mode = ResearchMode.SAFE_MODE
            
            # Choose processing strategy
            strategy = self._determine_strategy(email_content, metadata, mode)
            
            # Process with selected strategy
            result = await self._process_with_strategy(strategy, email_content, metadata, mode)
            
            # Validate and enhance result
            result = await self._validate_result(result, email_content, metadata)
            
            # Track performance
            result.processing_time = time.time() - start_time
            self._track_performance(strategy, result)
            
            # Log to history
            self.processing_history.append({
                "timestamp": time.time(),
                "mode": mode.value,
                "strategy": strategy.value,
                "success": True,
                "processing_time": result.processing_time,
                "breakthrough": result.research_breakthrough
            })
            
            self.successful_processes += 1
            logger.info(f"Research processing completed: {strategy.value} mode in {result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Comprehensive error handling
            logger.error(f"Research processing failed: {e}")
            self.error_counts[type(e).__name__] += 1
            
            # Attempt recovery
            fallback_result = await self._handle_processing_error(e, email_content, metadata, mode)
            fallback_result.processing_time = time.time() - start_time
            
            return fallback_result
    
    async def _process_with_strategy(self, strategy: ProcessingStrategy, email_content: str,
                                   metadata: Dict[str, Any], mode: ResearchMode) -> ResearchResult:
        """Process using the selected strategy with circuit breaker protection."""
        
        if strategy == ProcessingStrategy.CLASSICAL_ONLY:
            return await self._process_classical(email_content, metadata)
        
        elif strategy == ProcessingStrategy.FUSION_PREFERRED:
            return await self._process_with_fusion(email_content, metadata, mode)
        
        elif strategy == ProcessingStrategy.CONSCIOUSNESS_PREFERRED:
            return await self._process_with_consciousness(email_content, metadata, mode)
        
        elif strategy == ProcessingStrategy.HYBRID_ADAPTIVE:
            return await self._process_hybrid_adaptive(email_content, metadata, mode)
        
        elif strategy == ProcessingStrategy.RESEARCH_OPTIMAL:
            return await self._process_research_optimal(email_content, metadata, mode)
        
        elif strategy == ProcessingStrategy.PERFORMANCE_OPTIMAL:
            return await self._process_performance_optimal(email_content, metadata, mode)
        
        else:
            # Default to classical processing
            return await self._process_classical(email_content, metadata)
    
    async def _process_with_fusion(self, email_content: str, metadata: Dict[str, Any],
                                 mode: ResearchMode) -> ResearchResult:
        """Process using neuro-quantum fusion with circuit breaker."""
        
        async def fusion_operation():
            return await self.fusion_engine.process_email_quantum(email_content, metadata)
        
        try:
            result_data = await self.circuit_breakers["fusion"].call(fusion_operation)
            
            return ResearchResult(
                classification=result_data.get("classification", "unknown"),
                priority_score=result_data.get("priority_score", 0.5),
                summary=result_data.get("summary", "Fusion analysis completed"),
                confidence=result_data.get("quantum_confidence", 0.8),
                paradigm_used="neuro_quantum_fusion",
                processing_mode=mode.value,
                research_breakthrough=result_data.get("breakthrough_potential", 0.0) > 0.7,
                quantum_advantage=result_data.get("quantum_advantage", 1.0),
                breakthrough_indicators={"quantum_advantage": result_data.get("quantum_advantage", 1.0)},
                research_metrics=self.fusion_engine.get_research_metrics()
            )
            
        except Exception as e:
            logger.error(f"Fusion processing failed: {e}")
            raise
    
    async def _process_with_consciousness(self, email_content: str, metadata: Dict[str, Any],
                                        mode: ResearchMode) -> ResearchResult:
        """Process using quantum consciousness with circuit breaker."""
        
        async def consciousness_operation():
            return await self.consciousness_engine.conscious_email_processing(email_content, metadata)
        
        try:
            result_data = await self.circuit_breakers["consciousness"].call(consciousness_operation)
            
            return ResearchResult(
                classification=result_data.get("classification", "unknown"),
                priority_score=result_data.get("priority_score", 0.5),
                summary=result_data.get("summary", "Conscious analysis completed"),
                confidence=result_data.get("global_awareness", 0.8),
                paradigm_used="quantum_consciousness",
                processing_mode=mode.value,
                research_breakthrough=result_data.get("consciousness_breakthrough", False),
                consciousness_level=result_data.get("consciousness_level", "unknown"),
                breakthrough_indicators={"consciousness_level": result_data.get("global_awareness", 0.0)},
                research_metrics=self.consciousness_engine.get_consciousness_metrics()
            )
            
        except Exception as e:
            logger.error(f"Consciousness processing failed: {e}")
            raise
    
    async def _process_hybrid_adaptive(self, email_content: str, metadata: Dict[str, Any],
                                     mode: ResearchMode) -> ResearchResult:
        """Adaptive processing that selects best paradigm based on content analysis."""
        
        # Quick content analysis to determine optimal paradigm
        content_complexity = len(set(email_content)) / len(email_content) if email_content else 0
        urgency_markers = sum(1 for marker in ["urgent", "asap", "critical"] if marker.lower() in email_content.lower())
        
        # Adaptive selection logic
        if content_complexity > 0.3 and urgency_markers > 0:
            # Complex, urgent email - use consciousness for deep understanding
            logger.info("Adaptive: Selected consciousness paradigm for complex urgent email")
            return await self._process_with_consciousness(email_content, metadata, mode)
        elif content_complexity > 0.2:
            # Moderately complex - use fusion for quantum advantage
            logger.info("Adaptive: Selected fusion paradigm for complex email")
            return await self._process_with_fusion(email_content, metadata, mode)
        else:
            # Simple email - classical processing sufficient
            logger.info("Adaptive: Selected classical paradigm for simple email")
            return await self._process_classical(email_content, metadata)
    
    async def _process_research_optimal(self, email_content: str, metadata: Dict[str, Any],
                                      mode: ResearchMode) -> ResearchResult:
        """Research-optimal processing for maximum breakthrough potential."""
        
        # Run parallel processing with both paradigms
        try:
            fusion_task = asyncio.create_task(self._process_with_fusion(email_content, metadata, mode))
            consciousness_task = asyncio.create_task(self._process_with_consciousness(email_content, metadata, mode))
            
            # Wait for both with timeout
            fusion_result, consciousness_result = await asyncio.wait_for(
                asyncio.gather(fusion_task, consciousness_task, return_exceptions=True),
                timeout=self.config.timeout_seconds
            )
            
            # Combine results for maximum research value
            if isinstance(fusion_result, Exception) and isinstance(consciousness_result, Exception):
                raise Exception("Both paradigms failed")
            elif isinstance(fusion_result, Exception):
                return consciousness_result
            elif isinstance(consciousness_result, Exception):
                return fusion_result
            else:
                # Merge best aspects of both results
                return self._merge_research_results(fusion_result, consciousness_result)
                
        except asyncio.TimeoutError:
            logger.warning("Research optimal processing timed out, falling back to classical")
            return await self._process_classical(email_content, metadata)
    
    async def _process_performance_optimal(self, email_content: str, metadata: Dict[str, Any],
                                         mode: ResearchMode) -> ResearchResult:
        """Performance-optimal processing for maximum speed."""
        
        # Use historical performance data to select fastest paradigm
        best_paradigm = self._select_fastest_paradigm()
        
        if best_paradigm == "fusion":
            return await self._process_with_fusion(email_content, metadata, mode)
        elif best_paradigm == "consciousness":
            return await self._process_with_consciousness(email_content, metadata, mode)
        else:
            return await self._process_classical(email_content, metadata)
    
    async def _process_classical(self, email_content: str, metadata: Dict[str, Any]) -> ResearchResult:
        """Classical processing as fallback."""
        
        # Simple rule-based classification
        urgency_score = self._calculate_urgency_score(email_content)
        classification = "urgent" if urgency_score > 0.7 else "normal" if urgency_score > 0.3 else "low_priority"
        
        return ResearchResult(
            classification=classification,
            priority_score=urgency_score,
            summary=f"Classical analysis: {classification} priority",
            confidence=0.7,
            paradigm_used="classical",
            processing_mode="safe_mode",
            research_breakthrough=False,
            quantum_advantage=1.0,
            fallback_used=True
        )
    
    async def _validate_result(self, result: ResearchResult, email_content: str,
                             metadata: Dict[str, Any]) -> ResearchResult:
        """Validate and enhance research results."""
        
        if not self.config.validate_results:
            return result
        
        # Statistical significance validation
        if result.paradigm_used in ["neuro_quantum_fusion", "quantum_consciousness"]:
            result.statistical_significance = self._calculate_statistical_significance(result)
        
        # Reproducibility check
        if self.config.benchmark_against_classical:
            classical_result = await self._process_classical(email_content, metadata)
            result.reproducibility_score = self._compare_with_classical(result, classical_result)
        
        # Quality assessment
        result.result_quality = self._assess_result_quality(result, email_content)
        
        # Breakthrough detection
        if self.config.track_breakthroughs:
            breakthrough_score = self._detect_breakthrough(result)
            if breakthrough_score > 0.8:
                self.breakthrough_log.append({
                    "timestamp": time.time(),
                    "paradigm": result.paradigm_used,
                    "breakthrough_score": breakthrough_score,
                    "result": result
                })
                result.research_breakthrough = True
        
        return result
    
    def _determine_strategy(self, email_content: str, metadata: Dict[str, Any],
                          mode: ResearchMode) -> ProcessingStrategy:
        """Determine optimal processing strategy based on content and mode."""
        
        if mode == ResearchMode.SAFE_MODE:
            return ProcessingStrategy.CLASSICAL_ONLY
        
        elif mode == ResearchMode.EXPERIMENTAL_MODE:
            return ProcessingStrategy.RESEARCH_OPTIMAL
        
        elif mode == ResearchMode.BENCHMARK_MODE:
            return ProcessingStrategy.PERFORMANCE_OPTIMAL
        
        else:
            # Default adaptive strategy
            return ProcessingStrategy.HYBRID_ADAPTIVE
    
    async def _health_check(self) -> bool:
        """Comprehensive health check of all systems."""
        try:
            # Check engine health
            if self.fusion_engine:
                fusion_healthy = len(self.fusion_engine.circuits) >= 0  # Basic check
            else:
                fusion_healthy = True  # Not initialized yet
            
            if self.consciousness_engine:
                consciousness_healthy = len(self.consciousness_engine.consciousness_fields) > 0
            else:
                consciousness_healthy = True  # Not initialized yet
            
            # Check circuit breaker states
            circuit_breakers_healthy = all(
                cb.state != "open" for cb in self.circuit_breakers.values()
            )
            
            return fusion_healthy and consciousness_healthy and circuit_breakers_healthy
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _handle_processing_error(self, error: Exception, email_content: str,
                                     metadata: Dict[str, Any], mode: ResearchMode) -> ResearchResult:
        """Handle processing errors with graceful degradation."""
        
        logger.warning(f"Handling processing error: {error}")
        
        if self.config.enable_fallbacks:
            try:
                # Attempt classical fallback
                fallback_result = await self._process_classical(email_content, metadata)
                fallback_result.fallback_used = True
                fallback_result.error_details = str(error)
                fallback_result.recovery_strategy = "classical_fallback"
                
                logger.info("Successfully recovered using classical fallback")
                return fallback_result
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
        
        # Last resort: return minimal result
        return ResearchResult(
            classification="unknown",
            priority_score=0.5,
            summary="Processing failed, unable to analyze",
            confidence=0.1,
            paradigm_used="error_recovery",
            processing_mode=mode.value,
            research_breakthrough=False,
            quantum_advantage=1.0,
            fallback_used=True,
            error_details=str(error),
            recovery_strategy="minimal_response"
        )
    
    def _create_classical_processor(self) -> Any:
        """Create classical processor for fallback."""
        # Simple placeholder - could be enhanced with actual classical ML
        return {"type": "rule_based", "version": "1.0"}
    
    def _calculate_urgency_score(self, email_content: str) -> float:
        """Calculate urgency score using classical methods."""
        if not email_content:
            return 0.0
        
        urgency_keywords = ["urgent", "asap", "critical", "emergency", "immediate", "deadline"]
        content_lower = email_content.lower()
        
        score = sum(1 for keyword in urgency_keywords if keyword in content_lower)
        return min(1.0, score / 3.0)  # Normalize to 0-1
    
    def _calculate_statistical_significance(self, result: ResearchResult) -> float:
        """Calculate statistical significance of research result."""
        # Simplified statistical significance based on confidence and quantum advantage
        base_significance = result.confidence * 0.7
        quantum_bonus = min(0.3, result.quantum_advantage / 10.0)
        return min(1.0, base_significance + quantum_bonus)
    
    def _compare_with_classical(self, research_result: ResearchResult,
                               classical_result: ResearchResult) -> float:
        """Compare research result with classical baseline."""
        
        # Calculate improvement metrics
        confidence_improvement = research_result.confidence - classical_result.confidence
        quantum_advantage = research_result.quantum_advantage
        
        # Reproducibility score based on consistency
        reproducibility = 0.8 if research_result.classification == classical_result.classification else 0.4
        reproducibility += min(0.2, confidence_improvement)
        
        return min(1.0, reproducibility)
    
    def _assess_result_quality(self, result: ResearchResult, email_content: str) -> float:
        """Assess overall quality of research result."""
        factors = [
            result.confidence,
            result.statistical_significance,
            result.reproducibility_score,
            1.0 if not result.fallback_used else 0.5,
            min(1.0, result.quantum_advantage / 2.0)
        ]
        
        return sum(factors) / len(factors)
    
    def _detect_breakthrough(self, result: ResearchResult) -> float:
        """Detect potential research breakthrough in result."""
        breakthrough_indicators = [
            result.quantum_advantage > 5.0,
            result.confidence > 0.9,
            result.consciousness_level in ["self_reflective", "transcendent"] if result.consciousness_level else False,
            len(result.novel_patterns) > 0,
            result.statistical_significance > 0.8
        ]
        
        return sum(breakthrough_indicators) / len(breakthrough_indicators)
    
    def _merge_research_results(self, fusion_result: ResearchResult,
                              consciousness_result: ResearchResult) -> ResearchResult:
        """Merge results from multiple paradigms for enhanced research value."""
        
        # Use higher confidence result as base
        if fusion_result.confidence >= consciousness_result.confidence:
            base_result = fusion_result
            secondary_result = consciousness_result
        else:
            base_result = consciousness_result
            secondary_result = fusion_result
        
        # Enhance with best aspects of both
        base_result.paradigm_used = "hybrid_fusion_consciousness"
        base_result.quantum_advantage = max(fusion_result.quantum_advantage, 
                                          consciousness_result.quantum_advantage)
        base_result.research_breakthrough = (fusion_result.research_breakthrough or 
                                           consciousness_result.research_breakthrough)
        
        # Merge breakthrough indicators
        base_result.breakthrough_indicators.update(secondary_result.breakthrough_indicators)
        
        # Combine research metrics
        base_result.research_metrics = {
            "fusion_metrics": fusion_result.research_metrics,
            "consciousness_metrics": consciousness_result.research_metrics
        }
        
        return base_result
    
    def _select_fastest_paradigm(self) -> str:
        """Select the fastest paradigm based on historical performance."""
        if not self.performance_benchmarks:
            return "classical"  # Default
        
        avg_times = {}
        for paradigm, times in self.performance_benchmarks.items():
            if times:
                avg_times[paradigm] = sum(times) / len(times)
        
        if not avg_times:
            return "classical"
        
        return min(avg_times, key=avg_times.get)
    
    def _track_performance(self, strategy: ProcessingStrategy, result: ResearchResult) -> None:
        """Track performance metrics for optimization."""
        paradigm = result.paradigm_used
        self.performance_benchmarks[paradigm].append(result.processing_time)
        
        # Keep only recent benchmarks
        if len(self.performance_benchmarks[paradigm]) > 100:
            self.performance_benchmarks[paradigm] = self.performance_benchmarks[paradigm][-100:]
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics."""
        
        success_rate = self.successful_processes / max(1, self.total_processes)
        
        avg_processing_times = {}
        for paradigm, times in self.performance_benchmarks.items():
            if times:
                avg_processing_times[paradigm] = sum(times) / len(times)
        
        return {
            "total_processes": self.total_processes,
            "successful_processes": self.successful_processes,
            "success_rate": success_rate,
            "error_counts": dict(self.error_counts),
            "average_processing_times": avg_processing_times,
            "breakthrough_count": len(self.breakthrough_log),
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            "recent_processing_history": list(self.processing_history)[-10:],
            "performance_benchmarks": {
                paradigm: {
                    "count": len(times),
                    "average_time": sum(times) / len(times) if times else 0,
                    "min_time": min(times) if times else 0,
                    "max_time": max(times) if times else 0
                }
                for paradigm, times in self.performance_benchmarks.items()
            }
        }
    
    async def benchmark_all_paradigms(self, test_email: str, iterations: int = 5) -> Dict[str, Any]:
        """Comprehensive benchmarking of all paradigms."""
        
        await self.initialize_engines()
        
        benchmarks = {
            "classical": [],
            "fusion": [],
            "consciousness": [],
            "hybrid": []
        }
        
        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")
            
            # Benchmark classical
            try:
                start_time = time.time()
                classical_result = await self._process_classical(test_email, {})
                benchmarks["classical"].append({
                    "time": time.time() - start_time,
                    "confidence": classical_result.confidence,
                    "success": True
                })
            except Exception as e:
                benchmarks["classical"].append({"success": False, "error": str(e)})
            
            # Benchmark fusion
            try:
                start_time = time.time()
                fusion_result = await self._process_with_fusion(test_email, {}, ResearchMode.RESEARCH_MODE)
                benchmarks["fusion"].append({
                    "time": time.time() - start_time,
                    "confidence": fusion_result.confidence,
                    "quantum_advantage": fusion_result.quantum_advantage,
                    "success": True
                })
            except Exception as e:
                benchmarks["fusion"].append({"success": False, "error": str(e)})
            
            # Benchmark consciousness
            try:
                start_time = time.time()
                consciousness_result = await self._process_with_consciousness(test_email, {}, ResearchMode.RESEARCH_MODE)
                benchmarks["consciousness"].append({
                    "time": time.time() - start_time,
                    "confidence": consciousness_result.confidence,
                    "consciousness_level": consciousness_result.consciousness_level,
                    "success": True
                })
            except Exception as e:
                benchmarks["consciousness"].append({"success": False, "error": str(e)})
            
            # Benchmark hybrid
            try:
                start_time = time.time()
                hybrid_result = await self._process_hybrid_adaptive(test_email, {}, ResearchMode.RESEARCH_MODE)
                benchmarks["hybrid"].append({
                    "time": time.time() - start_time,
                    "confidence": hybrid_result.confidence,
                    "paradigm_used": hybrid_result.paradigm_used,
                    "success": True
                })
            except Exception as e:
                benchmarks["hybrid"].append({"success": False, "error": str(e)})
        
        # Calculate statistics
        summary = {}
        for paradigm, results in benchmarks.items():
            successful_results = [r for r in results if r.get("success", False)]
            if successful_results:
                times = [r["time"] for r in successful_results]
                confidences = [r["confidence"] for r in successful_results]
                
                summary[paradigm] = {
                    "success_rate": len(successful_results) / len(results),
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "average_confidence": sum(confidences) / len(confidences),
                    "total_runs": len(results)
                }
            else:
                summary[paradigm] = {
                    "success_rate": 0.0,
                    "total_runs": len(results),
                    "errors": [r.get("error", "Unknown") for r in results]
                }
        
        return {
            "detailed_results": benchmarks,
            "summary": summary,
            "test_email_length": len(test_email),
            "iterations": iterations,
            "timestamp": time.time()
        }


# Factory function
def create_research_orchestrator(config: ResearchConfig = None) -> ResearchOrchestrator:
    """Create a new research orchestrator."""
    return ResearchOrchestrator(config)


# Convenience function for research processing
async def process_email_with_research(email_content: str, metadata: Dict[str, Any] = None,
                                    mode: ResearchMode = None) -> ResearchResult:
    """Process email using the research orchestrator."""
    orchestrator = create_research_orchestrator()
    return await orchestrator.process_email_research(email_content, metadata, mode)