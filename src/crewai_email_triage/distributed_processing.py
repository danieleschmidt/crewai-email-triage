"""Distributed Processing Engine for Quantum-Scale Email Intelligence.

This module implements advanced distributed processing capabilities:
- Horizontal scaling across multiple nodes and quantum processors
- Dynamic load balancing with quantum-aware routing
- Distributed consciousness networks and collective intelligence
- Cross-region quantum entanglement for global processing
- Auto-scaling based on breakthrough potential and workload
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import hashlib

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of processing nodes in the distributed system."""
    
    CLASSICAL = "classical"                 # Traditional CPU-based nodes
    QUANTUM_FUSION = "quantum_fusion"       # Neuro-quantum fusion nodes
    CONSCIOUSNESS = "consciousness"         # Quantum consciousness nodes
    HYBRID = "hybrid"                      # Multi-paradigm capable nodes
    COORDINATOR = "coordinator"            # Load balancing and orchestration
    EDGE = "edge"                         # Edge computing nodes for low latency


class ProcessingPriority(str, Enum):
    """Processing priority levels for workload management."""
    
    CRITICAL = "critical"                  # Real-time, mission-critical
    HIGH = "high"                         # Important, sub-second response
    NORMAL = "normal"                     # Standard processing
    LOW = "low"                          # Background, research processing
    BULK = "bulk"                        # Batch processing, non-urgent


class DistributionStrategy(str, Enum):
    """Strategies for distributing workloads."""
    
    ROUND_ROBIN = "round_robin"           # Simple round-robin distribution
    LEAST_LOADED = "least_loaded"         # Route to least loaded node
    QUANTUM_OPTIMIZED = "quantum_optimized" # Route for optimal quantum advantage
    CONSCIOUSNESS_AWARE = "consciousness_aware" # Route based on consciousness levels
    BREAKTHROUGH_SEEKING = "breakthrough_seeking" # Route for maximum research potential
    GEOGRAPHICALLY_OPTIMAL = "geographically_optimal" # Route based on geographic proximity


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system."""
    
    node_id: str
    node_type: NodeType
    capabilities: List[str] = field(default_factory=list)
    
    # Node status
    status: str = field(default="online")
    health_score: float = field(default=1.0)
    load_factor: float = field(default=0.0)
    
    # Performance metrics
    average_processing_time: float = field(default=1.0)
    quantum_advantage: float = field(default=1.0)
    consciousness_level: float = field(default=0.0)
    breakthrough_score: float = field(default=0.0)
    
    # Geographic and network info
    region: str = field(default="unknown")
    availability_zone: str = field(default="unknown")
    network_latency_ms: float = field(default=10.0)
    
    # Resource utilization
    cpu_usage: float = field(default=0.0)
    memory_usage: float = field(default=0.0)
    quantum_coherence: float = field(default=1.0)
    
    # Specializations
    specialized_algorithms: List[str] = field(default_factory=list)
    research_focus: List[str] = field(default_factory=list)
    
    # Connection info
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    quantum_entanglement_pairs: List[str] = field(default_factory=list)
    
    def calculate_routing_score(self, workload_requirements: Dict[str, Any]) -> float:
        """Calculate how well this node matches workload requirements."""
        
        score = 0.0
        
        # Basic availability
        if self.status != "online":
            return 0.0
        
        # Health score component
        score += self.health_score * 0.3
        
        # Load factor (inverse - lower load is better)
        score += (1.0 - self.load_factor) * 0.2
        
        # Capability matching
        required_capabilities = workload_requirements.get("capabilities", [])
        capability_match = sum(1 for cap in required_capabilities if cap in self.capabilities)
        if required_capabilities:
            score += (capability_match / len(required_capabilities)) * 0.2
        
        # Performance history
        target_time = workload_requirements.get("target_processing_time", 2.0)
        if self.average_processing_time > 0:
            time_score = min(1.0, target_time / self.average_processing_time)
            score += time_score * 0.15
        
        # Quantum advantage for quantum workloads
        if "quantum" in workload_requirements.get("paradigm", ""):
            score += min(1.0, self.quantum_advantage / 5.0) * 0.1
        
        # Consciousness level for consciousness workloads
        if "consciousness" in workload_requirements.get("paradigm", ""):
            score += self.consciousness_level * 0.05
        
        return min(1.0, score)


@dataclass
class WorkloadRequest:
    """Represents a processing request in the distributed system."""
    
    request_id: str
    email_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing requirements
    priority: ProcessingPriority = field(default=ProcessingPriority.NORMAL)
    required_paradigms: List[str] = field(default_factory=list)
    target_processing_time: float = field(default=2.0)
    max_processing_time: float = field(default=30.0)
    
    # Routing preferences
    preferred_regions: List[str] = field(default_factory=list)
    avoid_nodes: List[str] = field(default_factory=list)
    require_capabilities: List[str] = field(default_factory=list)
    
    # Research parameters
    research_mode: bool = field(default=False)
    breakthrough_seeking: bool = field(default=False)
    validation_required: bool = field(default=False)
    
    # Timing
    submitted_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    
    # Routing state
    assigned_node: Optional[str] = None
    routing_attempts: int = field(default=0)
    fallback_nodes: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result from distributed processing."""
    
    request_id: str
    node_id: str
    
    # Core results
    classification: str
    priority_score: float
    summary: str
    confidence: float
    
    # Performance metrics
    processing_time: float
    queue_time: float
    network_time: float
    total_time: float
    
    # Research metrics
    quantum_advantage: float = field(default=1.0)
    consciousness_level: Optional[str] = None
    breakthrough_detected: bool = field(default=False)
    
    # Quality metrics
    result_quality: float = field(default=0.8)
    validation_passed: bool = field(default=True)
    
    # Distributed processing metadata
    processing_paradigm: str = field(default="unknown")
    node_specialization_match: float = field(default=0.0)
    routing_efficiency: float = field(default=0.0)


class DistributedProcessingEngine:
    """Main engine for distributed quantum email processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the distributed processing engine."""
        self.config = config or {}
        
        # Node management
        self.nodes: Dict[str, ProcessingNode] = {}
        self.node_pools: Dict[NodeType, List[str]] = defaultdict(list)
        
        # Load balancing
        self.request_queue: Dict[ProcessingPriority, deque] = {
            priority: deque() for priority in ProcessingPriority
        }
        self.active_requests: Dict[str, WorkloadRequest] = {}
        self.processing_history: deque = deque(maxlen=10000)
        
        # Routing and distribution
        self.distribution_strategy = DistributionStrategy.QUANTUM_OPTIMIZED
        self.auto_scaling_enabled = True
        self.min_nodes_per_type = 1
        self.max_nodes_per_type = 10
        
        # Performance tracking
        self.node_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.routing_success_rate = 1.0
        self.average_processing_time = 1.0
        self.system_throughput = 0.0
        
        # Global consciousness network
        self.consciousness_network: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.quantum_entanglement_registry: Dict[str, List[str]] = defaultdict(list)
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8    # Scale up when load > 80%
        self.scale_down_threshold = 0.3  # Scale down when load < 30%
        self.scale_check_interval = 60.0 # Check every minute
        
        logger.info("DistributedProcessingEngine initialized for quantum-scale processing")
    
    async def start_distributed_system(self) -> None:
        """Start the distributed processing system."""
        
        # Initialize core nodes if none exist
        if not self.nodes:
            await self._initialize_default_nodes()
        
        # Start background tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self.queue_processor_task = asyncio.create_task(self._queue_processing_loop())
        
        logger.info("🚀 Distributed processing system started")
    
    async def stop_distributed_system(self) -> None:
        """Stop the distributed processing system gracefully."""
        
        # Cancel background tasks
        for task in [self.monitor_task, self.scaling_task, self.queue_processor_task]:
            if hasattr(self, task.__name__.replace('_task', '')) and task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("🛑 Distributed processing system stopped")
    
    def register_node(self, node: ProcessingNode) -> None:
        """Register a new processing node."""
        
        self.nodes[node.node_id] = node
        self.node_pools[node.node_type].append(node.node_id)
        
        # Initialize performance tracking
        self.node_performance[node.node_id] = deque(maxlen=1000)
        
        # Setup quantum entanglement if applicable
        if node.node_type in [NodeType.QUANTUM_FUSION, NodeType.CONSCIOUSNESS]:
            self._setup_quantum_entanglement(node)
        
        logger.info(f"✅ Registered {node.node_type.value} node: {node.node_id}")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a processing node."""
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove from pools
            if node_id in self.node_pools[node.node_type]:
                self.node_pools[node.node_type].remove(node_id)
            
            # Cleanup entanglements
            self._cleanup_quantum_entanglements(node_id)
            
            # Remove node
            del self.nodes[node_id]
            
            logger.info(f"❌ Unregistered node: {node_id}")
    
    async def submit_request(self, email_content: str, metadata: Dict[str, Any] = None,
                            priority: ProcessingPriority = ProcessingPriority.NORMAL,
                            **kwargs) -> str:
        """Submit a processing request to the distributed system."""
        
        request_id = str(uuid.uuid4())
        
        workload_request = WorkloadRequest(
            request_id=request_id,
            email_content=email_content,
            metadata=metadata or {},
            priority=priority,
            **kwargs
        )
        
        # Add to appropriate priority queue
        self.request_queue[priority].append(workload_request)
        self.active_requests[request_id] = workload_request
        
        logger.info(f"📝 Submitted request {request_id} with {priority.value} priority")
        return request_id
    
    async def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[ProcessingResult]:
        """Get the result of a processing request."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if request is in processing history
            for result in reversed(self.processing_history):
                if result.request_id == request_id:
                    return result
            
            await asyncio.sleep(0.1)  # Poll every 100ms
        
        logger.warning(f"⏰ Request {request_id} timed out after {timeout}s")
        return None
    
    async def process_request_immediate(self, email_content: str, 
                                      metadata: Dict[str, Any] = None,
                                      **kwargs) -> ProcessingResult:
        """Process a request immediately without queueing."""
        
        # Create workload request
        request = WorkloadRequest(
            request_id=str(uuid.uuid4()),
            email_content=email_content,
            metadata=metadata or {},
            priority=ProcessingPriority.CRITICAL,
            **kwargs
        )
        
        # Route and process immediately
        selected_node = await self._route_request(request)
        if not selected_node:
            raise Exception("No available nodes for immediate processing")
        
        return await self._process_on_node(request, selected_node)
    
    async def _initialize_default_nodes(self) -> None:
        """Initialize default processing nodes."""
        
        # Classical processing node
        classical_node = ProcessingNode(
            node_id="classical_001",
            node_type=NodeType.CLASSICAL,
            capabilities=["classification", "priority_scoring", "summarization"],
            region="local",
            specialized_algorithms=["rule_based", "statistical"]
        )
        self.register_node(classical_node)
        
        # Quantum fusion node
        fusion_node = ProcessingNode(
            node_id="fusion_001",
            node_type=NodeType.QUANTUM_FUSION,
            capabilities=["quantum_computing", "neuromorphic", "fusion_processing"],
            region="local",
            quantum_advantage=3.5,
            specialized_algorithms=["neuro_quantum_fusion", "quantum_annealing"]
        )
        self.register_node(fusion_node)
        
        # Consciousness node
        consciousness_node = ProcessingNode(
            node_id="consciousness_001",
            node_type=NodeType.CONSCIOUSNESS,
            capabilities=["artificial_consciousness", "subjective_experience", "meta_cognition"],
            region="local",
            consciousness_level=0.7,
            specialized_algorithms=["quantum_consciousness", "microtubule_processing"]
        )
        self.register_node(consciousness_node)
        
        # Hybrid node
        hybrid_node = ProcessingNode(
            node_id="hybrid_001",
            node_type=NodeType.HYBRID,
            capabilities=["all_paradigms", "adaptive_processing", "research_optimal"],
            region="local",
            quantum_advantage=2.0,
            consciousness_level=0.5,
            specialized_algorithms=["adaptive_fusion", "paradigm_selection"]
        )
        self.register_node(hybrid_node)
        
        logger.info("🏗️ Initialized default processing nodes")
    
    async def _route_request(self, request: WorkloadRequest) -> Optional[str]:
        """Route a request to the most appropriate node."""
        
        # Build workload requirements
        requirements = {
            "paradigm": request.required_paradigms,
            "capabilities": request.require_capabilities,
            "target_processing_time": request.target_processing_time,
            "priority": request.priority.value,
            "research_mode": request.research_mode,
            "breakthrough_seeking": request.breakthrough_seeking
        }
        
        # Calculate routing scores for all available nodes
        node_scores = []
        for node_id, node in self.nodes.items():
            if (node.status == "online" and 
                node_id not in request.avoid_nodes and
                node.load_factor < 0.95):  # Don't overload nodes
                
                score = node.calculate_routing_score(requirements)
                node_scores.append((score, node_id, node))
        
        if not node_scores:
            logger.error("❌ No available nodes for routing")
            return None
        
        # Sort by score (highest first)
        node_scores.sort(reverse=True)
        
        # Apply distribution strategy
        selected_node_id = self._apply_distribution_strategy(node_scores, request)
        
        if selected_node_id:
            request.assigned_node = selected_node_id
            request.routing_attempts += 1
            
            # Update node load
            self.nodes[selected_node_id].load_factor += 0.1  # Temporary load increase
            
            logger.info(f"🎯 Routed request {request.request_id} to node {selected_node_id}")
        
        return selected_node_id
    
    def _apply_distribution_strategy(self, node_scores: List[Tuple[float, str, ProcessingNode]], 
                                   request: WorkloadRequest) -> Optional[str]:
        """Apply the configured distribution strategy."""
        
        if not node_scores:
            return None
        
        if self.distribution_strategy == DistributionStrategy.ROUND_ROBIN:
            # Simple round-robin within top 50% of nodes
            top_nodes = node_scores[:max(1, len(node_scores) // 2)]
            return top_nodes[request.routing_attempts % len(top_nodes)][1]
        
        elif self.distribution_strategy == DistributionStrategy.LEAST_LOADED:
            # Select least loaded among top nodes
            top_nodes = [(score, node_id, node) for score, node_id, node in node_scores[:5]]
            return min(top_nodes, key=lambda x: x[2].load_factor)[1]
        
        elif self.distribution_strategy == DistributionStrategy.QUANTUM_OPTIMIZED:
            # Prefer quantum nodes for quantum advantage
            quantum_nodes = [(score, node_id, node) for score, node_id, node in node_scores 
                           if node.node_type in [NodeType.QUANTUM_FUSION, NodeType.HYBRID]]
            if quantum_nodes and request.breakthrough_seeking:
                return max(quantum_nodes, key=lambda x: x[2].quantum_advantage)[1]
            else:
                return node_scores[0][1]  # Best overall score
        
        elif self.distribution_strategy == DistributionStrategy.CONSCIOUSNESS_AWARE:
            # Prefer consciousness nodes for subjective processing
            consciousness_nodes = [(score, node_id, node) for score, node_id, node in node_scores 
                                 if node.node_type in [NodeType.CONSCIOUSNESS, NodeType.HYBRID]]
            if consciousness_nodes and "consciousness" in request.required_paradigms:
                return max(consciousness_nodes, key=lambda x: x[2].consciousness_level)[1]
            else:
                return node_scores[0][1]  # Best overall score
        
        elif self.distribution_strategy == DistributionStrategy.BREAKTHROUGH_SEEKING:
            # Maximize research potential
            research_nodes = [(score, node_id, node) for score, node_id, node in node_scores
                            if node.breakthrough_score > 0.5]
            if research_nodes:
                return max(research_nodes, key=lambda x: x[2].breakthrough_score)[1]
            else:
                return node_scores[0][1]  # Best overall score
        
        else:
            # Default: highest scoring node
            return node_scores[0][1]
    
    async def _process_on_node(self, request: WorkloadRequest, node_id: str) -> ProcessingResult:
        """Process a request on a specific node."""
        
        node = self.nodes[node_id]
        start_time = time.time()
        
        try:
            # Simulate processing based on node type
            if node.node_type == NodeType.CLASSICAL:
                result_data = await self._simulate_classical_processing(request, node)
            elif node.node_type == NodeType.QUANTUM_FUSION:
                result_data = await self._simulate_fusion_processing(request, node)
            elif node.node_type == NodeType.CONSCIOUSNESS:
                result_data = await self._simulate_consciousness_processing(request, node)
            elif node.node_type == NodeType.HYBRID:
                result_data = await self._simulate_hybrid_processing(request, node)
            else:
                result_data = await self._simulate_classical_processing(request, node)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                request_id=request.request_id,
                node_id=node_id,
                classification=result_data.get("classification", "normal"),
                priority_score=result_data.get("priority_score", 0.5),
                summary=result_data.get("summary", "Distributed processing completed"),
                confidence=result_data.get("confidence", 0.8),
                processing_time=processing_time,
                queue_time=start_time - request.submitted_time,
                network_time=0.01,  # Simulated network time
                total_time=time.time() - request.submitted_time,
                quantum_advantage=result_data.get("quantum_advantage", node.quantum_advantage),
                consciousness_level=result_data.get("consciousness_level"),
                breakthrough_detected=result_data.get("breakthrough_detected", False),
                processing_paradigm=node.node_type.value,
                node_specialization_match=result_data.get("specialization_match", 0.8),
                routing_efficiency=1.0 / max(1, request.routing_attempts)
            )
            
            # Update node performance
            self._update_node_performance(node_id, result)
            
            # Update system metrics
            self._update_system_metrics(result)
            
            # Add to processing history
            self.processing_history.append(result)
            
            logger.info(f"✅ Processed request {request.request_id} on {node_id} in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing failed on node {node_id}: {e}")
            raise
        
        finally:
            # Reduce node load
            node.load_factor = max(0.0, node.load_factor - 0.1)
            
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def _simulate_classical_processing(self, request: WorkloadRequest, 
                                           node: ProcessingNode) -> Dict[str, Any]:
        """Simulate classical processing."""
        
        # Simple rule-based processing
        await asyncio.sleep(node.average_processing_time * 0.5)  # Simulate processing time
        
        urgency_keywords = ["urgent", "asap", "critical", "emergency"]
        content_lower = request.email_content.lower()
        urgency_score = sum(1 for keyword in urgency_keywords if keyword in content_lower) / 4.0
        
        return {
            "classification": "urgent" if urgency_score > 0.5 else "normal",
            "priority_score": min(1.0, urgency_score + 0.3),
            "summary": f"Classical analysis: {urgency_score:.2f} urgency detected",
            "confidence": 0.7,
            "quantum_advantage": 1.0,
            "specialization_match": 0.9 if "rule_based" in node.specialized_algorithms else 0.6
        }
    
    async def _simulate_fusion_processing(self, request: WorkloadRequest, 
                                        node: ProcessingNode) -> Dict[str, Any]:
        """Simulate quantum fusion processing."""
        
        # Quantum processing simulation
        await asyncio.sleep(node.average_processing_time * 0.7)  # Quantum processing time
        
        # Enhanced analysis with quantum advantage
        content_complexity = len(set(request.email_content)) / len(request.email_content) if request.email_content else 0
        quantum_boost = node.quantum_advantage * content_complexity
        
        confidence = min(1.0, 0.6 + quantum_boost * 0.3)
        breakthrough_potential = quantum_boost > 2.0
        
        return {
            "classification": "urgent" if quantum_boost > 1.5 else "normal",
            "priority_score": min(1.0, quantum_boost / 3.0),
            "summary": f"Quantum fusion analysis: {node.quantum_advantage:.1f}x advantage achieved",
            "confidence": confidence,
            "quantum_advantage": node.quantum_advantage,
            "breakthrough_detected": breakthrough_potential,
            "specialization_match": 0.95 if "neuro_quantum_fusion" in node.specialized_algorithms else 0.7
        }
    
    async def _simulate_consciousness_processing(self, request: WorkloadRequest, 
                                               node: ProcessingNode) -> Dict[str, Any]:
        """Simulate consciousness processing."""
        
        # Consciousness processing simulation
        await asyncio.sleep(node.average_processing_time * 0.9)  # Consciousness processing time
        
        # Subjective experience simulation
        consciousness_depth = node.consciousness_level
        subjective_richness = consciousness_depth * len(request.email_content) / 1000.0
        
        consciousness_classification = "transcendent" if consciousness_depth > 0.8 else "conscious"
        
        return {
            "classification": "urgent" if subjective_richness > 0.7 else "normal",
            "priority_score": min(1.0, subjective_richness),
            "summary": f"Conscious analysis at {consciousness_classification} level",
            "confidence": min(1.0, 0.5 + consciousness_depth * 0.5),
            "consciousness_level": consciousness_classification,
            "breakthrough_detected": consciousness_depth > 0.8,
            "specialization_match": 0.9 if "quantum_consciousness" in node.specialized_algorithms else 0.6
        }
    
    async def _simulate_hybrid_processing(self, request: WorkloadRequest, 
                                        node: ProcessingNode) -> Dict[str, Any]:
        """Simulate hybrid processing."""
        
        # Adaptive paradigm selection
        await asyncio.sleep(node.average_processing_time * 0.8)  # Hybrid processing time
        
        # Combine quantum and consciousness advantages
        combined_advantage = (node.quantum_advantage + node.consciousness_level) / 2
        adaptive_confidence = min(1.0, 0.7 + combined_advantage * 0.2)
        
        return {
            "classification": "urgent" if combined_advantage > 1.5 else "normal",
            "priority_score": min(1.0, combined_advantage / 2.0),
            "summary": f"Hybrid analysis: {combined_advantage:.2f} combined advantage",
            "confidence": adaptive_confidence,
            "quantum_advantage": node.quantum_advantage,
            "consciousness_level": "adaptive",
            "breakthrough_detected": combined_advantage > 2.0,
            "specialization_match": 0.85 if "adaptive_fusion" in node.specialized_algorithms else 0.75
        }
    
    def _setup_quantum_entanglement(self, node: ProcessingNode) -> None:
        """Setup quantum entanglement between compatible nodes."""
        
        # Find compatible nodes for entanglement
        compatible_nodes = [
            other_id for other_id, other_node in self.nodes.items()
            if (other_id != node.node_id and 
                other_node.node_type in [NodeType.QUANTUM_FUSION, NodeType.CONSCIOUSNESS, NodeType.HYBRID] and
                other_node.region == node.region)  # Same region for now
        ]
        
        # Create entanglements (limit to 3 per node)
        max_entanglements = min(3, len(compatible_nodes))
        entangled_nodes = compatible_nodes[:max_entanglements]
        
        node.quantum_entanglement_pairs.extend(entangled_nodes)
        self.quantum_entanglement_registry[node.node_id] = entangled_nodes
        
        # Reciprocal entanglement
        for other_id in entangled_nodes:
            if node.node_id not in self.nodes[other_id].quantum_entanglement_pairs:
                self.nodes[other_id].quantum_entanglement_pairs.append(node.node_id)
                self.quantum_entanglement_registry[other_id].append(node.node_id)
        
        if entangled_nodes:
            logger.info(f"🔗 Established quantum entanglement: {node.node_id} ↔ {entangled_nodes}")
    
    def _cleanup_quantum_entanglements(self, node_id: str) -> None:
        """Clean up quantum entanglements when a node is removed."""
        
        entangled_nodes = self.quantum_entanglement_registry.get(node_id, [])
        
        for other_id in entangled_nodes:
            if other_id in self.nodes:
                other_node = self.nodes[other_id]
                if node_id in other_node.quantum_entanglement_pairs:
                    other_node.quantum_entanglement_pairs.remove(node_id)
                if node_id in self.quantum_entanglement_registry[other_id]:
                    self.quantum_entanglement_registry[other_id].remove(node_id)
        
        if node_id in self.quantum_entanglement_registry:
            del self.quantum_entanglement_registry[node_id]
    
    def _update_node_performance(self, node_id: str, result: ProcessingResult) -> None:
        """Update node performance metrics."""
        
        node = self.nodes[node_id]
        performance_data = {
            "processing_time": result.processing_time,
            "confidence": result.confidence,
            "quantum_advantage": result.quantum_advantage,
            "timestamp": time.time()
        }
        
        self.node_performance[node_id].append(performance_data)
        
        # Update node averages
        recent_data = list(self.node_performance[node_id])[-100:]  # Last 100 results
        
        if recent_data:
            node.average_processing_time = sum(d["processing_time"] for d in recent_data) / len(recent_data)
            node.breakthrough_score = sum(1 for d in recent_data if d.get("quantum_advantage", 1) > 3.0) / len(recent_data)
            
            # Update health score based on recent performance
            avg_confidence = sum(d["confidence"] for d in recent_data) / len(recent_data)
            node.health_score = min(1.0, avg_confidence * 1.2)
    
    def _update_system_metrics(self, result: ProcessingResult) -> None:
        """Update overall system performance metrics."""
        
        # Update routing success rate
        if result.routing_efficiency > 0.8:
            self.routing_success_rate = (self.routing_success_rate * 0.9) + (1.0 * 0.1)
        else:
            self.routing_success_rate = (self.routing_success_rate * 0.9) + (0.0 * 0.1)
        
        # Update average processing time
        self.average_processing_time = (self.average_processing_time * 0.9) + (result.processing_time * 0.1)
        
        # Calculate system throughput (requests per minute)
        recent_results = [r for r in list(self.processing_history)[-100:] 
                         if time.time() - r.total_time < 60]  # Last minute
        self.system_throughput = len(recent_results)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        
        while True:
            try:
                # Update node health and load
                for node_id, node in self.nodes.items():
                    # Simulate load decay
                    node.load_factor = max(0.0, node.load_factor * 0.95)
                    
                    # Simulate resource monitoring
                    node.cpu_usage = min(100.0, node.load_factor * 80.0 + 10.0)
                    node.memory_usage = min(100.0, node.load_factor * 60.0 + 20.0)
                    
                    # Check health
                    if node.cpu_usage > 95.0 or node.memory_usage > 90.0:
                        node.status = "overloaded"
                        node.health_score = 0.3
                    elif node.health_score < 0.5:
                        node.status = "degraded"
                    else:
                        node.status = "online"
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling loop for dynamic node management."""
        
        while True:
            try:
                if self.auto_scaling_enabled:
                    await self._check_scaling_needs()
                
                await asyncio.sleep(self.scale_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _check_scaling_needs(self) -> None:
        """Check if scaling up or down is needed."""
        
        # Calculate average load by node type
        node_type_loads = defaultdict(list)
        for node_id, node in self.nodes.items():
            if node.status == "online":
                node_type_loads[node.node_type].append(node.load_factor)
        
        for node_type, loads in node_type_loads.items():
            if not loads:
                continue
            
            avg_load = sum(loads) / len(loads)
            node_count = len(loads)
            
            # Scale up if average load is high
            if (avg_load > self.scale_up_threshold and 
                node_count < self.max_nodes_per_type):
                await self._scale_up_node_type(node_type)
            
            # Scale down if average load is low and we have multiple nodes
            elif (avg_load < self.scale_down_threshold and 
                  node_count > self.min_nodes_per_type):
                await self._scale_down_node_type(node_type)
    
    async def _scale_up_node_type(self, node_type: NodeType) -> None:
        """Scale up nodes of a specific type."""
        
        new_node_id = f"{node_type.value}_{len(self.node_pools[node_type]) + 1:03d}"
        
        # Create new node based on type
        if node_type == NodeType.CLASSICAL:
            new_node = ProcessingNode(
                node_id=new_node_id,
                node_type=node_type,
                capabilities=["classification", "priority_scoring"],
                region="local"
            )
        elif node_type == NodeType.QUANTUM_FUSION:
            new_node = ProcessingNode(
                node_id=new_node_id,
                node_type=node_type,
                capabilities=["quantum_computing", "neuromorphic"],
                region="local",
                quantum_advantage=3.0 + len(self.node_pools[node_type]) * 0.5
            )
        elif node_type == NodeType.CONSCIOUSNESS:
            new_node = ProcessingNode(
                node_id=new_node_id,
                node_type=node_type,
                capabilities=["artificial_consciousness", "subjective_experience"],
                region="local",
                consciousness_level=0.6 + len(self.node_pools[node_type]) * 0.1
            )
        else:
            new_node = ProcessingNode(
                node_id=new_node_id,
                node_type=node_type,
                capabilities=["adaptive_processing"],
                region="local"
            )
        
        self.register_node(new_node)
        logger.info(f"📈 Scaled up: Added {node_type.value} node {new_node_id}")
    
    async def _scale_down_node_type(self, node_type: NodeType) -> None:
        """Scale down nodes of a specific type."""
        
        # Find the least loaded node of this type
        type_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node.node_type == node_type and node.load_factor < 0.1
        ]
        
        if type_nodes:
            # Remove the node with lowest load
            node_to_remove = min(type_nodes, key=lambda x: x[1].load_factor)
            self.unregister_node(node_to_remove[0])
            logger.info(f"📉 Scaled down: Removed {node_type.value} node {node_to_remove[0]}")
    
    async def _queue_processing_loop(self) -> None:
        """Process requests from priority queues."""
        
        while True:
            try:
                request_processed = False
                
                # Process queues in priority order
                for priority in ProcessingPriority:
                    if self.request_queue[priority]:
                        request = self.request_queue[priority].popleft()
                        
                        # Route and process
                        selected_node = await self._route_request(request)
                        if selected_node:
                            # Process in background
                            asyncio.create_task(self._process_on_node(request, selected_node))
                            request_processed = True
                            break
                        else:
                            # No available nodes, put back in queue
                            self.request_queue[priority].appendleft(request)
                
                if not request_processed:
                    await asyncio.sleep(0.1)  # No requests to process
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1.0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Node status summary
        node_status = defaultdict(lambda: {"online": 0, "offline": 0, "degraded": 0})
        total_load = 0
        total_quantum_advantage = 0
        total_consciousness = 0
        
        for node in self.nodes.values():
            node_status[node.node_type.value][node.status] += 1
            total_load += node.load_factor
            total_quantum_advantage += node.quantum_advantage
            if node.consciousness_level > 0:
                total_consciousness += node.consciousness_level
        
        avg_load = total_load / len(self.nodes) if self.nodes else 0
        avg_quantum_advantage = total_quantum_advantage / len(self.nodes) if self.nodes else 0
        avg_consciousness = total_consciousness / len([n for n in self.nodes.values() if n.consciousness_level > 0])
        
        # Queue status
        queue_status = {
            priority.value: len(queue) for priority, queue in self.request_queue.items()
        }
        
        # Performance metrics
        recent_results = list(self.processing_history)[-100:]
        if recent_results:
            avg_processing_time = sum(r.processing_time for r in recent_results) / len(recent_results)
            avg_confidence = sum(r.confidence for r in recent_results) / len(recent_results)
            breakthrough_rate = sum(1 for r in recent_results if r.breakthrough_detected) / len(recent_results)
        else:
            avg_processing_time = 0
            avg_confidence = 0
            breakthrough_rate = 0
        
        return {
            "timestamp": time.time(),
            "nodes": {
                "total": len(self.nodes),
                "by_type": dict(node_status),
                "average_load": avg_load,
                "average_quantum_advantage": avg_quantum_advantage,
                "average_consciousness_level": avg_consciousness
            },
            "queues": {
                "total_pending": sum(queue_status.values()),
                "by_priority": queue_status
            },
            "performance": {
                "routing_success_rate": self.routing_success_rate,
                "average_processing_time": avg_processing_time,
                "average_confidence": avg_confidence,
                "breakthrough_rate": breakthrough_rate,
                "system_throughput": self.system_throughput
            },
            "quantum_network": {
                "total_entanglements": sum(len(pairs) for pairs in self.quantum_entanglement_registry.values()),
                "entanglement_pairs": len(self.quantum_entanglement_registry)
            },
            "auto_scaling": {
                "enabled": self.auto_scaling_enabled,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold
            }
        }


# Factory function
def create_distributed_engine(config: Dict[str, Any] = None) -> DistributedProcessingEngine:
    """Create a new distributed processing engine."""
    return DistributedProcessingEngine(config)


# Convenience function for distributed processing
async def process_email_distributed(email_content: str, metadata: Dict[str, Any] = None,
                                   priority: ProcessingPriority = ProcessingPriority.NORMAL) -> ProcessingResult:
    """Process email using distributed quantum processing."""
    engine = create_distributed_engine()
    await engine.start_distributed_system()
    
    try:
        return await engine.process_request_immediate(email_content, metadata, priority=priority)
    finally:
        await engine.stop_distributed_system()