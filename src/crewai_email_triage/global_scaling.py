"""Global Scaling and Multi-Region Architecture for Email Triage.

This module implements enterprise-grade global scaling capabilities including:
- Multi-region deployment and failover
- Intelligent load balancing and traffic routing
- Data sovereignty and compliance management
- Edge computing optimization
- Cross-region synchronization and consistency
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .enhanced_pipeline import EnhancedTriageResult, ProcessingConfig
from .performance import get_performance_tracker

logger = logging.getLogger(__name__)


class Region(str, Enum):
    """Global regions for deployment."""

    # North America
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    CA_CENTRAL_1 = "ca-central-1"

    # Europe
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"

    # Asia Pacific
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_SOUTH_1 = "ap-south-1"

    # Other regions
    SA_EAST_1 = "sa-east-1"
    AF_SOUTH_1 = "af-south-1"
    ME_SOUTH_1 = "me-south-1"


class ComplianceFramework(str, Enum):
    """Data compliance frameworks."""

    GDPR = "gdpr"           # EU General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    SOX = "sox"             # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"     # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001" # Information Security Management
    SOC_2 = "soc_2"         # Service Organization Control 2


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"         # Simple round-robin distribution
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weighted by capacity
    LEAST_CONNECTIONS = "least_connections"        # Route to least busy
    GEOGRAPHIC = "geographic"           # Route by geographic proximity
    LATENCY_BASED = "latency_based"     # Route by lowest latency
    RESOURCE_BASED = "resource_based"   # Route by resource availability
    AI_OPTIMIZED = "ai_optimized"       # AI-driven routing decisions


@dataclass
class RegionCapacity:
    """Resource capacity information for a region."""

    region: Region
    max_concurrent_requests: int = 1000
    current_load: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_latency_ms: float = 0.0

    # Performance metrics
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0

    # Health status
    is_healthy: bool = True
    is_available: bool = True
    last_health_check: float = field(default_factory=time.time)

    # Compliance
    supported_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    data_residency_enforced: bool = False

    def utilization_score(self) -> float:
        """Calculate overall utilization score (0.0 to 1.0)."""
        if not self.is_healthy or not self.is_available:
            return 1.0  # Fully utilized if unhealthy

        load_util = self.current_load / max(self.max_concurrent_requests, 1)
        resource_util = (self.cpu_utilization + self.memory_utilization) / 2

        return min(1.0, max(load_util, resource_util))

    def can_handle_request(self, compliance_required: Optional[Set[ComplianceFramework]] = None) -> bool:
        """Check if region can handle a new request."""
        if not self.is_healthy or not self.is_available:
            return False

        if self.current_load >= self.max_concurrent_requests:
            return False

        if compliance_required and not compliance_required.issubset(self.supported_frameworks):
            return False

        return True


@dataclass
class RoutingDecision:
    """Routing decision for a request."""

    target_region: Region
    routing_reason: str
    expected_latency_ms: float
    load_factor: float
    compliance_satisfied: bool
    fallback_regions: List[Region] = field(default_factory=list)

    # Decision metadata
    decision_time_ms: float = 0.0
    decision_confidence: float = 1.0
    alternative_options: int = 0


class GlobalLoadBalancer:
    """Intelligent global load balancer with AI-driven routing."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.AI_OPTIMIZED):
        self.strategy = strategy
        self.region_capacities: Dict[Region, RegionCapacity] = {}
        self.routing_history = deque(maxlen=10000)

        # Performance tracking
        self.performance_tracker = get_performance_tracker()
        self.routing_metrics = {
            "total_requests": 0,
            "successful_routings": 0,
            "failed_routings": 0,
            "avg_decision_time": 0.0,
            "strategy_effectiveness": {}
        }

        # AI learning for routing optimization
        self.routing_patterns = defaultdict(list)  # Request patterns to region performance
        self.latency_predictions = defaultdict(list)  # Predicted vs actual latency

        # Initialize regions
        self._initialize_regions()

        logger.info(f"GlobalLoadBalancer initialized with strategy: {strategy.value}")

    def _initialize_regions(self) -> None:
        """Initialize region capacities with default values."""

        region_configs = {
            # North America - High capacity, low latency for US traffic
            Region.US_EAST_1: {
                "max_concurrent_requests": 2000,
                "supported_frameworks": {ComplianceFramework.CCPA, ComplianceFramework.HIPAA, ComplianceFramework.SOX},
                "data_residency_enforced": True
            },
            Region.US_WEST_2: {
                "max_concurrent_requests": 1500,
                "supported_frameworks": {ComplianceFramework.CCPA, ComplianceFramework.HIPAA},
                "data_residency_enforced": True
            },
            Region.CA_CENTRAL_1: {
                "max_concurrent_requests": 800,
                "supported_frameworks": {ComplianceFramework.CCPA},
                "data_residency_enforced": True
            },

            # Europe - GDPR compliance focus
            Region.EU_WEST_1: {
                "max_concurrent_requests": 1200,
                "supported_frameworks": {ComplianceFramework.GDPR, ComplianceFramework.ISO_27001, ComplianceFramework.SOC_2},
                "data_residency_enforced": True
            },
            Region.EU_CENTRAL_1: {
                "max_concurrent_requests": 1000,
                "supported_frameworks": {ComplianceFramework.GDPR, ComplianceFramework.ISO_27001},
                "data_residency_enforced": True
            },
            Region.EU_NORTH_1: {
                "max_concurrent_requests": 600,
                "supported_frameworks": {ComplianceFramework.GDPR},
                "data_residency_enforced": True
            },

            # Asia Pacific - Growing capacity
            Region.AP_SOUTHEAST_1: {
                "max_concurrent_requests": 800,
                "supported_frameworks": {ComplianceFramework.ISO_27001, ComplianceFramework.SOC_2},
                "data_residency_enforced": False
            },
            Region.AP_NORTHEAST_1: {
                "max_concurrent_requests": 1000,
                "supported_frameworks": {ComplianceFramework.ISO_27001},
                "data_residency_enforced": False
            },
            Region.AP_SOUTH_1: {
                "max_concurrent_requests": 500,
                "supported_frameworks": {ComplianceFramework.ISO_27001},
                "data_residency_enforced": False
            },

            # Other regions - Smaller capacity
            Region.SA_EAST_1: {
                "max_concurrent_requests": 300,
                "supported_frameworks": set(),
                "data_residency_enforced": False
            },
            Region.AF_SOUTH_1: {
                "max_concurrent_requests": 200,
                "supported_frameworks": set(),
                "data_residency_enforced": False
            },
            Region.ME_SOUTH_1: {
                "max_concurrent_requests": 200,
                "supported_frameworks": set(),
                "data_residency_enforced": False
            }
        }

        for region, config in region_configs.items():
            self.region_capacities[region] = RegionCapacity(
                region=region,
                max_concurrent_requests=config["max_concurrent_requests"],
                supported_frameworks=config["supported_frameworks"],
                data_residency_enforced=config["data_residency_enforced"]
            )

    async def route_request(
        self,
        request_metadata: Dict[str, Any],
        compliance_requirements: Optional[Set[ComplianceFramework]] = None,
        preferred_region: Optional[Region] = None
    ) -> RoutingDecision:
        """Route a request to the optimal region."""

        start_time = time.time()

        # Extract request characteristics
        request_size = request_metadata.get("content_length", 1000)
        user_location = request_metadata.get("user_location", "unknown")
        priority = request_metadata.get("priority", 5)
        session_id = request_metadata.get("session_id")

        # Apply routing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            decision = await self._route_round_robin(compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            decision = await self._route_weighted_round_robin(compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            decision = await self._route_least_connections(compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            decision = await self._route_geographic(user_location, compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.LATENCY_BASED:
            decision = await self._route_latency_based(compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            decision = await self._route_resource_based(compliance_requirements)

        elif self.strategy == LoadBalancingStrategy.AI_OPTIMIZED:
            decision = await self._route_ai_optimized(request_metadata, compliance_requirements)

        else:
            # Fallback to least connections
            decision = await self._route_least_connections(compliance_requirements)

        # Handle preferred region if specified
        if preferred_region and preferred_region in self.region_capacities:
            capacity = self.region_capacities[preferred_region]
            if capacity.can_handle_request(compliance_requirements):
                decision.target_region = preferred_region
                decision.routing_reason = f"Preferred region override: {preferred_region.value}"

        # Calculate decision time
        decision_time = (time.time() - start_time) * 1000
        decision.decision_time_ms = decision_time

        # Update routing metrics
        self.routing_metrics["total_requests"] += 1
        if decision.target_region:
            self.routing_metrics["successful_routings"] += 1

            # Update average decision time
            total_successful = self.routing_metrics["successful_routings"]
            current_avg = self.routing_metrics["avg_decision_time"]
            self.routing_metrics["avg_decision_time"] = (
                (current_avg * (total_successful - 1) + decision_time) / total_successful
            )
        else:
            self.routing_metrics["failed_routings"] += 1

        # Store routing history for learning
        self.routing_history.append({
            "timestamp": time.time(),
            "request_metadata": request_metadata,
            "decision": decision,
            "compliance_requirements": list(compliance_requirements) if compliance_requirements else [],
            "decision_time_ms": decision_time
        })

        # Track performance
        self.performance_tracker.record_operation(
            f"routing_{self.strategy.value}",
            decision_time / 1000,
            {
                "target_region": decision.target_region.value if decision.target_region else "none",
                "compliance_satisfied": decision.compliance_satisfied,
                "load_factor": decision.load_factor
            }
        )

        logger.debug(
            f"Routing decision: {decision.target_region.value if decision.target_region else 'FAILED'} "
            f"({decision.routing_reason}, {decision_time:.2f}ms)"
        )

        return decision

    async def _route_round_robin(self, compliance: Optional[Set[ComplianceFramework]]) -> RoutingDecision:
        """Simple round-robin routing."""
        available_regions = [
            region for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # Simple round-robin selection based on request count
        region_index = self.routing_metrics["total_requests"] % len(available_regions)
        selected_region = available_regions[region_index]
        capacity = self.region_capacities[selected_region]

        return RoutingDecision(
            target_region=selected_region,
            routing_reason="Round-robin selection",
            expected_latency_ms=capacity.avg_response_time_ms,
            load_factor=capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=available_regions[region_index+1:] + available_regions[:region_index]
        )

    async def _route_weighted_round_robin(self, compliance: Optional[Set[ComplianceFramework]]) -> RoutingDecision:
        """Weighted round-robin based on capacity."""
        available_regions = [
            (region, capacity) for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # Calculate weights based on available capacity
        weights = []
        for region, capacity in available_regions:
            available_capacity = capacity.max_concurrent_requests - capacity.current_load
            weight = max(1, available_capacity)  # Minimum weight of 1
            weights.append(weight)

        # Weighted selection
        total_weight = sum(weights)
        random_value = (self.routing_metrics["total_requests"] * 17) % total_weight  # Pseudo-random

        cumulative_weight = 0
        for i, (region, capacity) in enumerate(available_regions):
            cumulative_weight += weights[i]
            if random_value < cumulative_weight:
                return RoutingDecision(
                    target_region=region,
                    routing_reason=f"Weighted selection (weight: {weights[i]}/{total_weight})",
                    expected_latency_ms=capacity.avg_response_time_ms,
                    load_factor=capacity.utilization_score(),
                    compliance_satisfied=True,
                    fallback_regions=[r for r, _ in available_regions if r != region]
                )

        # Fallback to first region
        region, capacity = available_regions[0]
        return RoutingDecision(
            target_region=region,
            routing_reason="Weighted fallback",
            expected_latency_ms=capacity.avg_response_time_ms,
            load_factor=capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=[r for r, _ in available_regions[1:]]
        )

    async def _route_least_connections(self, compliance: Optional[Set[ComplianceFramework]]) -> RoutingDecision:
        """Route to region with least connections."""
        available_regions = [
            (region, capacity) for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # Find region with lowest current load
        best_region, best_capacity = min(available_regions, key=lambda x: x[1].current_load)

        return RoutingDecision(
            target_region=best_region,
            routing_reason=f"Least connections ({best_capacity.current_load}/{best_capacity.max_concurrent_requests})",
            expected_latency_ms=best_capacity.avg_response_time_ms,
            load_factor=best_capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=[r for r, _ in available_regions if r != best_region],
            alternative_options=len(available_regions) - 1
        )

    async def _route_geographic(
        self,
        user_location: str,
        compliance: Optional[Set[ComplianceFramework]]
    ) -> RoutingDecision:
        """Route based on geographic proximity."""

        # Geographic affinity mapping
        location_to_regions = {
            "us": [Region.US_EAST_1, Region.US_WEST_2, Region.CA_CENTRAL_1],
            "canada": [Region.CA_CENTRAL_1, Region.US_EAST_1, Region.US_WEST_2],
            "eu": [Region.EU_WEST_1, Region.EU_CENTRAL_1, Region.EU_NORTH_1],
            "asia": [Region.AP_SOUTHEAST_1, Region.AP_NORTHEAST_1, Region.AP_SOUTH_1],
            "south_america": [Region.SA_EAST_1, Region.US_EAST_1],
            "africa": [Region.AF_SOUTH_1, Region.EU_WEST_1],
            "middle_east": [Region.ME_SOUTH_1, Region.EU_CENTRAL_1]
        }

        # Find preferred regions for user location
        location_key = user_location.lower()
        preferred_regions = []

        for key, regions in location_to_regions.items():
            if key in location_key:
                preferred_regions = regions
                break

        if not preferred_regions:
            # Default to all regions if no geographic match
            preferred_regions = list(self.region_capacities.keys())

        # Filter by availability and compliance
        available_regions = [
            region for region in preferred_regions
            if (region in self.region_capacities and
                self.region_capacities[region].can_handle_request(compliance))
        ]

        if not available_regions:
            # Fallback to any available region
            return await self._route_least_connections(compliance)

        # Select first available region (closest geographically)
        selected_region = available_regions[0]
        capacity = self.region_capacities[selected_region]

        return RoutingDecision(
            target_region=selected_region,
            routing_reason=f"Geographic proximity to {user_location}",
            expected_latency_ms=capacity.network_latency_ms,
            load_factor=capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=available_regions[1:],
            alternative_options=len(available_regions) - 1
        )

    async def _route_latency_based(self, compliance: Optional[Set[ComplianceFramework]]) -> RoutingDecision:
        """Route to region with lowest expected latency."""
        available_regions = [
            (region, capacity) for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # Find region with lowest combined latency
        def calculate_total_latency(region, capacity):
            network_latency = capacity.network_latency_ms
            processing_latency = capacity.avg_response_time_ms
            load_penalty = capacity.utilization_score() * 50  # Add latency for high load
            return network_latency + processing_latency + load_penalty

        best_region, best_capacity = min(
            available_regions,
            key=lambda x: calculate_total_latency(x[0], x[1])
        )

        expected_latency = calculate_total_latency(best_region, best_capacity)

        return RoutingDecision(
            target_region=best_region,
            routing_reason=f"Lowest expected latency ({expected_latency:.1f}ms)",
            expected_latency_ms=expected_latency,
            load_factor=best_capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=[r for r, _ in available_regions if r != best_region],
            alternative_options=len(available_regions) - 1
        )

    async def _route_resource_based(self, compliance: Optional[Set[ComplianceFramework]]) -> RoutingDecision:
        """Route based on resource availability."""
        available_regions = [
            (region, capacity) for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # Calculate resource score (lower is better)
        def calculate_resource_score(capacity):
            cpu_factor = capacity.cpu_utilization
            memory_factor = capacity.memory_utilization
            load_factor = capacity.current_load / max(capacity.max_concurrent_requests, 1)

            # Weight factors
            return 0.4 * cpu_factor + 0.3 * memory_factor + 0.3 * load_factor

        best_region, best_capacity = min(
            available_regions,
            key=lambda x: calculate_resource_score(x[1])
        )

        resource_score = calculate_resource_score(best_capacity)

        return RoutingDecision(
            target_region=best_region,
            routing_reason=f"Best resource availability (score: {resource_score:.3f})",
            expected_latency_ms=best_capacity.avg_response_time_ms,
            load_factor=resource_score,
            compliance_satisfied=True,
            fallback_regions=[r for r, _ in available_regions if r != best_region],
            alternative_options=len(available_regions) - 1
        )

    async def _route_ai_optimized(
        self,
        request_metadata: Dict[str, Any],
        compliance: Optional[Set[ComplianceFramework]]
    ) -> RoutingDecision:
        """AI-optimized routing using learned patterns."""

        available_regions = [
            (region, capacity) for region, capacity in self.region_capacities.items()
            if capacity.can_handle_request(compliance)
        ]

        if not available_regions:
            return RoutingDecision(
                target_region=None,
                routing_reason="No available regions",
                expected_latency_ms=0,
                load_factor=1.0,
                compliance_satisfied=False
            )

        # AI scoring based on multiple factors
        def calculate_ai_score(region, capacity):
            # Base factors
            latency_score = 1.0 - min(1.0, capacity.avg_response_time_ms / 1000.0)  # Normalized
            capacity_score = 1.0 - capacity.utilization_score()
            health_score = 1.0 if capacity.is_healthy else 0.0
            success_score = capacity.success_rate

            # Historical performance
            history_score = self._get_historical_performance_score(region, request_metadata)

            # Pattern matching
            pattern_score = self._get_pattern_matching_score(region, request_metadata)

            # Weighted combination
            weights = {
                "latency": 0.25,
                "capacity": 0.20,
                "health": 0.15,
                "success": 0.15,
                "history": 0.15,
                "pattern": 0.10
            }

            ai_score = (
                weights["latency"] * latency_score +
                weights["capacity"] * capacity_score +
                weights["health"] * health_score +
                weights["success"] * success_score +
                weights["history"] * history_score +
                weights["pattern"] * pattern_score
            )

            return ai_score

        # Score all available regions
        region_scores = [
            (region, capacity, calculate_ai_score(region, capacity))
            for region, capacity in available_regions
        ]

        # Sort by AI score (descending)
        region_scores.sort(key=lambda x: x[2], reverse=True)

        best_region, best_capacity, best_score = region_scores[0]

        return RoutingDecision(
            target_region=best_region,
            routing_reason=f"AI optimization (score: {best_score:.3f})",
            expected_latency_ms=best_capacity.avg_response_time_ms,
            load_factor=best_capacity.utilization_score(),
            compliance_satisfied=True,
            fallback_regions=[r for r, _, _ in region_scores[1:]],
            alternative_options=len(region_scores) - 1,
            decision_confidence=best_score
        )

    def _get_historical_performance_score(self, region: Region, request_metadata: Dict[str, Any]) -> float:
        """Get historical performance score for region."""

        if not self.routing_history:
            return 0.5  # Neutral score

        # Find similar requests routed to this region
        similar_requests = []
        request_size = request_metadata.get("content_length", 1000)
        request_priority = request_metadata.get("priority", 5)

        for history_entry in self.routing_history:
            if history_entry["decision"].target_region == region:
                # Check similarity
                hist_metadata = history_entry["request_metadata"]
                hist_size = hist_metadata.get("content_length", 1000)
                hist_priority = hist_metadata.get("priority", 5)

                size_similarity = 1.0 - abs(request_size - hist_size) / max(request_size, hist_size, 1)
                priority_similarity = 1.0 - abs(request_priority - hist_priority) / 10.0

                overall_similarity = (size_similarity + priority_similarity) / 2

                if overall_similarity > 0.7:  # Threshold for similarity
                    similar_requests.append(history_entry)

        if not similar_requests:
            return 0.5  # Neutral score

        # Calculate average performance for similar requests
        recent_requests = similar_requests[-10:]  # Last 10 similar requests

        # Use decision time as a proxy for performance (lower is better)
        avg_decision_time = statistics.mean(req["decision_time_ms"] for req in recent_requests)

        # Normalize to 0-1 score (lower decision time = higher score)
        performance_score = max(0.0, 1.0 - (avg_decision_time / 100.0))  # Assuming 100ms is poor

        return min(1.0, performance_score)

    def _get_pattern_matching_score(self, region: Region, request_metadata: Dict[str, Any]) -> float:
        """Get pattern matching score for region."""

        # Create request pattern signature
        user_location = request_metadata.get("user_location", "unknown")
        content_length = request_metadata.get("content_length", 1000)
        priority = request_metadata.get("priority", 5)

        pattern_key = f"{user_location}:{content_length//1000}k:{priority}"

        if pattern_key not in self.routing_patterns:
            return 0.5  # Neutral score for new patterns

        # Get historical routing success for this pattern to this region
        pattern_history = self.routing_patterns[pattern_key]
        region_successes = sum(1 for entry in pattern_history if entry["region"] == region and entry["success"])
        region_total = sum(1 for entry in pattern_history if entry["region"] == region)

        if region_total == 0:
            return 0.5  # No history for this region

        success_rate = region_successes / region_total
        return success_rate

    async def update_region_status(
        self,
        region: Region,
        metrics: Dict[str, Any]
    ) -> None:
        """Update region status and metrics."""

        if region not in self.region_capacities:
            logger.warning(f"Unknown region: {region}")
            return

        capacity = self.region_capacities[region]

        # Update basic metrics
        capacity.current_load = metrics.get("current_load", capacity.current_load)
        capacity.cpu_utilization = metrics.get("cpu_utilization", capacity.cpu_utilization)
        capacity.memory_utilization = metrics.get("memory_utilization", capacity.memory_utilization)
        capacity.network_latency_ms = metrics.get("network_latency_ms", capacity.network_latency_ms)

        # Update performance metrics
        capacity.avg_response_time_ms = metrics.get("avg_response_time_ms", capacity.avg_response_time_ms)
        capacity.success_rate = metrics.get("success_rate", capacity.success_rate)
        capacity.error_rate = metrics.get("error_rate", capacity.error_rate)

        # Update health status
        capacity.is_healthy = metrics.get("is_healthy", capacity.is_healthy)
        capacity.is_available = metrics.get("is_available", capacity.is_available)
        capacity.last_health_check = time.time()

        logger.debug(f"Updated region {region.value}: Load {capacity.current_load}/{capacity.max_concurrent_requests}, Health {capacity.is_healthy}")

    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global load balancer status."""

        region_statuses = {}
        for region, capacity in self.region_capacities.items():
            region_statuses[region.value] = {
                "current_load": capacity.current_load,
                "max_capacity": capacity.max_concurrent_requests,
                "utilization": capacity.utilization_score(),
                "is_healthy": capacity.is_healthy,
                "is_available": capacity.is_available,
                "avg_response_time_ms": capacity.avg_response_time_ms,
                "success_rate": capacity.success_rate,
                "supported_frameworks": [f.value for f in capacity.supported_frameworks],
                "data_residency_enforced": capacity.data_residency_enforced,
                "last_health_check_age_seconds": time.time() - capacity.last_health_check
            }

        # Calculate global statistics
        total_capacity = sum(c.max_concurrent_requests for c in self.region_capacities.values())
        total_load = sum(c.current_load for c in self.region_capacities.values())
        healthy_regions = sum(1 for c in self.region_capacities.values() if c.is_healthy)

        return {
            "load_balancer_strategy": self.strategy.value,
            "global_metrics": {
                "total_capacity": total_capacity,
                "total_current_load": total_load,
                "global_utilization": total_load / max(total_capacity, 1),
                "healthy_regions": healthy_regions,
                "total_regions": len(self.region_capacities),
                "health_percentage": healthy_regions / len(self.region_capacities)
            },
            "routing_metrics": self.routing_metrics,
            "region_statuses": region_statuses,
            "recent_routing_decisions": [
                {
                    "timestamp": entry["timestamp"],
                    "target_region": entry["decision"].target_region.value if entry["decision"].target_region else None,
                    "routing_reason": entry["decision"].routing_reason,
                    "decision_time_ms": entry["decision_time_ms"],
                    "compliance_satisfied": entry["decision"].compliance_satisfied
                }
                for entry in list(self.routing_history)[-10:]
            ]
        }


class GlobalComplianceManager:
    """Manages data sovereignty and compliance across regions."""

    def __init__(self):
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {}
        self.region_compliance: Dict[Region, Set[ComplianceFramework]] = {}
        self.data_residency_rules: Dict[str, Set[Region]] = {}

        self._initialize_compliance_rules()

        logger.info("GlobalComplianceManager initialized")

    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance framework rules."""

        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                "data_residency_required": True,
                "allowed_regions": {Region.EU_WEST_1, Region.EU_CENTRAL_1, Region.EU_NORTH_1},
                "retention_max_days": 1095,  # 3 years
                "consent_required": True,
                "encryption_required": True,
                "audit_logging_required": True,
                "data_portability_required": True
            },
            ComplianceFramework.CCPA: {
                "data_residency_required": False,
                "allowed_regions": {Region.US_EAST_1, Region.US_WEST_2, Region.CA_CENTRAL_1},
                "retention_max_days": 365,
                "consent_required": True,
                "encryption_required": False,
                "audit_logging_required": True,
                "data_portability_required": True
            },
            ComplianceFramework.HIPAA: {
                "data_residency_required": True,
                "allowed_regions": {Region.US_EAST_1, Region.US_WEST_2},
                "retention_max_days": 2555,  # 7 years
                "consent_required": True,
                "encryption_required": True,
                "audit_logging_required": True,
                "access_logging_required": True
            },
            ComplianceFramework.ISO_27001: {
                "data_residency_required": False,
                "allowed_regions": set(Region),  # All regions
                "retention_max_days": 2555,
                "encryption_required": True,
                "audit_logging_required": True,
                "security_monitoring_required": True
            },
            ComplianceFramework.SOC_2: {
                "data_residency_required": False,
                "allowed_regions": set(Region),  # All regions
                "audit_logging_required": True,
                "access_control_required": True,
                "security_monitoring_required": True
            }
        }

        # Initialize region compliance mapping
        for region in Region:
            self.region_compliance[region] = set()

        # Set region compliance based on rules
        for framework, rules in self.compliance_rules.items():
            allowed_regions = rules.get("allowed_regions", set())
            for region in allowed_regions:
                if region in self.region_compliance:
                    self.region_compliance[region].add(framework)

    def validate_compliance(
        self,
        request_metadata: Dict[str, Any],
        target_region: Region,
        required_frameworks: Optional[Set[ComplianceFramework]] = None
    ) -> Dict[str, Any]:
        """Validate compliance requirements for a request."""

        validation_result = {
            "compliant": True,
            "violations": [],
            "requirements_met": [],
            "additional_requirements": []
        }

        if not required_frameworks:
            return validation_result

        region_frameworks = self.region_compliance.get(target_region, set())

        for framework in required_frameworks:
            if framework not in region_frameworks:
                validation_result["compliant"] = False
                validation_result["violations"].append(
                    f"{framework.value} not supported in region {target_region.value}"
                )
            else:
                validation_result["requirements_met"].append(framework.value)

                # Check specific requirements
                framework_rules = self.compliance_rules.get(framework, {})

                # Data residency check
                if framework_rules.get("data_residency_required"):
                    allowed_regions = framework_rules.get("allowed_regions", set())
                    if target_region not in allowed_regions:
                        validation_result["compliant"] = False
                        validation_result["violations"].append(
                            f"{framework.value} requires data residency in specific regions"
                        )

                # Add additional requirements
                if framework_rules.get("encryption_required"):
                    validation_result["additional_requirements"].append("Data encryption required")

                if framework_rules.get("audit_logging_required"):
                    validation_result["additional_requirements"].append("Audit logging required")

                if framework_rules.get("consent_required"):
                    validation_result["additional_requirements"].append("User consent required")

        return validation_result

    def get_compliant_regions(
        self,
        required_frameworks: Set[ComplianceFramework]
    ) -> List[Region]:
        """Get list of regions that support all required compliance frameworks."""

        compliant_regions = []

        for region, supported_frameworks in self.region_compliance.items():
            if required_frameworks.issubset(supported_frameworks):
                compliant_regions.append(region)

        return compliant_regions

    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report."""

        framework_coverage = {}
        for framework in ComplianceFramework:
            supporting_regions = [
                region.value for region, frameworks in self.region_compliance.items()
                if framework in frameworks
            ]
            framework_coverage[framework.value] = {
                "supporting_regions": supporting_regions,
                "coverage_percentage": len(supporting_regions) / len(Region) * 100,
                "rules": self.compliance_rules.get(framework, {})
            }

        region_compliance_summary = {}
        for region, frameworks in self.region_compliance.items():
            region_compliance_summary[region.value] = {
                "supported_frameworks": [f.value for f in frameworks],
                "compliance_count": len(frameworks),
                "compliance_percentage": len(frameworks) / len(ComplianceFramework) * 100
            }

        return {
            "framework_coverage": framework_coverage,
            "region_compliance": region_compliance_summary,
            "total_frameworks": len(ComplianceFramework),
            "total_regions": len(Region),
            "global_compliance_score": sum(
                len(frameworks) for frameworks in self.region_compliance.values()
            ) / (len(ComplianceFramework) * len(Region)) * 100
        }


class GlobalScalingManager:
    """Manages global scaling and orchestration."""

    def __init__(self):
        self.load_balancer = GlobalLoadBalancer()
        self.compliance_manager = GlobalComplianceManager()
        self.performance_tracker = get_performance_tracker()

        # Scaling metrics
        self.scaling_history = deque(maxlen=1000)
        self.global_metrics = {
            "total_requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_global_latency": 0.0,
            "peak_concurrent_load": 0
        }

        logger.info("GlobalScalingManager initialized")

    async def process_request_globally(
        self,
        request_content: str,
        request_headers: Optional[Dict[str, str]] = None,
        processing_config: Optional[ProcessingConfig] = None,
        compliance_requirements: Optional[Set[ComplianceFramework]] = None,
        preferred_region: Optional[Region] = None
    ) -> Tuple[EnhancedTriageResult, Dict[str, Any]]:
        """Process request using global scaling and routing."""

        start_time = time.time()

        # Prepare request metadata
        request_metadata = {
            "content_length": len(request_content),
            "user_location": request_headers.get("location", "unknown") if request_headers else "unknown",
            "priority": processing_config.quality.value if processing_config else "balanced",
            "session_id": f"global_{int(start_time * 1000)}",
            "timestamp": start_time
        }

        # Route request to optimal region
        routing_decision = await self.load_balancer.route_request(
            request_metadata,
            compliance_requirements,
            preferred_region
        )

        # Validate compliance
        if routing_decision.target_region:
            compliance_validation = self.compliance_manager.validate_compliance(
                request_metadata,
                routing_decision.target_region,
                compliance_requirements
            )

            if not compliance_validation["compliant"]:
                # Handle compliance violation
                logger.error(f"Compliance violation: {compliance_validation['violations']}")

                # Try to find compliant region
                compliant_regions = self.compliance_manager.get_compliant_regions(
                    compliance_requirements or set()
                )

                if compliant_regions:
                    # Re-route to compliant region
                    routing_decision.target_region = compliant_regions[0]
                    routing_decision.routing_reason = "Compliance-driven re-routing"
                    routing_decision.compliance_satisfied = True
                else:
                    routing_decision.compliance_satisfied = False

        # Process request (simulate regional processing)
        processing_result = await self._simulate_regional_processing(
            request_content,
            request_headers,
            processing_config,
            routing_decision.target_region
        )

        # Calculate global processing metrics
        processing_time = (time.time() - start_time) * 1000

        # Update global metrics
        self.global_metrics["total_requests_processed"] += 1

        if processing_result.category != "error":
            self.global_metrics["successful_requests"] += 1
        else:
            self.global_metrics["failed_requests"] += 1

        # Update average latency
        total_requests = self.global_metrics["total_requests_processed"]
        current_avg = self.global_metrics["avg_global_latency"]
        self.global_metrics["avg_global_latency"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

        # Track peak load
        current_global_load = sum(
            capacity.current_load
            for capacity in self.load_balancer.region_capacities.values()
        )
        self.global_metrics["peak_concurrent_load"] = max(
            self.global_metrics["peak_concurrent_load"],
            current_global_load
        )

        # Create global processing summary
        global_summary = {
            "routing_decision": {
                "target_region": routing_decision.target_region.value if routing_decision.target_region else None,
                "routing_reason": routing_decision.routing_reason,
                "expected_latency_ms": routing_decision.expected_latency_ms,
                "compliance_satisfied": routing_decision.compliance_satisfied,
                "decision_time_ms": routing_decision.decision_time_ms,
                "fallback_regions": [r.value for r in routing_decision.fallback_regions],
                "alternative_options": routing_decision.alternative_options
            },
            "compliance_validation": self.compliance_manager.validate_compliance(
                request_metadata,
                routing_decision.target_region,
                compliance_requirements
            ) if routing_decision.target_region else None,
            "processing_metrics": {
                "total_processing_time_ms": processing_time,
                "regional_processing_time_ms": processing_result.processing_time_ms,
                "routing_overhead_ms": routing_decision.decision_time_ms,
                "network_latency_ms": routing_decision.expected_latency_ms
            },
            "global_context": {
                "total_regions_available": len([
                    c for c in self.load_balancer.region_capacities.values()
                    if c.is_available
                ]),
                "global_load_factor": current_global_load / sum(
                    c.max_concurrent_requests
                    for c in self.load_balancer.region_capacities.values()
                ),
                "compliance_frameworks_required": [
                    f.value for f in (compliance_requirements or set())
                ]
            }
        }

        # Store scaling history
        self.scaling_history.append({
            "timestamp": start_time,
            "request_metadata": request_metadata,
            "routing_decision": routing_decision,
            "processing_result": processing_result,
            "global_summary": global_summary
        })

        # Track performance
        self.performance_tracker.record_operation(
            "global_request_processing",
            processing_time / 1000,
            {
                "target_region": routing_decision.target_region.value if routing_decision.target_region else "none",
                "compliance_satisfied": routing_decision.compliance_satisfied,
                "success": processing_result.category != "error"
            }
        )

        logger.info(
            f"Global request processed: {routing_decision.target_region.value if routing_decision.target_region else 'FAILED'} "
            f"({processing_time:.2f}ms total, compliance: {routing_decision.compliance_satisfied})"
        )

        return processing_result, global_summary

    async def _simulate_regional_processing(
        self,
        content: str,
        headers: Optional[Dict[str, str]],
        config: Optional[ProcessingConfig],
        target_region: Optional[Region]
    ) -> EnhancedTriageResult:
        """Simulate regional processing (in real implementation, this would call regional service)."""

        if not target_region:
            return EnhancedTriageResult(
                category="error",
                priority=5,
                summary="No available region for processing",
                response_suggestion="System temporarily unavailable",
                confidence_score=0.0,
                processing_mode=config.mode if config else ProcessingMode.STANDARD,
                processing_quality=config.quality if config else ProcessingQuality.BALANCED,
                processing_time_ms=0.0,
                health_status="error",
                error_details={"error": "No available region"}
            )

        # Simulate processing delay based on region
        regional_latencies = {
            Region.US_EAST_1: 50,
            Region.US_WEST_2: 75,
            Region.EU_WEST_1: 100,
            Region.AP_SOUTHEAST_1: 150
        }

        base_latency = regional_latencies.get(target_region, 100)
        processing_delay = base_latency + len(content) * 0.01  # Content-dependent delay

        # Simulate processing
        await asyncio.sleep(processing_delay / 1000)  # Convert to seconds

        # Create result based on content analysis
        if any(word in content.lower() for word in ["urgent", "critical", "emergency"]):
            category = "urgent"
            priority = 9
        elif any(word in content.lower() for word in ["meeting", "schedule"]):
            category = "meeting"
            priority = 6
        elif any(word in content.lower() for word in ["invoice", "payment", "billing"]):
            category = "billing"
            priority = 7
        else:
            category = "general"
            priority = 5

        return EnhancedTriageResult(
            category=category,
            priority=priority,
            summary=f"Email processed in {target_region.value} region",
            response_suggestion=f"Thank you for your {category} message. Processing completed in {target_region.value}.",
            confidence_score=0.8,
            processing_mode=config.mode if config else ProcessingMode.STANDARD,
            processing_quality=config.quality if config else ProcessingQuality.BALANCED,
            processing_time_ms=processing_delay,
            health_status="healthy",
            model_used=f"regional_model_{target_region.value}"
        )

    async def update_regional_health(self, region: Region, health_metrics: Dict[str, Any]) -> None:
        """Update health metrics for a specific region."""
        await self.load_balancer.update_region_status(region, health_metrics)

    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global scaling status."""

        load_balancer_status = self.load_balancer.get_global_status()
        compliance_report = self.compliance_manager.get_compliance_report()

        return {
            "global_metrics": self.global_metrics,
            "load_balancing": load_balancer_status,
            "compliance": compliance_report,
            "scaling_history_size": len(self.scaling_history),
            "performance_summary": {
                "success_rate": (
                    self.global_metrics["successful_requests"] /
                    max(self.global_metrics["total_requests_processed"], 1)
                ),
                "avg_latency_ms": self.global_metrics["avg_global_latency"],
                "peak_concurrent_load": self.global_metrics["peak_concurrent_load"],
                "current_global_load": sum(
                    c.current_load for c in self.load_balancer.region_capacities.values()
                ),
                "total_global_capacity": sum(
                    c.max_concurrent_requests for c in self.load_balancer.region_capacities.values()
                )
            },
            "recommendations": self._generate_scaling_recommendations()
        }

    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations based on current metrics."""
        recommendations = []

        # Check global success rate
        success_rate = (
            self.global_metrics["successful_requests"] /
            max(self.global_metrics["total_requests_processed"], 1)
        )

        if success_rate < 0.95:
            recommendations.append(
                f"Global success rate ({success_rate:.1%}) is below optimal. Consider adding capacity or improving health monitoring."
            )

        # Check average latency
        if self.global_metrics["avg_global_latency"] > 2000:  # 2 seconds
            recommendations.append(
                "High average latency detected. Consider optimizing routing strategy or adding regional capacity."
            )

        # Check regional distribution
        region_loads = [c.current_load for c in self.load_balancer.region_capacities.values()]
        if region_loads and max(region_loads) > 0:
            load_imbalance = max(region_loads) - statistics.mean(region_loads)
            if load_imbalance > 100:  # Significant imbalance
                recommendations.append(
                    "Significant load imbalance detected between regions. Consider adjusting routing weights."
                )

        # Check compliance coverage
        compliance_report = self.compliance_manager.get_compliance_report()
        if compliance_report["global_compliance_score"] < 50:
            recommendations.append(
                "Low global compliance score. Consider enabling more compliance frameworks in additional regions."
            )

        return recommendations


# Global instance
_global_scaling_manager: Optional[GlobalScalingManager] = None


def get_global_scaling_manager() -> GlobalScalingManager:
    """Get the global scaling manager instance."""
    global _global_scaling_manager
    if _global_scaling_manager is None:
        _global_scaling_manager = GlobalScalingManager()
    return _global_scaling_manager


# Convenience functions
async def process_email_globally(
    content: str,
    headers: Optional[Dict[str, str]] = None,
    config: Optional[ProcessingConfig] = None,
    compliance: Optional[Set[ComplianceFramework]] = None,
    preferred_region: Optional[Region] = None
) -> Tuple[EnhancedTriageResult, Dict[str, Any]]:
    """Process email using global scaling infrastructure."""
    manager = get_global_scaling_manager()
    return await manager.process_request_globally(
        content, headers, config, compliance, preferred_region
    )


def get_global_insights() -> Dict[str, Any]:
    """Get comprehensive global scaling insights."""
    manager = get_global_scaling_manager()
    return manager.get_global_status()
