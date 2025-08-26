"""
Global Deployment Manager
Manages multi-region deployments with intelligent routing and failover.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import time
import random


class RegionStatus(Enum):
    """Deployment region status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"


@dataclass
class Region:
    """Deployment region configuration."""
    code: str
    name: str
    endpoint: str
    primary: bool = False
    capacity: int = 100
    current_load: int = 0
    status: RegionStatus = RegionStatus.HEALTHY
    latency_ms: float = 0.0
    error_rate: float = 0.0
    compliance_zones: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    version: str
    strategy: DeploymentStrategy
    regions: List[str]
    rollout_percentage: int = 100
    health_check_path: str = "/health"
    max_error_rate: float = 0.05
    max_latency_ms: float = 2000
    auto_rollback: bool = True


class GlobalDeploymentManager:
    """Manages global deployments across multiple regions."""
    
    def __init__(self):
        self.logger = logging.getLogger("global_deployment_manager")
        self.regions: Dict[str, Region] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self._initialize_regions()
    
    def _initialize_regions(self):
        """Initialize deployment regions."""
        
        regions_config = [
            {
                "code": "us-east-1",
                "name": "US East (N. Virginia)",
                "endpoint": "https://api-us-east-1.crewai-triage.com",
                "primary": True,
                "compliance_zones": ["us", "global"],
                "languages": ["en", "es"],
                "capacity": 1000
            },
            {
                "code": "eu-west-1", 
                "name": "EU West (Ireland)",
                "endpoint": "https://api-eu-west-1.crewai-triage.com",
                "primary": False,
                "compliance_zones": ["eu", "uk", "global"],
                "languages": ["en", "de", "fr", "es", "it"],
                "capacity": 800
            },
            {
                "code": "ap-southeast-1",
                "name": "Asia Pacific (Singapore)", 
                "endpoint": "https://api-ap-southeast-1.crewai-triage.com",
                "primary": False,
                "compliance_zones": ["sg", "au", "global"],
                "languages": ["en", "zh", "ja", "ko"],
                "capacity": 600
            }
        ]
        
        for region_config in regions_config:
            region = Region(
                code=region_config["code"],
                name=region_config["name"],
                endpoint=region_config["endpoint"],
                primary=region_config["primary"],
                capacity=region_config["capacity"],
                compliance_zones=region_config["compliance_zones"],
                languages=region_config["languages"]
            )
            self.regions[region.code] = region
    
    def get_optimal_region(self, user_location: Optional[str] = None, 
                          compliance_zone: Optional[str] = None,
                          language: Optional[str] = None) -> Optional[Region]:
        """Get optimal region based on user requirements."""
        
        available_regions = [
            region for region in self.regions.values() 
            if region.status == RegionStatus.HEALTHY and region.current_load < region.capacity * 0.9
        ]
        
        if not available_regions:
            self.logger.warning("No healthy regions available")
            return None
        
        # Filter by compliance zone
        if compliance_zone:
            compliant_regions = [
                region for region in available_regions 
                if compliance_zone in region.compliance_zones or "global" in region.compliance_zones
            ]
            if compliant_regions:
                available_regions = compliant_regions
        
        # Filter by language support
        if language:
            language_regions = [
                region for region in available_regions 
                if language in region.languages
            ]
            if language_regions:
                available_regions = language_regions
        
        # Score regions based on multiple factors
        scored_regions = []
        for region in available_regions:
            score = self._calculate_region_score(region, user_location)
            scored_regions.append((region, score))
        
        # Sort by score (higher is better)
        scored_regions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_regions[0][0] if scored_regions else None
    
    def _calculate_region_score(self, region: Region, user_location: Optional[str] = None) -> float:
        """Calculate region score based on multiple factors."""
        
        score = 100.0
        
        # Latency factor (lower is better)
        if region.latency_ms > 0:
            latency_penalty = min(region.latency_ms / 100, 50)  # Max 50 point penalty
            score -= latency_penalty
        
        # Load factor (lower is better)
        load_ratio = region.current_load / region.capacity
        load_penalty = load_ratio * 30  # Max 30 point penalty
        score -= load_penalty
        
        # Error rate factor (lower is better)
        error_penalty = region.error_rate * 100  # Max penalty based on error rate
        score -= error_penalty
        
        # Primary region bonus
        if region.primary:
            score += 10
        
        # Geographic proximity bonus (simplified)
        if user_location:
            proximity_bonus = self._get_proximity_bonus(region.code, user_location)
            score += proximity_bonus
        
        return max(score, 0)
    
    def _get_proximity_bonus(self, region_code: str, user_location: str) -> float:
        """Get proximity bonus based on user location."""
        
        proximity_map = {
            ("us-east-1", "us"): 20,
            ("us-east-1", "ca"): 15,
            ("eu-west-1", "eu"): 20,
            ("eu-west-1", "uk"): 25,
            ("eu-west-1", "africa"): 10,
            ("ap-southeast-1", "sg"): 25,
            ("ap-southeast-1", "au"): 20,
            ("ap-southeast-1", "jp"): 15,
            ("ap-southeast-1", "asia"): 15,
        }
        
        return proximity_map.get((region_code, user_location.lower()), 0)
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global deployment status."""
        
        total_capacity = sum(region.capacity for region in self.regions.values())
        total_load = sum(region.current_load for region in self.regions.values())
        
        healthy_regions = len([r for r in self.regions.values() if r.status == RegionStatus.HEALTHY])
        total_regions = len(self.regions)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_regions': total_regions,
            'healthy_regions': healthy_regions,
            'global_health_percentage': (healthy_regions / total_regions) * 100,
            'global_capacity': total_capacity,
            'global_load': total_load,
            'global_utilization': (total_load / total_capacity) * 100,
            'regions': {
                code: {
                    'name': region.name,
                    'status': region.status.value,
                    'load_percentage': (region.current_load / region.capacity) * 100,
                    'latency_ms': region.latency_ms,
                    'error_rate': region.error_rate * 100
                }
                for code, region in self.regions.items()
            },
            'recent_deployments': len(self.deployment_history[-10:])
        }


# Global deployment manager instance
_deployment_manager: Optional[GlobalDeploymentManager] = None


def get_deployment_manager() -> GlobalDeploymentManager:
    """Get or create global deployment manager instance."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = GlobalDeploymentManager()
    return _deployment_manager