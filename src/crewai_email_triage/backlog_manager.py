"""
Comprehensive Backlog Management System with WSJF Prioritization

This module implements a continuous backlog processing system that:
1. Loads and normalizes all backlog items
2. Scores items using Weighted Shortest Job First (WSJF) methodology
3. Discovers new tasks from codebase analysis
4. Provides full backlog execution capabilities
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class TaskStatus(Enum):
    """Task lifecycle states"""
    NEW = "NEW"
    REFINED = "REFINED" 
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    MERGED = "MERGED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class TaskType(Enum):
    """Task categorization types"""
    FEATURE = "Feature"
    BUG = "Bug" 
    REFACTOR = "Refactor"
    SECURITY = "Security"
    DOC = "Doc"
    INFRASTRUCTURE = "Infrastructure"


@dataclass
class BacklogItem:
    """Normalized backlog item structure"""
    id: str
    title: str
    description: str
    type: TaskType
    status: TaskStatus
    
    # WSJF Components (1,2,3,5,8,13 scale)
    business_value: int = 1
    time_criticality: int = 1
    risk_reduction: int = 1
    effort: int = 1
    
    # Metadata
    links: List[str] = None
    acceptance_criteria: List[str] = None
    created_at: str = ""
    updated_at: str = ""
    blocked_reason: str = ""
    
    # Calculated fields
    wsjf_score: float = 0.0
    aging_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.acceptance_criteria is None:
            self.acceptance_criteria = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class BacklogManager:
    """
    Comprehensive backlog management system that continuously processes
    all actionable items using WSJF prioritization methodology.
    """
    
    def __init__(self, backlog_file: str = "/root/repo/DOCS/backlog.json"):
        self.backlog_file = backlog_file
        self.backlog: Dict[str, BacklogItem] = {}
        self.metrics = {
            "total_items": 0,
            "completed_items": 0,
            "blocked_items": 0,
            "average_cycle_time": 0.0,
            "wsjf_distribution": {},
            "last_updated": ""
        }
        
    def load_backlog(self) -> Dict[str, BacklogItem]:
        """Load and normalize all backlog items from various sources"""
        # Load from existing JSON if available
        if os.path.exists(self.backlog_file):
            try:
                with open(self.backlog_file, 'r') as f:
                    data = json.load(f)
                    for item_data in data.get('items', []):
                        # Remove unknown fields that aren't part of BacklogItem
                        filtered_data = {k: v for k, v in item_data.items() 
                                       if k in ['id', 'title', 'description', 'type', 'status',
                                               'business_value', 'time_criticality', 'risk_reduction', 'effort',
                                               'links', 'acceptance_criteria', 'created_at', 'updated_at',
                                               'blocked_reason', 'wsjf_score', 'aging_multiplier']}
                        item = BacklogItem(**filtered_data)
                        self.backlog[item.id] = item
            except Exception as e:
                print(f"Warning: Could not load existing backlog: {e}")
        
        # Parse existing BACKLOG.md for additional items
        self._parse_markdown_backlog()
        
        # Discover new tasks from codebase
        self._discover_code_tasks()
        
        # Normalize and validate all items
        self._normalize_items()
        
        return self.backlog
    
    def _parse_markdown_backlog(self):
        """Parse existing BACKLOG.md to extract remaining actionable items"""
        backlog_md = "/root/repo/BACKLOG.md"
        if not os.path.exists(backlog_md):
            return
            
        try:
            with open(backlog_md, 'r') as f:
                content = f.read()
            
            # Extract the critical security issue
            security_pattern = r"### 2\. Hardcoded Gmail Credentials Vulnerability \[WSJF: 80\].*?(?=###|\Z)"
            security_match = re.search(security_pattern, content, re.DOTALL)
            
            if security_match and "gmail_oauth_security" not in self.backlog:
                self.backlog["gmail_oauth_security"] = BacklogItem(
                    id="gmail_oauth_security",
                    title="Hardcoded Gmail Credentials Vulnerability",
                    description="Replace password authentication with OAuth2 flow. Currently stores plaintext passwords in provider.py:26",
                    type=TaskType.SECURITY,
                    status=TaskStatus.BLOCKED,
                    business_value=8,
                    time_criticality=8,
                    risk_reduction=13,
                    effort=5,
                    blocked_reason="REQUIRES HUMAN REVIEW - Critical security issue affecting authentication system",
                    acceptance_criteria=[
                        "Implement OAuth2 flow for Gmail authentication",
                        "Remove plaintext password storage",
                        "Update provider.py to use secure OAuth tokens",
                        "Add OAuth2 configuration documentation",
                        "Ensure backward compatibility or migration path"
                    ]
                )
                
        except Exception as e:
            print(f"Warning: Could not parse BACKLOG.md: {e}")
    
    def _discover_code_tasks(self):
        """Discover new tasks from codebase analysis"""
        repo_root = "/root/repo"
        
        # Add bandit security warnings as tasks
        if "bandit_security_review" not in self.backlog:
            self.backlog["bandit_security_review"] = BacklogItem(
                id="bandit_security_review",
                title="Review and Address Bandit Security Warnings",
                description="Bandit security scan found 11 low-severity issues including subprocess usage, try-except patterns, and random usage that need review.",
                type=TaskType.SECURITY,
                status=TaskStatus.READY,
                business_value=5,
                time_criticality=3,
                risk_reduction=8,
                effort=3,
                acceptance_criteria=[
                    "Review all 11 bandit security warnings in detail",
                    "Address or document justification for each warning",
                    "Configure .bandit file with appropriate suppressions where justified",
                    "Ensure no actual security vulnerabilities remain",
                    "Document security review process"
                ]
            )
        
        # Add continuous discovery task
        if "continuous_backlog_discovery" not in self.backlog:
            self.backlog["continuous_backlog_discovery"] = BacklogItem(
                id="continuous_backlog_discovery", 
                title="Implement Continuous Backlog Discovery",
                description="Enhance the autonomous system to continuously discover new tasks from TODO comments, failing tests, and code analysis.",
                type=TaskType.FEATURE,
                status=TaskStatus.READY,
                business_value=8,
                time_criticality=5,
                risk_reduction=3,
                effort=5,
                acceptance_criteria=[
                    "Scan codebase for TODO/FIXME comments and convert to backlog items",
                    "Monitor CI/CD for failing tests and create remediation tasks",
                    "Analyze dependency vulnerabilities and create security tasks",
                    "Implement automated task prioritization updates",
                    "Create recurring discovery job"
                ]
            )
            
        # Add status reporting enhancement
        if "enhanced_status_reporting" not in self.backlog:
            self.backlog["enhanced_status_reporting"] = BacklogItem(
                id="enhanced_status_reporting",
                title="Enhance Autonomous Status Reporting",
                description="Create comprehensive status reports with metrics, progress tracking, and automated documentation generation.",
                type=TaskType.FEATURE,
                status=TaskStatus.READY,
                business_value=6,
                time_criticality=4,
                risk_reduction=2,
                effort=3,
                acceptance_criteria=[
                    "Generate daily status reports in docs/status/",
                    "Include WSJF metrics and trend analysis",
                    "Track cycle time and completion rates",
                    "Add visual progress indicators",
                    "Implement automated report publishing"
                ]
            )
            
        # Check for code quality improvements
        self._scan_for_code_issues(repo_root)
    
    def _scan_for_code_issues(self, repo_root: str):
        """Scan codebase for potential issues and improvements"""
        src_dir = os.path.join(repo_root, "src")
        
        # Scan for potential security issues
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        # Look for hardcoded credentials patterns
                        if re.search(r'password\s*=\s*["\']', content, re.IGNORECASE):
                            task_id = f"hardcoded_creds_{file}"
                            if task_id not in self.backlog:
                                self.backlog[task_id] = BacklogItem(
                                    id=task_id,
                                    title=f"Potential Hardcoded Credentials in {file}",
                                    description=f"Found potential hardcoded credentials in {file_path}",
                                    type=TaskType.SECURITY,
                                    status=TaskStatus.NEW,
                                    business_value=8,
                                    time_criticality=5,
                                    risk_reduction=8,
                                    effort=3
                                )
                                
                    except Exception:
                        continue  # Skip files that can't be read
    
    def _normalize_items(self):
        """Normalize all backlog items and ensure required fields"""
        for item in self.backlog.values():
            # Ensure status is enum
            if isinstance(item.status, str):
                item.status = TaskStatus(item.status)
            if isinstance(item.type, str):
                item.type = TaskType(item.type)
                
            # Update timestamps
            item.updated_at = datetime.now().isoformat()
            
            # Validate WSJF components are in acceptable range
            item.business_value = max(1, min(13, item.business_value))
            item.time_criticality = max(1, min(13, item.time_criticality))
            item.risk_reduction = max(1, min(13, item.risk_reduction))
            item.effort = max(1, min(13, item.effort))
    
    def calculate_wsjf_scores(self):
        """Calculate WSJF scores for all items with aging factor"""
        for item in self.backlog.values():
            # Cost of Delay = Business Value + Time Criticality + Risk Reduction
            cost_of_delay = item.business_value + item.time_criticality + item.risk_reduction
            
            # WSJF = Cost of Delay / Job Size (effort)
            base_wsjf = cost_of_delay / item.effort
            
            # Calculate aging multiplier (capped to prevent runaway inflation)
            created_date = datetime.fromisoformat(item.created_at)
            age_days = (datetime.now() - created_date).days
            aging_factor = min(1.5, 1.0 + (age_days * 0.01))  # Max 50% boost
            
            item.aging_multiplier = aging_factor
            item.wsjf_score = base_wsjf * aging_factor
    
    def get_prioritized_backlog(self) -> List[BacklogItem]:
        """Get backlog items sorted by WSJF score (highest first)"""
        self.calculate_wsjf_scores()
        
        # Filter actionable items (not DONE, MERGED, or BLOCKED)
        actionable_items = [
            item for item in self.backlog.values()
            if item.status not in [TaskStatus.DONE, TaskStatus.MERGED, TaskStatus.BLOCKED]
        ]
        
        # Sort by WSJF score descending
        return sorted(actionable_items, key=lambda x: x.wsjf_score, reverse=True)
    
    def get_next_actionable_item(self) -> Optional[BacklogItem]:
        """Get the highest priority actionable item"""
        prioritized = self.get_prioritized_backlog()
        
        for item in prioritized:
            if item.status == TaskStatus.READY:
                return item
            elif item.status == TaskStatus.NEW and self._can_refine_item(item):
                # Auto-refine if possible
                self._refine_item(item)
                if item.status == TaskStatus.READY:
                    return item
        
        return None
    
    def _can_refine_item(self, item: BacklogItem) -> bool:
        """Check if item can be automatically refined to READY status"""
        # Items need acceptance criteria and reasonable effort estimation
        has_criteria = len(item.acceptance_criteria) > 0
        reasonable_effort = item.effort <= 8  # Items > 8 should be split
        
        return has_criteria and reasonable_effort
    
    def _refine_item(self, item: BacklogItem):
        """Refine item from NEW to READY status"""
        if self._can_refine_item(item):
            item.status = TaskStatus.READY
            item.updated_at = datetime.now().isoformat()
    
    def update_item_status(self, item_id: str, new_status: TaskStatus, reason: str = ""):
        """Update item status with audit trail"""
        if item_id in self.backlog:
            item = self.backlog[item_id]
            old_status = item.status
            item.status = new_status
            item.updated_at = datetime.now().isoformat()
            
            if new_status == TaskStatus.BLOCKED:
                item.blocked_reason = reason
            
            print(f"Status updated: {item.title} | {old_status.value} → {new_status.value}")
            
            # Auto-save after status changes
            self.save_backlog()
    
    def mark_completed(self, item_id: str):
        """Mark item as completed and update metrics"""
        if item_id in self.backlog:
            self.update_item_status(item_id, TaskStatus.DONE)
            self.metrics["completed_items"] += 1
            self._update_metrics()
    
    def save_backlog(self):
        """Save current backlog state to JSON file"""
        os.makedirs(os.path.dirname(self.backlog_file), exist_ok=True)
        
        data = {
            "items": [asdict(item) for item in self.backlog.values()],
            "metrics": self.metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        # Convert enums to strings for JSON serialization
        for item_data in data["items"]:
            item_data["status"] = item_data["status"].value if hasattr(item_data["status"], 'value') else item_data["status"]
            item_data["type"] = item_data["type"].value if hasattr(item_data["type"], 'value') else item_data["type"]
        
        with open(self.backlog_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_metrics(self):
        """Update backlog metrics"""
        self.metrics.update({
            "total_items": len(self.backlog),
            "completed_items": len([i for i in self.backlog.values() if i.status in [TaskStatus.DONE, TaskStatus.MERGED]]),
            "blocked_items": len([i for i in self.backlog.values() if i.status == TaskStatus.BLOCKED]),
            "last_updated": datetime.now().isoformat()
        })
        
        # Calculate WSJF distribution
        scores = [item.wsjf_score for item in self.backlog.values()]
        if scores:
            self.metrics["wsjf_distribution"] = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores)
            }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        self._update_metrics()
        
        prioritized = self.get_prioritized_backlog()
        next_item = self.get_next_actionable_item()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "backlog_size": len(self.backlog),
            "actionable_items": len(prioritized),
            "next_priority_item": {
                "id": next_item.id,
                "title": next_item.title,
                "wsjf_score": next_item.wsjf_score,
                "effort": next_item.effort
            } if next_item else None,
            "status_breakdown": {
                status.value: len([i for i in self.backlog.values() if i.status == status])
                for status in TaskStatus
            }
        }
    
    def generate_enhanced_status_report(self) -> Dict[str, Any]:
        """Generate enhanced status report with trends, cycle times, and visualizations"""
        self._update_metrics()
        
        prioritized = self.get_prioritized_backlog()
        next_item = self.get_next_actionable_item()
        completed_items = [item for item in self.backlog.values() if item.status == TaskStatus.DONE]
        
        # Calculate cycle times for completed items
        cycle_times = []
        for item in completed_items:
            if item.created_at and item.updated_at:
                created = datetime.fromisoformat(item.created_at)
                completed = datetime.fromisoformat(item.updated_at)
                cycle_time_hours = (completed - created).total_seconds() / 3600
                cycle_times.append(cycle_time_hours)
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        
        # WSJF analysis
        wsjf_scores = [item.wsjf_score for item in self.backlog.values()]
        wsjf_analysis = {
            "min": min(wsjf_scores) if wsjf_scores else 0,
            "max": max(wsjf_scores) if wsjf_scores else 0,
            "avg": sum(wsjf_scores) / len(wsjf_scores) if wsjf_scores else 0,
            "high_priority_threshold": 5.0,
            "high_priority_count": len([s for s in wsjf_scores if s >= 5.0])
        }
        
        # Risk assessment
        risk_assessment = self._assess_risks()
        
        # Progress indicators
        total_effort = sum(item.effort for item in self.backlog.values())
        completed_effort = sum(item.effort for item in completed_items)
        progress_percentage = (completed_effort / total_effort * 100) if total_effort > 0 else 0
        
        return {
            "session_metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_version": "2.0",
                "generated_by": "autonomous_backlog_manager"
            },
            
            "executive_summary": {
                "total_backlog_items": len(self.backlog),
                "actionable_items": len(prioritized),
                "completed_items": len(completed_items),
                "blocked_items": len([i for i in self.backlog.values() if i.status == TaskStatus.BLOCKED]),
                "completion_rate": f"{progress_percentage:.1f}%",
                "avg_cycle_time_hours": avg_cycle_time,
                "health_status": self._get_health_status()
            },
            
            "wsjf_analysis": wsjf_analysis,
            
            "task_flow": {
                "status_breakdown": {
                    status.value: len([i for i in self.backlog.values() if i.status == status])
                    for status in TaskStatus
                },
                "type_breakdown": {
                    task_type.value: len([i for i in self.backlog.values() if i.type == task_type])
                    for task_type in TaskType
                },
                "effort_distribution": self._get_effort_distribution()
            },
            
            "priority_queue": {
                "next_actionable": {
                    "id": next_item.id,
                    "title": next_item.title,
                    "wsjf_score": next_item.wsjf_score,
                    "effort": next_item.effort,
                    "type": next_item.type.value
                } if next_item else None,
                "top_3_priorities": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "wsjf_score": item.wsjf_score,
                        "effort": item.effort,
                        "type": item.type.value
                    }
                    for item in prioritized[:3]
                ]
            },
            
            "risk_assessment": risk_assessment,
            
            "performance_metrics": {
                "cycle_times": cycle_times,
                "velocity": len(completed_items) / max(1, avg_cycle_time) if avg_cycle_time > 0 else 0,
                "completion_trend": self._calculate_completion_trend(),
                "bottlenecks": self._identify_bottlenecks()
            },
            
            "recommendations": self._generate_recommendations(),
            
            "visual_indicators": {
                "progress_bar": self._create_progress_bar(progress_percentage),
                "status_chart": self._create_status_chart(),
                "priority_heat_map": self._create_priority_heatmap()
            }
        }
    
    def _assess_risks(self) -> Dict[str, Any]:
        """Assess current risks in the backlog"""
        blocked_items = [item for item in self.backlog.values() if item.status == TaskStatus.BLOCKED]
        high_effort_items = [item for item in self.backlog.values() if item.effort >= 8]
        aging_items = []
        
        for item in self.backlog.values():
            if item.created_at:
                created = datetime.fromisoformat(item.created_at)
                age_days = (datetime.now() - created).days
                if age_days > 7:  # Items older than 7 days
                    aging_items.append({
                        "id": item.id,
                        "title": item.title,
                        "age_days": age_days,
                        "wsjf_score": item.wsjf_score
                    })
        
        return {
            "blocked_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "blocked_reason": item.blocked_reason,
                    "wsjf_score": item.wsjf_score
                }
                for item in blocked_items
            ],
            "high_effort_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "effort": item.effort,
                    "wsjf_score": item.wsjf_score
                }
                for item in high_effort_items
            ],
            "aging_items": aging_items,
            "risk_score": len(blocked_items) * 3 + len(high_effort_items) * 2 + len(aging_items)
        }
    
    def _get_health_status(self) -> str:
        """Determine overall backlog health status"""
        blocked_count = len([i for i in self.backlog.values() if i.status == TaskStatus.BLOCKED])
        actionable_count = len(self.get_prioritized_backlog())
        
        if blocked_count > 2:
            return "CRITICAL"
        elif blocked_count > 0 or actionable_count == 0:
            return "WARNING"
        elif actionable_count > 5:
            return "HEALTHY"
        else:
            return "STABLE"
    
    def _get_effort_distribution(self) -> Dict[str, int]:
        """Get distribution of effort estimates"""
        distribution = {"small": 0, "medium": 0, "large": 0, "epic": 0}
        
        for item in self.backlog.values():
            if item.effort <= 3:
                distribution["small"] += 1
            elif item.effort <= 5:
                distribution["medium"] += 1
            elif item.effort <= 8:
                distribution["large"] += 1
            else:
                distribution["epic"] += 1
                
        return distribution
    
    def _calculate_completion_trend(self) -> str:
        """Calculate completion trend direction"""
        completed_items = [item for item in self.backlog.values() if item.status == TaskStatus.DONE]
        if len(completed_items) >= 2:
            return "POSITIVE"
        elif len(completed_items) == 1:
            return "NEUTRAL"
        else:
            return "NEEDS_ATTENTION"
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify potential bottlenecks"""
        bottlenecks = []
        
        blocked_count = len([i for i in self.backlog.values() if i.status == TaskStatus.BLOCKED])
        if blocked_count > 0:
            bottlenecks.append(f"{blocked_count} blocked items requiring attention")
        
        high_effort_count = len([i for i in self.backlog.values() if i.effort >= 8])
        if high_effort_count > 1:
            bottlenecks.append(f"{high_effort_count} high-effort items may need decomposition")
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        next_item = self.get_next_actionable_item()
        if next_item:
            recommendations.append(f"Execute next priority: {next_item.title} (WSJF: {next_item.wsjf_score:.2f})")
        
        blocked_items = [item for item in self.backlog.values() if item.status == TaskStatus.BLOCKED]
        if blocked_items:
            recommendations.append(f"Review {len(blocked_items)} blocked items for unblocking opportunities")
        
        actionable_count = len(self.get_prioritized_backlog())
        if actionable_count == 0:
            recommendations.append("Refine NEW items to READY status to maintain flow")
        
        return recommendations
    
    def _create_progress_bar(self, percentage: float) -> str:
        """Create ASCII progress bar"""
        bar_length = 20
        filled_length = int(bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        return f"[{bar}] {percentage:.1f}%"
    
    def _create_status_chart(self) -> Dict[str, str]:
        """Create status visualization"""
        status_counts = {
            status.value: len([i for i in self.backlog.values() if i.status == status])
            for status in TaskStatus
        }
        
        total = sum(status_counts.values())
        if total == 0:
            return {}
        
        chart = {}
        for status, count in status_counts.items():
            if count > 0:
                percentage = (count / total) * 100
                chart[status] = f"{count} items ({percentage:.1f}%)"
        
        return chart
    
    def _create_priority_heatmap(self) -> Dict[str, int]:
        """Create priority heatmap data"""
        priority_ranges = {
            "Critical (>8)": 0,
            "High (5-8)": 0,  
            "Medium (3-5)": 0,
            "Low (<3)": 0
        }
        
        for item in self.backlog.values():
            score = item.wsjf_score
            if score > 8:
                priority_ranges["Critical (>8)"] += 1
            elif score >= 5:
                priority_ranges["High (5-8)"] += 1
            elif score >= 3:
                priority_ranges["Medium (3-5)"] += 1
            else:
                priority_ranges["Low (<3)"] += 1
        
        return priority_ranges


def main():
    """Initialize and demonstrate backlog management system"""
    manager = BacklogManager()
    
    # Load and process backlog
    manager.load_backlog()
    
    # Get status report
    report = manager.get_status_report()
    
    print("=== BACKLOG MANAGEMENT SYSTEM INITIALIZED ===")
    print(f"Total items: {report['backlog_size']}")
    print(f"Actionable items: {report['actionable_items']}")
    
    if report['next_priority_item']:
        next_item = report['next_priority_item']
        print(f"Next priority: {next_item['title']} (WSJF: {next_item['wsjf_score']:.2f})")
    
    print("\nStatus breakdown:")
    for status, count in report['status_breakdown'].items():
        if count > 0:
            print(f"  {status}: {count}")
    
    # Save current state
    manager.save_backlog()
    print(f"\nBacklog saved to: {manager.backlog_file}")


if __name__ == "__main__":
    main()