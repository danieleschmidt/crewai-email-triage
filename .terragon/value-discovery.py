#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes work items using WSJF, ICE, and technical debt analysis.
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# import yaml  # Use json for config instead

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    category: str
    source: str
    priority: str
    effort_estimate: float
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    discovered_at: str
    status: str = "pending"
    files_affected: List[str] = None
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []

class ValueDiscoveryEngine:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        # Use default config if file doesn't exist or has issues
        default_config = {
            "scoring": {
                "weights": {
                    "advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}
                },
                "thresholds": {"securityBoost": 2.0}
            },
            "maturity": {"level": "advanced"}
        }
        return default_config
    
    def _load_metrics(self) -> Dict:
        """Load existing value metrics."""
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return {
            "executionHistory": [],
            "backlogMetrics": {
                "totalItems": 0,
                "averageAge": 0,
                "debtRatio": 0,
                "velocityTrend": "unknown"
            }
        }
    
    def discover_work_items(self) -> List[WorkItem]:
        """Discover work items from multiple sources."""
        work_items = []
        
        # Discover from code comments
        work_items.extend(self._discover_from_code_comments())
        
        # Discover from static analysis
        work_items.extend(self._discover_from_static_analysis())
        
        # Discover from security analysis
        work_items.extend(self._discover_from_security_analysis())
        
        # Discover from dependency analysis  
        work_items.extend(self._discover_from_dependencies())
        
        # Discover from performance analysis
        work_items.extend(self._discover_from_performance())
        
        return work_items
    
    def _discover_from_code_comments(self) -> List[WorkItem]:
        """Discover TODO, FIXME, HACK items from code."""
        work_items = []
        patterns = [
            (r'TODO[:\s]+(.+)', 'technical-debt', 'Medium'),
            (r'FIXME[:\s]+(.+)', 'bug-fix', 'High'),  
            (r'HACK[:\s]+(.+)', 'technical-debt', 'High'),
            (r'XXX[:\s]+(.+)', 'technical-debt', 'Medium'),
            (r'DEPRECATED[:\s]+(.+)', 'modernization', 'Low')
        ]
        
        try:
            result = subprocess.run(['grep', '-r', '-n', '-E', 'TODO|FIXME|HACK|XXX|DEPRECATED', 
                                   str(self.repo_path / 'src')], 
                                  capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                for pattern, category, priority in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        file_path = line.split(':')[0]
                        line_num = line.split(':')[1]
                        description = match.group(1).strip()
                        
                        work_item = WorkItem(
                            id=f"code-comment-{hash(line) % 10000}",
                            title=f"{category.title()}: {description[:50]}...",
                            description=f"Found in {file_path}:{line_num} - {description}",
                            category=category,
                            source="code-comments",
                            priority=priority,
                            effort_estimate=self._estimate_effort(description, category),
                            wsjf_score=0,
                            ice_score=0, 
                            technical_debt_score=0,
                            composite_score=0,
                            discovered_at=datetime.now().isoformat(),
                            files_affected=[file_path]
                        )
                        work_items.append(work_item)
                        break
        except Exception as e:
            logger.warning(f"Could not scan code comments: {e}")
            
        return work_items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Discover issues from static analysis tools."""
        work_items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run(['ruff', 'check', '--output-format=json', str(self.repo_path / 'src')],
                                  capture_output=True, text=True)
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:10]:  # Limit to top 10
                    work_item = WorkItem(
                        id=f"ruff-{issue.get('code', 'unknown')}-{hash(str(issue)) % 10000}",
                        title=f"Code Quality: {issue.get('message', 'Unknown issue')[:50]}",
                        description=f"Ruff violation {issue.get('code')} in {issue.get('filename')}:{issue.get('location', {}).get('row')} - {issue.get('message')}",
                        category="code-quality",
                        source="static-analysis",
                        priority="Medium",
                        effort_estimate=0.5,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        discovered_at=datetime.now().isoformat(),
                        files_affected=[issue.get('filename', '')]
                    )
                    work_items.append(work_item)
        except Exception as e:
            logger.warning(f"Could not run ruff analysis: {e}")
            
        return work_items
        
    def _discover_from_security_analysis(self) -> List[WorkItem]:
        """Discover security issues."""
        work_items = []
        
        # Run bandit for security issues
        try:
            result = subprocess.run(['bandit', '-f', 'json', '-r', str(self.repo_path / 'src')],
                                  capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                for issue in data.get('results', [])[:5]:  # Limit to top 5
                    work_item = WorkItem(
                        id=f"security-{issue.get('test_id', 'unknown')}-{hash(str(issue)) % 10000}",
                        title=f"Security: {issue.get('issue_text', 'Security issue')[:50]}",
                        description=f"Bandit {issue.get('test_id')} in {issue.get('filename')}:{issue.get('line_number')} - {issue.get('issue_text')}",
                        category="security",
                        source="security-analysis", 
                        priority="High",
                        effort_estimate=2.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        discovered_at=datetime.now().isoformat(),
                        files_affected=[issue.get('filename', '')]
                    )
                    work_items.append(work_item)
        except Exception as e:
            logger.warning(f"Could not run bandit analysis: {e}")
            
        return work_items
    
    def _discover_from_dependencies(self) -> List[WorkItem]:
        """Discover dependency update opportunities.""" 
        work_items = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(['safety', 'check', '--json'], capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                for vulnerability in data[:3]:  # Limit to top 3
                    work_item = WorkItem(
                        id=f"dependency-{vulnerability.get('id', 'unknown')}",
                        title=f"Dependency: Update {vulnerability.get('package_name', 'package')}",
                        description=f"Security vulnerability in {vulnerability.get('package_name')} {vulnerability.get('analyzed_version')} - {vulnerability.get('advisory', 'No details')}",
                        category="dependency-update",
                        source="dependency-analysis",
                        priority="High",
                        effort_estimate=1.0,
                        wsjf_score=0,
                        ice_score=0,
                        technical_debt_score=0,
                        composite_score=0,
                        discovered_at=datetime.now().isoformat()
                    )
                    work_items.append(work_item)
        except Exception as e:
            logger.warning(f"Could not run safety check: {e}")
            
        return work_items
    
    def _discover_from_performance(self) -> List[WorkItem]:
        """Discover performance optimization opportunities."""
        work_items = []
        
        # Check if benchmarks exist and can be run
        benchmark_file = self.repo_path / "tests" / "test_performance_benchmarks.py"
        if benchmark_file.exists():
            work_item = WorkItem(
                id="performance-optimization-001",
                title="Performance: Run and analyze benchmarks for optimization opportunities",
                description="Execute performance benchmarks to identify optimization opportunities and baseline performance regression prevention",
                category="performance",
                source="performance-analysis",
                priority="Medium",
                effort_estimate=3.0,
                wsjf_score=0,
                ice_score=0,
                technical_debt_score=0,
                composite_score=0,
                discovered_at=datetime.now().isoformat(),
                files_affected=[str(benchmark_file)]
            )
            work_items.append(work_item)
            
        return work_items
    
    def _estimate_effort(self, description: str, category: str) -> float:
        """Estimate effort in story points/hours."""
        base_effort = {
            'bug-fix': 2.0,
            'technical-debt': 3.0,
            'security': 4.0,
            'feature': 5.0,
            'performance': 3.0,
            'code-quality': 1.0,
            'dependency-update': 1.5,
            'modernization': 4.0
        }
        
        effort = base_effort.get(category, 2.0)
        
        # Adjust based on description complexity
        if any(word in description.lower() for word in ['refactor', 'rewrite', 'redesign']):
            effort *= 2
        elif any(word in description.lower() for word in ['simple', 'minor', 'trivial']):
            effort *= 0.5
            
        return max(0.5, effort)
    
    def calculate_scores(self, work_items: List[WorkItem]) -> List[WorkItem]:
        """Calculate WSJF, ICE, and composite scores for work items."""
        maturity_level = self.config.get('maturity', {}).get('level', 'developing')
        weights = self.config.get('scoring', {}).get('weights', {}).get(maturity_level, {
            'wsjf': 0.5, 'ice': 0.1, 'technicalDebt': 0.3, 'security': 0.1
        })
        
        for item in work_items:
            # Calculate WSJF (Cost of Delay / Job Size)
            business_value = self._score_business_value(item)
            time_criticality = self._score_time_criticality(item)
            risk_reduction = self._score_risk_reduction(item)
            
            cost_of_delay = business_value + time_criticality + risk_reduction
            item.wsjf_score = cost_of_delay / max(item.effort_estimate, 0.5)
            
            # Calculate ICE 
            impact = self._score_impact(item)
            confidence = self._score_confidence(item)
            ease = self._score_ease(item)
            item.ice_score = impact * confidence * ease
            
            # Calculate Technical Debt Score
            item.technical_debt_score = self._score_technical_debt(item)
            
            # Apply security/compliance boosts
            security_boost = 1.0
            if item.category == 'security':
                security_boost = self.config.get('scoring', {}).get('thresholds', {}).get('securityBoost', 2.0)
            
            # Calculate composite score
            item.composite_score = (
                weights.get('wsjf', 0.5) * self._normalize_score(item.wsjf_score, 0, 100) +
                weights.get('ice', 0.1) * self._normalize_score(item.ice_score, 0, 1000) +
                weights.get('technicalDebt', 0.3) * self._normalize_score(item.technical_debt_score, 0, 100) +
                weights.get('security', 0.1) * 50  # Base security score
            ) * security_boost
            
        return sorted(work_items, key=lambda x: x.composite_score, reverse=True)
    
    def _score_business_value(self, item: WorkItem) -> int:
        """Score business value impact (1-10)."""
        category_scores = {
            'security': 9,
            'bug-fix': 8,
            'performance': 7,
            'feature': 6,
            'technical-debt': 5,
            'code-quality': 4,
            'dependency-update': 3,
            'modernization': 4
        }
        return category_scores.get(item.category, 5)
    
    def _score_time_criticality(self, item: WorkItem) -> int:
        """Score time criticality (1-10)."""
        priority_scores = {'High': 8, 'Medium': 5, 'Low': 2}
        return priority_scores.get(item.priority, 5)
    
    def _score_risk_reduction(self, item: WorkItem) -> int:
        """Score risk reduction value (1-10)."""
        if item.category in ['security', 'bug-fix']:
            return 8
        elif item.category in ['technical-debt', 'dependency-update']:
            return 6
        return 3
    
    def _score_impact(self, item: WorkItem) -> int:
        """Score implementation impact (1-10)."""
        return self._score_business_value(item)
    
    def _score_confidence(self, item: WorkItem) -> int:
        """Score execution confidence (1-10)."""
        effort_confidence = {0.5: 9, 1.0: 8, 2.0: 7, 3.0: 6, 4.0: 5, 5.0: 4}
        return effort_confidence.get(item.effort_estimate, 5)
    
    def _score_ease(self, item: WorkItem) -> int: 
        """Score implementation ease (1-10)."""
        return max(1, 11 - int(item.effort_estimate * 2))
    
    def _score_technical_debt(self, item: WorkItem) -> int:
        """Score technical debt impact (1-100)."""
        if item.category == 'technical-debt':
            return 80
        elif item.category in ['code-quality', 'modernization']:
            return 60
        elif item.category == 'bug-fix':
            return 40  
        return 20
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range."""
        if max_val == min_val:
            return 50.0
        return max(0, min(100, ((score - min_val) / (max_val - min_val)) * 100))
    
    def update_backlog(self, work_items: List[WorkItem]) -> None:
        """Update BACKLOG.md with discovered and scored work items."""
        content = []
        content.append("# 📊 Autonomous Value Discovery Backlog")
        content.append("")
        content.append(f"Last Updated: {datetime.now().isoformat()}")
        content.append(f"Items Discovered: {len(work_items)}")
        content.append("")
        
        if work_items:
            content.append("## 🎯 Next Best Value Item")
            top_item = work_items[0]
            content.append(f"**[{top_item.id}] {top_item.title}**")
            content.append(f"- **Composite Score**: {top_item.composite_score:.1f}")
            content.append(f"- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.0f} | **Tech Debt**: {top_item.technical_debt_score:.0f}")
            content.append(f"- **Estimated Effort**: {top_item.effort_estimate} hours")
            content.append(f"- **Category**: {top_item.category} | **Priority**: {top_item.priority}")
            content.append(f"- **Description**: {top_item.description}")
            content.append("")
            
            content.append("## 📋 Top 10 Backlog Items")
            content.append("")
            content.append("| Rank | ID | Title | Score | Category | Priority | Est. Hours |")
            content.append("|------|-----|--------|---------|----------|----------|------------|")
            
            for i, item in enumerate(work_items[:10], 1):
                title_short = item.title[:40] + "..." if len(item.title) > 40 else item.title
                content.append(f"| {i} | {item.id} | {title_short} | {item.composite_score:.1f} | {item.category} | {item.priority} | {item.effort_estimate} |")
            
            content.append("")
            
            # Category breakdown
            categories = {}
            for item in work_items:
                categories[item.category] = categories.get(item.category, 0) + 1
            
            content.append("## 📈 Discovery Summary")
            content.append("")
            content.append("### Items by Category")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                content.append(f"- **{category.title()}**: {count} items")
            
            content.append("")
            content.append("### Items by Priority")
            priorities = {}
            for item in work_items:
                priorities[item.priority] = priorities.get(item.priority, 0) + 1
            for priority, count in sorted(priorities.items()):
                content.append(f"- **{priority}**: {count} items")
        
        # Write to backlog file
        backlog_content = "\n".join(content)
        
        # Append to existing backlog instead of replacing
        if self.backlog_path.exists():
            with open(self.backlog_path, 'r') as f:
                existing_content = f.read()
            
            # Insert new backlog at the top after the first header
            lines = existing_content.split('\n')
            header_found = False
            insert_index = 0
            
            for i, line in enumerate(lines):
                if line.startswith('# ') and header_found:
                    insert_index = i
                    break
                elif line.startswith('# '):
                    header_found = True
            
            if insert_index > 0:
                lines.insert(insert_index, "\n" + backlog_content + "\n")
                final_content = "\n".join(lines)
            else:
                final_content = backlog_content + "\n\n" + existing_content
        else:
            final_content = backlog_content
            
        with open(self.backlog_path, 'w') as f:
            f.write(final_content)
        
        logger.info(f"Updated backlog with {len(work_items)} items")
    
    def save_metrics(self, work_items: List[WorkItem]) -> None:
        """Save value discovery metrics."""
        self.metrics["backlogMetrics"] = {
            "totalItems": len(work_items),
            "averageAge": 0,  # All items are new
            "debtRatio": len([i for i in work_items if i.category == 'technical-debt']) / max(len(work_items), 1),
            "velocityTrend": "discovering",
            "lastDiscovery": datetime.now().isoformat(),
            "topScore": work_items[0].composite_score if work_items else 0,
            "categories": {item.category: 1 for item in work_items}
        }
        
        # Ensure directory exists
        self.metrics_path.parent.mkdir(exist_ok=True)
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def run_discovery(self) -> List[WorkItem]:
        """Run the complete value discovery process."""
        logger.info("Starting autonomous value discovery...")
        
        # Discover work items
        work_items = self.discover_work_items()
        logger.info(f"Discovered {len(work_items)} work items")
        
        # Calculate scores
        work_items = self.calculate_scores(work_items)
        logger.info("Calculated scores for all items")
        
        # Update backlog
        self.update_backlog(work_items)
        
        # Save metrics
        self.save_metrics(work_items)
        
        logger.info("Value discovery complete")
        return work_items

def main():
    """Main entry point for value discovery."""
    engine = ValueDiscoveryEngine()
    work_items = engine.run_discovery()
    
    if work_items:
        print(f"\n🎯 Next Best Value Item:")
        top_item = work_items[0]
        print(f"   {top_item.title}")
        print(f"   Score: {top_item.composite_score:.1f} | Effort: {top_item.effort_estimate}h")
        print(f"   Category: {top_item.category} | Priority: {top_item.priority}")
    else:
        print("\n✅ No high-value work items discovered at this time")

if __name__ == "__main__":
    main()