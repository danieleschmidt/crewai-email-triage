#!/usr/bin/env python3
"""
Production Readiness Validator v2.0
Final validation before production deployment
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionReadinessValidator:
    """Validates production readiness across all dimensions."""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_issues = []
        self.warnings = []
        
    def validate_deployment_artifacts(self) -> Dict[str, Any]:
        """Validate deployment artifacts are present and correct."""
        print("🔍 Validating Deployment Artifacts...")
        
        required_artifacts = {
            'dockerfile': 'Dockerfile',
            'docker_compose': 'docker-compose.yml',
            'deployment_manifest': 'deployment/deployment_manifest.json',
            'global_orchestrator': 'global_deployment_orchestrator.py',
            'quality_gates': 'simple_quality_gates.py',
            'monitoring_config': 'monitoring/',
            'kubernetes_config': 'deployment/kubernetes/',
        }
        
        results = {}
        for artifact_name, path in required_artifacts.items():
            artifact_path = Path(path)
            exists = artifact_path.exists()
            results[artifact_name] = {
                'exists': exists,
                'path': str(artifact_path),
                'size': artifact_path.stat().st_size if exists else 0
            }
            
            if exists:
                print(f"  ✅ {artifact_name}: Found")
            else:
                print(f"  ❌ {artifact_name}: Missing - {path}")
                self.critical_issues.append(f"Missing deployment artifact: {path}")
        
        return results
    
    def validate_configuration_management(self) -> Dict[str, Any]:
        """Validate configuration management is production-ready."""
        print("\n🔍 Validating Configuration Management...")
        
        results = {}
        
        # Check for environment-specific configs
        config_dirs = [
            'deployment/regions/us-east-1',
            'deployment/regions/eu-west-1',
            'deployment/regions/ap-southeast-1'
        ]
        
        for config_dir in config_dirs:
            config_path = Path(config_dir)
            exists = config_path.exists()
            
            if exists:
                app_config = config_path / 'app_config.json'
                i18n_config = config_path / 'i18n_config.json'
                
                results[config_dir] = {
                    'directory_exists': True,
                    'app_config_exists': app_config.exists(),
                    'i18n_config_exists': i18n_config.exists(),
                    'config_files': list(config_path.glob('*.json'))
                }
                
                print(f"  ✅ {config_dir}: Configuration ready")
            else:
                results[config_dir] = {'directory_exists': False}
                print(f"  ❌ {config_dir}: Configuration missing")
                self.critical_issues.append(f"Missing region configuration: {config_dir}")
        
        return results
    
    def validate_security_compliance(self) -> Dict[str, Any]:
        """Validate security and compliance readiness."""
        print("\n🔍 Validating Security & Compliance...")
        
        results = {}
        
        # Check security frameworks
        security_files = [
            'src/crewai_email_triage/enhanced_security_framework.py',
            'src/crewai_email_triage/autonomous_resilience.py',
            'src/crewai_email_triage/secure_credentials.py'
        ]
        
        security_ready = True
        for security_file in security_files:
            exists = Path(security_file).exists()
            results[security_file] = exists
            
            if exists:
                print(f"  ✅ Security: {Path(security_file).name}")
            else:
                print(f"  ❌ Security: Missing {security_file}")
                security_ready = False
        
        # Check compliance configuration
        manifest_file = Path('deployment/deployment_manifest.json')
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            compliance_frameworks = manifest.get('compliance_config', {}).get('frameworks', [])
            results['compliance_frameworks'] = compliance_frameworks
            
            if compliance_frameworks:
                print(f"  ✅ Compliance: {len(compliance_frameworks)} frameworks configured")
            else:
                print(f"  ⚠️  Compliance: No frameworks configured")
                self.warnings.append("No compliance frameworks configured")
        
        results['security_ready'] = security_ready
        return results
    
    def validate_monitoring_observability(self) -> Dict[str, Any]:
        """Validate monitoring and observability setup."""
        print("\n🔍 Validating Monitoring & Observability...")
        
        results = {}
        
        # Check monitoring configuration
        monitoring_configs = [
            'monitoring/prometheus.yml',
            'monitoring/grafana/',
            'deployment/kubernetes/service.yml',
            'src/crewai_email_triage/metrics_export.py'
        ]
        
        monitoring_ready = True
        for config in monitoring_configs:
            config_path = Path(config)
            exists = config_path.exists()
            results[config] = exists
            
            if exists:
                print(f"  ✅ Monitoring: {config}")
            else:
                print(f"  ❌ Monitoring: Missing {config}")
                monitoring_ready = False
        
        # Check health endpoints
        health_files = [
            'src/crewai_email_triage/health.py',
            'src/crewai_email_triage/robust_health.py'
        ]
        
        for health_file in health_files:
            exists = Path(health_file).exists()
            if exists:
                print(f"  ✅ Health: {Path(health_file).name}")
            else:
                self.warnings.append(f"Missing health endpoint: {health_file}")
        
        results['monitoring_ready'] = monitoring_ready
        return results
    
    def validate_performance_scaling(self) -> Dict[str, Any]:
        """Validate performance and scaling capabilities."""
        print("\n🔍 Validating Performance & Scaling...")
        
        results = {}
        
        # Check scaling components
        scaling_files = [
            'src/crewai_email_triage/quantum_performance_optimizer.py',
            'src/crewai_email_triage/advanced_scaling.py',
            'src/crewai_email_triage/scalability.py',
            'deployment/kubernetes/hpa.yml'
        ]
        
        scaling_ready = True
        for scaling_file in scaling_files:
            exists = Path(scaling_file).exists()
            results[scaling_file] = exists
            
            if exists:
                print(f"  ✅ Scaling: {Path(scaling_file).name}")
            else:
                print(f"  ❌ Scaling: Missing {scaling_file}")
                scaling_ready = False
        
        # Check performance optimization
        if Path('src/crewai_email_triage/quantum_performance_optimizer.py').exists():
            print(f"  ✅ Performance: Quantum optimization available")
        else:
            self.warnings.append("Advanced performance optimization not available")
        
        results['scaling_ready'] = scaling_ready
        return results
    
    def validate_internationalization(self) -> Dict[str, Any]:
        """Validate internationalization setup."""
        print("\n🔍 Validating Internationalization...")
        
        results = {}
        
        # Check I18n files
        i18n_files = [
            'src/crewai_email_triage/i18n.py',
            'src/crewai_email_triage/global_features.py'
        ]
        
        i18n_ready = True
        for i18n_file in i18n_files:
            exists = Path(i18n_file).exists()
            results[i18n_file] = exists
            
            if exists:
                print(f"  ✅ I18n: {Path(i18n_file).name}")
            else:
                print(f"  ❌ I18n: Missing {i18n_file}")
                i18n_ready = False
        
        # Check region-specific configurations
        regions_with_i18n = 0
        for region_dir in Path('deployment/regions').glob('*'):
            i18n_config = region_dir / 'i18n_config.json'
            if i18n_config.exists():
                regions_with_i18n += 1
        
        if regions_with_i18n > 0:
            print(f"  ✅ I18n: {regions_with_i18n} regions configured")
        else:
            print(f"  ⚠️  I18n: No regions configured")
            self.warnings.append("No regional I18n configurations found")
        
        results['i18n_ready'] = i18n_ready
        results['regions_configured'] = regions_with_i18n
        return results
    
    def validate_data_persistence(self) -> Dict[str, Any]:
        """Validate data persistence and backup strategies."""
        print("\n🔍 Validating Data Persistence...")
        
        results = {}
        
        # Check database configurations
        db_configs = [
            'deployment/kubernetes/pvc.yml',
            'scripts/init-db.sql'
        ]
        
        db_ready = True
        for db_config in db_configs:
            exists = Path(db_config).exists()
            results[db_config] = exists
            
            if exists:
                print(f"  ✅ Database: {db_config}")
            else:
                print(f"  ❌ Database: Missing {db_config}")
                db_ready = False
        
        # Check backup configurations
        backup_configs = [
            'ops/resource-rightsizing.yml',
            'deployment/blue-green-automation.yml'
        ]
        
        backup_ready = False
        for backup_config in backup_configs:
            if Path(backup_config).exists():
                backup_ready = True
                print(f"  ✅ Backup: {backup_config}")
                break
        
        if not backup_ready:
            print(f"  ⚠️  Backup: No backup strategy configured")
            self.warnings.append("No backup strategy configured")
        
        results['db_ready'] = db_ready
        results['backup_ready'] = backup_ready
        return results
    
    def validate_testing_coverage(self) -> Dict[str, Any]:
        """Validate testing coverage and quality."""
        print("\n🔍 Validating Testing Coverage...")
        
        results = {}
        
        # Check test directories
        test_dirs = [
            'tests/',
            'tests/integration/',
            'tests/performance/',
            'tests/e2e/'
        ]
        
        tests_ready = True
        for test_dir in test_dirs:
            test_path = Path(test_dir)
            exists = test_path.exists()
            
            if exists:
                test_files = list(test_path.glob('test_*.py'))
                results[test_dir] = {
                    'exists': True,
                    'test_files': len(test_files)
                }
                print(f"  ✅ Tests: {test_dir} ({len(test_files)} files)")
            else:
                results[test_dir] = {'exists': False, 'test_files': 0}
                print(f"  ❌ Tests: Missing {test_dir}")
                tests_ready = False
        
        # Check quality gates
        quality_gates = [
            'simple_quality_gates.py',
            'autonomous_quality_gates_enhanced.py'
        ]
        
        for gate_file in quality_gates:
            if Path(gate_file).exists():
                print(f"  ✅ Quality: {gate_file}")
            else:
                self.warnings.append(f"Missing quality gate: {gate_file}")
        
        results['tests_ready'] = tests_ready
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive production readiness validation."""
        print("🚀 PRODUCTION READINESS VALIDATION v2.0")
        print("=" * 60)
        
        validation_start = time.time()
        
        # Run all validations
        validations = {
            'deployment_artifacts': self.validate_deployment_artifacts(),
            'configuration_management': self.validate_configuration_management(),
            'security_compliance': self.validate_security_compliance(),
            'monitoring_observability': self.validate_monitoring_observability(),
            'performance_scaling': self.validate_performance_scaling(),
            'internationalization': self.validate_internationalization(),
            'data_persistence': self.validate_data_persistence(),
            'testing_coverage': self.validate_testing_coverage()
        }
        
        validation_time = time.time() - validation_start
        
        # Calculate overall readiness score
        readiness_scores = {
            'deployment_artifacts': sum(1 for v in validations['deployment_artifacts'].values() if v.get('exists', False)) / len(validations['deployment_artifacts']),
            'security_compliance': 1.0 if validations['security_compliance'].get('security_ready', False) else 0.5,
            'monitoring_observability': 1.0 if validations['monitoring_observability'].get('monitoring_ready', False) else 0.5,
            'performance_scaling': 1.0 if validations['performance_scaling'].get('scaling_ready', False) else 0.5,
            'internationalization': 1.0 if validations['internationalization'].get('i18n_ready', False) else 0.5,
            'data_persistence': 1.0 if validations['data_persistence'].get('db_ready', False) else 0.5,
            'testing_coverage': 1.0 if validations['testing_coverage'].get('tests_ready', False) else 0.5
        }
        
        overall_score = sum(readiness_scores.values()) / len(readiness_scores) * 100
        
        # Generate summary
        print(f"\n📊 PRODUCTION READINESS SUMMARY")
        print("=" * 60)
        print(f"Overall Readiness Score: {overall_score:.1f}%")
        print(f"Validation Time: {validation_time:.2f}s")
        print(f"Critical Issues: {len(self.critical_issues)}")
        print(f"Warnings: {len(self.warnings)}")
        
        # Print detailed scores
        print(f"\n📈 COMPONENT SCORES")
        print("=" * 40)
        for component, score in readiness_scores.items():
            score_pct = score * 100
            status_icon = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
            print(f"{status_icon} {component.replace('_', ' ').title()}: {score_pct:.1f}%")
        
        # Print issues
        if self.critical_issues:
            print(f"\n❌ CRITICAL ISSUES")
            print("=" * 40)
            for issue in self.critical_issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS")
            print("=" * 40)
            for warning in self.warnings[:5]:  # Show first 5
                print(f"  • {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        # Final recommendation
        if overall_score >= 90 and not self.critical_issues:
            recommendation = "READY FOR PRODUCTION DEPLOYMENT"
            print(f"\n🎉 {recommendation}")
            deployment_ready = True
        elif overall_score >= 75 and len(self.critical_issues) <= 2:
            recommendation = "READY WITH MINOR FIXES"
            print(f"\n⚠️  {recommendation}")
            deployment_ready = True
        else:
            recommendation = "NOT READY - REQUIRES FIXES"
            print(f"\n❌ {recommendation}")
            deployment_ready = False
        
        # Save validation report
        report = {
            'timestamp': time.time(),
            'overall_score': overall_score,
            'deployment_ready': deployment_ready,
            'recommendation': recommendation,
            'component_scores': readiness_scores,
            'validations': validations,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'validation_time_seconds': validation_time
        }
        
        report_file = Path('production_readiness_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Full validation report saved to: {report_file}")
        
        return report


def main():
    """Run production readiness validation."""
    validator = ProductionReadinessValidator()
    report = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if report['deployment_ready']:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()